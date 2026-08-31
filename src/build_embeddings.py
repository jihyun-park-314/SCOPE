# -*- coding: utf-8 -*-
"""
[embed] prepare_dataset.py 산출물 + semantic_card.py 카드 -> 동결 텍스트 인코더 임베딩.

산출(--out_dir):
  item_embs.npy, item_texts.json, embedding_meta.json,
  {train,valid,test}_instances.jsonl, {train,valid,test}_query_embs.npy

[도메인 하드코딩 제거]
예전 이 스크립트는 아이템 텍스트 앞머리(instruction)와 빈 아이템 폴백 문구에 "book"을
하드코딩하고 있었다("Represent this book ..." / "general book recommendation") — Books를
인코딩할 때만 우연히 맞는 문구라, 같은 스크립트를 Video_Games/Beauty에 쓰면 조용히 잘못된
instruction으로 인코딩됐다(구 HALO의 final/build_halo_lite_embeddings.py에서 발견된 것과 동일 버그).
이제 문구는 prompts/item_instruction_{domain}.txt에서 읽고, 도메인은 --domain 또는
--dataset에서 유도한다. 3개 instruction 파일 모두 동봉된 임베딩을 만든 문자열과
byte-identical임을 실측 확인했으므로 재실행해도 같은 텍스트가 인코딩된다.

[아이템 한 개가 실제로 인코딩되는 문자열]
  "passage: " + instruction + "\n\n" + (카드 본문  또는  리뷰 풀 조각들)
  · 카드가 있으면 **리뷰 풀 텍스트를 덮어쓴다**(카드 우선, build_item_texts 참조).
  · 카드도 리뷰도 없는 아이템만 empty_item_text가 되는데, 이 경로에만 instruction이 붙지 않는다.
  · index 0은 "padding item"이지만 인코딩 직후 item_embs[0] = 0으로 덮이므로 값 자체는 무의미하다.
  · "passage: "/"query: "는 E5 규약이라 도메인이 아니라 **모델**에 종속된다 — --model_name을
    prefix를 쓰지 않는 모델로 바꾸면 이 접두사가 그대로 붙는다(현재 검사하지 않음).
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

from config import (PROMPT_DIR, cards_path, domain_of,
                    embed_dir as default_embed_dir, processed_dir as default_processed_dir)
from utils import load_jsonl


def load_item_instruction(domain: str) -> str:
    """아이템 텍스트 앞에 붙일 instruction을 prompts/item_instruction_{domain}.txt에서 읽는다.
    review2query.py/semantic_card.py의 프롬프트 파일 관례와 동일 — 없으면 즉시 에러."""
    slug = domain.replace(" ", "_")
    path = os.path.join(PROMPT_DIR, f"item_instruction_{slug}.txt")
    if not os.path.exists(path):
        available = sorted(
            f[len("item_instruction_"):-len(".txt")]
            for f in os.listdir(PROMPT_DIR)
            if f.startswith("item_instruction_") and f.endswith(".txt")
        )
        raise FileNotFoundError(
            f"[embed] prompts/item_instruction_{slug}.txt 없음 (--domain '{domain}'). "
            f"사용 가능한 도메인: {available} — 새 도메인이면 그 파일을 먼저 만드세요.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def safe_text(x):
    if x is None:
        return ""
    return str(x).strip()


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "to_list"):
        try:
            return x.to_list()
        except Exception:
            pass
    if isinstance(x, str):
        return [x] if x.strip() else []
    return [x]


def review_piece(title: str, text: str, max_chars: int):
    """리뷰 한 건의 텍스트 조각. 제목·본문이 둘 다 비면 None (호출부에서 건너뛴다).
    aggregated / row-level 두 분기에 같은 코드가 복붙돼 있던 것을 합친 것으로, 조립 규칙은 동일하다."""
    if not title and not text:
        return None
    piece = ""
    if title:
        piece += f"Review title: {title}\n"
    if text:
        piece += f"Review text: {text[:max_chars]}"
    return piece.strip()


def load_card_texts(card_path, item_map):
    asin_to_iid = dict(zip(item_map["item_id"].to_list(), item_map["iid"].to_list()))

    card_texts = {}
    for obj in load_jsonl(card_path):
        iid = asin_to_iid.get(obj["asin"])
        if iid is not None:
            card_texts[iid] = safe_text(obj["card"])

    return card_texts


def build_item_texts(pool_path, item_map_path, instruction, empty_item_text,
                     max_reviews_per_item=5, max_chars_per_review=500, card_path=None):
    df = pl.read_parquet(pool_path)
    item_map = pl.read_parquet(item_map_path)
    num_items = int(item_map["iid"].max())

    print("[Item pool columns]", df.columns)
    print("[num_items]", num_items)

    item_texts = [""] * (num_items + 1)

    if "review_texts" in df.columns:
        print("[Info] aggregated item pool detected")

        for row in df.iter_rows(named=True):
            iid = int(row["iid"])
            titles = as_list(row.get("review_titles", []))
            texts = as_list(row.get("review_texts", []))

            pieces = []
            n = min(len(texts), max_reviews_per_item)

            for i in range(n):
                title = safe_text(titles[i]) if i < len(titles) else ""
                text = safe_text(texts[i])
                piece = review_piece(title, text, max_chars_per_review)
                if piece is None:
                    continue
                pieces.append(piece)

            if pieces:
                item_texts[iid] = instruction + "\n\n" + "\n\n".join(pieces)

    else:
        print("[Info] row-level item pool detected")

        title_col = None
        for c in ["review_title", "title"]:
            if c in df.columns:
                title_col = c
                break

        text_col = None
        for c in ["review_text", "text"]:
            if c in df.columns:
                text_col = c
                break

        if text_col is None:
            raise ValueError("review_text/text 또는 review_texts 컬럼이 필요합니다.")

        item_to_reviews = defaultdict(list)

        selected = ["iid", text_col]
        if title_col is not None:
            selected.append(title_col)

        for row in df.select(selected).iter_rows(named=True):
            iid = int(row["iid"])
            title = safe_text(row.get(title_col, "")) if title_col else ""
            text = safe_text(row.get(text_col, ""))
            piece = review_piece(title, text, max_chars_per_review)
            if piece is None:
                continue

            if len(item_to_reviews[iid]) < max_reviews_per_item:
                item_to_reviews[iid].append(piece)

        for iid in range(1, num_items + 1):
            pieces = item_to_reviews.get(iid, [])
            if pieces:
                item_texts[iid] = instruction + "\n\n" + "\n\n".join(pieces)

    n_review_pool = sum(1 for x in item_texts[1:] if x)

    if card_path is not None:
        print("[Info] LLM semantic card detected:", card_path)
        card_texts = load_card_texts(card_path, item_map)

        n_card = 0
        for iid, card_text in card_texts.items():
            if 1 <= iid <= num_items and card_text:
                # ★ 카드가 있으면 위에서 만든 리뷰 풀 텍스트를 **덮어쓴다**(카드 우선).
                #   리뷰 풀은 카드가 없는 아이템의 폴백으로만 남는다.
                item_texts[iid] = instruction + "\n\n" + card_text
                n_card += 1

        print(f"[Card coverage] {n_card:,}/{num_items:,} items got LLM card "
              f"({max(0, n_review_pool - n_card):,} fell back to review pool)")

    # 카드도 리뷰도 없는 아이템. ★ 이 경로에만 instruction이 붙지 않는다(위 세 경로와 비대칭).
    for iid in range(1, num_items + 1):
        if not item_texts[iid]:
            item_texts[iid] = empty_item_text

    item_texts[0] = "padding item"   # 인코딩 직후 item_embs[0] = 0으로 덮이므로 값 자체는 무의미

    print("[items with text]", sum(1 for x in item_texts[1:] if x != empty_item_text))

    return item_texts, num_items


def encode_texts(model, texts, batch_size, prefix):
    prefixed = [prefix + t for t in texts]
    emb = model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.astype(np.float32)


def save_queries_and_embeddings(model, instance_path, out_jsonl, out_npy, batch_size):
    rows = load_jsonl(instance_path)

    kept = []
    queries = []

    for obj in rows:
        q = safe_text(obj.get("query", ""))
        if not q:
            continue
        kept.append(obj)
        queries.append(q)

    print(f"[Queries] {instance_path}: {len(kept):,}")

    q_embs = encode_texts(
        model=model,
        texts=queries,
        batch_size=batch_size,
        prefix="query: ",
    )

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    np.save(out_npy, q_embs)

    print("[Saved]", out_jsonl)
    print("[Saved]", out_npy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                         help="데이터셋 키 (books / video_games / beauty). "
                              "--data_dir/--out_dir/--card_path/--domain을 여기서 유도한다.")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="미지정 시 data/preprocessed/{dataset}/processed")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="미지정 시 data/preprocessed/{dataset}/embeddings")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_reviews_per_item", type=int, default=5)
    parser.add_argument("--max_chars_per_review", type=int, default=500)
    parser.add_argument("--card_path", type=str, default=None,
                         help="semantic_card.py 산출물. 미지정 시 "
                              "data/preprocessed/{dataset}/cards.jsonl (없으면 카드 없이 리뷰 풀만 사용)")
    parser.add_argument("--no_card", action="store_true",
                         help="카드를 쓰지 않고 리뷰 풀 텍스트만으로 인코딩 (w/o LLM-Card 절제)")
    parser.add_argument("--domain", type=str, default=None,
                         help="아이템 instruction 문구의 도메인 — prompts/item_instruction_{domain}.txt를 "
                              "읽는다. 미지정 시 --dataset에서 유도 (books->book).")
    args = parser.parse_args()

    domain = args.domain or domain_of(args.dataset)
    args.data_dir = args.data_dir or default_processed_dir(args.dataset)
    args.out_dir = args.out_dir or default_embed_dir(args.dataset)
    if args.no_card:
        args.card_path = None
    elif args.card_path is None:
        cp = cards_path(args.dataset)
        args.card_path = cp if os.path.exists(cp) else None
        if args.card_path is None:
            print(f"[Warn] 카드 파일이 없어 리뷰 풀만 사용합니다: {cp}")
    instruction = load_item_instruction(domain)
    empty_item_text = f"general {domain} recommendation"  # review2query.py의 FALLBACK_TEXT와 동일 관례
    print(f"[Domain] {domain}\n[Instruction] {instruction}\n[Empty-item text] {empty_item_text}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)
    print("[Model]", args.model_name)

    model = SentenceTransformer(args.model_name, device=device)

    item_texts, num_items = build_item_texts(
        pool_path=data_dir / "item_card_review_pool_train_only.parquet",
        item_map_path=data_dir / "item_map.parquet",
        instruction=instruction,
        empty_item_text=empty_item_text,
        max_reviews_per_item=args.max_reviews_per_item,
        max_chars_per_review=args.max_chars_per_review,
        card_path=args.card_path,
    )

    print("[Encode] item texts")
    item_embs = encode_texts(
        model=model,
        texts=item_texts,
        batch_size=args.batch_size,
        prefix="passage: ",   # E5 규약. --model_name을 바꿔도 이 접두사는 따라 바뀌지 않는다
    )

    # padding row는 0으로 둔다.
    item_embs[0] = 0.0

    np.save(out_dir / "item_embs.npy", item_embs)

    with open(out_dir / "item_texts.json", "w", encoding="utf-8") as f:
        json.dump(item_texts, f, ensure_ascii=False, indent=2)

    meta = {
        "model_name": args.model_name,
        "num_items": num_items,
        "emb_dim": int(item_embs.shape[1]),
        "dataset": args.dataset,
        "domain": domain,
        "item_instruction": instruction,
        "card_path": args.card_path,
    }

    with open(out_dir / "embedding_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[Saved]", out_dir / "item_embs.npy")

    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}_query_instances.jsonl"
        if not path.exists():
            print("[Skip]", path)
            continue

        save_queries_and_embeddings(
            model=model,
            instance_path=path,
            out_jsonl=out_dir / f"{split}_instances.jsonl",
            out_npy=out_dir / f"{split}_query_embs.npy",
            batch_size=args.batch_size,
        )

    print("[Done]")


if __name__ == "__main__":
    main()