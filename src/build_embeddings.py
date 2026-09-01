# -*- coding: utf-8 -*-

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
    slug = domain.replace(" ", "_")
    path = os.path.join(PROMPT_DIR, f"item_instruction_{slug}.txt")
    if not os.path.exists(path):
        available = sorted(
            f[len("item_instruction_"):-len(".txt")]
            for f in os.listdir(PROMPT_DIR)
            if f.startswith("item_instruction_") and f.endswith(".txt")
        )
        raise FileNotFoundError(
            f"[embed] prompts/item_instruction_{slug}.txt not found (--domain '{domain}'). "
            f"available domains: {available} — create that file first for a new domain.")
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
            raise ValueError("a review_text / text or review_texts column is required.")

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
                item_texts[iid] = instruction + "\n\n" + card_text
                n_card += 1

        print(f"[Card coverage] {n_card:,}/{num_items:,} items got LLM card "
              f"({max(0, n_review_pool - n_card):,} fell back to review pool)")

    for iid in range(1, num_items + 1):
        if not item_texts[iid]:
            item_texts[iid] = empty_item_text

    item_texts[0] = "padding item"

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
                         help="dataset key (books / video_games / beauty); --data_dir, --out_dir, "
                              "--card_path and --domain are derived from it.")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="defaults to data/preprocessed/{dataset}/processed")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="defaults to data/preprocessed/{dataset}/embeddings")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_reviews_per_item", type=int, default=5)
    parser.add_argument("--max_chars_per_review", type=int, default=500)
    parser.add_argument("--card_path", type=str, default=None,
                         help="output of semantic_card.py; defaults to "
                              "data/preprocessed/{dataset}/cards.jsonl (without it, only the review "
                              "pool text is encoded)")
    parser.add_argument("--no_card", action="store_true",
                         help="encode the review pool text only, without cards (the w/o LLM-Card ablation)")
    parser.add_argument("--domain", type=str, default=None,
                         help="domain of the item instruction text; reads "
                              "prompts/item_instruction_{domain}.txt. Derived from --dataset when omitted.")
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
            print(f"[warn] no card file, encoding the review pool only: {cp}")
    instruction = load_item_instruction(domain)
    empty_item_text = f"general {domain} recommendation"
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
        prefix="passage: ",
    )

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
