"""
[card] 아이템 시맨틱 카드 생성 (prompt3.md §0-(2)/(3)/(4), §3).

src/ 파이프라인에서의 위치: `preprocessing.py`가 만든 `interactions.pkl`(아이템 목록)과
`split_manifest.json`(valid/test 타깃 리뷰 제외 키, immutable)만 참조한다 —
split을 독립적으로 재계산하지 않는다. Ollama 호출은 `src/ollama_client.py`를 통한다.

핵심 설계(구 HALO 저장소의 `03_semantic_cards.py`에서 무수정 이식 — 이 저장소에는 없다):
  · 카드 원본은 샘플이 아니라 원본 전체(--reviews/--meta, 기본값은 --dataset에서 유도한
    data/raw/{원본카테고리}_{reviews,meta}.parquet)에서 가져온다 — 추천 시점에 이미 존재하는
    다른 유저들의 리뷰이므로 누수가 아니다(§0-(2)).
  · stream_reviews_by_item: reserve 캡으로 "parquet 순서상 먼저 나온 리뷰"만 보던 편향을 없애고,
    min-heap(card_pool_cap)으로 helpful_vote 상위를 유지하며 아이템의 전체 리뷰 풀을 본다(§0-(3)).
  · valid/test 타깃으로 쓰인 리뷰 행(R_ui)은 그 아이템의 카드 풀에서만 행 단위로 제외한다(§0-(4)).
    R_ui_selected_rate: 제외하지 않았다면 top-8 후보에 실제로 들었을 비율(측정, 가정 아님).
  · 카드는 asin으로 저장/재개한다(iid는 샘플마다 바뀌므로 카드 캐시가 asin 기준이어야 재사용 가능).

Ollama 호출(재시도·동시성·URL 파싱·진행표시)은 src/ollama_client.py가 담당한다 — review2query.py와
같은 코드를 쓰고, 단계별로 다른 것은 프롬프트와 max_new_tokens(카드 300 / 쿼리 220)뿐이다.

프롬프트: prompts/card_prompt_{domain}.txt 하나만 필요하다. LLM이 빈 응답을 준 아이템에 쓸
       폴백 카드는 그 프롬프트의 필드명에서 생성하므로 별도 파일이 없다(build_fallback_card).

실행:  python src/semantic_card.py --dataset books
       (--domain/--reviews/--meta/--pkl/--manifest는 전부 --dataset에서 유도된다.
        원본이 다른 카테고리로 새는 사고를 막으려고, 예전처럼 config에 하드코딩된
        Video_Games 경로로 폴백하지 않고 --dataset에서 data/raw/{원본}_*.parquet를 만든다.)
산출:  data/preprocessed/{dataset}/cards.jsonl
       results/card_stats_{dataset}.json
"""
import argparse
import heapq
import itertools
import json
import os
import pickle
import re
import time
from collections import defaultdict

import ollama_client
import pandas as pd
import pyarrow.parquet as pq
from config import (CFG, PROMPT_DIR, cards_path, dataset_root, domain_of,
                    interactions_path, manifest_path as default_manifest_path,
                    meta_fields_of, raw_paths, stats_path)
from utils import norm_text, sha1_16


def load_card_prompt_template(domain: str) -> str:
    """도메인별 카드 프롬프트를 prompts/card_prompt_{domain}.txt에서 읽는다 (도메인의 공백은 밑줄로
    치환). review2query.py의 --domain 관례를 그대로 따르되, 프롬프트 문구 자체는 코드가 아니라
    검토/수정 가능한 텍스트 파일로 관리한다 — 도메인마다 별도 파일이 필요하며, 없으면 즉시 에러."""
    slug = domain.replace(" ", "_")
    path = os.path.join(PROMPT_DIR, f"card_prompt_{slug}.txt")
    if not os.path.exists(path):
        available = sorted(
            f[len("card_prompt_"):-len(".txt")]
            for f in os.listdir(PROMPT_DIR) if f.startswith("card_prompt_") and f.endswith(".txt")
        )
        raise FileNotFoundError(
            f"[card] prompts/card_prompt_{slug}.txt 없음 (--domain '{domain}'). "
            f"사용 가능한 도메인: {available} — 새 도메인이면 prompts/card_prompt_{slug}.txt를 먼저 만드세요."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_card_fields(template: str) -> tuple:
    """카드 프롬프트가 요구하는 필드명을 템플릿에서 그대로 읽는다.

    필드 줄만 `Genre: <...>`처럼 꺾쇠 자리표시자가 따라오므로 `Rules:`/`[METADATA]`는 걸리지 않는다."""
    fields = tuple(re.findall(r"^([A-Za-z][A-Za-z ]*): <", template, flags=re.MULTILINE))
    if not fields:
        raise ValueError(
            "[card] 카드 프롬프트에서 필드를 하나도 찾지 못했습니다. 각 필드는 "
            "`Genre: <설명>`처럼 꺾쇠 자리표시자가 따라오는 줄이어야 합니다.")
    return fields


def build_fallback_card(fields: tuple) -> str:
    """LLM이 빈 응답을 준 아이템에 쓸 카드를 카드 프롬프트의 필드명에서 그대로 생성한다.

    ★ 예전에는 이 텍스트가 prompts/fallback_card_{domain}.txt 3개 파일에 따로 있었고,
    필드명이 card_prompt_와 어긋나면(Beauty에 Genre를 쓰는 식) 실패한 아이템만 다른 스키마의
    카드가 섞였다. 그래서 로드 시점에 두 파일을 대조하는 검사까지 붙어 있었다.
    필드명을 프롬프트에서 바로 가져오면 어긋남 자체가 구조적으로 불가능해지므로, 파일 3개와
    대조 검사를 모두 없앴다 — 도메인을 추가할 때 관리할 파일도 하나 줄어든다.

    값은 전 필드 "unknown"이다. 이 카드가 쓰이는 상황은 정의상 그 아이템에 대해 아무것도
    모르는 경우다. (동봉된 세 데이터셋 67,752장 중 이 경로를 탄 카드는 0장이다.)"""
    return "\n".join(f"{f}: unknown" for f in fields)


def _fmt_meta(row, fields) -> str:
    """카드 프롬프트의 [METADATA] 블록을 만든다. `fields`는 config.meta_fields_of(dataset)에서
    오며 튜플 순서가 곧 출력 순서다 — 예전에는 이 목록이 이 함수 안에 박혀 있어서, 메타 스키마가
    다른 카테고리를 추가하면 없는 컬럼이 row.get()에서 조용히 None으로 건너뛰어졌다.

    title/subtitle이 목록에 없는 것은 의도적이다: title은 아이템 식별자라 쿼리-target 표면 일치
    경로를 만들고(제목 3-gram 공유율 무작위 대비 37배), subtitle은 91.4%가 판형/발행 문자열이다.
    author는 그룹 속성이므로 유지한다.
    주의: 현재 쓰는 halo_lite_newcard 카드는 이 변경 이전(title/subtitle 포함) 입력으로 생성됨."""
    parts = []
    for k in fields:
        v = row.get(k)
        if v is not None and len(str(v)) > 2:
            parts.append(f"{k.capitalize()}: {str(v)[:500]}")
    if row.get("average_rating") is not None:
        parts.append(f"AvgRating: {row['average_rating']} ({row.get('rating_number', '?')} ratings)")
    return "\n".join(parts)


def build_exclusion_keys(P: dict, scope: str):
    """valid/test 타깃 행(R_ui)의 (asin,user_id,timestamp)/(asin,user_id,hash) 키 집합.

    manifest 없이 pkl에서 직접 만드는 하위호환 경로다(--manifest '' 로 명시했을 때만 쓰인다).
    scope가 "eval"/"valid_test"가 아니면 제외 키가 빈 집합이 되어 아무것도 제외되지 않는다 —
    CFG.exclude_scope의 유일한 값이 "eval"이라 현재 그 경로로는 들어가지 않는다."""
    target_qidxs = set()
    if scope in ("eval", "valid_test"):
        for split in ("valid", "test"):
            for _hist, _tgt, q in P[split].values():
                target_qidxs.add(q)
    key1, key2 = set(), set()
    for q in target_qidxs:
        user_id, asin, ts, h = P["inter_meta"][q]
        key1.add((asin, user_id, ts))
        key2.add((asin, user_id, h))
    return key1, key2


def build_exclusion_keys_from_manifest(manifest_path: str):
    """
    preprocessing.py가 만든 immutable manifest(split_manifest.json)에서 R_ui 제외 키를 만든다.
    interactions.pkl을 직접 읽는 build_exclusion_keys()와 동일한
    (asin,user_id,timestamp)/(asin,user_id,hash) 키를 만들지만, prepare_dataset.py가 참조하는
    것과 정확히 같은 manifest 파일을 참조하므로 두 스크립트의 split이 어긋날 수 없다.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    key1, key2 = set(), set()
    for row in manifest["valid"] + manifest["test"]:
        key1.add((row["item_id"], row["user_id"], row["timestamp"]))
        key2.add((row["item_id"], row["user_id"], row["review_hash"]))
    return key1, key2


# 리뷰 한 행의 표현: row = (helpful_vote, rating, title, text, content_hash)
#   [0] select8()의 정렬 키          [1][2][3] format_reviews()의 출력 필드
#   [4] leakage assertion의 비교 키
# 인덱스가 세 함수에 흩어져 있으므로 순서를 바꾸려면 세 곳을 함께 고쳐야 한다.
ROW_HELPFUL_VOTE, ROW_RATING, ROW_TITLE, ROW_TEXT, ROW_HASH = range(5)


def stream_reviews_by_item(path: str, asins: set, pool_cap: int, key1: set, key2: set,
                            chunk_rows: int = 1_000_000):
    """asin별 전체 리뷰 풀(helpful_vote 상위 pool_cap, min-heap)과 R_ui 제외 행을 함께 수집.

    key2(asin,user_id,hash)만으로는 부족하다 — 원본 전체 리뷰(--reviews)에는 콘텐츠 중복 리뷰가
    여전히 남아있다(Amazon 원본 자체의 문제: 같은 물리적 리뷰가 접미사 붙은 다른 user_id로 재등장,
    실측 확인됨). manifest의 타깃 review_hash가 그 dedup 승자 user_id로만 기록되므로, user_id가
    다른 나머지 복제본은 key2로 못 걸러진다 — (asin,hash)만으로 보는 key3(content-only, user_id
    무시)를 추가해야 최종 leakage assertion(코드도 (asin,hash)만 비교)과 제외 기준이 일치한다."""
    key3 = {(asin, h) for (asin, _uid, h) in key2}
    heaps = defaultdict(list)          # asin -> [(helpful_vote, tie, row)]
    excluded = defaultdict(list)       # asin -> [row, ...]  (R_ui, 카드 풀에서 제외)
    counter = itertools.count()
    pf = pq.ParquetFile(path)
    avail = pf.schema.names
    cols = [c for c in ["parent_asin", "user_id", "timestamp", "rating", "title", "text", "helpful_vote"]
            if c in avail]
    seen = 0
    for batch in pf.iter_batches(batch_size=chunk_rows, columns=cols):
        df = batch.to_pandas()
        df = df[df["parent_asin"].isin(asins)]
        for c, default in [("rating", 0.0), ("helpful_vote", 0), ("title", ""), ("text", ""),
                            ("user_id", ""), ("timestamp", 0)]:
            if c not in df.columns:
                df[c] = default
        for r in df.itertuples():
            asin = r.parent_asin
            title = str(getattr(r, "title", ""))[:80]
            text = str(getattr(r, "text", ""))[:400]
            hv = int(getattr(r, "helpful_vote", 0) or 0)
            rating = float(getattr(r, "rating", 0) or 0)
            uid = getattr(r, "user_id", "")
            ts = int(getattr(r, "timestamp", 0) or 0)
            # ★ 해시는 위에서 자른 title/text가 아니라 **절단 전 원문**으로 계산한다 —
            #   preprocessing.build_pkl()의 review_hash와 같은 문자열이어야 manifest의
            #   제외 키(key2/key3)와 매칭된다. 여기서 title/text 변수를 재사용하면 안 된다.
            h = sha1_16(norm_text(f"{getattr(r, 'title', '')}. {getattr(r, 'text', '')}"))
            row = (hv, rating, title, text, h)
            if (asin, uid, ts) in key1 or (asin, uid, h) in key2 or (asin, h) in key3:
                excluded[asin].append(row)
                continue
            heap = heaps[asin]
            item = (hv, next(counter), row)
            if len(heap) < pool_cap:
                heapq.heappush(heap, item)
            else:
                heapq.heappushpop(heap, item)
        seen += len(batch)
        if seen % 5_000_000 < chunk_rows:
            print(f"[card] scanned {seen:,} review rows ...")
    pools = {asin: [row for _, _, row in heap] for asin, heap in heaps.items()}
    return pools, excluded


def select8(rows: list, per_item: int) -> list:
    """helpful_vote 상위 절반 + 나머지에서 균등 간격 다양성 샘플링 (원본과 동일 로직)."""
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda x: -x[0])
    top = rows_sorted[: max(1, per_item // 2)]
    rest = rows_sorted[len(top):]
    step = max(1, len(rest) // max(1, per_item - len(top)))
    div = rest[::step][: per_item - len(top)]
    return top + div


def format_reviews(selected: list) -> list:
    return [f"({r[1]}/5) {r[2]} — {r[3]}" for r in selected]


def clean_card(raw: str, fallback: str) -> str:
    q = (raw or "").strip()
    return q if q else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True,
                     help="데이터셋 키 (books / video_games / beauty).")
    ap.add_argument("--max_items", type=int, default=0, help="테스트용: 대상 아이템 수 제한 (0=전체)")
    ap.add_argument("--skip_llm", action="store_true", help="테스트용: 풀/제외 로직만 확인, LLM 호출 안 함")
    ap.add_argument("--pkl", type=str, default=None,
                     help="preprocessing.py가 만든 pkl 경로. "
                          "기본값: data/preprocessed/{dataset}/interactions.pkl")
    ap.add_argument("--manifest", type=str, default=None,
                     help="preprocessing.py가 만든 split_manifest.json 경로. "
                          "기본값: data/preprocessed/{dataset}/split_manifest.json — prepare_dataset.py가 "
                          "참조하는 것과 동일한 파일이므로 두 스크립트의 split이 어긋날 수 없다. "
                          "빈 문자열('')로 명시하면 pkl 직접 재계산 방식으로 폴백한다. "
                          "manifest가 없으면 preprocessing.py --from_pkl로 먼저 복원할 것.")
    ap.add_argument("--reviews", type=str, default=None,
                     help="카드 원본 리뷰 parquet 경로. 미지정 시 --dataset에서 유도 "
                          "(data/raw/{원본카테고리}_reviews.parquet).")
    ap.add_argument("--meta", type=str, default=None,
                     help="카드 메타데이터 parquet 경로. 미지정 시 --dataset에서 유도 "
                          "(data/raw/{원본카테고리}_meta.parquet).")
    ap.add_argument("--model", type=str, default=CFG.ollama_model, help="Ollama 모델 태그")
    ap.add_argument("--ollama_urls", "--ollama-urls", type=str, default=CFG.ollama_urls,
                     help="Ollama 서버 base URL 목록(쉼표 구분). 예전에는 CLI가 없어 서버를 "
                          "바꾸려면 config.py를 고쳐야 했고, 그래서 쿼리 단계와 카드 단계가 "
                          "서로 다른 서버를 볼 수 있었다.")
    ap.add_argument("--requests_per_server", type=int, default=1,
                     help="Ollama 서버 1대당 동시 요청 수(OLLAMA_NUM_PARALLEL과 맞춰서 설정)")
    ap.add_argument("--domain", type=str, default=None,
                     help="카드 프롬프트에 쓰일 도메인 명사(review2query.py --domain과 동일 관례). "
                          "미지정 시 --dataset에서 유도한다 (books->book, video_games->'video game', "
                          "beauty->'beauty product').")
    args = ap.parse_args()

    ds = args.dataset
    domain = args.domain or domain_of(ds)
    prompt_template = load_card_prompt_template(domain)
    card_fields = parse_card_fields(prompt_template)
    fallback_card = build_fallback_card(card_fields)
    print(f"[card] 카드 필드: {card_fields}  (prompts/card_prompt_{domain.replace(' ', '_')}.txt)")

    pkl_path = args.pkl or interactions_path(ds)
    default_reviews, default_meta = raw_paths(ds)
    reviews_path = args.reviews or default_reviews
    meta_path = args.meta or default_meta
    print(f"[card] dataset={ds}  domain={domain}\n"
          f"       pkl     ={pkl_path}\n"
          f"       reviews ={reviews_path}\n"
          f"       meta    ={meta_path}")

    with open(pkl_path, "rb") as f:
        P = pickle.load(f)

    asins = list(P["i2id"].keys())
    if args.max_items > 0:
        asins = asins[: args.max_items]
    asin_set = set(asins)
    print(f"[card] items to card: {len(asins):,} (dataset={ds})")

    meta = pd.read_parquet(meta_path)
    meta = meta[meta["parent_asin"].isin(asin_set)].set_index("parent_asin")

    # 선언된 메타 컬럼이 실제 parquet에 다 있는지 확인한다. 없으면 그 컬럼은 카드 입력에서
    # 통째로 빠지는데, 예전에는 아무 신호 없이 그렇게 됐다(row.get() -> None -> skip).
    # 카테고리에 따라 실제로 없을 수 있는 컬럼이라 중단하지는 않고 경고만 남긴다.
    meta_fields = meta_fields_of(ds)
    missing_meta = [k for k in meta_fields if k not in meta.columns]
    print(f"[card] 메타 필드: {meta_fields}" +
          (f"  ⚠ parquet에 없어 카드 입력에서 빠짐: {missing_meta}" if missing_meta else ""))

    mf_path = default_manifest_path(ds) if args.manifest is None else args.manifest

    key1, key2 = set(), set()
    if CFG.exclude_query_review_from_card:
        if mf_path and not os.path.exists(mf_path):
            # 예전엔 여기서 open()이 그냥 FileNotFoundError를 던졌다 — 무엇을 해야 하는지
            # 알 수 없는 에러였다. manifest는 split의 단일 진실 공급원이므로, 없으면
            # 복원 방법을 알려주고 멈춘다(조용히 pkl 재계산으로 새는 쪽이 훨씬 위험하다:
            # 22k Books에서 이 어긋남으로 788건 누락 / 112건 실제 leak이 났다).
            raise FileNotFoundError(
                f"[card] manifest가 없습니다: {mf_path}\n"
                f"       preprocessing.py를 아직 안 돌렸다면:\n"
                f"         python src/preprocessing.py --dataset {ds}\n"
                f"       이미 pkl은 있고 manifest만 없다면 (구버전 산출물) 재샘플링하지 말고 복원만:\n"
                f"         python src/preprocessing.py --dataset {ds} --from_pkl\n"
                f"       의도적으로 pkl 직접 재계산을 쓰려면 --manifest '' 로 명시하세요.")
        if mf_path:
            key1, key2 = build_exclusion_keys_from_manifest(mf_path)
            print(f"[card] R_ui 제외 키 source: manifest={mf_path}")
        else:
            key1, key2 = build_exclusion_keys(P, CFG.exclude_scope)
            print(f"[card] R_ui 제외 키 source: {pkl_path} (manifest 미지정 — 하위호환 경로)")
    print(f"[card] R_ui 제외 키: {len(key1):,}건 (scope={CFG.exclude_scope})")
    # (asin, hash) 쌍으로 scope를 좁힌다 — hash만 쓰면 서로 다른 아이템의 리뷰가 우연히
    # 같은 문구(예: "Great game!")를 써서 생기는 hash 충돌을 진짜 leakage로 오탐할 수 있다.
    eval_source_review_ids = {(asin, h) for (asin, _uid, h) in key2}

    t0 = time.time()
    pools, excluded = stream_reviews_by_item(
        reviews_path, asin_set, CFG.card_pool_cap, key1, key2)
    print(f"[card] 원본 리뷰 스캔 완료 ({time.time() - t0:.0f}s) — "
          f"풀 보유 아이템={len(pools):,}, R_ui 보유 아이템={len(excluded):,}")

    # ---- R_ui_selected_rate 진단: 제외하지 않았다면 top-8에 실제로 들었을 비율 ----
    n_with_rui, n_would_be_selected = len(excluded), 0
    for asin, rui_rows in excluded.items():
        full_rows = pools.get(asin, []) + rui_rows
        selected_full = select8(full_rows, CFG.max_reviews_per_item)
        if any(r in rui_rows for r in selected_full):
            n_would_be_selected += 1
    r_ui_selected_rate = (n_would_be_selected / n_with_rui) if n_with_rui else None

    # ---- 카드용 리뷰 선정 (R_ui 제외된 풀 기준) ----
    selected_rows_by_asin = {asin: select8(pools.get(asin, []), CFG.max_reviews_per_item)
                              for asin in asins}
    rev_selected = {asin: format_reviews(rows) for asin, rows in selected_rows_by_asin.items()}
    n_no_review = sum(1 for asin in asins if not rev_selected.get(asin))
    n_capped = sum(1 for asin in asins if len(pools.get(asin, [])) >= CFG.card_pool_cap)

    # ---- leakage 최종 assertion: 카드에 실제로 뽑힌 리뷰와 eval 타깃 리뷰가 절대 겹치면 안 된다 ----
    # (asin, hash)로 scope를 좁혀서, 서로 다른 아이템의 리뷰가 우연히 같은 문구를 써서 생기는
    # hash 충돌을 진짜 leakage로 오탐하지 않도록 한다.
    card_source_review_ids = {(asin, row[4]) for asin, rows in selected_rows_by_asin.items() for row in rows}
    leaked_hashes = eval_source_review_ids & card_source_review_ids
    if leaked_hashes:
        raise AssertionError(
            f"[card][LEAKAGE] eval_source_review_ids ∩ card_source_review_ids = {len(leaked_hashes):,}건 "
            f"(0이어야 함). R_ui 제외 키가 최종 valid/test split과 어긋났을 가능성이 큽니다 — "
            f"--manifest가 prepare_dataset.py와 동일한 파일인지 확인하세요."
        )
    print(f"[card] leakage assertion 통과: eval_source_review_ids({len(eval_source_review_ids):,}) "
          f"∩ card_source_review_ids({len(card_source_review_ids):,}) = 0")

    out_path = cards_path(ds)
    os.makedirs(dataset_root(ds), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["asin"] for line in f}
    todo = [a for a in asins if a not in done]
    print(f"[card] to generate: {len(todo):,} (skipped {len(done):,})")

    if not args.skip_llm and todo:
        urls = ollama_client.parse_urls(args.ollama_urls)
        batch_size = CFG.gemma_batch
        # append 모드 + 배치마다 flush = 체크포인트. 중간에 죽어도 위의 done 집합이
        # 이미 쓴 asin을 걸러내므로 그 지점부터 이어진다.
        with open(out_path, "a", encoding="utf-8") as fout:
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                prompts = []
                for asin in chunk:
                    # meta에 없는 asin은 빈 메타로 간다 — {"title": asin}은 title이 meta_fields에서
                    # 빠진 뒤로 _fmt_meta()가 전부 건너뛰어 빈 문자열이 된다(의도된 현재 동작).
                    m = meta.loc[asin].to_dict() if asin in meta.index else {"title": asin}
                    reviews = rev_selected.get(asin, [])
                    prompts.append(prompt_template.format(
                        meta=_fmt_meta(m, meta_fields), reviews="\n".join(reviews) or "(no reviews)"))
                raw = ollama_client.generate_batch(
                    urls, args.model, prompts, CFG.card_max_new_tokens,
                    desc=f"cards {min(start + batch_size, len(todo))}/{len(todo)}",
                    requests_per_server=args.requests_per_server)
                cards = [clean_card(r, fallback_card) for r in raw]
                for asin, card in zip(chunk, cards):
                    fout.write(json.dumps({"asin": asin, "card": card}) + "\n")
                fout.flush()
    elif args.skip_llm:
        print("[card] --skip_llm: LLM 호출 생략 (풀/제외 로직 검증용)")

    stats = {
        "dataset": ds, "domain": domain,
        "n_items": len(asins), "n_items_no_review": n_no_review,
        "n_items_pool_capped": n_capped, "card_pool_cap": CFG.card_pool_cap,
        "exclude_query_review_from_card": CFG.exclude_query_review_from_card,
        "exclude_scope": CFG.exclude_scope,
        "n_items_with_rui": n_with_rui,
        "r_ui_selected_rate": r_ui_selected_rate,
    }
    st_path = stats_path(ds, "card")
    os.makedirs(CFG.result_dir, exist_ok=True)
    with open(st_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[card] stats -> {st_path}: {stats}")
    print(f"[card] done -> {out_path}")


if __name__ == "__main__":
    main()
