"""
[review2query] 리뷰 -> 검색쿼리(A 프롬프트) 생성.

프롬프트 검증에서 A(1인칭 rephrase)가 GO 판정을 받아 채택됐다. 이 스크립트는 그 A 프롬프트
하나만 사용한다(B/옛 rephrase는 폐기).

이 스크립트는 split을 만들지도, 읽지도 않는다. (user_id, parent_asin, timestamp)를 그대로
보존한 채 query 컬럼만 채워 넣고, train/valid/test 판정은 preprocessing.py가 고정한
split_manifest.json이 단일 출처로 나중에 prepare_dataset.py에서 join된다.
※ 예전에는 이 산출물로 "유저별 timestamp 정렬 -> leave-last-out"을 다시 계산했다. 그 재계산이
   semantic_card.py의 제외 키와 어긋나 22k Books에서 타깃 788건이 카드 제외에서 빠지고
   112건이 실제로 카드에 leak된 원인이었다(2026-07-24). 지금 구조에는 그 경로가 없다.

두 가지 실행 경로:
  · --fixed_input (SCOPE 표준) — preprocessing.py가 확정한 sample.parquet의 행에만 쿼리를
    채운다. k-core/유저선정(Step 1-3)을 통째로 건너뛰므로 --reviews를 아예 읽지 않는다.
  · 독립 실행 — 원본 리뷰에서 직접 k-core를 돌리고 생존 유저 중 고정 시드로 --sample_users 명을
    무작위 선정한다. 무거운 전체 스캔이라 결과를 data/preprocessed/_cache/에 캐시한다.

★ 두 경로는 리뷰 본문 길이가 다르다: --fixed_input은 sample.parquet의 텍스트를 그대로 쓰지만,
  독립 실행은 load_texts_for_users()가 title을 200자, text를 800자로 자른 뒤 프롬프트에 넣는다.
  같은 리뷰라도 경로에 따라 다른 쿼리가 나올 수 있다.

재개: 출력 parquet + {out}.parts/ 의 샤드 parquet에 이미 있는 (user_id, parent_asin, timestamp)
키는 스킵한다. 체크포인트 청크마다 샤드를 즉시 저장하므로 중간에 죽어도 그 청크만 다시 하면 된다.

실행 예 (SCOPE 표준 경로):
  python src/review2query.py --dataset books \
    --fixed_input data/preprocessed/books/sample.parquet \
    --ollama-urls http://localhost:11434,http://localhost:11435

  --dataset만 주면 --domain(book) / --reviews(data/raw/Books_reviews.parquet) /
  --out(data/preprocessed/books/queries.parquet)이 전부 유도된다.

  # 원본에서 직접 k-core + 유저 샘플링까지 하는 독립 실행:
  python src/review2query.py --dataset books \
    --min_user_inter 10 --min_item_inter 10 --sample_users 5000
"""
import os
import re
import json
import time
import hashlib
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ollama_client
from config import (CFG, PROMPT_DIR, dataset_root, domain_of, queries_path, raw_paths,
                    scan_cache_path)
from utils import kcore_filter

# ── 채택된 A 프롬프트를 도메인별 텍스트 파일(prompts/query_prompt_{domain}.txt)에서 읽는다 —
# semantic_card.py의 카드 프롬프트와 동일한 관례. --domain으로 그 도메인의 파일을 고른다.
# book/video_game/beauty_product 파일은 이전에 코드로 생성하던 문구와 byte-identical(확인됨).
# PROMPT_DIR은 config.py가 repo root 기준으로 고정한 <repo>/prompts를 그대로 쓴다.
# (예전에는 이 파일에서 os.path.dirname(__file__)/prompts로 잡아 src/prompts를 가리켰다 —
#  스크립트가 src/로 내려온 뒤로는 프롬프트를 못 찾는 경로였다. semantic_card.py는 이미
#  repo root 기준이었어서 두 스크립트의 기준이 서로 달랐다.)


def load_query_prompt_template(domain: str) -> str:
    slug = domain.replace(" ", "_")
    path = os.path.join(PROMPT_DIR, f"query_prompt_{slug}.txt")
    if not os.path.exists(path):
        available = sorted(
            f[len("query_prompt_"):-len(".txt")]
            for f in os.listdir(PROMPT_DIR) if f.startswith("query_prompt_") and f.endswith(".txt")
        )
        raise FileNotFoundError(
            f"[review2query] prompts/query_prompt_{slug}.txt 없음 (--domain '{domain}'). "
            f"사용 가능한 도메인: {available} — 새 도메인이면 prompts/query_prompt_{slug}.txt를 먼저 만드세요."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


KEY_COLS = ["user_id", "parent_asin", "timestamp"]
BATCH_USERS = 1000              # 이 명수 단위로 진행상황을 보고하고, 다 끝난 배치는 통째로 스킵
CHECKPOINT_CHUNK = 200          # 배치 내부 저장 단위(크래시 시 손실을 최소화)


def format_seconds(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"


# ── 독립 실행 경로: k-core 필터링(Step 1) + 대상 유저 텍스트 스트리밍(Step 3) ──
def load_or_build_kcore(reviews_path: str, ku: int, ki: int) -> pd.DataFrame:
    """29M행 전체 스캔 + 반복 k-core는 비용이 크므로 결과를 캐시해 재실행/재개 시 재스캔을 피한다."""
    cache_path = scan_cache_path(reviews_path, f"kcore_u{ku}_i{ki}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        lite = pd.read_parquet(cache_path)
        print(f"         k-core 캐시 재사용 -> {cache_path} ({len(lite):,} interactions)")
        return lite
    lite = pd.read_parquet(reviews_path, columns=KEY_COLS).dropna()
    print(f"         Raw interactions: {len(lite):,}")
    lite, _ = kcore_filter(lite, ku, ki)
    lite.to_parquet(cache_path, index=False)
    print(f"         k-core 캐시 저장 -> {cache_path}")
    return lite


def load_texts_for_users(path: str, users: set, chunk_rows: int = 1_000_000):
    """대상 유저 행만 청크 스트리밍으로 수집 — 본문 전체 상주 회피."""
    pf = pq.ParquetFile(path)
    cols = ["user_id", "parent_asin", "timestamp", "title", "text"]
    parts = []
    seen = 0
    for batch in pf.iter_batches(batch_size=chunk_rows, columns=cols):
        df = batch.to_pandas()
        df = df[df["user_id"].isin(users)]
        if len(df):
            df["title"] = df["title"].astype(str).str.slice(0, 200)
            df["text"] = df["text"].astype(str).str.slice(0, 800)
            parts.append(df)
        seen += len(batch)
        if seen % 5_000_000 < chunk_rows:
            print(f"     scanned {seen:,} rows ...")
    return pd.concat(parts, ignore_index=True)


# ── 독립 실행 경로: 대상 유저 선정(Step 2) ──
def select_sample_users(surv_users: set, n: int, seed: int) -> set:
    """k-core 생존 유저에서 고정 시드로 n명을 무작위 선정한다."""
    rng = np.random.default_rng(seed)
    pool = sorted(surv_users)
    print(f"[select] k-core 생존 유저: {len(pool):,}")
    n_sample = min(n, len(pool))
    selected = set(rng.choice(pool, n_sample, replace=False).tolist())
    print(f"[select] 최종 선정: {len(selected):,}명 (목표 {n:,})")
    if len(selected) < n:
        print(f"[select] ⚠ 생존 유저 풀이 목표보다 작아 {len(selected):,}명 전부 사용")
    return selected


# ── 응답 후처리(Step 5) — HTTP 호출과 재시도/동시성은 src/ollama_client.py가 담당 ──
def clean_query(raw: str, fallback: str) -> str:
    """폴백 인식 정리: 첫 줄만 취하고 따옴표/프리앰블 제거, 빈 출력은 폴백 텍스트로.

    fallback은 main()이 도메인에서 만들어 넘긴다(`general {domain} recommendation`).
    예전에는 모듈 레벨 FALLBACK_TEXT를 main()이 global로 덮어쓰는 구조였는데, 초기값
    "general recommendation"이 어느 도메인에서도 맞지 않아 main()을 거치지 않고 이 함수를
    부르면 조용히 그 값이 쓰였다."""
    q = (raw or "").strip()
    if not q:
        return fallback
    q = q.splitlines()[0].strip().strip('"').strip("'")
    lower = q.lower()
    for prefix in ("query:", "search query:", "output:"):
        if lower.startswith(prefix):
            q = q[len(prefix):].strip()
            break
    return q if q else fallback


def is_fallback(query: str, fallback: str) -> bool:
    """공백 정규화 후 폴백 문자열과 정확히 같은가 (queries.parquet의 is_fallback 컬럼)."""
    return re.sub(r"\s+", " ", query.strip().lower()) == fallback


# ── 재개(Step 4): 이미 처리된 키 스킵 ──
def load_done_keys(out_path, parts_dir) -> set:
    done = set()
    if os.path.exists(out_path):
        d = pd.read_parquet(out_path, columns=KEY_COLS)
        done.update(zip(d["user_id"], d["parent_asin"], d["timestamp"]))
    if os.path.isdir(parts_dir):
        for fn in sorted(os.listdir(parts_dir)):
            if fn.endswith(".parquet"):
                d = pd.read_parquet(os.path.join(parts_dir, fn), columns=KEY_COLS)
                done.update(zip(d["user_id"], d["parent_asin"], d["timestamp"]))
    return done


def merge_and_save(out_path: str, parts_dir: str):
    """지금까지의 out + 모든 샤드를 합쳐 바로 다음 단계(03/04 등)에 쓸 수 있는 단일 parquet으로
    저장한다. 배치(1,000명)마다 호출 — 중간에 중단돼도 그 시점 결과물로 연구를 계속 진행할 수 있게.
    쓰기는 tmp 파일 + os.replace로 원자적으로 처리해 중단 시 out_path가 깨지는 것을 막고,
    이번에 병합된 샤드는 저장 성공 직후 지워서 다음 호출이 같은 샤드를 반복해서 읽지 않게 한다
    (안 지우면 배치가 진행될수록 매번 읽어야 할 샤드 수가 계속 늘어난다)."""
    merged = []
    if os.path.exists(out_path):
        merged.append(pd.read_parquet(out_path))
    shard_files = sorted(fn for fn in os.listdir(parts_dir) if fn.endswith(".parquet"))
    for fn in shard_files:
        merged.append(pd.read_parquet(os.path.join(parts_dir, fn)))
    if not merged:
        return None

    final_df = pd.concat(merged, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=KEY_COLS, keep="last")
    final_df = final_df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)

    assert final_df[KEY_COLS].notna().all().all(), "[review2query] 병합 결과에 키 결측치가 있습니다."
    for c in ["user_id", "parent_asin", "timestamp", "query", "is_fallback"]:
        assert c in final_df.columns, f"[review2query] 필수 컬럼 누락: {c}"

    tmp_path = out_path + ".tmp"
    final_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    for fn in shard_files:
        os.remove(os.path.join(parts_dir, fn))
    return final_df


# ── 메인 실행 함수 ──
def main():
    ap = argparse.ArgumentParser()
    # --dataset 하나로 도메인/원본/출력 경로가 전부 유도된다. 개별 override는 아래 인자로.
    ap.add_argument("--dataset", required=True,
                    help="데이터셋 키 (books / video_games / beauty). "
                         "--domain/--reviews/--out의 기본값을 이 값에서 유도한다.")
    ap.add_argument("--reviews", default=None,
                    help="리뷰 parquet 경로. 미지정 시 --dataset에서 유도 "
                         "(data/raw/{원본카테고리}_reviews.parquet). --fixed_input 모드에선 안 읽는다.")
    ap.add_argument("--out", default=None,
                    help="출력 경로. 미지정 시 data/preprocessed/{dataset}/queries.parquet")
    ap.add_argument("--min_user_inter", type=int, default=10, help="유저별 최소 상호작용 수")
    ap.add_argument("--min_item_inter", type=int, default=10, help="아이템별 최소 상호작용 수")
    ap.add_argument("--sample_users", type=int, default=5000, help="선정할 샘플 유저 수")
    ap.add_argument("--sample_seed", type=int, default=42, help="무작위 충원 고정 시드")
    ap.add_argument("--model", default=CFG.ollama_model, help="Ollama 모델 태그")
    ap.add_argument("--ollama_urls", "--ollama-urls", default=CFG.ollama_urls,
                    help="Ollama 서버 base URL 목록(쉼표 구분)")
    ap.add_argument("--requests_per_server", type=int, default=1,
                    help="Ollama 서버 1대당 동시 요청 수(OLLAMA_NUM_PARALLEL과 맞춰서 설정)")
    ap.add_argument("--exclude_users_file", default=None,
                    help="이미 처리한 user_id 목록 JSON 파일 — 선정 유저에서 제외(예: 5,000명 샘플 이후 "
                         "나머지 생존 유저 전체를 돌릴 때 중복 생성 방지)")
    ap.add_argument("--domain", default=None,
                    help="쿼리 리라이트 프롬프트의 대상 도메인 — prompts/query_prompt_{domain}.txt를 "
                         "읽는다(공백은 밑줄로 치환). 미지정 시 --dataset에서 유도한다 "
                         "(books->book, video_games->'video game'). 새 도메인이면 그 파일을 "
                         "먼저 만들고 config.DATASETS에 등록할 것.")
    ap.add_argument("--fixed_input", default=None,
                    help="이미 확정된 (user_id, parent_asin, timestamp, title, text) parquet을 그대로 "
                         "사용 — k-core/유저선정(Step 1-3)을 건너뛰고 이 파일의 행에만 쿼리를 생성한다 "
                         "(SCOPE 표준 경로: preprocessing.py가 만든 sample.parquet에 쿼리를 붙일 때)")
    args = ap.parse_args()

    # --dataset에서 유도되는 기본값들 (명시된 값이 항상 우선)
    domain = args.domain or domain_of(args.dataset)
    args.reviews = args.reviews or raw_paths(args.dataset)[0]
    args.out = args.out or queries_path(args.dataset)
    os.makedirs(dataset_root(args.dataset), exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[review2query] dataset={args.dataset}  domain={domain}\n"
          f"               reviews={args.reviews}\n"
          f"               out={args.out}")

    prompt_template = load_query_prompt_template(domain)
    fallback = f"general {domain} recommendation"

    urls = ollama_client.parse_urls(args.ollama_urls)
    parts_dir = args.out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    excl_users = set()
    excl_tag = "none"
    if args.exclude_users_file:
        with open(args.exclude_users_file) as f:
            excl_users = set(json.load(f))
        excl_tag = hashlib.sha1(",".join(sorted(excl_users)).encode()).hexdigest()[:8]
        print(f"[review2query] 제외 유저 로드: {len(excl_users):,}명 ({args.exclude_users_file})")

    # Pass-1(k-core)·Pass-2(선정 유저 본문 스트리밍)는 둘 다 29M행 전체를 훑는 무거운 작업이라,
    # (reviews, core 파라미터, sample_users, seed, 제외목록)이 같으면 재실행 때 통째로 건너뛴다.
    pass2_cache = scan_cache_path(
        args.reviews,
        f"pass2_u{args.min_user_inter}_i{args.min_item_inter}"
        f"_n{args.sample_users}_s{args.sample_seed}_ex{excl_tag}")

    if args.fixed_input:
        print(f"\n[Step 1-3] --fixed_input 지정됨 -> k-core/유저선정을 건너뛰고 "
              f"{args.fixed_input}의 행 그대로 사용")
        df = pd.read_parquet(args.fixed_input, columns=KEY_COLS + ["title", "text"]).dropna(subset=KEY_COLS)
        df["_review"] = df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
        print(f"         rows={len(df):,}  users={df['user_id'].nunique():,}  "
              f"items={df['parent_asin'].nunique():,}")
    elif os.path.exists(pass2_cache):
        print(f"\n[Step 1-3] 캐시 재사용 -> {pass2_cache} (Pass-1/Pass-2/유저선정 스킵)")
        df = pd.read_parquet(pass2_cache)
        print(f"         interactions={len(df):,}  users={df['user_id'].nunique():,}")
    else:
        # [Step 1] Pass-1: 경량 컬럼으로 k-core (자체 캐시로 재스캔 방지)
        print(f"\n[Step 1] Pass-1: light columns for {args.min_user_inter}/{args.min_item_inter}-core ...")
        lite = load_or_build_kcore(args.reviews, args.min_user_inter, args.min_item_inter)
        surv_users = set(lite["user_id"].unique())
        print(f"         After {args.min_user_inter}-core: {len(lite):,} interactions ({len(surv_users):,} users)")

        # [Step 2] 대상 유저 선정
        print(f"\n[Step 2] 유저 선정 (무작위, seed={args.sample_seed}) ...")
        selected_users = select_sample_users(surv_users, args.sample_users, args.sample_seed)

        if excl_users:
            before = len(selected_users)
            selected_users = selected_users - excl_users
            print(f"[Step 2]   제외 적용: {before:,} -> {len(selected_users):,}명 "
                  f"({before - len(selected_users):,}명 제외, 이미 처리됨)")

        # 선정 유저로 exact-match 키 축소 (Pass-2 스트리밍/메모리 범위를 5,000명으로 제한 -> OOM 회피)
        lite_sel = lite[lite["user_id"].isin(selected_users)]
        surv_keys = set(zip(lite_sel["user_id"], lite_sel["parent_asin"], lite_sel["timestamp"]))
        del lite, lite_sel

        # [Step 3] Pass-2: 선정된 5,000명 텍스트만 스트리밍
        print(f"\n[Step 3] Pass-2: streaming texts for {len(selected_users):,} selected users ...")
        df = load_texts_for_users(args.reviews, selected_users)
        key = list(zip(df["user_id"], df["parent_asin"], df["timestamp"]))
        df = df[[k in surv_keys for k in key]]
        df = df.drop_duplicates(subset=KEY_COLS).reset_index(drop=True)
        print(f"         Joined interactions (exact {args.min_user_inter}/{args.min_item_inter}-core, "
              f"{len(selected_users):,}-user sample): {len(df):,}")

        for col in ("title", "text"):
            if col not in df.columns:
                df[col] = ""
        df["_review"] = df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)

        df.to_parquet(pass2_cache, index=False)
        print(f"[Step 3] Pass-2 결과 캐시 저장 -> {pass2_cache}")

    assert df[KEY_COLS].notna().all().all(), "[review2query] user_id/parent_asin/timestamp에 결측치가 있습니다."
    print(f"[review2query] timestamp 보존 확인: dtype={df['timestamp'].dtype}, null=0, "
          f"range=[{df['timestamp'].min()}, {df['timestamp'].max()}]")
    # 세 갈래(--fixed_input / 캐시 / 독립 실행) 어느 쪽으로 왔든, 이 시점부터는 df에 실제로
    # 들어 있는 유저만 본다 (Step 2의 selected_users와 구분하려고 이름을 달리 둔다).
    users_in_df = set(df["user_id"].unique())

    # [Step 4] 재개: 이미 처리된 (user_id, parent_asin, timestamp) 스킵 (O(1) 조회를 위해 set로 보관)
    done_keys = load_done_keys(args.out, parts_dir)
    keys_all = list(zip(df["user_id"], df["parent_asin"], df["timestamp"]))
    todo_idx_set = {i for i, k in enumerate(keys_all) if k not in done_keys}
    print(f"\n[Step 4] resume: {len(done_keys):,} rows already done, "
          f"{len(todo_idx_set):,}/{len(df):,} to generate")

    # [Step 5] 1,000명 단위 배치로 진행 — 배치가 이미 다 끝났으면 통째로 스킵(빠른 재개),
    #          배치 내부는 CHECKPOINT_CHUNK 단위로 저장해 중단 시 손실을 그 청크만으로 제한.
    reviews_all = df["_review"].tolist()
    base_cols = ["user_id", "parent_asin", "timestamp", "title", "text"]
    # 유저별 행 위치를 한 번만 인덱싱 — 배치(140개)마다 3M+ 행 전체를 np.isin으로 다시
    # 훑으면(문자열 user_id 배열이라 더 느림) 재개할 때마다 배치스킵 구간이 병목이 된다.
    user_to_rows = df.groupby("user_id", sort=False).indices
    users_sorted = sorted(users_in_df)
    user_batches = [users_sorted[i:i + BATCH_USERS] for i in range(0, len(users_sorted), BATCH_USERS)]
    n_todo_total = len(todo_idx_set)
    t0 = time.time()
    done_so_far = 0
    for bi, batch_users in enumerate(user_batches):
        row_idx_parts = [user_to_rows[u] for u in batch_users if u in user_to_rows]
        batch_row_idx = np.sort(np.concatenate(row_idx_parts)) if row_idx_parts else np.empty(0, dtype=int)
        batch_todo = [i for i in batch_row_idx if i in todo_idx_set]

        if not batch_todo:
            print(f"[Step 5] batch {bi + 1}/{len(user_batches)} ({len(batch_users):,}명) "
                  f"— 이미 완료, 스킵")
            continue

        print(f"[Step 5] batch {bi + 1}/{len(user_batches)}: {len(batch_users):,}명, "
              f"{len(batch_todo):,}행 생성 필요")
        for start in range(0, len(batch_todo), CHECKPOINT_CHUNK):
            chunk_idx = batch_todo[start:start + CHECKPOINT_CHUNK]
            chunk_prompts = [prompt_template.format(review=reviews_all[i]) for i in chunk_idx]
            raw = ollama_client.generate_batch(
                urls, args.model, chunk_prompts, CFG.query_max_new_tokens,
                desc=f"batch {bi + 1}/{len(user_batches)} chunk",
                requests_per_server=args.requests_per_server)
            outs = [clean_query(r, fallback) for r in raw]

            shard = df.iloc[chunk_idx][base_cols].copy()
            shard["query"] = outs
            shard["is_fallback"] = [is_fallback(q, fallback) for q in outs]
            # 샤드 파일명은 df 안에서의 절대 행 위치(chunk_idx[0])로 짓는다 — 재개 때마다
            # todo 목록이 줄어들며 `start`가 다시 0부터 시작하면 이전 실행의 샤드 파일명과
            # 겹쳐 덮어써버리는(데이터 손실) 문제가 있었다.
            shard_path = os.path.join(parts_dir, f"batch{bi:03d}_row{chunk_idx[0]:08d}.parquet")
            shard.to_parquet(shard_path, index=False)

            done_so_far += len(chunk_idx)
            elapsed = time.time() - t0
            rate = done_so_far / elapsed if elapsed > 0 else 0
            eta = (n_todo_total - done_so_far) / rate if rate > 0 else float("inf")
            print(f"[Step 5]   checkpoint -> {shard_path} "
                  f"({done_so_far:,}/{n_todo_total:,} total, "
                  f"elapsed={format_seconds(elapsed)}, ETA={format_seconds(eta)})")

        # 배치(1,000명)가 끝날 때마다 바로 --out에 완성된 parquet으로 병합 저장.
        # 여기서 중단돼도 이 시점의 --out을 그대로 다음 파이프라인 단계에 쓸 수 있다.
        batch_final = merge_and_save(args.out, parts_dir)
        n_fb = int(batch_final["is_fallback"].sum())
        print(f"[Step 5] batch {bi + 1}/{len(user_batches)} 완료 -> {args.out} 갱신 "
              f"(누적 {len(batch_final):,}행, {batch_final['user_id'].nunique():,}명, "
              f"fallback={n_fb:,} [{n_fb / len(batch_final) * 100:.1f}%])")

    # 전체 배치 종료 -> 샤드는 이미 매 배치마다 --out에 반영됐으니 정리만 한다.
    final_df = merge_and_save(args.out, parts_dir)
    if final_df is None:
        print("\n[Step 6] 생성된 행이 없습니다 (대상 0건).")
        return
    for fn in os.listdir(parts_dir):
        os.remove(os.path.join(parts_dir, fn))
    os.rmdir(parts_dir)

    n_fb = int(final_df["is_fallback"].sum())
    print(f"\n[Step 6] 전체 완료 -> {args.out}")
    print(f"         rows={len(final_df):,}  users={final_df['user_id'].nunique():,}  "
          f"fallback={n_fb:,} ({n_fb / len(final_df) * 100:.1f}%)")

    print("\n[Sample Results]")
    for i in range(min(3, len(final_df))):
        print(f"  REVIEW: {str(final_df['text'].iloc[i])[:100]}...")
        print(f"  QUERY : {final_df['query'].iloc[i]}  (fallback={final_df['is_fallback'].iloc[i]})\n")


if __name__ == "__main__":
    main()
