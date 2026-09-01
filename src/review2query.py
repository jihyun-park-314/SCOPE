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

def load_query_prompt_template(domain: str) -> str:
    slug = domain.replace(" ", "_")
    path = os.path.join(PROMPT_DIR, f"query_prompt_{slug}.txt")
    if not os.path.exists(path):
        available = sorted(
            f[len("query_prompt_"):-len(".txt")]
            for f in os.listdir(PROMPT_DIR) if f.startswith("query_prompt_") and f.endswith(".txt")
        )
        raise FileNotFoundError(
            f"[review2query] prompts/query_prompt_{slug}.txt not found (--domain '{domain}'). "
            f"available domains: {available} — for a new domain, create "
            f"prompts/query_prompt_{slug}.txt first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

KEY_COLS = ["user_id", "parent_asin", "timestamp"]
BATCH_USERS = 1000
CHECKPOINT_CHUNK = 200

def format_seconds(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

def load_or_build_kcore(reviews_path: str, ku: int, ki: int) -> pd.DataFrame:
    cache_path = scan_cache_path(reviews_path, f"kcore_u{ku}_i{ki}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        lite = pd.read_parquet(cache_path)
        print(f"         reusing the k-core cache -> {cache_path} ({len(lite):,} interactions)")
        return lite
    lite = pd.read_parquet(reviews_path, columns=KEY_COLS).dropna()
    print(f"         Raw interactions: {len(lite):,}")
    lite, _ = kcore_filter(lite, ku, ki)
    lite.to_parquet(cache_path, index=False)
    print(f"         k-core cache saved -> {cache_path}")
    return lite

def load_texts_for_users(path: str, users: set, chunk_rows: int = 1_000_000):
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

def select_sample_users(surv_users: set, n: int, seed: int) -> set:
    rng = np.random.default_rng(seed)
    pool = sorted(surv_users)
    print(f"[select] users surviving k-core: {len(pool):,}")
    n_sample = min(n, len(pool))
    selected = set(rng.choice(pool, n_sample, replace=False).tolist())
    print(f"[select] selected: {len(selected):,} users (target {n:,})")
    if len(selected) < n:
        print(f"[select] the surviving pool is smaller than the target; using all {len(selected):,}")
    return selected

def clean_query(raw: str, fallback: str) -> str:
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
    return re.sub(r"\s+", " ", query.strip().lower()) == fallback

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

    assert final_df[KEY_COLS].notna().all().all(), "[review2query] missing key values after the merge."
    for c in ["user_id", "parent_asin", "timestamp", "query", "is_fallback"]:
        assert c in final_df.columns, f"[review2query] required column missing: {c}"

    tmp_path = out_path + ".tmp"
    final_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    for fn in shard_files:
        os.remove(os.path.join(parts_dir, fn))
    return final_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="dataset key (books / video_games / beauty); the defaults of --domain, "
                         "--reviews and --out are derived from it.")
    ap.add_argument("--reviews", default=None,
                    help="path to the reviews parquet; derived from --dataset when omitted "
                         "(data/raw/{source_category}_reviews.parquet). Unused in --fixed_input mode.")
    ap.add_argument("--out", default=None,
                    help="output path; defaults to data/preprocessed/{dataset}/queries.parquet")
    ap.add_argument("--min_user_inter", type=int, default=10, help="minimum interactions per user")
    ap.add_argument("--min_item_inter", type=int, default=10, help="minimum interactions per item")
    ap.add_argument("--sample_users", type=int, default=5000, help="number of users to sample")
    ap.add_argument("--sample_seed", type=int, default=42, help="seed for the random top-up")
    ap.add_argument("--model", default=CFG.ollama_model, help="Ollama model tag")
    ap.add_argument("--ollama_urls", "--ollama-urls", default=CFG.ollama_urls,
                    help="comma-separated list of Ollama base URLs")
    ap.add_argument("--requests_per_server", type=int, default=1,
                    help="concurrent requests per Ollama server (match OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--exclude_users_file", default=None,
                    help="JSON list of already-processed user_ids to exclude from selection, so a "
                         "follow-up run does not regenerate them")
    ap.add_argument("--domain", default=None,
                    help="prompt domain for the query rewrite; reads prompts/query_prompt_{domain}.txt "
                         "(spaces become underscores). Derived from --dataset when omitted "
                         "(books->book, video_games->'video game').")
    ap.add_argument("--fixed_input", default=None,
                    help="use a fixed (user_id, parent_asin, timestamp, title, text) parquet as-is: "
                         "skip k-core and user selection (steps 1-3) and generate queries only for "
                         "its rows. This is the standard path, applied to sample.parquet from stage [2].")
    args = ap.parse_args()

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
        print(f"[review2query] excluded users loaded: {len(excl_users):,} ({args.exclude_users_file})")

    pass2_cache = scan_cache_path(
        args.reviews,
        f"pass2_u{args.min_user_inter}_i{args.min_item_inter}"
        f"_n{args.sample_users}_s{args.sample_seed}_ex{excl_tag}")

    if args.fixed_input:
        print(f"\n[Step 1-3] --fixed_input given -> skipping k-core and user selection; using the rows "
              f"of {args.fixed_input} as-is")
        df = pd.read_parquet(args.fixed_input, columns=KEY_COLS + ["title", "text"]).dropna(subset=KEY_COLS)
        df["_review"] = df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
        print(f"         rows={len(df):,}  users={df['user_id'].nunique():,}  "
              f"items={df['parent_asin'].nunique():,}")
    elif os.path.exists(pass2_cache):
        print(f"\n[Step 1-3] reusing the cache -> {pass2_cache} (pass 1, pass 2 and selection skipped)")
        df = pd.read_parquet(pass2_cache)
        print(f"         interactions={len(df):,}  users={df['user_id'].nunique():,}")
    else:
        print(f"\n[Step 1] Pass-1: light columns for {args.min_user_inter}/{args.min_item_inter}-core ...")
        lite = load_or_build_kcore(args.reviews, args.min_user_inter, args.min_item_inter)
        surv_users = set(lite["user_id"].unique())
        print(f"         After {args.min_user_inter}-core: {len(lite):,} interactions ({len(surv_users):,} users)")

        print(f"\n[Step 2] selecting users (random, seed={args.sample_seed}) ...")
        selected_users = select_sample_users(surv_users, args.sample_users, args.sample_seed)

        if excl_users:
            before = len(selected_users)
            selected_users = selected_users - excl_users
            print(f"[Step 2]   exclusions applied: {before:,} -> {len(selected_users):,} users "
                  f"({before - len(selected_users):,} already processed)")

        lite_sel = lite[lite["user_id"].isin(selected_users)]
        surv_keys = set(zip(lite_sel["user_id"], lite_sel["parent_asin"], lite_sel["timestamp"]))
        del lite, lite_sel

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
        print(f"[Step 3] pass-2 result cached -> {pass2_cache}")

    assert df[KEY_COLS].notna().all().all(), "[review2query] missing user_id / parent_asin / timestamp."
    print(f"[review2query] timestamps preserved: dtype={df['timestamp'].dtype}, null=0, "
          f"range=[{df['timestamp'].min()}, {df['timestamp'].max()}]")
    users_in_df = set(df["user_id"].unique())

    done_keys = load_done_keys(args.out, parts_dir)
    keys_all = list(zip(df["user_id"], df["parent_asin"], df["timestamp"]))
    todo_idx_set = {i for i, k in enumerate(keys_all) if k not in done_keys}
    print(f"\n[Step 4] resume: {len(done_keys):,} rows already done, "
          f"{len(todo_idx_set):,}/{len(df):,} to generate")

    reviews_all = df["_review"].tolist()
    base_cols = ["user_id", "parent_asin", "timestamp", "title", "text"]
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
            print(f"[Step 5] batch {bi + 1}/{len(user_batches)} ({len(batch_users):,} users) "
                  f"— already done, skipping")
            continue

        print(f"[Step 5] batch {bi + 1}/{len(user_batches)}: {len(batch_users):,} users, "
              f"{len(batch_todo):,} rows to generate")
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
            shard_path = os.path.join(parts_dir, f"batch{bi:03d}_row{chunk_idx[0]:08d}.parquet")
            shard.to_parquet(shard_path, index=False)

            done_so_far += len(chunk_idx)
            elapsed = time.time() - t0
            rate = done_so_far / elapsed if elapsed > 0 else 0
            eta = (n_todo_total - done_so_far) / rate if rate > 0 else float("inf")
            print(f"[Step 5]   checkpoint -> {shard_path} "
                  f"({done_so_far:,}/{n_todo_total:,} total, "
                  f"elapsed={format_seconds(elapsed)}, ETA={format_seconds(eta)})")

        batch_final = merge_and_save(args.out, parts_dir)
        n_fb = int(batch_final["is_fallback"].sum())
        print(f"[Step 5] batch {bi + 1}/{len(user_batches)} done -> {args.out} updated "
              f"(cumulative {len(batch_final):,} rows, {batch_final['user_id'].nunique():,} users, "
              f"fallback={n_fb:,} [{n_fb / len(batch_final) * 100:.1f}%])")

    final_df = merge_and_save(args.out, parts_dir)
    if final_df is None:
        print("\n[Step 6] nothing was generated (no target rows).")
        return
    for fn in os.listdir(parts_dir):
        os.remove(os.path.join(parts_dir, fn))
    os.rmdir(parts_dir)

    n_fb = int(final_df["is_fallback"].sum())
    print(f"\n[Step 6] all done -> {args.out}")
    print(f"         rows={len(final_df):,}  users={final_df['user_id'].nunique():,}  "
          f"fallback={n_fb:,} ({n_fb / len(final_df) * 100:.1f}%)")

    print("\n[Sample Results]")
    for i in range(min(3, len(final_df))):
        print(f"  REVIEW: {str(final_df['text'].iloc[i])[:100]}...")
        print(f"  QUERY : {final_df['query'].iloc[i]}  (fallback={final_df['is_fallback'].iloc[i]})\n")

if __name__ == "__main__":
    main()
