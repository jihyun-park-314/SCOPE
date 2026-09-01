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
    slug = domain.replace(" ", "_")
    path = os.path.join(PROMPT_DIR, f"card_prompt_{slug}.txt")
    if not os.path.exists(path):
        available = sorted(
            f[len("card_prompt_"):-len(".txt")]
            for f in os.listdir(PROMPT_DIR) if f.startswith("card_prompt_") and f.endswith(".txt")
        )
        raise FileNotFoundError(
            f"[card] prompts/card_prompt_{slug}.txt not found (--domain '{domain}'). "
            f"available domains: {available} — for a new domain, create "
            f"prompts/card_prompt_{slug}.txt first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_card_fields(template: str) -> tuple:
    fields = tuple(re.findall(r"^([A-Za-z][A-Za-z ]*): <", template, flags=re.MULTILINE))
    if not fields:
        raise ValueError(
            "[card] no fields found in the card prompt. Each field must be a line of the form "
            "`Genre: <description>`, with an angle-bracket placeholder.")
    return fields

def build_fallback_card(fields: tuple) -> str:
    return "\n".join(f"{f}: unknown" for f in fields)

def _fmt_meta(row, fields) -> str:
    parts = []
    for k in fields:
        v = row.get(k)
        if v is not None and len(str(v)) > 2:
            parts.append(f"{k.capitalize()}: {str(v)[:500]}")
    if row.get("average_rating") is not None:
        parts.append(f"AvgRating: {row['average_rating']} ({row.get('rating_number', '?')} ratings)")
    return "\n".join(parts)

def build_exclusion_keys(P: dict, scope: str):
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
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    key1, key2 = set(), set()
    for row in manifest["valid"] + manifest["test"]:
        key1.add((row["item_id"], row["user_id"], row["timestamp"]))
        key2.add((row["item_id"], row["user_id"], row["review_hash"]))
    return key1, key2

ROW_HELPFUL_VOTE, ROW_RATING, ROW_TITLE, ROW_TEXT, ROW_HASH = range(5)

def stream_reviews_by_item(path: str, asins: set, pool_cap: int, key1: set, key2: set,
                            chunk_rows: int = 1_000_000):
    key3 = {(asin, h) for (asin, _uid, h) in key2}
    heaps = defaultdict(list)
    excluded = defaultdict(list)
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
                     help="dataset key (books / video_games / beauty).")
    ap.add_argument("--max_items", type=int, default=0, help="limit the number of items, for testing (0 = all)")
    ap.add_argument("--skip_llm", action="store_true", help="check the pool and exclusion logic only, without calling the LLM")
    ap.add_argument("--pkl", type=str, default=None,
                     help="path to the pkl written by preprocessing.py "
                          "(default: data/preprocessed/{dataset}/interactions.pkl)")
    ap.add_argument("--manifest", type=str, default=None,
                     help="path to the split_manifest.json written by preprocessing.py (default: "
                          "data/preprocessed/{dataset}/split_manifest.json). It is the same file "
                          "prepare_dataset.py reads, so the two cannot disagree on the split. Pass an "
                          "empty string to fall back to recomputing from the pkl.")
    ap.add_argument("--reviews", type=str, default=None,
                     help="reviews parquet used as the card source; derived from --dataset when omitted "
                          "(data/raw/{source_category}_reviews.parquet).")
    ap.add_argument("--meta", type=str, default=None,
                     help="metadata parquet used as the card source; derived from --dataset when omitted "
                          "(data/raw/{source_category}_meta.parquet).")
    ap.add_argument("--model", type=str, default=CFG.ollama_model, help="Ollama model tag")
    ap.add_argument("--ollama_urls", "--ollama-urls", type=str, default=CFG.ollama_urls,
                     help="comma-separated list of Ollama base URLs")
    ap.add_argument("--requests_per_server", type=int, default=1,
                     help="concurrent requests per Ollama server (match OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--domain", type=str, default=None,
                     help="domain noun used in the card prompt (same convention as review2query.py "
                          "--domain). Derived from --dataset when omitted (books->book, video_games->'video game', "
                          "beauty->'beauty product').")
    args = ap.parse_args()

    ds = args.dataset
    domain = args.domain or domain_of(ds)
    prompt_template = load_card_prompt_template(domain)
    card_fields = parse_card_fields(prompt_template)
    fallback_card = build_fallback_card(card_fields)
    print(f"[card] card fields: {card_fields}  (prompts/card_prompt_{domain.replace(' ', '_')}.txt)")

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

    meta_fields = meta_fields_of(ds)
    missing_meta = [k for k in meta_fields if k not in meta.columns]
    print(f"[card] metadata fields: {meta_fields}" +
          (f"  (absent from the parquet, so omitted from the card input: {missing_meta})" if missing_meta else ""))

    mf_path = default_manifest_path(ds) if args.manifest is None else args.manifest

    key1, key2 = set(), set()
    if CFG.exclude_query_review_from_card:
        if mf_path and not os.path.exists(mf_path):
            raise FileNotFoundError(
                f"[card] manifest not found: {mf_path}\n"
                f"       if preprocessing.py has not run yet:\n"
                f"         python src/preprocessing.py --dataset {ds}\n"
                f"       if the pkl exists but the manifest does not, restore it without re-sampling:\n"
                f"         python src/preprocessing.py --dataset {ds} --from_pkl\n"
                f"       to deliberately recompute from the pkl, pass --manifest '' explicitly.")
        if mf_path:
            key1, key2 = build_exclusion_keys_from_manifest(mf_path)
            print(f"[card] R_ui exclusion keys from: manifest={mf_path}")
        else:
            key1, key2 = build_exclusion_keys(P, CFG.exclude_scope)
            print(f"[card] R_ui exclusion keys from: {pkl_path} (no manifest given — fallback path)")
    print(f"[card] R_ui exclusion keys: {len(key1):,} (scope={CFG.exclude_scope})")
    eval_source_review_ids = {(asin, h) for (asin, _uid, h) in key2}

    t0 = time.time()
    pools, excluded = stream_reviews_by_item(
        reviews_path, asin_set, CFG.card_pool_cap, key1, key2)
    print(f"[card] corpus scan finished ({time.time() - t0:.0f}s) — "
          f"items with a review pool={len(pools):,}, items with R_ui={len(excluded):,}")

    n_with_rui, n_would_be_selected = len(excluded), 0
    for asin, rui_rows in excluded.items():
        full_rows = pools.get(asin, []) + rui_rows
        selected_full = select8(full_rows, CFG.max_reviews_per_item)
        if any(r in rui_rows for r in selected_full):
            n_would_be_selected += 1
    r_ui_selected_rate = (n_would_be_selected / n_with_rui) if n_with_rui else None

    selected_rows_by_asin = {asin: select8(pools.get(asin, []), CFG.max_reviews_per_item)
                              for asin in asins}
    rev_selected = {asin: format_reviews(rows) for asin, rows in selected_rows_by_asin.items()}
    n_no_review = sum(1 for asin in asins if not rev_selected.get(asin))
    n_capped = sum(1 for asin in asins if len(pools.get(asin, [])) >= CFG.card_pool_cap)

    card_source_review_ids = {(asin, row[4]) for asin, rows in selected_rows_by_asin.items() for row in rows}
    leaked_hashes = eval_source_review_ids & card_source_review_ids
    if leaked_hashes:
        raise AssertionError(
            f"[card][LEAKAGE] eval_source_review_ids ∩ card_source_review_ids = {len(leaked_hashes):,} "
            f"(must be 0). The R_ui exclusion keys probably disagree with the final valid/test split — "
            f"check that --manifest is the same file prepare_dataset.py reads."
        )
    print(f"[card] leakage assertion passed: eval_source_review_ids({len(eval_source_review_ids):,}) "
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
        with open(out_path, "a", encoding="utf-8") as fout:
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                prompts = []
                for asin in chunk:
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
        print("[card] --skip_llm: no LLM calls (pool and exclusion logic only)")

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
