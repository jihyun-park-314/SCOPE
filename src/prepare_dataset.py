# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from config import (canonical_n, canonical_eval_id, canonical_sha256, dataset_root,
                    leak_json_path, manifest_path as default_manifest_path, queries_path)

def safe_text(x):
    if x is None:
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()

def print_shape(tag: str, df: pl.DataFrame) -> None:
    print(f"[{tag}] rows={df.height:,}, users={df['user_id'].n_unique():,}, "
          f"items={df['item_id'].n_unique():,}")

def iterative_kcore(df: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    prev_n = -1
    step = 0

    while prev_n != df.height:
        prev_n = df.height
        step += 1

        user_keep = (
            df.group_by("user_id")
            .len()
            .filter(pl.col("len") >= k)
            .select("user_id")
        )

        item_keep = (
            df.group_by("item_id")
            .len()
            .filter(pl.col("len") >= k)
            .select("item_id")
        )

        df = (
            df.join(user_keep, on="user_id", how="inner")
            .join(item_keep, on="item_id", how="inner")
        )

        print(f"    k-core step {step}: rows={df.height:,}")

    return df

def sample_users(df: pl.DataFrame, n_users: int, seed: int = 42) -> pl.DataFrame:
    users = df.select("user_id").unique().to_series().to_list()

    if n_users is None or n_users <= 0:
        print("[sample] no user sampling")
        return df

    if n_users >= len(users):
        print(f"[sample] requested {n_users:,}, but only {len(users):,} users exist. Keep all.")
        return df

    rng = np.random.default_rng(seed)
    sampled_users = rng.choice(users, size=n_users, replace=False).tolist()

    sampled_df = df.filter(pl.col("user_id").is_in(sampled_users))

    print(f"[sample] users: {len(users):,} -> {n_users:,}")
    print(f"[sample] rows : {df.height:,} -> {sampled_df.height:,}")

    return sampled_df

def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {src: dst for src, dst in (("parent_asin", "item_id"),
                                            ("title", "review_title"),
                                            ("text", "review_text"))
                  if src in df.columns}
    df = df.rename(rename_map)

    required = ["user_id", "item_id", "timestamp", "query"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col, default in (("review_title", ""), ("review_text", ""), ("is_c4_user", False)):
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    df = df.with_columns(
        [
            pl.col("user_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Int64),
            pl.col("query").cast(pl.Utf8),
            pl.col("review_title").cast(pl.Utf8),
            pl.col("review_text").cast(pl.Utf8),
        ]
    )

    df = df.filter(
        pl.col("query").is_not_null()
        & (pl.col("query").str.strip_chars().str.len_chars() > 0)
    )

    return df

def deduplicate_user_item(df: pl.DataFrame, protected_keys=None) -> pl.DataFrame:
    before = df.height

    if protected_keys:
        keys = pl.DataFrame(
            {"user_id": [k[0] for k in protected_keys],
             "item_id": [k[1] for k in protected_keys],
             "timestamp": [k[2] for k in protected_keys]},
            schema={"user_id": pl.Utf8, "item_id": pl.Utf8, "timestamp": pl.Int64},
        ).with_columns(pl.lit(0, dtype=pl.Int8).alias("_keep_first"))
        df = (df.join(keys, on=["user_id", "item_id", "timestamp"], how="left")
                .with_columns(pl.col("_keep_first").fill_null(1)))
    else:
        df = df.with_columns(pl.lit(1, dtype=pl.Int8).alias("_keep_first"))

    protected_before = int((df["_keep_first"] == 0).sum())
    df = (
        df.sort(["user_id", "item_id", "_keep_first", "timestamp"])
        .unique(subset=["user_id", "item_id"], keep="first", maintain_order=True)
    )
    protected_after = int((df["_keep_first"] == 0).sum())
    df = df.drop("_keep_first")

    print(f"[dedup] rows: {before:,} -> {df.height:,}")
    if protected_keys:
        print(f"[dedup] manifest targets protected: {protected_before:,} -> {protected_after:,} "
              f"({protected_before - protected_after:,} had more than one target per (user, item))")
    return df

def load_manifest_split_keys(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    valid_keys = {(r["user_id"], r["item_id"], r["timestamp"]) for r in manifest["valid"]}
    test_keys = {(r["user_id"], r["item_id"], r["timestamp"]) for r in manifest["test"]}
    return valid_keys, test_keys

def add_split_and_ids(df: pl.DataFrame, valid_keys, test_keys):
    if not valid_keys or not test_keys:
        raise ValueError(
            "add_split_and_ids: valid_keys / test_keys are empty. Pass the manifest written by "
            "preprocessing.py with --manifest; recomputing the split here is not supported."
        )

    df = df.sort(["user_id", "timestamp", "item_id"])

    print(f"[split] using the manifest split: valid_keys={len(valid_keys):,} test_keys={len(test_keys):,}")

    key_df = pl.concat(
        [
            pl.DataFrame(
                list(valid_keys), schema=["user_id", "item_id", "timestamp"], orient="row"
            ).with_columns(pl.lit("valid").alias("_manifest_split")),
            pl.DataFrame(
                list(test_keys), schema=["user_id", "item_id", "timestamp"], orient="row"
            ).with_columns(pl.lit("test").alias("_manifest_split")),
        ]
    ).with_columns(
        [
            pl.col("user_id").cast(pl.Utf8),
            pl.col("item_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Int64),
        ]
    )

    df = df.join(key_df, on=["user_id", "item_id", "timestamp"], how="left")
    df = df.with_columns(
        pl.col("_manifest_split").fill_null("train").alias("split")
    ).drop("_manifest_split")

    n_valid = df.filter(pl.col("split") == "valid").height
    n_test = df.filter(pl.col("split") == "test").height
    if n_valid != len(valid_keys) or n_test != len(test_keys):
        print(
            f"[split][WARN] some manifest valid/test keys are absent from this dataframe "
            f"(manifest valid={len(valid_keys):,} -> matched {n_valid:,}, "
            f"manifest test={len(test_keys):,} -> matched {n_test:,}). "
            f"k-core / dedup / sampling in this script may have produced a different row set than the manifest."
        )

    df = df.with_columns(
        [
            pl.len().over("user_id").alias("seq_len"),
            (pl.col("item_id").cum_count().over("user_id") - 1).alias("pos"),
        ]
    )

    users = (
        df.select("user_id")
        .unique()
        .sort("user_id")
        .with_row_index("uid", offset=1)
    )

    items = (
        df.select("item_id")
        .unique()
        .sort("item_id")
        .with_row_index("iid", offset=1)
    )

    df = df.join(users, on="user_id", how="left")
    df = df.join(items, on="item_id", how="left")

    return df, users, items

def write_train_sequences(df: pl.DataFrame, output_path: Path, max_seq_len: int):
    train = df.filter(pl.col("split") == "train")

    train_seq = (
        train.sort(["uid", "timestamp", "iid"])
        .group_by(["uid", "user_id"], maintain_order=True)
        .agg(
            [
                pl.col("iid").tail(max_seq_len).alias("item_seq"),
                pl.col("item_id").tail(max_seq_len).alias("raw_item_seq"),
                pl.col("timestamp").tail(max_seq_len).alias("time_seq"),
            ]
        )
    )

    train_seq.write_ndjson(output_path)
    print(f"[train_sequences] saved to {output_path}")

def write_query_instances(
    df: pl.DataFrame,
    split_name: str,
    output_path: Path,
    max_seq_len: int,
):
    use_cols = [
        "uid",
        "iid",
        "user_id",
        "item_id",
        "timestamp",
        "query",
        "review_title",
        "review_text",
        "is_c4_user",
        "split",
    ]

    pdf = (
        df.select(use_cols)
        .sort(["uid", "timestamp", "iid"])
        .to_pandas()
    )

    n_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for _uid, g in tqdm(pdf.groupby("uid", sort=False), desc=f"Writing {split_name} instances"):
            g = g.sort_values(["timestamp", "iid"]).reset_index(drop=True)

            for idx, row in g.iterrows():
                if row["split"] != split_name:
                    continue

                hist = g.iloc[:idx].tail(max_seq_len)

                if len(hist) == 0:
                    continue

                obj = {
                    "uid": int(row["uid"]),
                    "user_id": row["user_id"],
                    "target_iid": int(row["iid"]),
                    "target_item_id": row["item_id"],
                    "target_timestamp": int(row["timestamp"]),
                    "history_iids": [int(x) for x in hist["iid"].tolist()],
                    "history_item_ids": hist["item_id"].tolist(),
                    "history_timestamps": [int(x) for x in hist["timestamp"].tolist()],
                    "query": safe_text(row["query"]),
                    "split": split_name,
                    "is_c4_user": bool(row["is_c4_user"]),
                }

                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"[query_instances] {split_name}: {n_written:,} rows -> {output_path}")

def write_item_card_pool(df: pl.DataFrame, output_path: Path, max_reviews_per_item: int = 5):
    train = df.filter(pl.col("split") == "train")

    pool = (
        train.sort(["item_id", "timestamp"])
        .group_by(["iid", "item_id"], maintain_order=True)
        .agg(
            [
                pl.col("review_title").head(max_reviews_per_item).alias("review_titles"),
                pl.col("review_text").head(max_reviews_per_item).alias("review_texts"),
                pl.col("query").head(max_reviews_per_item).alias("train_queries"),
            ]
        )
    )

    pool.write_parquet(output_path)
    print(f"[item_card_pool] saved to {output_path}")

def verify_final_split_consistency(processed_dir: Path, valid_keys: set, test_keys: set):
    def read_keys(path):
        keys = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                keys.add((obj["user_id"], obj["target_item_id"], obj["target_timestamp"]))
        return keys

    final_valid = read_keys(processed_dir / "valid_query_instances.jsonl")
    final_test = read_keys(processed_dir / "test_query_instances.jsonl")

    valid_not_in_manifest = final_valid - valid_keys
    test_not_in_manifest = final_test - test_keys
    valid_in_test_keys = final_valid & test_keys
    test_in_valid_keys = final_test & valid_keys

    ok = True
    if valid_not_in_manifest:
        ok = False
        print(f"[verify][FAIL] {len(valid_not_in_manifest):,} keys in valid_query_instances.jsonl are not in the manifest")
    if test_not_in_manifest:
        ok = False
        print(f"[verify][FAIL] {len(test_not_in_manifest):,} keys in test_query_instances.jsonl are not in the manifest")
    if valid_in_test_keys:
        ok = False
        print(f"[verify][FAIL] {len(valid_in_test_keys):,} valid instances collide with manifest test keys")
    if test_in_valid_keys:
        ok = False
        print(f"[verify][FAIL] {len(test_in_valid_keys):,} test instances collide with manifest valid keys")

    print(f"[verify] manifest valid={len(valid_keys):,} -> final valid_query_instances={len(final_valid):,} "
          f"(instances without history are skipped; difference {len(valid_keys) - len(final_valid):,})")
    print(f"[verify] manifest test ={len(test_keys):,} -> final test_query_instances ={len(final_test):,} "
          f"(difference {len(test_keys) - len(final_test):,})")

    assert ok, "the final valid/test instance split disagrees with the manifest — see the [FAIL] lines above."
    print("[verify] OK: the final valid/test instance split is exactly a subset of the manifest.")

def check_canonical_population(processed_dir: Path, dataset: str) -> dict:
    expected_n, expected_id = canonical_n(dataset), canonical_sha256(dataset)
    with open(processed_dir / "test_query_instances.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    leak_path = leak_json_path(dataset)
    dropped = set()
    if os.path.exists(leak_path):
        with open(leak_path, encoding="utf-8") as f:
            dropped = {int(u) for u in json.load(f)["combined_dropped_uids_union_both_splits"]}
    kept = [o for o in rows if int(o["uid"]) not in dropped]
    obtained_n = len(kept)
    obtained_id = canonical_eval_id(kept)

    n_ok = expected_n is None or obtained_n == expected_n
    id_ok = expected_id is None or obtained_id == expected_id
    result = {"expected_n": expected_n, "obtained_n": obtained_n,
              "expected_sha256": expected_id, "obtained_sha256": obtained_id,
              "test_instances": len(rows), "leak_dropped_users": len(dropped),
              "match": n_ok and id_ok and not (expected_n is None and expected_id is None)}

    exp_str = f"{expected_n:,}" if expected_n is not None else "(not registered in config.DATASETS)"
    print(f"[canonical] Expected canonical {dataset} evaluation population: {exp_str}")
    print(f"[canonical] Obtained: {obtained_n:,}  "
          f"(test instances {len(rows):,} - leak-dropped users {len(dropped):,})")
    print(f"[canonical] sha256   expected={expected_id or '(not registered)'}")
    print(f"[canonical]          obtained={obtained_id}")
    if expected_n is None and expected_id is None:
        print("[canonical] SKIP — nothing to compare against. For a new dataset, register the count and "
              "sha256 above as canonical_n / canonical_sha256 in config.DATASETS and later runs verify themselves.")
    elif result["match"]:
        print("[canonical] PASS — count and content (users, targets, histories) match the reported evaluation set.")
    elif not n_ok:
        print(f"[canonical] FAIL — count mismatch {obtained_n:,} != {expected_n:,} ({obtained_n - expected_n:+,})")
    else:
        print("[canonical] FAIL — same count but different content (sha256 mismatch): the targets or "
              "histories differ from the reported evaluation set.")
    return result

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                         help="dataset key (books / video_games / beauty). "
                              "the defaults of --input, --manifest and --out_dir are derived from it.")
    parser.add_argument("--input", type=str, default=None,
                         help="output of review2query.py (parquet with the query column filled). "
                              "Defaults to data/preprocessed/{dataset}/queries.parquet")
    parser.add_argument("--manifest", type=str, default=None,
                         help="path to the split_manifest.json written by preprocessing.py. The split "
                              "comes only from this manifest and is never recomputed. Defaults to "
                              "data/preprocessed/{dataset}/split_manifest.json")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="parent directory for processed/. Defaults to data/preprocessed/{dataset}/")
    parser.add_argument("--k_core", type=int, default=0,
                         help="0 (default) does not re-apply k-core: the manifest is the single source "
                              "of truth for the split, and the released artifacts were built without "
                              "re-applying it. Pass a positive value to re-densify a new dataset.")
    parser.add_argument("--sample_users", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_reviews_per_item", type=int, default=5)

    parser.add_argument("--allow_new_population", action="store_true",
                         help="continue even when the evaluation population differs from canonical_n in "
                              "config.DATASETS. Use it for a dataset newly built with stages [1]-[4], "
                              "whose evaluation population is not the one behind the reported tables.")
    parser.add_argument("--skip_kcore", action="store_true",
                         help="same as --k_core 0 (backwards-compatible alias).")
    parser.add_argument("--skip_dedup", action="store_true",
                         help="keep duplicate (user, item) rows instead of reducing them to one.")

    args = parser.parse_args()

    args.input = args.input or queries_path(args.dataset)
    args.manifest = args.manifest or default_manifest_path(args.dataset)
    args.out_dir = args.out_dir or dataset_root(args.dataset)
    if not os.path.exists(args.manifest):
        raise FileNotFoundError(
            f"[prepare] manifest not found: {args.manifest}\n"
            f"       if preprocessing has not run yet:  python src/preprocessing.py --dataset {args.dataset}\n"
            f"       if only the pkl exists (older output; do not re-sample):\n"
            f"         python src/preprocessing.py --dataset {args.dataset} --from_pkl")
    print(f"[prepare] dataset={args.dataset}\n"
          f"          input   ={args.input}\n"
          f"          manifest={args.manifest}\n"
          f"          out_dir ={args.out_dir}")

    valid_keys, test_keys = load_manifest_split_keys(args.manifest)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    processed_dir = out_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {input_path}")
    df = pl.read_parquet(input_path)
    print(f"[load] raw shape: {df.shape}")
    print(f"[load] columns: {df.columns}")

    df = normalize_columns(df)

    print("[after normalize]")
    print(f"rows : {df.height:,}")
    print(f"users: {df['user_id'].n_unique():,}")
    print(f"items: {df['item_id'].n_unique():,}")

    if not args.skip_dedup:
        df = deduplicate_user_item(df, protected_keys=valid_keys | test_keys)

    skip_kcore = args.skip_kcore or args.k_core <= 0
    if not skip_kcore:
        print(f"[k-core] first {args.k_core}-core")
        df = iterative_kcore(df, k=args.k_core)
        print_shape("after first k-core", df)

    if args.sample_users and args.sample_users > 0:
        print("[sample] user-level sampling")
        df = sample_users(df, n_users=args.sample_users, seed=args.seed)

        if not skip_kcore:
            print(f"[k-core] re-{args.k_core}-core after sampling")
            df = iterative_kcore(df, k=args.k_core)
            print_shape("after re-k-core", df)

    df, users, items = add_split_and_ids(df, valid_keys, test_keys)

    keep_cols = [
        "uid",
        "iid",
        "user_id",
        "item_id",
        "timestamp",
        "review_title",
        "review_text",
        "query",
        "is_c4_user",
        "seq_len",
        "pos",
        "split",
    ]

    parts = {name: df.filter(pl.col("split") == name).select(keep_cols)
             for name in ("train", "valid", "test")}
    for name, part in parts.items():
        part.write_parquet(processed_dir / f"{name}.parquet")

    users.write_parquet(processed_dir / "user_map.parquet")
    items.write_parquet(processed_dir / "item_map.parquet")

    print("[split stats]")
    print(f"train rows: {parts['train'].height:,}")
    print(f"valid rows: {parts['valid'].height:,}")
    print(f"test rows : {parts['test'].height:,}")
    print(f"users     : {df['uid'].n_unique():,}")
    print(f"items     : {df['iid'].n_unique():,}")

    write_train_sequences(
        df=df,
        output_path=processed_dir / "train_sequences.jsonl",
        max_seq_len=args.max_seq_len,
    )

    for split_name in ("train", "valid", "test"):
        write_query_instances(
            df=df,
            split_name=split_name,
            output_path=processed_dir / f"{split_name}_query_instances.jsonl",
            max_seq_len=args.max_seq_len,
        )

    write_item_card_pool(
        df=df,
        output_path=processed_dir / "item_card_review_pool_train_only.parquet",
        max_reviews_per_item=args.max_reviews_per_item,
    )

    print("[verify] re-checking the final valid/test instance split against the manifest ...")
    verify_final_split_consistency(processed_dir, valid_keys, test_keys)

    population = check_canonical_population(processed_dir, args.dataset)

    preprocess_config = {
        "dataset": args.dataset,
        "input": str(input_path),
        "manifest": str(args.manifest),
        "item_id": "parent_asin -> item_id",
        "query_source": "existing query column in input parquet",
        "split": "manifest-joined (preprocessing.py; no independent recomputation)",
        "k_core": None if skip_kcore else args.k_core,
        "dedup": not args.skip_dedup,
        "sample_users": args.sample_users,
        "max_seq_len": args.max_seq_len,
        "canonical_population": population,
        "leakage_policy": {
            "train_sequences": "train interactions only",
            "item_card_pool": "train reviews only",
            "valid_test_query": "existing query column only; review text is not included in query instance files",
            "split_source_of_truth": str(args.manifest),
        },
    }

    with open(processed_dir / "preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_config, f, ensure_ascii=False, indent=2)

    print(f"[done] processed data saved to {processed_dir}")

    known = population["expected_n"] is not None or population["expected_sha256"] is not None
    if known and not population["match"] and not args.allow_new_population:
        raise SystemExit(
            f"[canonical][abort] the evaluation set differs from the reported one "
            f"(count {population['obtained_n']:,} vs {population['expected_n']:,}, "
            f"sha256 {population['obtained_sha256'][:16]}… vs "
            f"{(population['expected_sha256'] or '')[:16]}…).\n"
            f"  Continuing would train and evaluate on a different population, and "
            f"test.py --leak_drop would stop for the same reason once training is done.\n"
            f"  · Reproducing from the released artifacts: check that the inputs are the distributed "
            f"queries.parquet / split_manifest.json.\n"
            f"  · Building a new dataset with stages [1]-[4]: a different population is expected — "
            f"pass --allow_new_population, and treat the numbers as this build's, not the paper's.")

if __name__ == "__main__":
    main()
