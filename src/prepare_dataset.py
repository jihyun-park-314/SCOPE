# -*- coding: utf-8 -*-
"""
[bridge] src/ 파이프라인 산출물(review2query.py의 쿼리채움 parquet + preprocessing.py의
split_manifest.json) -> build_embeddings.py가 바로 읽는 표준 스키마로 변환.

구 HALO의 final/prepare_from_existing_query_parquet.py에서 가져온 것 — 변환/split/leakage 로직은
무수정이다.
SCOPE 정리판에서 바뀐 것은 경로 기본값뿐이다: --dataset 하나만 주면 --input/--manifest/--out_dir이
data/preprocessed/{dataset}/ 아래의 표준 파일명으로 유도된다(예전에는 세 경로를 매번 손으로
맞춰야 했고, 그 과정에서 manifest를 빠뜨리면 split이 어긋났다).

입력:
  --input     review2query.py --fixed_input ... --out <이 경로>로 만든, query 컬럼이 채워진
              parquet (user_id, parent_asin, timestamp, query, 선택적 title/text/is_c4_user).
              is_c4_user는 구버전 Books 산출물에만 실제 값이 들어있는 유산 컬럼이다
              (동봉 processed_Books_sample22k_queryA.parquet에 True 4,261건). 그 값을 잃지
              않으려고 읽기 경로만 남겨두었고, 이 컬럼을 새로 만드는 코드는 없다 —
              컬럼이 없는 입력에는 아래에서 False로 채운다.
  --manifest  preprocessing.py가 만든 split_manifest.json — split(valid/test)의
              단일 진실 공급원. 이 스크립트는 split을 독립적으로 재계산하지 않고 이 manifest의
              (user_id, item_id, timestamp) 키에만 의존한다(add_split_and_ids).

출력(--out_dir/processed/ 아래):
  train.parquet, valid.parquet, test.parquet, user_map.parquet, item_map.parquet,
  train_sequences.jsonl, {train,valid,test}_query_instances.jsonl,
  item_card_review_pool_train_only.parquet  <- build_embeddings.py --data_dir가 이 processed/를
  가리키면 item_map.parquet(asin<->iid)과 이 리뷰 풀을 바로 읽는다. LLM 시맨틱 카드
  (semantic_card.py가 만든 cards.jsonl)는 이 스크립트가 만들지 않고
  build_embeddings.py --card_path로 별도 결합된다(item_map.parquet의 asin<->iid로 매핑) —
  구 HALO의 final/build_halo_lite_embeddings.py가 쓰던 것과 동일한 연결 방식.

★ 이 스크립트는 k-core를 한 번 더 돌린다 (--k_core 5, 기본 켜짐).
  preprocessing.py[2]가 이미 in-sample (5,5)-core를 맞춰놨는데도 여기서 또 도는 이유는 그 사이에
  normalize_columns()가 **query가 빈 행을 버리기** 때문이다 — 그만큼 밀도가 떨어져 core가 깨질 수
  있다. 다만 그 결과로 manifest의 valid/test 타깃 행까지 떨어져 나가면 add_split_and_ids()가
  [split][WARN]을 찍고, 최종 인스턴스 수가 manifest보다 줄어든다.
  끄려면 --skip_kcore. 로직이므로 기본값은 그대로 둔다.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from config import (dataset_root, manifest_path as default_manifest_path,
                    queries_path)


def safe_text(x):
    if x is None:
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def print_shape(tag: str, df: pl.DataFrame) -> None:
    print(f"[{tag}] rows={df.height:,}, users={df['user_id'].n_unique():,}, "
          f"items={df['item_id'].n_unique():,}")


def iterative_kcore(df: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    """
    user/item iterative k-core.
    user와 item이 모두 k개 이상 interaction을 가질 때까지 반복.
    """
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
    """
    review2query.py 산출물(queries.parquet)의 컬럼을 HALO-SR용 이름으로 정리한다.
    parent_asin -> item_id / title -> review_title / text -> review_text
    """
    rename_map = {src: dst for src, dst in (("parent_asin", "item_id"),
                                            ("title", "review_title"),
                                            ("text", "review_text"))
                  if src in df.columns}
    df = df.rename(rename_map)

    required = ["user_id", "item_id", "timestamp", "query"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # optional columns 보강 (is_c4_user는 구버전 Books 산출물에만 실제 값이 있는 유산 컬럼)
    for col, default in (("review_title", ""), ("review_text", ""), ("is_c4_user", False)):
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    # 타입 정리
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

    # query 비어 있는 row 제거
    df = df.filter(
        pl.col("query").is_not_null()
        & (pl.col("query").str.strip_chars().str.len_chars() > 0)
    )

    return df


def deduplicate_user_item(df: pl.DataFrame) -> pl.DataFrame:
    """
    같은 유저가 같은 item에 여러 번 등장하면 가장 이른 timestamp만 유지.
    """
    before = df.height

    df = (
        df.sort(["user_id", "item_id", "timestamp"])
        .unique(subset=["user_id", "item_id"], keep="first", maintain_order=True)
    )

    after = df.height
    print(f"[dedup] rows: {before:,} -> {after:,}")

    return df


def load_manifest_split_keys(manifest_path: str):
    """
    preprocessing.py가 만든 immutable manifest에서 valid/test 타깃
    (user_id, item_id, timestamp) 키 집합을 읽는다.

    이 manifest가 카테고리의 split을 정의하는 단일 진실 공급원이다 — 카드 생성
    (semantic_card.py)도 같은 manifest를 읽으므로, 여기서 split을 독립적으로
    재계산하면(과거에 leave-last-two-out을 다시 계산했던 방식) 두 스크립트의 split이
    어긋날 수 있다. 실제로 22k Books에서 이 어긋남 때문에 788건의 valid/test 타깃 리뷰가
    카드 제외 대상에서 빠졌고, 그중 112건이 실제로 카드에 leak됐다.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    valid_keys = {(r["user_id"], r["item_id"], r["timestamp"]) for r in manifest["valid"]}
    test_keys = {(r["user_id"], r["item_id"], r["timestamp"]) for r in manifest["test"]}
    return valid_keys, test_keys


def add_split_and_ids(df: pl.DataFrame, valid_keys, test_keys):
    """
    uid/iid mapping 생성 + split 부여.

    split은 오직 manifest(load_manifest_split_keys)에서만 가져온다 — 이 함수는 더 이상
    leave-last-out을 독립적으로 재계산하지 않는다. valid_keys/test_keys는 필수 인자다.
    """
    if not valid_keys or not test_keys:
        raise ValueError(
            "add_split_and_ids: valid_keys/test_keys가 비어 있습니다. "
            "preprocessing.py로 만든 manifest를 --manifest로 지정했는지 확인하세요. "
            "split을 자체 재계산하는 경로는 (leakage 재발 방지를 위해) 제거됐습니다."
        )

    df = df.sort(["user_id", "timestamp", "item_id"])

    print(f"[split] manifest split 사용: valid_keys={len(valid_keys):,} test_keys={len(test_keys):,}")

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
            f"[split][WARN] manifest의 valid/test 키 중 일부가 이 df에 없습니다 "
            f"(manifest valid={len(valid_keys):,} -> matched {n_valid:,}, "
            f"manifest test={len(test_keys):,} -> matched {n_test:,}). "
            f"이 스크립트 자체의 k-core/dedup/sampling이 manifest와 다른 행 집합을 만들었을 수 있습니다."
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
    """
    SASRec 기본 학습용 sequence.
    train interaction만 사용.
    """
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
    """
    query-aware 학습/평가용 instance 생성.

    train:
      각 train target마다 이전 train history를 붙임.
      예: [10] + query_for_25 -> 25

    valid:
      valid target 이전 history를 붙임.

    test:
      test target 이전 history를 붙임.
      이때 history에는 valid item까지 포함됨.

    history는 split으로 거르지 않고 그 유저의 **이전 행 전부**(g.iloc[:idx])에서 가져온다.
    valid/test 타깃이 유저 시퀀스의 마지막 두 개라, train 타깃의 이전 행은 자동으로 전부 train이
    되고 test의 이전 행에는 valid가 포함된다 — 위 세 줄이 그 결과다.
    """
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

                # history가 없는 첫 interaction은 next-item/query-aware 학습에 부적합
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
    """
    item semantic card 생성용 review pool.
    반드시 train review만 사용한다.
    valid/test review는 leakage 방지를 위해 제외.
    """
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
    """
    최종 valid/test query instance 파일을 다시 읽어, manifest의 valid/test 키와
    정확히 일치하는지 재검사한다 (write_query_instances가 "history 없는 첫 interaction"을
    거르므로 최종 건수는 manifest보다 적을 수 있지만, 그 차이는 반드시 이 필터 때문이어야
    하고, 다른 split으로 새거나 하면 안 된다).
    """
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
        print(f"[verify][FAIL] valid_query_instances.jsonl에 manifest에 없는 키 {len(valid_not_in_manifest):,}건")
    if test_not_in_manifest:
        ok = False
        print(f"[verify][FAIL] test_query_instances.jsonl에 manifest에 없는 키 {len(test_not_in_manifest):,}건")
    if valid_in_test_keys:
        ok = False
        print(f"[verify][FAIL] valid instance인데 manifest test 키와 겹치는 것 {len(valid_in_test_keys):,}건")
    if test_in_valid_keys:
        ok = False
        print(f"[verify][FAIL] test instance인데 manifest valid 키와 겹치는 것 {len(test_in_valid_keys):,}건")

    print(f"[verify] manifest valid={len(valid_keys):,} -> final valid_query_instances={len(final_valid):,} "
          f"(no-history 등으로 누락 가능, {len(valid_keys) - len(final_valid):,}건 차이)")
    print(f"[verify] manifest test ={len(test_keys):,} -> final test_query_instances ={len(final_test):,} "
          f"({len(test_keys) - len(final_test):,}건 차이)")

    assert ok, "최종 valid/test instance split이 manifest와 어긋납니다 — 위 [FAIL] 항목을 확인하세요."
    print("[verify] OK: 최종 valid/test instance split이 manifest와 정확히 일치(부분집합)합니다.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                         help="데이터셋 키 (books / video_games / beauty). "
                              "--input/--manifest/--out_dir의 기본값을 이 값에서 유도한다.")
    parser.add_argument("--input", type=str, default=None,
                         help="review2query.py 산출물(query 컬럼이 채워진 parquet). 미지정 시 "
                              "data/preprocessed/{dataset}/queries.parquet")
    parser.add_argument("--manifest", type=str, default=None,
                         help="preprocessing.py가 만든 split_manifest.json 경로. "
                              "split은 이 manifest에서만 가져오며 독립 재계산하지 않는다. "
                              "미지정 시 data/preprocessed/{dataset}/split_manifest.json")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="processed/ 를 만들 상위 폴더. 미지정 시 data/preprocessed/{dataset}/")
    parser.add_argument("--k_core", type=int, default=5)
    parser.add_argument("--sample_users", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_reviews_per_item", type=int, default=5)

    parser.add_argument("--skip_kcore", action="store_true")
    parser.add_argument("--skip_dedup", action="store_true")

    args = parser.parse_args()

    args.input = args.input or queries_path(args.dataset)
    args.manifest = args.manifest or default_manifest_path(args.dataset)
    args.out_dir = args.out_dir or dataset_root(args.dataset)
    if not os.path.exists(args.manifest):
        raise FileNotFoundError(
            f"[prepare] manifest가 없습니다: {args.manifest}\n"
            f"       아직 전처리를 안 돌렸다면:  python src/preprocessing.py --dataset {args.dataset}\n"
            f"       pkl만 있고 manifest가 없다면(구버전 산출물, 재샘플링 금지):\n"
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
        df = deduplicate_user_item(df)

    if not args.skip_kcore:
        print(f"[k-core] first {args.k_core}-core")
        df = iterative_kcore(df, k=args.k_core)
        print_shape("after first k-core", df)

    if args.sample_users and args.sample_users > 0:
        print("[sample] user-level sampling")
        df = sample_users(df, n_users=args.sample_users, seed=args.seed)

        if not args.skip_kcore:
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

    print("[verify] 최종 valid/test instance split을 manifest와 재대조합니다...")
    verify_final_split_consistency(processed_dir, valid_keys, test_keys)

    preprocess_config = {
        "dataset": args.dataset,
        "input": str(input_path),
        "manifest": str(args.manifest),
        "item_id": "parent_asin -> item_id",
        "query_source": "existing query column in input parquet",
        "split": "manifest-joined (preprocessing.py; no independent recomputation)",
        "k_core": None if args.skip_kcore else args.k_core,
        "sample_users": args.sample_users,
        "max_seq_len": args.max_seq_len,
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


if __name__ == "__main__":
    main()
