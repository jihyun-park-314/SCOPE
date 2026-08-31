"""
[02] data/raw/{cat}_reviews.parquet -> dedup -> priority 유저 풀 -> 샘플링 -> in-sample k-core ->
     chronological split -> manifest 고정 -> interactions.pkl

이 스크립트가 끝나면 user/item/interaction 구성과 train/valid/test split이 완전히 고정되며
(8단계, manifest), 이후 어떤 스크립트도 split을 다시 계산하지 않는다 — review2query.py[3]은
이 pkl이 확정한 행에 쿼리만 채워 넣고, semantic_card.py[4]는 이 pkl의 아이템 목록과
manifest만 참조한다.

8단계:
  1. Dedup — 실행 순서대로 (a) -> (b)다.
     (a) 콘텐츠 해시 dedup: 동일 리뷰(title+text)가 접미사 붙은 다른 user_id로 원본에 재등장하는
         경우 제거. 키는 (parent_asin, timestamp, content_hash) (Video_Games 실측: 78행/141그룹).
         이걸 빼면 semantic_card.py의 카드 leakage assertion이 다시 실패한다(2026-07-24 사건).
         ★ keep="first"를 정렬하지 않은 원본 순서에 적용하므로, 살아남는 행이 parquet의 행 순서에
           의존한다. 같은 원본 파일을 쓰는 한 결정적이지만, 원본을 다시 받아 행 순서가 달라지면
           같은 그룹에서 다른 행이 남을 수 있다(rating이 다르면 아래에서 중단하므로, 남는 차이는
           동일 rating·동일 콘텐츠 행들 사이의 선택뿐이다).
     (b) (user_id, parent_asin) 중복 제거: timestamp 오름차순 정렬 후 가장 이른 것만 유지.
  2. Priority user pool: rating>=sample_priority_min_rating & len(text)>=sample_priority_min_textlen
     인 리뷰만으로 in-sample (incore_user, incore_item)-core가 되는 유저 집합.
  3. 사용자 샘플링: priority 유저 우선 선택, 부족하면 (전역 (incore_user,incore_item)-core를
     만족하는) 일반 후보 유저에서 무작위 보충. 전역 core로 후보를 한 번 좁혀두지 않으면 무작위로
     뽑힌 저밀도 유저 대부분이 5번 in-sample 재수렴에서 탈락해 sample_users_pool을 아무리 키워도
     target_final_users에 도달하기 어렵다 — 8단계 스펙의 "일반 후보"를 이 전역 core 풀로 해석.
  4. 선택된 사용자의 전체 interaction 복원: rating/text 품질과 무관하게 원본 그대로 전부 포함.
  5. In-sample iterative k-core 재수렴: user>=incore_user, item>=incore_item 동시 만족까지 반복.
  6. Chronological split: 유저별 timestamp 오름차순 정렬 -> 마지막=test, 두 번째 마지막=valid,
     나머지=train.
  7. Warm-start 정리: valid/test 타깃 아이템이 전체 유저의 train 카탈로그 어디에도 없으면 그
     인스턴스만 평가에서 제거한다 — 그 유저의 train 시퀀스는 그대로 두고, 타깃을 train으로
     옮기지 않는다(이전 02e_sample_to_pkl.py는 옮겼음 — 사용자 지적으로 변경, 2026-07-24).
  8. Split manifest 고정: user_id, history(item_id 목록), valid target, test target, source
     review_hash를 split_manifest.json에 저장. 이후 과정에서 변경하지 않는다.

실행:
  python src/preprocessing.py --dataset books

  # 이미 만들어진 pkl에서 manifest만 복원 (1~7단계 재실행 없음, 8단계만 수행):
  python src/preprocessing.py --dataset books --from_pkl

산출 (전부 data/preprocessed/{dataset}/ 아래):
  interactions.pkl      (학습용 — query는 아직 placeholder, review2query.py가 채움)
  interactions_raw.pkl  (필터 전, 진단용, 학습엔 안 씀)
  sample.parquet        (review2query.py의 --fixed_input 입력)
  split_manifest.json   (immutable, 8번 — semantic_card.py/prepare_dataset.py의 단일 진실 공급원)
  results/sample_stats_{dataset}.json

[--from_pkl: manifest 복원 모드]
★ 이미 interactions.pkl이 있는 데이터셋에 이 스크립트를 다시 돌릴 때는 반드시 --from_pkl을 붙인다.
  안 붙이면 1~7단계가 처음부터 다시 돌아 pkl/manifest/sample.parquet을 전부 덮어쓰는데,
  --sample_seed가 같아도 구버전 스크립트와 코드 경로가 달라 뽑히는 유저 집합이 바뀔 수 있다.
  그러면 이미 만들어둔 queries.parquet/cards.jsonl/embeddings와 짝이 맞지 않게 된다.

manifest는 8단계에서 P(pkl)만 보고 만들어지므로(build_manifest), 어떤 경로로 만든 pkl이든
u2id/i2id/valid/test/inter_meta만 온전하면 동일한 manifest를 재샘플링 없이 복원할 수 있다.
--from_pkl은 1~7단계를 통째로 건너뛰고 8단계만 수행한다.
"""
import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from config import (CFG, dataset_root, interactions_path, interactions_raw_path,
                    manifest_path as default_manifest_path, raw_paths, sample_path,
                    stats_path)
from utils import kcore_filter, norm_text, sha1_16

KEY_COLS = ["user_id", "parent_asin", "timestamp"]
RAW_COLS = KEY_COLS + ["rating", "title", "text"]


# ---------------- Step 1: dedup ----------------

def dedup_user_item(df: pd.DataFrame) -> pd.DataFrame:
    """동일 (user_id, parent_asin) 중복 제거 — 가장 이른 timestamp의 interaction만 유지."""
    before = len(df)
    df = df.sort_values("timestamp", kind="stable")
    df = df.drop_duplicates(subset=["user_id", "parent_asin"], keep="first").reset_index(drop=True)
    print(f"[02][dedup][user-item] (user_id,parent_asin) 중복 제거(최이른 timestamp 유지): "
          f"{before:,} -> {len(df):,}")
    return df


def dedup_content_duplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amazon 원본 데이터 자체에 같은 물리적 리뷰가 접미사 붙은 다른 user_id로 재등장하는 콘텐츠
    중복이 있다(Video_Games 실측: AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ vs
    AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ_2_2_1 — 같은 아이템·같은 timestamp·같은 title/text). dedup_
    user_item()은 user_id가 다르면 못 잡으므로 (parent_asin, timestamp, content_hash) 기준으로
    2차 dedup한다. rating까지 다르면 진짜 다른 이벤트일 수 있으므로 자동으로 고르지 않고 중단한다.
    """
    content_hash = (df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
                     ).map(lambda t: sha1_16(norm_text(t)))
    df = df.assign(_content_hash=content_hash)
    content_key_cols = ["parent_asin", "timestamp", "_content_hash"]

    dup_mask = df.duplicated(subset=content_key_cols, keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows == 0:
        print("[02][dedup][content-hash] key=(parent_asin,timestamp,content_hash): 중복 없음")
        return df.drop(columns=["_content_hash"])

    # 중복 그룹을 한 번만 순회하면서 (i) 교차 user_id 그룹 수와 (ii) rating 충돌 그룹을 함께 센다.
    # 예전에는 같은 groupby를 두 번(루프 + transform("nunique")) 돌았다.
    dup_rows = df[dup_mask]
    conflict_groups = []
    n_cross_user_groups = 0
    for key, g in dup_rows.groupby(content_key_cols):
        if g["user_id"].nunique() == 1:
            continue  # user_id까지 같으면 dedup_user_item()이 이미 처리했어야 할 케이스
        n_cross_user_groups += 1
        # dropna=False: rating은 KEY_COLS가 아니라 결측이 남아 있을 수 있고, 결측도 서로 다른
        # 값으로 취급해야 예전의 drop_duplicates() 판정과 같아진다.
        if g["rating"].nunique(dropna=False) > 1:
            conflict_groups.append((key, g))

    if conflict_groups:
        print(f"[02][dedup][content-hash][CONFLICT] 같은 리뷰 콘텐츠·다른 user_id인데 rating이 "
              f"다른 그룹 {len(conflict_groups):,}개 발견 — 자동으로 고르지 않고 중단합니다.")
        for key, g in conflict_groups[:10]:
            print(f"  key={key}")
            print(g[["user_id", "parent_asin", "timestamp", "rating"]].to_string())
        raise AssertionError(
            f"[02][dedup][content-hash] {len(conflict_groups):,}개의 충돌 그룹을 자동으로 "
            f"해소할 수 없습니다."
        )

    before = len(df)
    df = df.drop_duplicates(subset=content_key_cols, keep="first").reset_index(drop=True)
    df = df.drop(columns=["_content_hash"])
    print(f"[02][dedup][content-hash] 서로 다른 user_id로 중복된 동일 콘텐츠 리뷰 "
          f"{n_dup_rows:,}행({n_cross_user_groups:,}개 서로다른user_id 그룹) 제거 -> "
          f"{before:,} -> {len(df):,}행")
    return df


# ---------------- Step 2: priority user pool ----------------

def build_priority_user_pool(df: pd.DataFrame, ku: int, ki: int,
                              min_rating: float, min_textlen: int) -> set:
    """rating>=min_rating & len(text)>=min_textlen인 행만으로 in-sample core가 되는 유저 집합."""
    text_len = df["text"].astype(str).str.len()
    good_mask = (df["rating"] >= min_rating) & (text_len >= min_textlen)
    good_df = df.loc[good_mask, KEY_COLS]
    print(f"[02][priority] rating>={min_rating} & len(text)>={min_textlen}: "
          f"{good_mask.sum():,}/{len(df):,}행 ({good_mask.mean() * 100:.2f}%)")
    good_core, n_iter = kcore_filter(good_df, ku, ki)
    users = set(good_core["user_id"].unique().tolist())
    print(f"[02][priority] good-only {ki}-core/{ku}-user 수렴(iters={n_iter}) -> "
          f"users={len(users):,} items={good_core['parent_asin'].nunique():,} rows={len(good_core):,}")
    return users


# ---------------- Step 3: sampling ----------------

def sample_users(df: pd.DataFrame, priority_users: set, ku: int, ki: int,
                  n_target: int, seed: int) -> set:
    """priority 유저 우선 선택, 부족하면 전역 (ku,ki)-core 생존 풀에서 무작위 보충."""
    global_core, n_iter = kcore_filter(df[KEY_COLS], ku, ki)
    candidate_pool = set(global_core["user_id"].unique().tolist())
    print(f"[02][sample] 전역 {ki}-core/{ku}-user 후보 풀(iters={n_iter}): {len(candidate_pool):,}명")

    priority_in_pool = np.array(sorted(priority_users & candidate_pool))
    rng = np.random.default_rng(seed)
    n_sample = min(n_target, len(candidate_pool))
    if n_sample < n_target:
        print(f"[02][sample] ⚠ 후보 풀({len(candidate_pool):,}명)이 목표({n_target:,}명)보다 작아 "
              f"풀 전체를 사용합니다")

    if len(priority_in_pool) >= n_sample:
        sampled = set(rng.choice(priority_in_pool, size=n_sample, replace=False).tolist())
        print(f"[02][sample] 우선순위 유저({len(priority_in_pool):,}명)가 목표({n_sample:,}명) "
              f"이상 -> 우선순위 유저만으로 샘플링")
    else:
        remaining = np.array(sorted(candidate_pool - set(priority_in_pool.tolist())))
        n_fill = n_sample - len(priority_in_pool)
        fill = rng.choice(remaining, size=min(n_fill, len(remaining)), replace=False)
        sampled = set(priority_in_pool.tolist()) | set(fill.tolist())
        print(f"[02][sample] 우선순위 유저 {len(priority_in_pool):,}명 전부 채택 + 무작위 보충 "
              f"{len(fill):,}명 -> 총 {len(sampled):,}명")
    return sampled


# ---------------- Steps 6-8: split / cleanup / manifest ----------------

def item_user_stats(df: pd.DataFrame) -> dict:
    ic = df["parent_asin"].value_counts()
    uc = df["user_id"].value_counts()
    return {
        "n_users": int(df["user_id"].nunique()), "n_items": int(df["parent_asin"].nunique()),
        "n_inter": int(len(df)),
        "item_inter_mean": float(ic.mean()), "item_inter_median": float(ic.median()),
        "item_single_pct": float((ic == 1).mean() * 100),
        "user_inter_mean": float(uc.mean()), "user_inter_median": float(uc.median()),
    }


def build_pkl(df: pd.DataFrame) -> dict:
    """정렬 전 df -> u2id/i2id/시퀀스/leave-two-out 분할/inter_meta까지 만든 pkl dict.
    query는 아직 placeholder("") — review2query.py(03)가 나중에 채운다."""
    users = sorted(df["user_id"].unique())
    items = sorted(df["parent_asin"].unique())
    u2id = {u: i + 1 for i, u in enumerate(users)}
    i2id = {a: i + 1 for i, a in enumerate(items)}
    df = df.copy()
    df["uid"] = df["user_id"].map(u2id)
    df["iid"] = df["parent_asin"].map(i2id)

    df = df.sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)
    df["inter_idx"] = np.arange(len(df))

    seqs, ts_seqs, q_idx = {}, {}, {}
    for uid, g in df.groupby("uid", sort=False):
        seqs[uid] = g["iid"].tolist()[-CFG.max_seq_len - 2:]
        ts_seqs[uid] = g["timestamp"].tolist()[-CFG.max_seq_len - 2:]
        q_idx[uid] = g["inter_idx"].tolist()[-CFG.max_seq_len - 2:]

    train, valid, test = {}, {}, {}
    for uid, s in seqs.items():
        if len(s) < 3:
            train[uid] = s
            continue
        train[uid] = s[:-2]
        valid[uid] = (s[:-2], s[-2], q_idx[uid][-2])
        test[uid] = (s[:-1], s[-1], q_idx[uid][-1])

    pop = Counter(df["iid"])
    queries = [""] * len(df)

    review_text = df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
    text_hash = review_text.map(lambda t: sha1_16(norm_text(t)))
    inter_meta = {
        int(r.inter_idx): (r.user_id, r.parent_asin, int(r.timestamp), h)
        for r, h in zip(df.itertuples(), text_hash)
    }

    return dict(u2id=u2id, i2id=i2id, n_users=len(users), n_items=len(items),
                seqs=seqs, ts=ts_seqs, q_idx=q_idx,
                train=train, valid=valid, test=test,
                queries=queries, popularity=dict(pop), inter_meta=inter_meta)


def drop_unseen_targets(valid: dict, test: dict, train: dict) -> int:
    """item>=incore_item는 '전체 등장 수'만 보장하고 '최소 1회는 train에 남는다'는 보장은
    아니다 — 그런 타깃을 가진 인스턴스는 평가에서 제거한다. 유저의 train 시퀀스는 그대로
    두고, 타깃을 train으로 옮기지 않는다(2026-07-24, 사용자 지적으로 이전 동작에서 변경)."""
    all_train_items = set()
    for seq in train.values():
        all_train_items.update(seq)

    drop_uids = set()
    for uid, (_, tgt, _q) in test.items():
        if tgt not in all_train_items:
            drop_uids.add(uid)
    for uid, (_, tgt, _q) in valid.items():
        if tgt not in all_train_items:
            drop_uids.add(uid)

    for uid in drop_uids:
        valid.pop(uid, None)
        test.pop(uid, None)
    return len(drop_uids)


def unseen_target_rate(P: dict) -> dict:
    all_train_items = set()
    for seq in P["train"].values():
        all_train_items.update(seq)

    def rate(split):
        n_users = len(split)
        n_unseen = sum(1 for _hist, tgt, _q in split.values() if tgt not in all_train_items)
        return {"rate": (n_unseen / n_users) if n_users else None,
                "n_users": n_users, "n_unseen": n_unseen}
    return {"test": rate(P["test"]), "valid": rate(P["valid"])}


def print_unseen_rates(tag: str, stats: dict) -> None:
    """정리 전/후 진단 출력 (test -> valid 순서, 형식 동일)."""
    for split in ("test", "valid"):
        st = stats[split]
        print(f"[02][진단] {tag} unseen_{split}_target_rate: {st['rate'] * 100:.1f}% "
              f"({st['n_unseen']}/{st['n_users']})")


def build_manifest(P: dict, dataset: str) -> dict:
    """8번: user_id, history(item_id 목록), valid target, test target, source review_hash 고정."""
    id2item = {v: k for k, v in P["i2id"].items()}
    id2user = {v: k for k, v in P["u2id"].items()}

    def rows_for(split_name):
        rows = []
        for uid, (hist_iids, _tgt_iid, q) in P[split_name].items():
            _user_id, asin, ts, h = P["inter_meta"][q]   # user_id는 id2user[uid]로 쓴다
            rows.append({
                "user_id": id2user[uid],
                "item_id": asin,
                "timestamp": int(ts),
                "review_hash": h,
                "history": [id2item[i] for i in hist_iids],
            })
        return rows

    return {
        "dataset": dataset,
        "n_users": P["n_users"], "n_items": P["n_items"],
        "valid": rows_for("valid"), "test": rows_for("test"),
    }


def write_manifest(P: dict, dataset: str, manifest_path: str) -> dict:
    """8번 manifest를 만들어 저장한다 (main()과 --from_pkl 복원 모드가 공유)."""
    manifest = build_manifest(P, dataset)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"[02] saved -> {manifest_path} (valid={len(manifest['valid']):,}, "
          f"test={len(manifest['test']):,})")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", type=str, default=None,
                     help="원본 리뷰 parquet 경로. 미지정 시 --dataset에서 유도한다 "
                          "(data/raw/{원본카테고리}_reviews.parquet).")
    ap.add_argument("--dataset", type=str, required=True,
                     help="데이터셋 키 (books / video_games / beauty). 원본 경로·산출물 경로·"
                          "프롬프트 도메인이 전부 여기서 유도된다. config.py의 DATASETS 참조.")
    ap.add_argument("--incore_item", type=int, default=CFG.incore_item)
    ap.add_argument("--incore_user", type=int, default=CFG.incore_user)
    ap.add_argument("--sample_users_pool", type=int, default=CFG.sample_users_pool)
    ap.add_argument("--priority_min_rating", type=float, default=CFG.sample_priority_min_rating)
    ap.add_argument("--priority_min_textlen", type=int, default=CFG.sample_priority_min_textlen)
    ap.add_argument("--sample_seed", type=int, default=CFG.sample_seed)
    ap.add_argument("--from_pkl", action="store_true",
                     help="1~7단계를 건너뛰고, 이미 있는 interactions.pkl에서 8단계 manifest만 "
                          "복원한다. 구버전 스크립트로 만든 기존 산출물에 "
                          "manifest만 붙일 때 사용 — 재샘플링하면 유저 집합이 바뀌므로 반드시 이 모드로.")
    ap.add_argument("--pkl", type=str, default=None,
                     help="--from_pkl에서 읽을 pkl 경로 (기본: data/preprocessed/{dataset}/interactions.pkl)")
    ap.add_argument("--out_dir", type=str, default=None,
                     help="지정하면 pkl/manifest/sample-parquet/stats를 전부 이 폴더 하나에 저장 "
                          "(실제 data/preprocessed/{dataset}/를 건드리지 않고 테스트할 때 사용).")
    args = ap.parse_args()

    ku, ki = args.incore_user, args.incore_item
    ds = args.dataset
    # 산출물 경로는 여기서 한 번만 정하고, 아래 저장부는 전부 이 변수들만 쓴다
    # (예전에는 --out_dir을 계산만 하고 정작 저장은 CFG.data_dir로 나가던 버그가 있었다).
    # --out_dir을 주면 5개가 전부 그 폴더 하나에 떨어진다 — 실제 산출물을 건드리지 않고
    # 시험 실행할 때 쓴다. stats만 기본 경로에서 results/ 아래로 빠지는 점이 다르다.
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path, raw_path, mf_path, sample_parquet_path, st_path = (
            os.path.join(args.out_dir, name) for name in
            ("interactions.pkl", "interactions_raw.pkl", "split_manifest.json",
             "sample.parquet", "sample_stats.json"))
    else:
        os.makedirs(dataset_root(ds), exist_ok=True)
        os.makedirs(CFG.result_dir, exist_ok=True)
        out_path, raw_path = interactions_path(ds), interactions_raw_path(ds)
        mf_path, sample_parquet_path = default_manifest_path(ds), sample_path(ds)
        st_path = stats_path(ds, "sample")

    # ---- --from_pkl: 기존 pkl에서 manifest만 복원하고 종료 (1~7단계 스킵) ----
    if args.from_pkl:
        src_pkl = args.pkl or out_path
        print(f"[02][from_pkl] 기존 pkl에서 manifest만 복원합니다 (재샘플링 없음)\n"
              f"              src={src_pkl}")
        with open(src_pkl, "rb") as f:
            P = pickle.load(f)
        missing = [k for k in ("u2id", "i2id", "valid", "test", "inter_meta") if k not in P]
        if missing:
            raise KeyError(f"[02][from_pkl] pkl에 manifest 복원에 필요한 키가 없습니다: {missing}")
        P.setdefault("n_users", len(P["u2id"]))
        P.setdefault("n_items", len(P["i2id"]))
        write_manifest(P, ds, mf_path)
        return

    # ---- 원본 리뷰 경로: 명시 없으면 --dataset에서 유도 (카테고리 혼선 방지) ----
    reviews_path = args.reviews or raw_paths(ds)[0]

    df = pd.read_parquet(reviews_path, columns=RAW_COLS).dropna(subset=KEY_COLS)
    print(f"[02] loaded {len(df):,} rows, users={df['user_id'].nunique():,}, "
          f"items={df['parent_asin'].nunique():,} from {reviews_path}")

    # ---- Step 1: dedup (콘텐츠 해시 -> (user,item)) ----
    df = dedup_content_duplicate_reviews(df)
    df = dedup_user_item(df)

    # ---- Step 2: 우선순위 유저 풀 ----
    priority_users = build_priority_user_pool(
        df, ku, ki, args.priority_min_rating, args.priority_min_textlen)

    # ---- Step 3: 유저 샘플링 ----
    sampled_users = sample_users(df, priority_users, ku, ki, args.sample_users_pool, args.sample_seed)

    # ---- Step 4: 선택된 사용자의 전체 interaction 복원 (품질 필터 없이) ----
    sub = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)
    print(f"[02][restore] 선택 유저 전체 interaction: users={sub['user_id'].nunique():,} "
          f"items={sub['parent_asin'].nunique():,} rows={len(sub):,}")

    stats_before = item_user_stats(sub)
    print("[02] building raw(필터 전) pkl for diagnostics ...")
    P_raw = build_pkl(sub)

    # ---- Step 5: in-sample k-core 재수렴 ----
    filtered, n_iter = kcore_filter(sub.copy(), ku=ku, ki=ki)
    print(f"[02] in-sample k-core(item>={ki}, user>={ku}) 수렴까지 반복={n_iter}")
    stats_after = item_user_stats(filtered)
    print(f"[02] 필터 후: {stats_after}")

    # ---- Step 6: chronological split (build_pkl 내부에서 함께 계산) ----
    P = build_pkl(filtered)
    unseen_before = unseen_target_rate(P)
    print_unseen_rates("정리 전", unseen_before)

    # ---- Step 7: train에 한 번도 없는 타깃을 평가에서 제외 ----
    n_dropped = drop_unseen_targets(P["valid"], P["test"], P["train"])
    unseen_after = unseen_target_rate(P)
    print(f"[02] unseen 타깃 인스턴스 제거로 평가 제외된 유저: {n_dropped:,}명 "
          f"(train 시퀀스는 그대로 유지, 타깃을 train으로 옮기지 않음)")
    print_unseen_rates("정리 후", unseen_after)
    residual = [sp for sp in ("test", "valid") if unseen_after[sp]["rate"] not in (0.0, None)]
    if residual:
        # 예전 문구는 "중단 조건"이었는데 실제로 중단하지 않아 오해의 소지가 있었다.
        print(f"[02] ⚠⚠⚠ 정리 후에도 unseen_target_rate가 0%가 아닙니다 ({', '.join(residual)}) — "
              f"필터 로직을 점검하세요. (중단하지 않고 계속 진행합니다)")

    n_evaluable = len(P["test"])
    print(f"[02] 최종 유저={P['n_users']:,} (목표 target_final_users={CFG.target_final_users:,}), "
          f"평가 가능 유저(valid=test)={n_evaluable:,}, 아이템={P['n_items']:,}, "
          f"interactions={sum(len(s) for s in P['seqs'].values()):,}")

    # ---- Step 8: manifest 고정 (이후 아무도 split을 재계산하지 않는다) ----
    write_manifest(P, ds, mf_path)

    # ---- 저장: pkl, raw pkl, review2query.py 입력용 sample parquet ----
    with open(out_path, "wb") as f:
        pickle.dump(P, f)
    with open(raw_path, "wb") as f:
        pickle.dump(P_raw, f)

    sample_out = filtered[KEY_COLS + ["title", "text"]].copy()
    sample_out["query"] = ""
    sample_out["is_fallback"] = False
    sample_out.to_parquet(sample_parquet_path, index=False)

    stats_out = {
        "dataset": ds,
        "incore_item": ki, "incore_user": ku,
        "sample_users_pool": args.sample_users_pool,
        "priority_min_rating": args.priority_min_rating,
        "priority_min_textlen": args.priority_min_textlen,
        "n_iter_to_converge": n_iter,
        "before_incore": stats_before, "after_incore": stats_after,
        "n_dropped_unseen_instances": n_dropped,
        "n_evaluable_users": n_evaluable,
        "unseen_target_rate_before_cleanup": unseen_before,
        "unseen_target_rate_after_cleanup": unseen_after,
    }
    with open(st_path, "w") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)

    print(f"\n[02] saved -> {out_path} (학습용)")
    print(f"[02] saved -> {raw_path} (진단용, 학습엔 사용 안 함)")
    print(f"[02] saved -> {sample_parquet_path} (review2query.py --fixed_input 입력용)")
    print(f"[02] saved -> {st_path}")


if __name__ == "__main__":
    main()
