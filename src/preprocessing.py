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

def dedup_user_item(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values("timestamp", kind="stable")
    df = df.drop_duplicates(subset=["user_id", "parent_asin"], keep="first").reset_index(drop=True)
    print(f"[02][dedup][user-item] duplicate (user_id, parent_asin) rows reduced to the earliest timestamp: "
          f"{before:,} -> {len(df):,}")
    return df

def dedup_content_duplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    content_hash = (df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
                     ).map(lambda t: sha1_16(norm_text(t)))
    df = df.assign(_content_hash=content_hash)
    content_key_cols = ["parent_asin", "timestamp", "_content_hash"]

    dup_mask = df.duplicated(subset=content_key_cols, keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows == 0:
        print("[02][dedup][content-hash] key=(parent_asin,timestamp,content_hash): no duplicates")
        return df.drop(columns=["_content_hash"])

    dup_rows = df[dup_mask]
    conflict_groups = []
    n_cross_user_groups = 0
    for key, g in dup_rows.groupby(content_key_cols):
        if g["user_id"].nunique() == 1:
            continue
        n_cross_user_groups += 1
        if g["rating"].nunique(dropna=False) > 1:
            conflict_groups.append((key, g))

    if conflict_groups:
        print(f"[02][dedup][content-hash][CONFLICT] {len(conflict_groups):,} groups share review content "
              f"across user_ids but disagree on rating — stopping instead of picking one.")
        for key, g in conflict_groups[:10]:
            print(f"  key={key}")
            print(g[["user_id", "parent_asin", "timestamp", "rating"]].to_string())
        raise AssertionError(
            f"[02][dedup][content-hash] cannot resolve {len(conflict_groups):,} conflicting groups "
            f"automatically."
        )

    before = len(df)
    df = df.drop_duplicates(subset=content_key_cols, keep="first").reset_index(drop=True)
    df = df.drop(columns=["_content_hash"])
    print(f"[02][dedup][content-hash] removed {n_dup_rows:,} rows of identical review content "
          f"duplicated across user_ids ({n_cross_user_groups:,} groups): "
          f"{before:,} -> {len(df):,} rows")
    return df

def build_priority_user_pool(df: pd.DataFrame, ku: int, ki: int,
                              min_rating: float, min_textlen: int) -> set:
    text_len = df["text"].astype(str).str.len()
    good_mask = (df["rating"] >= min_rating) & (text_len >= min_textlen)
    good_df = df.loc[good_mask, KEY_COLS]
    print(f"[02][priority] rating>={min_rating} & len(text)>={min_textlen}: "
          f"{good_mask.sum():,}/{len(df):,} rows ({good_mask.mean() * 100:.2f}%)")
    good_core, n_iter = kcore_filter(good_df, ku, ki)
    users = set(good_core["user_id"].unique().tolist())
    print(f"[02][priority] good-only {ki}-core/{ku}-user converged (iters={n_iter}) -> "
          f"users={len(users):,} items={good_core['parent_asin'].nunique():,} rows={len(good_core):,}")
    return users

def sample_users(df: pd.DataFrame, priority_users: set, ku: int, ki: int,
                  n_target: int, seed: int) -> set:
    global_core, n_iter = kcore_filter(df[KEY_COLS], ku, ki)
    candidate_pool = set(global_core["user_id"].unique().tolist())
    print(f"[02][sample] global {ki}-core/{ku}-user candidate pool (iters={n_iter}): {len(candidate_pool):,} users")

    priority_in_pool = np.array(sorted(priority_users & candidate_pool))
    rng = np.random.default_rng(seed)
    n_sample = min(n_target, len(candidate_pool))
    if n_sample < n_target:
        print(f"[02][sample] candidate pool ({len(candidate_pool):,}) is smaller than the target "
              f"({n_target:,}); using the whole pool")

    if len(priority_in_pool) >= n_sample:
        sampled = set(rng.choice(priority_in_pool, size=n_sample, replace=False).tolist())
        print(f"[02][sample] priority users ({len(priority_in_pool):,}) already meet the target "
              f"({n_sample:,}); sampling from them only")
    else:
        remaining = np.array(sorted(candidate_pool - set(priority_in_pool.tolist())))
        n_fill = n_sample - len(priority_in_pool)
        fill = rng.choice(remaining, size=min(n_fill, len(remaining)), replace=False)
        sampled = set(priority_in_pool.tolist()) | set(fill.tolist())
        print(f"[02][sample] all {len(priority_in_pool):,} priority users kept + {len(fill):,} sampled at "
              f"random -> {len(sampled):,} users")
    return sampled

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
    for split in ("test", "valid"):
        st = stats[split]
        print(f"[02][diag] {tag} unseen_{split}_target_rate: {st['rate'] * 100:.1f}% "
              f"({st['n_unseen']}/{st['n_users']})")

def build_manifest(P: dict, dataset: str) -> dict:
    id2item = {v: k for k, v in P["i2id"].items()}
    id2user = {v: k for k, v in P["u2id"].items()}

    def rows_for(split_name):
        rows = []
        for uid, (hist_iids, _tgt_iid, q) in P[split_name].items():
            _user_id, asin, ts, h = P["inter_meta"][q]
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
                     help="path to the raw reviews parquet; derived from --dataset when omitted "
                          "(data/raw/{source_category}_reviews.parquet).")
    ap.add_argument("--dataset", type=str, required=True,
                     help="dataset key (books / video_games / beauty). Raw paths, output paths and the "
                          "prompt domain are all derived from it; see DATASETS in config.py.")
    ap.add_argument("--incore_item", type=int, default=CFG.incore_item)
    ap.add_argument("--incore_user", type=int, default=CFG.incore_user)
    ap.add_argument("--sample_users_pool", type=int, default=CFG.sample_users_pool)
    ap.add_argument("--priority_min_rating", type=float, default=CFG.sample_priority_min_rating)
    ap.add_argument("--priority_min_textlen", type=int, default=CFG.sample_priority_min_textlen)
    ap.add_argument("--sample_seed", type=int, default=CFG.sample_seed)
    ap.add_argument("--from_pkl", action="store_true",
                     help="skip stages 1-7 and restore only the manifest from an existing "
                          "interactions.pkl. Use this to attach a manifest to earlier outputs: "
                          "re-sampling would draw a different user set.")
    ap.add_argument("--pkl", type=str, default=None,
                     help="pkl to read in --from_pkl mode (default: data/preprocessed/{dataset}/interactions.pkl)")
    ap.add_argument("--out_dir", type=str, default=None,
                     help="write pkl / manifest / sample parquet / stats into this directory instead of "
                          "data/preprocessed/{dataset}/ (useful for testing).")
    args = ap.parse_args()

    ku, ki = args.incore_user, args.incore_item
    ds = args.dataset
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

    if args.from_pkl:
        src_pkl = args.pkl or out_path
        print(f"[02][from_pkl] restoring only the manifest from an existing pkl (no re-sampling)\n"
              f"              src={src_pkl}")
        with open(src_pkl, "rb") as f:
            P = pickle.load(f)
        missing = [k for k in ("u2id", "i2id", "valid", "test", "inter_meta") if k not in P]
        if missing:
            raise KeyError(f"[02][from_pkl] the pkl lacks keys needed to restore the manifest: {missing}")
        P.setdefault("n_users", len(P["u2id"]))
        P.setdefault("n_items", len(P["i2id"]))
        write_manifest(P, ds, mf_path)
        return

    reviews_path = args.reviews or raw_paths(ds)[0]

    df = pd.read_parquet(reviews_path, columns=RAW_COLS).dropna(subset=KEY_COLS)
    print(f"[02] loaded {len(df):,} rows, users={df['user_id'].nunique():,}, "
          f"items={df['parent_asin'].nunique():,} from {reviews_path}")

    df = dedup_content_duplicate_reviews(df)
    df = dedup_user_item(df)

    priority_users = build_priority_user_pool(
        df, ku, ki, args.priority_min_rating, args.priority_min_textlen)

    sampled_users = sample_users(df, priority_users, ku, ki, args.sample_users_pool, args.sample_seed)

    sub = df[df["user_id"].isin(sampled_users)].reset_index(drop=True)
    print(f"[02][restore] all interactions of the selected users: users={sub['user_id'].nunique():,} "
          f"items={sub['parent_asin'].nunique():,} rows={len(sub):,}")

    stats_before = item_user_stats(sub)
    print("[02] building the pre-filter pkl for diagnostics ...")
    P_raw = build_pkl(sub)

    filtered, n_iter = kcore_filter(sub.copy(), ku=ku, ki=ki)
    print(f"[02] in-sample k-core (item>={ki}, user>={ku}) converged in {n_iter} iterations")
    stats_after = item_user_stats(filtered)
    print(f"[02] after filtering: {stats_after}")

    P = build_pkl(filtered)
    unseen_before = unseen_target_rate(P)
    print_unseen_rates("before cleanup", unseen_before)

    n_dropped = drop_unseen_targets(P["valid"], P["test"], P["train"])
    unseen_after = unseen_target_rate(P)
    print(f"[02] users excluded from evaluation by removing unseen-target instances: {n_dropped:,} "
          f"(their train sequences are kept; targets are not moved into train)")
    print_unseen_rates("after cleanup", unseen_after)
    residual = [sp for sp in ("test", "valid") if unseen_after[sp]["rate"] not in (0.0, None)]
    if residual:
        print(f"[02] unseen_target_rate is still non-zero after cleanup ({', '.join(residual)}) — "
              f"check the filter logic. (continuing anyway)")

    n_evaluable = len(P["test"])
    print(f"[02] final users={P['n_users']:,} (target_final_users={CFG.target_final_users:,}), "
          f"evaluable users (valid=test)={n_evaluable:,}, items={P['n_items']:,}, "
          f"interactions={sum(len(s) for s in P['seqs'].values()):,}")

    write_manifest(P, ds, mf_path)

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

    print(f"\n[02] saved -> {out_path} (for training)")
    print(f"[02] saved -> {raw_path} (diagnostics only, not used for training)")
    print(f"[02] saved -> {sample_parquet_path} (input for review2query.py --fixed_input)")
    print(f"[02] saved -> {st_path}")

if __name__ == "__main__":
    main()
