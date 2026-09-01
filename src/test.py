# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from config import (canonical_n, canonical_eval_id, canonical_sha256,
                    embed_dir as default_embed_dir, leak_json_path,
                    processed_dir as default_processed_dir)
from model import (
    HaloSRSemanticAnchor, HaloQSHADataset, load_user_seen_items, evaluate, save_json,
)

def apply_leak_drop(test_data, dataset):
    path, expected_n = leak_json_path(dataset), canonical_n(dataset)
    expected_id = canonical_sha256(dataset)
    if not os.path.exists(path):
        raise SystemExit(f"leak-drop file not found: {path}\n"
                         f"  without --leak_drop the full test population is evaluated instead.")
    with open(path) as f:
        dropped = {int(u) for u in json.load(f)["combined_dropped_uids_union_both_splits"]}
    uids = [int(r["uid"]) for r in test_data.rows]
    keep = [i for i, u in enumerate(uids) if u not in dropped]
    if expected_n is not None and len(keep) != expected_n:
        raise SystemExit(f"leak-drop population mismatch: {len(keep)} != expected {expected_n} "
                         f"(total {len(uids)}, {len(dropped)} uids in the drop list). "
                         f"This is a different population from the reported one.")
    obtained_id = canonical_eval_id([test_data.rows[i] for i in keep])
    if expected_id is not None and obtained_id != expected_id:
        raise SystemExit(f"canonical evaluation-set fingerprint mismatch: sha256 {obtained_id[:16]}… != "
                         f"expected {expected_id[:16]}… (the count {len(keep):,} does match). "
                         f"The targets or histories differ from the reported evaluation set.")
    print(f"   [leak-drop] {len(uids):,} -> {len(keep):,}  ({len(uids)-len(keep)} users excluded; "
          f"canonical_n {expected_n} matches, sha256 {obtained_id[:16]}… matches)")
    return Subset(test_data, keep)

CTOR_KEYS = {
    "hidden_dim", "max_len", "num_blocks", "num_heads", "dropout", "topk", "soft_attn",
    "activate_on", "fusion_mode", "residual_alpha", "activation", "history_ablation",
    "fixed_gate", "objective", "backbone_fusion", "ln_eps",
}

def infer_paths(a: dict, cli_proc, cli_emb, dataset):
    if cli_proc and cli_emb:
        return Path(cli_proc), Path(cli_emb)
    if a.get("processed_dir") and a.get("embed_dir"):
        return Path(a["processed_dir"]), Path(a["embed_dir"])
    return (Path(cli_proc or default_processed_dir(dataset)),
            Path(cli_emb or default_embed_dir(dataset)))

def find_ckpt(d: Path):
    legacy = d / "best_halo_sr_semantic_anchor.pt"
    if legacy.exists():
        return legacy, None

    log_p = d / "train_log.json"
    if not log_p.exists():
        raise FileNotFoundError(f"{d}: no train_log.json, so the best epoch cannot be determined")
    raw = json.load(open(log_p))
    tr = [(r, r["valid_full"]) for r in raw if isinstance(r.get("valid_full"), dict)]
    if not tr:
        tr = [(r, r["valid"]) for r in raw if isinstance(r.get("valid"), dict)]
    if not tr:
        raise RuntimeError(f"{d}: no valid / valid_full records in the training log")
    best_r, best_v = max(tr, key=lambda x: x[1]["NDCG@10"])
    be = best_r["epoch"]

    p = d / f"epoch{be}.pt"
    if p.exists():
        return p, be

    have = {}
    for q in d.glob("epoch*.pt"):
        try:
            have[int(q.stem.replace("epoch", ""))] = q
        except ValueError:
            continue
    if have:
        cr, cv = max(((r, v) for r, v in tr if r["epoch"] in have), key=lambda x: x[1]["NDCG@10"])
        print(f"   [warn] {d}: no .pt for best epoch {be}; evaluating the best retained epoch "
              f"{cr['epoch']} instead (valid NDCG@10 {cv['NDCG@10']:.4f} "
              f"vs best {best_v['NDCG@10']:.4f})")
        return have[cr["epoch"]], cr["epoch"]
    if (d / "last.pt").exists():
        print(f"   [warn] {d}: no epoch*.pt; evaluating last.pt")
        return d / "last.pt", None
    raise FileNotFoundError(f"{d}: no usable checkpoint")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sdpa_math", action="store_true",
                    help="force the math SDPA kernel, as in training (evaluation is forward-only, so "
                         "results do not change; kept for consistency)")
    ap.add_argument("--out_name", type=str, default="test_result.json")
    ap.add_argument("--processed_dir", type=str, default=None)
    ap.add_argument("--embed_dir", type=str, default=None)
    ap.add_argument("--dataset", type=str, required=True,
                    help="dataset key (books / video_games / beauty). --processed_dir/--embed_dir/"
                         "the --leak_drop file is derived from it.")
    ap.add_argument("--leak_drop", action="store_true",
                    help="evaluate on the canonical population, excluding the uids listed in "
                         "population, checked against canonical_n in config.DATASETS.")
    ap.add_argument("--ks", type=str, default="10,50",
                    help="cutoffs for HR/NDCG (e.g. 1,5,10,20,50)")
    args = ap.parse_args()

    if args.sdpa_math:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[SDPA] math kernel forced")

    device = torch.device(args.device)
    rows = []

    for rd in args.run_dirs:
        d = Path(rd)
        ck_path, _ = find_ckpt(d)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        a = ck["args"] if isinstance(ck.get("args"), dict) else {}
        epoch = ck.get("epoch")
        state = ck.get("model_state_dict") or ck.get("state_dict") or ck

        processed_dir, embed_dir = infer_paths(a, args.processed_dir, args.embed_dir,
                                               args.dataset)
        item_embs = np.load(embed_dir / "item_embs.npy").astype(np.float32)
        num_items = item_embs.shape[0] - 1
        user_seen = load_user_seen_items(processed_dir)
        test_data = HaloQSHADataset(
            embed_dir / "test_instances.jsonl", embed_dir / "test_query_embs.npy",
            a.get("max_len", 200), num_items, user_seen,
        )

        kw = {k: v for k, v in a.items() if k in CTOR_KEYS}
        kw["use_dual_term"] = not a.get("no_dual_term", False)
        kw["no_backbone_fusion"] = True
        model = HaloSRSemanticAnchor(item_embs, item_embs.shape[1], num_items, **kw).to(device)
        model.load_state_dict(state)

        if args.leak_drop:
            test_data = apply_leak_drop(test_data, args.dataset)
        ks = tuple(int(x) for x in args.ks.split(","))
        full = evaluate(model, test_data, device, args.eval_batch_size, ks=ks, mode="full")
        qonly = evaluate(model, test_data, device, args.eval_batch_size, ks=ks, mode="query_only")

        save_json({"checkpoint": str(ck_path), "best_epoch": epoch,
                   "dataset": args.dataset, "leak_drop": bool(args.leak_drop), "ks": list(ks),
                   "processed_dir": str(processed_dir), "embed_dir": str(embed_dir), "args": a,
                   "test": full, "test_query_only_sanity": qonly},
                  d / args.out_name)
        print(f"\n── {d}  (best epoch {epoch})")
        print(f"   condition={a.get('condition')} gamma={a.get('gamma')} "
              f"history_ablation={a.get('history_ablation')}")
        print("   test        " + "  ".join(f"HR@{k}={full[f'HR@{k}']:.4f}" for k in ks)
              + "  " + "  ".join(f"N@{k}={full[f'NDCG@{k}']:.4f}" for k in ks)
              + f"  MRR={full['MRR']:.4f}  n={full['n_eval']:,}")
        print(f"   query-only  HR@10={qonly['HR@10']:.4f} NDCG@10={qonly['NDCG@10']:.4f}")
        if "rank_p50" in full: print(f"   rank        p50={full['rank_p50']}  mean={full.get('rank_mean',0):.1f}")
        print(f"   -> {d / args.out_name}")
        rows.append((str(d), a.get("condition"), a.get("gamma"), epoch, full))

    if len(rows) > 1:
        print("\n=== Summary (full-catalog test) ===")
        print(f"{'run':44s} {'cond':>5} {'gamma':>6} {'ep':>4} {'HR@10':>8} {'NDCG@10':>9} {'HR@50':>8} {'MRR':>8}")
        for name, cond, g, ep, m in rows:
            print(f"{name:44s} {str(cond):>5} {str(g):>6} {str(ep):>4} "
                  f"{m['HR@10']:>8.4f} {m['NDCG@10']:>9.4f} {m['HR@50']:>8.4f} {m['MRR']:>8.4f}")

if __name__ == "__main__":
    main()
