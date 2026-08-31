# -*- coding: utf-8 -*-
"""train.py가 남긴 best checkpoint 를 full-catalog test 로 평가한다.

v2 는 학습만 하고 test 평가를 하지 않으므로(train_log.json/checkpoint_manifest.json 만 저장),
anchor 쪽 evaluate() 를 그대로 재사용해 anchor 런과 동일한 프로토콜로 재평가한다.
  - mode="full"       : 전체 카탈로그(모든 아이템) 랭킹, 학습에서 본 아이템은 마스킹
  - mode="query_only" : Direct Semantic Path 만 (raw query-item cosine) — sanity 기준선

사용 (run_dir는 여러 개를 한 번에 줄 수 있고, 2개 이상이면 마지막에 요약표가 나온다):
    python src/test.py runs/books_T1_seed2026 --dataset books --leak_drop
    python src/test.py runs/books_T1_seed202{6,7,8,9} --dataset books --leak_drop \
        --sdpa_math --device cuda:0
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from config import (canonical_n, embed_dir as default_embed_dir, leak_json_path,
                    processed_dir as default_processed_dir)
from model import (
    HaloSRSemanticAnchor, HaloQSHADataset, load_user_seen_items, evaluate, save_json,
)

# leak-dropped uid 파일과 기대 모집단 크기는 --dataset에서 유도한다
# (data/preprocessed/{dataset}/leak_dropped_uids.json, config.DATASETS의
#  canonical_n). 예전에는 "reports/22k_books_problem_validation/configs/..." 같은 final/ 상대경로
# 테이블이 하드코딩돼 있어서 cwd가 final/이 아니면 깨졌다.


def apply_leak_drop(test_data, dataset):
    """canonical 모집단(leak 유저 제외)으로 test set 을 줄인다. 개수가 기대와 다르면 중단.

    이 목록의 유저들은 preprocessing.py 7단계 drop_unseen_targets가 제외하는 유저와 동일하다
    (Books 실측: 양쪽 다 정확히 같은 395명). 따라서 manifest 기반으로 새로 만든 데이터셋에는
    애초에 이들이 없어 이 단계가 no-op가 되고, manifest 도입 이전 산출물에만 실제로 작동한다."""
    path, expected_n = leak_json_path(dataset), canonical_n(dataset)
    if not os.path.exists(path):
        raise SystemExit(f"leak-drop 파일이 없다: {path}\n"
                         f"  --leak_drop 없이 돌리면 전체 test 모집단으로 평가한다.")
    with open(path) as f:
        dropped = {int(u) for u in json.load(f)["combined_dropped_uids_union_both_splits"]}
    uids = [int(r["uid"]) for r in test_data.rows]
    keep = [i for i, u in enumerate(uids) if u not in dropped]
    if expected_n is not None and len(keep) != expected_n:
        raise SystemExit(f"leak-drop 모집단 불일치: {len(keep)} != 기대 {expected_n} "
                         f"(전체 {len(uids)}, drop 정의 {len(dropped)}개). "
                         f"서로 다른 모집단끼리 비교하지 않는다.")
    print(f"   [leak-drop] {len(uids):,} -> {len(keep):,}  (제외 {len(uids)-len(keep)}명, "
          f"canonical_n {expected_n} 일치)")
    return Subset(test_data, keep)


CTOR_KEYS = {
    "hidden_dim", "max_len", "num_blocks", "num_heads", "dropout", "topk", "soft_attn",
    "activate_on", "fusion_mode", "residual_alpha", "activation", "history_ablation",
    "fixed_gate", "objective", "backbone_fusion", "ln_eps",
}


def infer_paths(a: dict, cli_proc, cli_emb, dataset):
    """train.py 의 save_ckpt 는 args 를 A5_ARGS 기반으로 저장해 embed_dir/processed_dir 을
    담지 않는다. 우선순위: CLI 명시 > 체크포인트 args > --dataset 유도.

    예전에는 run 디렉터리 이름에 "books"/"beauty"/"vg" 가 들어있는지로 도메인을 추측하고
    final/ 상대경로 테이블(DOMAIN_PATHS)을 참조했다 — run 디렉터리 이름을 바꾸면 조용히
    다른 도메인의 데이터로 채점될 수 있는 구조라, --dataset 유도로 바꿨다."""
    if cli_proc and cli_emb:
        return Path(cli_proc), Path(cli_emb)
    if a.get("processed_dir") and a.get("embed_dir"):      # anchor 런은 args 에 들어 있다
        return Path(a["processed_dir"]), Path(a["embed_dir"])
    return (Path(cli_proc or default_processed_dir(dataset)),
            Path(cli_emb or default_embed_dir(dataset)))


def find_ckpt(d: Path):
    """v2 는 best 파일을 따로 두지 않는다. train_log.json 의 valid NDCG@10 최고 epoch 를
    고르고, 그 epoch{N}.pt (joint top-3 로 보존된 것) 를 쓴다.
    anchor 런처럼 best_*.pt 가 있으면 그걸 그대로 쓴다."""
    legacy = d / "best_halo_sr_semantic_anchor.pt"
    if legacy.exists():
        return legacy, None

    log_p = d / "train_log.json"
    if not log_p.exists():
        raise FileNotFoundError(f"{d}: train_log.json 이 없어 best epoch 를 특정할 수 없다")
    # anchor 는 "valid", v2 는 "valid_full" 키를 쓴다.
    raw = json.load(open(log_p))
    tr = [(r, r["valid_full"]) for r in raw if isinstance(r.get("valid_full"), dict)]
    if not tr:
        tr = [(r, r["valid"]) for r in raw if isinstance(r.get("valid"), dict)]
    if not tr:
        raise RuntimeError(f"{d}: valid/valid_full 기록이 없다")
    best_r, best_v = max(tr, key=lambda x: x[1]["NDCG@10"])
    be = best_r["epoch"]

    p = d / f"epoch{be}.pt"
    if p.exists():
        return p, be

    # top-3 에서 밀려나 삭제된 경우: 남아 있는 epoch*.pt 중 valid 가 가장 좋은 것
    have = {}
    for q in d.glob("epoch*.pt"):
        try:
            have[int(q.stem.replace("epoch", ""))] = q
        except ValueError:
            continue
    if have:
        cr, cv = max(((r, v) for r, v in tr if r["epoch"] in have), key=lambda x: x[1]["NDCG@10"])
        print(f"   [주의] {d}: best epoch {be} 의 .pt 가 없어 남아 있는 최선 epoch "
              f"{cr['epoch']} 로 평가 (valid NDCG@10 {cv['NDCG@10']:.4f} "
              f"vs best {best_v['NDCG@10']:.4f})")
        return have[cr["epoch"]], cr["epoch"]
    if (d / "last.pt").exists():
        print(f"   [주의] {d}: epoch*.pt 가 없어 last.pt 로 평가")
        return d / "last.pt", None
    raise FileNotFoundError(f"{d}: 쓸 수 있는 체크포인트가 없다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sdpa_math", action="store_true",
                    help="학습과 동일하게 math 커널 강제 (평가는 forward 뿐이라 결과 차이는 없지만 일관성 유지)")
    ap.add_argument("--out_name", type=str, default="test_result.json")
    ap.add_argument("--processed_dir", type=str, default=None)
    ap.add_argument("--embed_dir", type=str, default=None)
    ap.add_argument("--dataset", type=str, required=True,
                    help="데이터셋 키 (books / video_games / beauty). --processed_dir/--embed_dir/"
                         "--leak_drop 대상 파일을 이 값에서 유도한다.")
    ap.add_argument("--leak_drop", action="store_true",
                    help="{dataset}/leak_dropped_uids.json 의 uid 를 제외한 canonical "
                         "모집단으로 평가 (config.DATASETS의 canonical_n과 대조 검증). "
                         "manifest 기반으로 새로 만든 데이터셋에는 해당 유저가 애초에 없다.")
    ap.add_argument("--ks", type=str, default="10,50",
                    help="HR/NDCG 를 계산할 k 목록 (예: 1,5,10,20,50)")
    args = ap.parse_args()

    if args.sdpa_math:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[SDPA] math 커널 강제")

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
        print("\n=== 요약 (full-catalog test) ===")
        print(f"{'run':44s} {'cond':>5} {'gamma':>6} {'ep':>4} {'HR@10':>8} {'NDCG@10':>9} {'HR@50':>8} {'MRR':>8}")
        for name, cond, g, ep, m in rows:
            print(f"{name:44s} {str(cond):>5} {str(g):>6} {str(ep):>4} "
                  f"{m['HR@10']:>8.4f} {m['NDCG@10']:>9.4f} {m['HR@50']:>8.4f} {m['MRR']:>8.4f}")


if __name__ == "__main__":
    main()
