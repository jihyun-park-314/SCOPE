# -*- coding: utf-8 -*-
import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import embed_dir as default_embed_dir, processed_dir as default_processed_dir
from model import (
    HaloSRSemanticAnchor, HaloQSHADataset, load_user_seen_items, collate_fn, set_seed, save_json,
    evaluate as official_evaluate, pad_left,
)
from metrics import metrics_from_ranks

A5_ARGS = dict(
    hidden_dim=128, max_len=200, num_blocks=2, num_heads=2, dropout=0.2, topk=5,
    soft_attn=False, activate_on="hidden", fusion_mode="gated", residual_alpha=0.4,
    activation="soft", history_ablation="u_act_only", fixed_gate=None, objective="bpr",
    backbone_fusion=False, epochs=200, patience=20, batch_size=128, eval_batch_size=256,
    lr=1e-3, weight_decay=1e-5, seed=2026,
)

def build_model(item_embs, device, history_ablation=None):
    ha = history_ablation if history_ablation is not None else A5_ARGS["history_ablation"]
    return HaloSRSemanticAnchor(
        item_embs=item_embs, emb_dim=item_embs.shape[1], num_items=item_embs.shape[0] - 1,
        hidden_dim=A5_ARGS["hidden_dim"], max_len=A5_ARGS["max_len"], num_blocks=A5_ARGS["num_blocks"],
        num_heads=A5_ARGS["num_heads"], dropout=A5_ARGS["dropout"], topk=A5_ARGS["topk"],
        soft_attn=A5_ARGS["soft_attn"], activate_on=A5_ARGS["activate_on"], fusion_mode=A5_ARGS["fusion_mode"],
        residual_alpha=A5_ARGS["residual_alpha"], activation=A5_ARGS["activation"], use_dual_term=True,
        history_ablation=ha, fixed_gate=A5_ARGS["fixed_gate"],
        objective=A5_ARGS["objective"], no_backbone_fusion=True, backbone_fusion=A5_ARGS["backbone_fusion"],
    ).to(device)

def act_raw_score(u_act, e_all, item_ids):
    act_n = F.normalize(u_act, dim=-1)
    return torch.sum(act_n * e_all[item_ids], dim=-1)

def joint_and_act_losses(model, history, delta_t, query_emb, pos, neg):
    u_star, q, e_all, diag = model.forward_user_vector(history, delta_t, query_emb, return_diag=True)
    pos_logits = model.score_against(u_star, q, pos, e_all)
    neg_logits = model.score_against(u_star, q, neg, e_all)
    L_joint = -F.logsigmoid(pos_logits - neg_logits).mean()

    u_act = diag["u_act"]
    act_pos = act_raw_score(u_act, e_all, pos)
    act_neg = act_raw_score(u_act, e_all, neg)
    L_act = -F.logsigmoid(act_pos - act_neg).mean()

    return L_joint, L_act, dict(u_star=u_star, q=q, e_all=e_all, diag=diag)

def inline_assert_act_raw_matches_u_star(u_star, e_all, diag, pos, neg):
    u_act = diag["u_act"]
    my_pos = act_raw_score(u_act, e_all, pos)
    my_neg = act_raw_score(u_act, e_all, neg)
    ref_pos = torch.sum(u_star * e_all[pos], dim=-1)
    ref_neg = torch.sum(u_star * e_all[neg], dim=-1)
    max_diff = max((my_pos - ref_pos).abs().max().item(), (my_neg - ref_neg).abs().max().item())
    print(f"[ASSERT inline] act_raw vs u_star-reconstructed: max_abs_diff={max_diff:.3e}")
    if max_diff >= 1e-5:
        raise RuntimeError(
            f"[ASSERT FAIL] act_raw != sum(u_star*e_all) (max_abs_diff={max_diff:.3e} >= 1e-5) "
            f"-- history_ablation / normalize_u_star differ from what was expected. Stopping."
        )

def _build_batch_no_neg(dataset, indices, max_len):
    hist_list, delta_list, target_list = [], [], []
    for idx in indices:
        obj = dataset.rows[idx]
        hist = [int(x) for x in obj["history_iids"]]
        hist_ts = [int(x) for x in obj.get("history_timestamps", [])]
        target = int(obj["target_iid"])
        target_ts = int(obj.get("target_timestamp", hist_ts[-1] if hist_ts else 0))
        hist_p = pad_left(hist, max_len, 0)
        ts_p = pad_left(hist_ts, max_len, 0)
        delta_days = [max(0.0, (target_ts - t) / 86_400_000.0) if t > 0 else 0.0 for t in ts_p]
        hist_list.append(hist_p)
        delta_list.append(delta_days)
        target_list.append(target)
    return {
        "history": torch.LongTensor(hist_list),
        "delta_t": torch.FloatTensor(delta_list),
        "query_emb": torch.FloatTensor(dataset.query_embs[indices]),
        "target": torch.LongTensor(target_list),
    }

def _mask_seen_(scores, history, target):
    scores[:, 0] = -1e9
    is_target = history.eq(target.view(-1, 1))
    seen_idx = torch.where(is_target, torch.zeros_like(history), history)
    scores.scatter_(1, seen_idx, torch.full_like(seen_idx, -1e9, dtype=scores.dtype))
    return scores

@torch.no_grad()
def evaluate_attention_only_no_rng(model, dataset, device, batch_size, max_len,
                                   ks=(10, 50), collect_gate_stats=False):
    was_training = model.training
    model.eval()
    n = len(dataset)
    ranks_all = []
    gate_vals = [] if collect_gate_stats else None
    captured = {}
    hook_handle = None
    if collect_gate_stats:
        def _hook(_module, _inp, out):
            captured["gate_logit"] = out.detach()
        hook_handle = model.W_f.register_forward_hook(_hook)
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        batch = _build_batch_no_neg(dataset, idxs, max_len)
        history = batch["history"].to(device)
        delta_t = batch["delta_t"].to(device)
        query_emb = batch["query_emb"].to(device)
        target = batch["target"].to(device)
        u_star, _q, e_all = model.forward_user_vector(history, delta_t, query_emb)
        if collect_gate_stats:
            gate_vals.append(torch.sigmoid(captured["gate_logit"]).reshape(-1).cpu())
        scores = u_star @ e_all.t()
        _mask_seen_(scores, history, target)
        sorted_items = torch.argsort(scores, dim=1, descending=True, stable=True)
        matches = sorted_items.eq(target.view(-1, 1))
        ranks_all.extend((matches.float().argmax(dim=1) + 1).cpu().tolist())
    if hook_handle is not None:
        hook_handle.remove()
    if was_training:
        model.train()
    metrics = metrics_from_ranks(ranks_all, ks=ks)
    gate_stats = None
    if collect_gate_stats and gate_vals:
        allg = torch.cat(gate_vals)
        gate_stats = {"mean": float(allg.mean()), "std": float(allg.std()),
                      "min": float(allg.min()), "max": float(allg.max())}
    return metrics, gate_stats

def batch_digest(history, pos, neg):
    h = hashlib.sha256()
    h.update(history.detach().cpu().numpy().tobytes())
    h.update(pos.detach().cpu().numpy().tobytes())
    h.update(neg.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def train_one_epoch(model, loader, optimizer, device, gamma, epoch, assert_state, history_ablation="u_act_only",
                    skip_nonfinite_step=False):
    n_skipped = 0
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="train")):
        history = batch["history"].to(device)
        delta_t = batch["delta_t"].to(device)
        query_emb = batch["query_emb"].to(device)
        pos = batch["target"].to(device)
        neg = batch["neg"].to(device)

        L_joint, L_act, extra = joint_and_act_losses(model, history, delta_t, query_emb, pos, neg)

        if epoch == 1 and batch_idx == 0 and not assert_state["done"]:
            if history_ablation == "u_act_only":
                inline_assert_act_raw_matches_u_star(extra["u_star"], extra["e_all"], extra["diag"], pos, neg)
            assert_state["done"] = True

        loss = L_joint + gamma * L_act

        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if skip_nonfinite_step and not torch.isfinite(gnorm):
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            continue
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
    if n_skipped:
        print(f"[skip] steps skipped for a non-finite grad norm: {n_skipped}")
    return total_loss / max(n_batches, 1)

def update_topk(topk_list, epoch, score, k=3):
    topk_list.append((score, epoch))
    topk_list.sort(key=lambda x: -x[0])
    return topk_list[:k]

def save_full_state(epoch, path, model, optimizer, best_joint, bad_count, joint_top3, attnonly_top3, log):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_joint": best_joint,
        "bad_count": bad_count,
        "joint_top3": joint_top3,
        "attnonly_top3": attnonly_top3,
        "log": log,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=str, required=True, choices=["T0", "T1", "M2", "M3", "M6"])
    ap.add_argument("--history_ablation", type=str, default="u_act_only",
                     choices=["h_n_only", "none", "u_act_only"],
                     help="h_n_only | none (learned gate) | u_act_only (default).")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dataset", type=str, required=True,
                    help="dataset key (books / video_games / beauty); the defaults of --processed_dir "
                         "and --embed_dir are derived from it.")
    ap.add_argument("--processed_dir", type=str, default=None,
                    help="defaults to data/preprocessed/{dataset}/processed")
    ap.add_argument("--embed_dir", type=str, default=None,
                    help="defaults to data/preprocessed/{dataset}/embeddings")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=A5_ARGS["epochs"])
    ap.add_argument("--patience", type=int, default=A5_ARGS["patience"])
    ap.add_argument("--batch_size", type=int, default=A5_ARGS["batch_size"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verify_only_batches", type=int, default=0,
                     help="0 trains normally; a positive value runs only that many batches, saves a "
                          "digest and the loss, and exits (a quick equivalence check).")
    ap.add_argument("--verify_out", type=str, default=None)
    ap.add_argument("--resume_from", type=str, default=None,
                     help="path to full_state_last.pt: restores model, optimizer, RNG and early-stop "
                          "state and continues from the epoch after the last completed one. The "
                          "last.pt / epoch{N}.pt files cannot be resumed from.")
    ap.add_argument("--skip_nonfinite_step", action="store_true",
                    help="skip steps whose grad norm is inf or nan (off by default).")
    ap.add_argument("--sdpa_math", action="store_true",
                    help="force the math kernel for scaled_dot_product_attention. The fused "
                         "mem-efficient kernel can produce exploding gradients on heavily padded "
                         "masks, which leads to NaNs (off by default).")
    args = ap.parse_args()

    args.processed_dir = args.processed_dir or default_processed_dir(args.dataset)
    args.embed_dir = args.embed_dir or default_embed_dir(args.dataset)
    print(f"[train] dataset={args.dataset}\n"
          f"        processed_dir={args.processed_dir}\n"
          f"        embed_dir    ={args.embed_dir}")

    if args.sdpa_math:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[SDPA] math kernel forced (flash / mem-efficient disabled)")

    NO_AUX_CONDITIONS = {"T0", "M2", "M3"}
    if args.condition in NO_AUX_CONDITIONS:
        assert args.gamma == 0.0, f"{args.condition} requires gamma=0 (L_joint only, no auxiliary BPR)"

    if args.condition == "M6":
        assert args.gamma > 0.0, "this condition requires gamma>0 (auxiliary BPR)"
        assert args.history_ablation == "none", (
            f"this condition keeps the learned gate, so history_ablation must be 'none' "
            f"(received {args.history_ablation!r})")
    device = torch.device(args.device)

    set_seed(args.seed)
    embed_dir = Path(args.embed_dir)
    processed_dir = Path(args.processed_dir)
    item_embs = np.load(embed_dir / "item_embs.npy").astype(np.float32)
    num_items = item_embs.shape[0] - 1
    user_seen_items = load_user_seen_items(processed_dir)

    train_data = HaloQSHADataset(embed_dir / "train_instances.jsonl", embed_dir / "train_query_embs.npy",
                                  A5_ARGS["max_len"], num_items, user_seen_items)
    valid_data = HaloQSHADataset(embed_dir / "valid_instances.jsonl", embed_dir / "valid_query_embs.npy",
                                  A5_ARGS["max_len"], num_items, user_seen_items)

    model = build_model(item_embs, device, history_ablation=args.history_ablation)

    resume_state = (torch.load(args.resume_from, map_location="cpu", weights_only=False)
                    if args.resume_from else None)

    if resume_state is not None:
        assert args.verify_only_batches == 0, "--resume_from cannot be combined with --verify_only_batches"
        model.load_state_dict(resume_state["model_state_dict"])
        print(f"[Resume] loaded model/optimizer/RNG from {args.resume_from} "
              f"(last completed epoch={resume_state['epoch']})")
    else:
        q_only_valid = official_evaluate(model, valid_data, device, A5_ARGS["eval_batch_size"], mode="query_only")
        print("[Sanity: epoch-0 query-only mode]", q_only_valid)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=A5_ARGS["lr"], weight_decay=A5_ARGS["weight_decay"])

    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        random.setstate(resume_state["rng_python"])
        np.random.set_state(resume_state["rng_numpy"])
        torch.set_rng_state(resume_state["rng_torch_cpu"])
        if torch.cuda.is_available() and resume_state.get("rng_torch_cuda") is not None:
            torch.cuda.set_rng_state_all(resume_state["rng_torch_cuda"])

    if args.verify_only_batches > 0:
        digests = []
        model.train()
        assert_state = {"done": False}
        it = iter(train_loader)
        for batch_idx in range(args.verify_only_batches):
            batch = next(it)
            history = batch["history"].to(device)
            delta_t = batch["delta_t"].to(device)
            query_emb = batch["query_emb"].to(device)
            pos = batch["target"].to(device)
            neg = batch["neg"].to(device)

            L_joint, L_act, extra = joint_and_act_losses(model, history, delta_t, query_emb, pos, neg)
            if batch_idx == 0 and not assert_state["done"]:
                if args.history_ablation == "u_act_only":
                    inline_assert_act_raw_matches_u_star(extra["u_star"], extra["e_all"], extra["diag"], pos, neg)
                assert_state["done"] = True
            loss = L_joint + args.gamma * L_act

            digests.append({
                "batch_idx": batch_idx, "digest": batch_digest(history, pos, neg),
                "loss": float(loss.item()),
            })

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        out_path = Path(args.verify_out) if args.verify_out else Path(args.out_dir) / "verify_first_batches.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json({"q_only_valid_sanity": q_only_valid, "batches": digests}, out_path)
        print(f"[Verify-only] saved {out_path}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_state_path = out_dir / "full_state_last.pt"

    if resume_state is not None:
        joint_top3 = resume_state["joint_top3"]
        attnonly_top3 = resume_state["attnonly_top3"]
        log = resume_state["log"]
        best_joint, bad_count = resume_state["best_joint"], resume_state["bad_count"]
        start_epoch = resume_state["epoch"] + 1
    else:
        joint_top3, attnonly_top3 = [], []
        log = []
        best_joint, bad_count = -1.0, 0
        start_epoch = 1
    last_path = out_dir / "last.pt"
    assert_state = {"done": resume_state is not None}

    def save_ckpt(epoch, path):
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "args": {**A5_ARGS, "history_ablation": args.history_ablation,
                             "condition": args.condition, "gamma": args.gamma, "seed": args.seed}}, path)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.gamma, epoch, assert_state,
                                     skip_nonfinite_step=args.skip_nonfinite_step,
                                      history_ablation=args.history_ablation)

        valid_result = official_evaluate(model, valid_data, device, A5_ARGS["eval_batch_size"], mode="full")
        attn_result, gate_stats = evaluate_attention_only_no_rng(
            model, valid_data, device, A5_ARGS["eval_batch_size"], A5_ARGS["max_len"],
            collect_gate_stats=(args.history_ablation == "none"))
        tau_sem = float(F.softplus(model.log_tau_sem).item())
        tau_beh = float(F.softplus(model.log_tau_beh).item())

        entry = {
            "epoch": epoch, "train_loss": train_loss, "valid_full": valid_result,
            "valid_attention_only": attn_result,
            "tau_sem": tau_sem, "tau_beh": tau_beh, "tau_ratio": tau_sem / tau_beh,
        }
        if gate_stats is not None:
            entry["gate_stats"] = gate_stats
        log.append(entry)
        save_json(log, out_dir / "train_log.json")
        gate_str = (f" gate(mean/std/min/max)={gate_stats['mean']:.3f}/{gate_stats['std']:.3f}/"
                    f"{gate_stats['min']:.3f}/{gate_stats['max']:.3f}") if gate_stats is not None else ""
        print(f"[Epoch {epoch}] loss={train_loss:.4f} valid_NDCG10={valid_result['NDCG@10']:.4f} "
              f"attn_only_NDCG10={attn_result['NDCG@10']:.4f} tau_ratio={tau_sem/tau_beh:.2f}{gate_str}")

        prev_joint_epochs = set(e for _, e in joint_top3)
        prev_attn_epochs = set(e for _, e in attnonly_top3)
        joint_top3 = update_topk(joint_top3, epoch, valid_result["NDCG@10"])
        attnonly_top3 = update_topk(attnonly_top3, epoch, attn_result["NDCG@10"])
        new_joint_epochs = set(e for _, e in joint_top3)
        new_attn_epochs = set(e for _, e in attnonly_top3)

        if epoch in new_joint_epochs or epoch in new_attn_epochs:
            save_ckpt(epoch, out_dir / f"epoch{epoch}.pt")
        evicted = (prev_joint_epochs | prev_attn_epochs) - (new_joint_epochs | new_attn_epochs)
        for e in evicted:
            p = out_dir / f"epoch{e}.pt"
            if p.exists():
                p.unlink()

        save_ckpt(epoch, last_path)

        if valid_result["NDCG@10"] > best_joint:
            best_joint, bad_count = valid_result["NDCG@10"], 0
        else:
            bad_count += 1

        save_full_state(epoch, full_state_path, model, optimizer, best_joint, bad_count,
                         joint_top3, attnonly_top3, log)

        if bad_count >= args.patience:
            print(f"[Early stop] epoch {epoch}, best_joint_valid_NDCG10={best_joint:.4f}")
            break

    manifest = {
        "condition": args.condition, "history_ablation": args.history_ablation,
        "gamma": args.gamma, "seed": args.seed,
        "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
        "joint_top3_epochs": [e for _, e in joint_top3],
        "attnonly_top3_epochs": [e for _, e in attnonly_top3],
        "last_epoch": log[-1]["epoch"], "best_joint_valid_NDCG10": best_joint,
    }
    save_json(manifest, out_dir / "checkpoint_manifest.json")
    print(f"[Done] {out_dir} manifest={manifest}")

if __name__ == "__main__":
    main()
