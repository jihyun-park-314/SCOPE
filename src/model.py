# -*- coding: utf-8 -*-

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import load_jsonl

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def pad_left(seq, max_len, pad_value=0):
    seq = [int(x) for x in seq]
    seq = seq[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq

def load_user_seen_items(processed_dir):
    path = Path(processed_dir) / "train_sequences.jsonl"
    rows = load_jsonl(path)

    seen = {}
    for obj in rows:
        uid = int(obj["uid"])
        seen[uid] = set(int(x) for x in obj.get("item_seq", []))

    return seen

class HaloQSHADataset(Dataset):

    def __init__(self, instance_path, query_emb_path, max_len, num_items, user_seen_items):
        rows = load_jsonl(instance_path)
        query_embs = np.load(query_emb_path).astype(np.float32)

        assert len(rows) == len(query_embs), (len(rows), len(query_embs))

        kept_rows, kept_embs = [], []
        for obj, emb in zip(rows, query_embs):
            if len(obj.get("history_iids", [])) == 0:
                continue
            if not obj.get("query", ""):
                continue
            kept_rows.append(obj)
            kept_embs.append(emb)

        self.rows = kept_rows
        self.query_embs = np.stack(kept_embs).astype(np.float32)
        self.max_len = max_len
        self.num_items = num_items
        self.user_seen_items = user_seen_items

        print(f"[Dataset] {instance_path}: {len(self.rows):,}")

    def __len__(self):
        return len(self.rows)

    def sample_negative(self, uid, target):
        exclude = self.user_seen_items.get(uid, set())

        for _ in range(100):
            x = random.randint(1, self.num_items)
            if x not in exclude and x != target:
                return x

        return random.randint(1, self.num_items)

    def __getitem__(self, idx):
        obj = self.rows[idx]

        hist = [int(x) for x in obj["history_iids"]]
        hist_ts = [int(x) for x in obj.get("history_timestamps", [])]
        target = int(obj["target_iid"])
        target_ts = int(obj.get("target_timestamp", hist_ts[-1] if hist_ts else 0))
        uid = int(obj["uid"])
        neg = self.sample_negative(uid, target)

        hist_p = pad_left(hist, self.max_len, 0)
        ts_p = pad_left(hist_ts, self.max_len, 0)

        delta_days = [
            max(0.0, (target_ts - t) / 86_400_000.0) if t > 0 else 0.0
            for t in ts_p
        ]

        return {
            "history": torch.LongTensor(hist_p),
            "delta_t": torch.FloatTensor(delta_days),
            "query_emb": torch.FloatTensor(self.query_embs[idx]),
            "target": torch.LongTensor([target]),
            "neg": torch.LongTensor([neg]),
        }

def collate_fn(batch):
    return {
        "history": torch.stack([x["history"] for x in batch], dim=0),
        "delta_t": torch.stack([x["delta_t"] for x in batch], dim=0),
        "query_emb": torch.stack([x["query_emb"] for x in batch], dim=0),
        "target": torch.cat([x["target"] for x in batch], dim=0),
        "neg": torch.cat([x["neg"] for x in batch], dim=0),
    }

class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(x.transpose(-1, -2))))))
        return out.transpose(-1, -2)

class SASRecBackbone(nn.Module):

    def __init__(self, hidden_dim, num_blocks, num_heads, dropout, ln_eps=1e-8):
        super().__init__()
        self.ln_eps = ln_eps
        self.attn_layernorms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.fwd_layernorms = nn.ModuleList()
        self.fwd_layers = nn.ModuleList()

        self.num_heads = num_heads

        for _ in range(num_blocks):
            self.attn_layernorms.append(nn.LayerNorm(hidden_dim, eps=ln_eps))
            self.attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.fwd_layernorms.append(nn.LayerNorm(hidden_dim, eps=ln_eps))
            self.fwd_layers.append(PointWiseFeedForward(hidden_dim, dropout))

        self.last_layernorm = nn.LayerNorm(hidden_dim, eps=ln_eps)

    def forward(self, x, pad_mask):
        seq_len = x.shape[1]
        device = x.device

        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

        combined = causal.unsqueeze(0) | pad_mask.unsqueeze(1)
        fully_masked = combined.all(dim=-1)
        diag = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
        combined = combined & ~(fully_masked.unsqueeze(-1) & diag)

        attn_mask = combined.repeat_interleave(self.num_heads, dim=0)

        keep = (~pad_mask).unsqueeze(-1).float()
        x = x * keep

        for i in range(len(self.attn_layers)):
            xn = self.attn_layernorms[i](x)
            attn_out, _ = self.attn_layers[i](
                xn, x, x, attn_mask=attn_mask, need_weights=False
            )
            x = x + attn_out
            x = x * keep

            xn2 = self.fwd_layernorms[i](x)
            ffn_out = self.fwd_layers[i](xn2)
            x = x + ffn_out
            x = x * keep

        return self.last_layernorm(x)

class HaloSRSemanticAnchor(nn.Module):
    def __init__(
        self,
        item_embs,
        emb_dim,
        num_items,
        hidden_dim=128,
        max_len=200,
        num_blocks=2,
        num_heads=2,
        dropout=0.2,
        topk=5,
        soft_attn=False,
        activate_on="hidden",
        fusion_mode="gated",
        residual_alpha=0.4,
        activation=None,
        use_dual_term=True,
        history_ablation="none",
        fixed_gate=None,
        objective="bpr",
        no_backbone_fusion=True,
        backbone_fusion=False,
        no_normalize_u_star=False,
        ln_eps=1e-8,
    ):
        super().__init__()

        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.topk = topk
        self.emb_scale = hidden_dim ** 0.5

        self.soft_attn = soft_attn
        assert activate_on in ("hidden", "raw")
        self.activate_on = activate_on

        assert activation in (None, "hard", "soft", "mean")
        self.activation = activation

        self.use_dual_term = use_dual_term

        assert history_ablation in ("none", "h_n_only", "u_act_only")
        self.history_ablation = history_ablation

        self.fixed_gate = fixed_gate

        assert objective in ("bpr", "bce")
        self.objective = objective

        assert no_backbone_fusion, (
            "this variant runs only with no_backbone_fusion=True: the shared W_p / s~_i are removed, "
            "so the card-gate fusion (g * s~_i) does not exist."
        )
        self.no_backbone_fusion = no_backbone_fusion

        self.backbone_fusion = backbone_fusion

        assert fusion_mode in ("gated", "residual")
        self.fusion_mode = fusion_mode
        self.residual_alpha = residual_alpha

        self.normalize_u_star = not no_normalize_u_star

        item_tensor = torch.tensor(item_embs, dtype=torch.float32)
        self.register_buffer("fixed_item_embs", item_tensor)

        self.W_attn = nn.Linear(emb_dim, hidden_dim)
        self.h_id = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        self.backbone = SASRecBackbone(hidden_dim, num_blocks, num_heads, dropout, ln_eps=ln_eps)

        if self.backbone_fusion:
            self.W_p_backbone = nn.Linear(emb_dim, hidden_dim)
            self.W_g_backbone = nn.Linear(hidden_dim * 2, hidden_dim)

        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(hidden_dim * 3, hidden_dim)

        self.log_gamma = nn.Parameter(torch.tensor(0.0))
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.log_tau_beh = nn.Parameter(torch.tensor(0.0))
        self.log_tau_sem = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def fused_item_vectors(self):
        if self.backbone_fusion:
            s_tilde_bb = self.W_p_backbone(self.fixed_item_embs)
            g = torch.sigmoid(self.W_g_backbone(torch.cat([self.h_id.weight, s_tilde_bb], dim=-1)))
            return self.h_id.weight + g * s_tilde_bb
        return self.h_id.weight

    def encode_history(self, history):
        e_all = self.fused_item_vectors()

        bsz, seq_len = history.shape
        x = e_all[history] * self.emb_scale

        positions = torch.arange(seq_len, device=history.device).unsqueeze(0).expand(bsz, seq_len)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        pad_mask = history.eq(0)
        h = self.backbone(x, pad_mask)

        return h, pad_mask, e_all, x

    def forward_user_vector(self, history, delta_t, query_emb, return_diag=False):
        h, pad_mask, e_all, raw_hist = self.encode_history(history)
        bsz, seq_len, hid = h.shape

        q_attn = self.W_attn(query_emb)

        qsha_source = h if self.activate_on == "hidden" else raw_hist

        k = self.W_k(qsha_source)
        v = self.W_v(qsha_source)

        gamma = F.softplus(self.log_gamma)
        tau = F.softplus(self.log_tau) + 1e-3

        sim = torch.einsum("bh,blh->bl", q_attn, k) / math.sqrt(hid)
        time_penalty = gamma * torch.log1p(delta_t.clamp_min(0.0))
        a = sim - time_penalty
        a = a.masked_fill(pad_mask, -1e9)

        dense = torch.softmax(a, dim=1)

        strategy = self.activation
        if strategy is None:
            strategy = "soft" if self.soft_attn else "hard"

        if strategy == "soft":
            alpha = dense
        elif strategy == "mean":
            nonpad_f = (~pad_mask).float()
            alpha = nonpad_f / nonpad_f.sum(dim=1, keepdim=True).clamp_min(1e-9)
        else:
            k_top = min(self.topk, seq_len)
            topk_vals, topk_idx = torch.topk(a, k=k_top, dim=1)

            hard = torch.zeros_like(a)
            hard.scatter_(1, topk_idx, torch.softmax(topk_vals / tau, dim=1))

            alpha = hard.detach() + dense - dense.detach()

        u_act = torch.einsum("bl,blh->bh", alpha, v)

        diag = None
        if return_diag:
            with torch.no_grad():
                alpha_safe = alpha.clamp_min(1e-12)
                ent = -(alpha_safe * alpha_safe.log()).sum(dim=1)
                eff_n = ent.exp()
                k_diag = min(2, seq_len)
                top_vals = torch.topk(alpha, k=k_diag, dim=1).values
                if k_diag == 2:
                    margin = top_vals[:, 0] - top_vals[:, 1]
                else:
                    margin = top_vals[:, 0]
                k_pos = min(3, seq_len)
                top_positions = torch.topk(alpha, k=k_pos, dim=1).indices
                nan_col = torch.full((bsz,), float("nan"), device=history.device)

                sim_masked = sim.masked_fill(pad_mask, -1e9)
                nonpad_idx_diag = torch.arange(seq_len, device=history.device).unsqueeze(0).expand(bsz, seq_len)
                nonpad_idx_diag = nonpad_idx_diag.masked_fill(pad_mask, -1)
                last_pos_diag = nonpad_idx_diag.max(dim=1).values.clamp_min(0)
                batch_idx_diag = torch.arange(bsz, device=history.device)
                sim_last = sim_masked[batch_idx_diag, last_pos_diag]
                sim_max = sim_masked.max(dim=1).values
                attn_top1_mass = top_vals[:, 0]
            diag = {
                "attn_entropy": ent,
                "attn_eff_n": eff_n,
                "attn_top1_top2_margin": margin,
                "attn_top_positions": top_positions,
                "gate_mean": nan_col,
                "gate_min": nan_col,
                "gate_max": nan_col,
                "h_n": None,
                "u_act": u_act,
                "alpha": alpha,
                "sim_last": sim_last,
                "sim_max": sim_max,
                "attn_top1_mass": attn_top1_mass,
                "u_star_raw_norm": nan_col,
            }

        if self.fusion_mode == "residual":
            u_star = q_attn + self.residual_alpha * u_act
            if self.normalize_u_star:
                u_star = F.normalize(u_star, dim=-1)
            if return_diag:
                return u_star, query_emb, e_all, diag
            return u_star, query_emb, e_all

        idx = torch.arange(seq_len, device=history.device).unsqueeze(0).expand(bsz, seq_len)
        idx = idx.masked_fill(pad_mask, -1)
        last_pos = idx.max(dim=1).values.clamp_min(0)
        batch_idx = torch.arange(bsz, device=history.device)
        h_n = h[batch_idx, last_pos]
        if return_diag:
            diag["h_n"] = h_n

        if self.history_ablation == "h_n_only":
            u_star = F.normalize(h_n, dim=-1) if self.normalize_u_star else h_n
            if return_diag:
                return u_star, query_emb, e_all, diag
            return u_star, query_emb, e_all
        if self.history_ablation == "u_act_only":
            u_star = F.normalize(u_act, dim=-1) if self.normalize_u_star else u_act
            if return_diag:
                return u_star, query_emb, e_all, diag
            return u_star, query_emb, e_all

        if self.fixed_gate is not None:
            g_u = torch.full_like(h_n[:, :1], float(self.fixed_gate)).expand_as(h_n)
        else:
            g_u = torch.sigmoid(self.W_f(torch.cat([h_n, u_act, q_attn], dim=-1)))

        u_star = g_u * h_n + (1.0 - g_u) * u_act
        if return_diag:
            diag["u_star_raw_norm"] = u_star.norm(dim=-1)
        if self.normalize_u_star:
            u_star = F.normalize(u_star, dim=-1)

        if return_diag:
            diag["gate_mean"] = g_u.mean(dim=1)
            diag["gate_min"] = g_u.min(dim=1).values
            diag["gate_max"] = g_u.max(dim=1).values
            return u_star, query_emb, e_all, diag
        return u_star, query_emb, e_all

    def score_against(self, u_star, query_emb, item_ids, e_all):
        tau_beh = F.softplus(self.log_tau_beh)
        beh_score = tau_beh * torch.sum(u_star * e_all[item_ids], dim=-1)

        has_dual = self.use_dual_term and self.fusion_mode != "residual"
        if not has_dual:
            return beh_score

        tau_sem = F.softplus(self.log_tau_sem)
        q_n = F.normalize(query_emb, dim=-1)
        item_n = F.normalize(self.fixed_item_embs[item_ids], dim=-1)
        sem_score = tau_sem * torch.sum(q_n * item_n, dim=-1)
        return beh_score + sem_score

    def full_scores(self, history, delta_t, query_emb, mode="full"):
        u_star, q, e_all = self.forward_user_vector(history, delta_t, query_emb)

        tau_sem = F.softplus(self.log_tau_sem)
        q_n = F.normalize(q, dim=-1)
        item_n = F.normalize(self.fixed_item_embs, dim=-1)
        sem_scores = tau_sem * (q_n @ item_n.t())

        if mode == "query_only":
            scores = sem_scores
        else:
            tau_beh = F.softplus(self.log_tau_beh)
            beh_scores = tau_beh * (u_star @ e_all.t())
            has_dual = self.use_dual_term and self.fusion_mode != "residual"
            scores = beh_scores + sem_scores if has_dual else beh_scores

        scores[:, 0] = -1e9
        return scores

@torch.no_grad()
def evaluate(model, dataset, device, batch_size=256, ks=(10, 50), mode="full"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    mrr = 0.0
    ranks_all = []
    n = 0

    for batch in tqdm(loader, desc=f"Eval ({mode})"):
        history = batch["history"].to(device)
        delta_t = batch["delta_t"].to(device)
        query_emb = batch["query_emb"].to(device)
        target = batch["target"].to(device)

        scores = model.full_scores(history, delta_t, query_emb, mode=mode)

        hist_cpu = history.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()

        for i in range(history.size(0)):
            seen = set(int(x) for x in hist_cpu[i].tolist() if int(x) != 0)
            for iid in seen:
                if iid != int(target_cpu[i]):
                    scores[i, iid] = -1e9

        sorted_items = torch.argsort(scores, dim=1, descending=True, stable=True)
        matches = sorted_items.eq(target.view(-1, 1))
        ranks = matches.float().argmax(dim=1) + 1

        for r in ranks.detach().cpu().tolist():
            r = int(r)
            ranks_all.append(r)
            n += 1
            mrr += 1.0 / r
            for k in ks:
                if r <= k:
                    hits[k] += 1.0
                    ndcgs[k] += 1.0 / math.log2(r + 1)

    result = {}
    for k in ks:
        result[f"HR@{k}"] = hits[k] / n
        result[f"NDCG@{k}"] = ndcgs[k] / n
    result["MRR"] = mrr / n
    result["n_eval"] = n

    ranks_sorted = sorted(ranks_all)
    result["rank_mean"] = sum(ranks_sorted) / len(ranks_sorted)
    result["rank_p50"] = ranks_sorted[int(len(ranks_sorted) * 0.50)]
    result["rank_p90"] = ranks_sorted[int(len(ranks_sorted) * 0.90)]

    return result
