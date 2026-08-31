# -*- coding: utf-8 -*-
"""
Semantic-Anchored M15: M15(HaloSRQSHA, train_eval_halo_sr_qsha.py)의 공용 투영 W_p가
query-history attention과 query-item semantic scoring에 동시에 쓰이던 것을 분리한 변형.

semantic_path_audit_report.md(PROJECTOR DISTORTION GO)가 확인한 대로 공용 W_p(768->128)는
raw E5 공간의 effective rank를 ~88% 붕괴시키고, candidate_residual_audit_report.md(CALIBRATION GO)는
그 붕괴가 M15 자체의 semantic-only 랭킹을 raw Query-only보다 이미 약하게 만든다는 것을 보였다.
이 파일은 그 가설을 직접 검증하기 위해 두 경로를 완전히 분리한다:

- Attention projector(W_attn, 768->128): QSHA의 query-history attention/게이트에만 사용.
  기존 M15의 attention 구조·게이트 블렌드·SASRec 백본은 이 프로젝터 교체 외에는 변경하지 않음.
- Direct Semantic Path: raw E5 공간(768-dim)에서 query와 item semantic embedding의 코사인
  유사도를 그대로 사용(투영 없음) — tau_sem(softplus 양수 제약 스칼라)로 가중.
  이 경로는 raw Query-only와 rank 기준 100% 동일해야 한다(회귀 검증 대상,
  semantic_anchor_regression_checks.py).
- Behavioral residual: 기존 M15 Behavioral Path(u_star @ e_ID) 그대로, tau_beh(softplus 스칼라)로 가중.

final_score = tau_beh * (u_star . e_ID) + tau_sem * cosine(query, item_semantic)

post-hoc/static fusion이 아니라 동일 BPR objective로 end-to-end 학습되는 단일 모델이다.
W_p/s~_i(카드 게이트 융합 대상)는 이 변형에서 완전히 제거되므로 no_backbone_fusion=True로만 동작한다.

[본문에 나오는 파일 이름들]
train_eval_halo_sr_qsha.py / train_eval_sasrec_official.py / semantic_anchor_regression_checks.py는
전부 **구 HALO 저장소**의 파일이고 SCOPE에는 없다. 아래 주석에서 "M15와 동일", "원본과 동일"이라고
할 때의 비교 대상이 그것들이며, 이 파일이 그 구조를 무수정으로 옮겨온 것임을 밝히는 표기다.

[이 파일은 라이브러리다 — 직접 실행하지 않는다]
학습은 src/train.py, 평가는 src/test.py가 엔트리포인트이고, 이 파일은 두 스크립트가 import하는
모델/데이터셋/evaluate만 제공한다. 예전에는 여기에도 자체 main()과 train_one_epoch()이 있었지만
train.py가 이를 대체한 뒤로 아무도 호출하지 않았고(RNG 소비 궤적이 달라 결과도 재현되지 않는다),
학습 예산 기본값이 두 곳에 따로 존재해 어느 쪽이 진짜인지 헷갈리는 원인이었다.
학습 하이퍼파라미터의 단일 출처는 train.py의 A5_ARGS다.
"""

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
    """
    negative sampling 제외용: train_sequences.jsonl의 uid -> train item 집합.
    """
    path = Path(processed_dir) / "train_sequences.jsonl"
    rows = load_jsonl(path)

    seen = {}
    for obj in rows:
        uid = int(obj["uid"])
        seen[uid] = set(int(x) for x in obj.get("item_seq", []))

    return seen


class HaloQSHADataset(Dataset):
    """train_eval_halo_sr_qsha.HaloQSHADataset과 완전히 동일 (동일 instance/split 파일, 동일
    negative sampler, 동일 candidate set을 그대로 재사용하기 위해 그대로 복제)."""

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

        # 밀리초 -> 일 단위. pad(0)는 top-k 계산 전에 마스킹되므로 값 자체는 무해함.
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
    """공식 SASRec 구조: Conv1d(k=1) - ReLU - Conv1d(k=1), 각각 dropout. (M15와 완전히 동일, 미변경)"""

    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [B, L, H]
        out = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(x.transpose(-1, -2))))))
        return out.transpose(-1, -2)


class SASRecBackbone(nn.Module):
    """
    pmixer/SASRec.pytorch의 log2feats 구조를 그대로 따름. (M15와 완전히 동일, 미변경)
    LayerNorm(Q) -> MultiheadAttention(Q,K=x,V=x, causal) -> 잔차 -> timeline mask
    -> LayerNorm -> PointWiseFeedForward -> 잔차 -> timeline mask -> (블록 반복) -> 최종 LayerNorm
    """

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
        # x: [B, L, H], pad_mask: [B, L] (True = padding)
        seq_len = x.shape[1]
        device = x.device

        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )  # [L, L], True = 미래 위치(차단)

        # 좌측 패딩 시퀀스에서는 causal + key padding mask를 합치면 어떤 행(자기 자신이
        # 패딩인 위치)은 attend 가능한 key가 하나도 안 남아 softmax(all -inf)=NaN이 된다.
        # 그 NaN이 out_proj의 weight gradient(배치 전체에 대한 외적 합)에 0*NaN=NaN으로
        # 섞여 전체 파라미터를 오염시키므로, 출력에서 지우는 게 아니라 애초에 완전히
        # 막힌 행에는 자기 자신(diagonal)만 예외로 열어줘 NaN 자체가 생기지 않게 한다.
        # 그 행의 출력값 자체는 의미 없지만(해당 위치는 패딩이라 keep=0으로 뒤에서 지워짐),
        # 유효한 위치들이 이 패딩 key를 보는 것은 key_padding_mask가 그대로 막아 정확성에 영향 없다.
        combined = causal.unsqueeze(0) | pad_mask.unsqueeze(1)  # [B, L, L], True = 차단
        fully_masked = combined.all(dim=-1)  # [B, L]
        diag = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)  # [1, L, L]
        combined = combined & ~(fully_masked.unsqueeze(-1) & diag)

        attn_mask = combined.repeat_interleave(self.num_heads, dim=0)  # [B*num_heads, L, L]

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

        # 아래 스위치들은 M15(train_eval_halo_sr_qsha.HaloSRQSHA)와 완전히 동일한 의미·기본값을
        # 유지한다 (behavioral path/attention 로직 자체는 이 실험에서 손대지 않음).
        self.soft_attn = soft_attn
        assert activate_on in ("hidden", "raw")
        self.activate_on = activate_on

        assert activation in (None, "hard", "soft", "mean")
        self.activation = activation

        # use_dual_term=False면 Direct Semantic Path 항을 강제로 끈다(behavioral-only 진단용).
        # 이 실험의 production 설정(Phase 1, seed 2026)에서는 항상 True로 학습한다.
        self.use_dual_term = use_dual_term

        assert history_ablation in ("none", "h_n_only", "u_act_only")
        self.history_ablation = history_ablation

        self.fixed_gate = fixed_gate

        # 손실 함수 자체는 train.py(joint_and_act_losses)가 갖는다. 이 필드는 생성자 인자로
        # 계속 받아야 한다 — train.py의 save_ckpt가 A5_ARGS를 그대로 args에 저장하고,
        # test.py가 그 args로 모델을 재구성하므로 기존 체크포인트와의 호환에 필요하다.
        assert objective in ("bpr", "bce")
        self.objective = objective

        # Semantic-Anchored 변형은 W_p/s~_i(카드 게이트 융합 대상)를 아예 갖지 않으므로
        # no_backbone_fusion=False(카드를 Personalized Sequential Path 입력에 게이트 융합)는
        # 구조적으로 구현 불가능하다 — 항상 e_i=h_ID_i만 사용.
        assert no_backbone_fusion, (
            "Semantic-Anchored M15는 no_backbone_fusion=True로만 동작한다: "
            "공용 W_p/s~_i를 제거했으므로 카드 게이트 융합(g*s~_i) 자체가 존재하지 않는다."
        )
        self.no_backbone_fusion = no_backbone_fusion

        # Exp 2 (Decoupled vs Semantic-Infused Backbone, problem-validation task): opt-in, additive
        # only. Default False preserves the original Semantic-Anchored behavior byte-for-byte.
        # When True, a THIRD projector (W_p_backbone/W_g_backbone) -- separate from W_attn
        # (attention-only) and from the raw Direct Semantic Path -- fuses semantic content into the
        # SASRec input, mirroring train_eval_halo_sr_qsha.py's original M8 fusion formula
        # (e = h_ID + sigmoid(W_g[h_ID||s~])*s~), so the only difference vs the decoupled model is
        # whether the semantic vector reaches the SASRec input.
        self.backbone_fusion = backbone_fusion

        assert fusion_mode in ("gated", "residual")
        self.fusion_mode = fusion_mode
        self.residual_alpha = residual_alpha

        # no_normalize_u_star=True: u_star를 L2-normalize하지 않고 raw dot product로 스코어링한다
        # (component-ablation study A0/A3용 -- semantic term이 없는 no_dual_term=True 조건에서는
        # normalize가 스케일 보정 근거 없이 표현력만 깎는다; 공식 SASRec baseline
        # (train_eval_sasrec_official.py)도 raw dot product를 쓴다). 기본값 False는 기존 Full
        # Model/A2/A5/A6의 동작을 byte-for-byte 그대로 유지한다.
        self.normalize_u_star = not no_normalize_u_star

        item_tensor = torch.tensor(item_embs, dtype=torch.float32)
        self.register_buffer("fixed_item_embs", item_tensor)  # [N+1, E] frozen semantic 임베딩 (raw, 투영 없음)

        # ---- Attention-only projector: Direct Semantic Path와 완전히 분리 (핵심 변경점) ----
        self.W_attn = nn.Linear(emb_dim, hidden_dim)  # M15의 W_p와 동일 shape/init, QSHA attention/게이트 전용
        self.h_id = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        self.backbone = SASRecBackbone(hidden_dim, num_blocks, num_heads, dropout, ln_eps=ln_eps)

        if self.backbone_fusion:
            # Exp 2 only: separate from W_attn and from the raw Direct Semantic Path.
            self.W_p_backbone = nn.Linear(emb_dim, hidden_dim)
            self.W_g_backbone = nn.Linear(hidden_dim * 2, hidden_dim)

        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(hidden_dim * 3, hidden_dim)  # 최종 게이트, M15와 동일 구조 (입력이 q_attn으로 교체됨)

        # gamma(시간감쇠), tau(top-k temperature)는 M15와 동일. log_lambda는 제거하고
        # tau_beh/tau_sem(각각 behavioral/semantic 경로 가중치)으로 대체 — 둘 다 softplus 양수 제약.
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
        """no_backbone_fusion=True 고정이므로 e_i = h_ID_i만 사용 (M15의 카드 게이트 융합
        분기 자체가 이 변형에는 존재하지 않음 — W_p/s~_i 제거).

        Exp 2 (backbone_fusion=True) only: e_i = h_ID_i + sigmoid(W_g_backbone[h_ID_i||s~_i])*s~_i,
        s~_i = W_p_backbone(raw_item_semantic) -- a projector used ONLY here, never shared with
        W_attn or the raw Direct Semantic Path. Mirrors train_eval_halo_sr_qsha.py's original M8
        fusion formula exactly."""
        if self.backbone_fusion:
            s_tilde_bb = self.W_p_backbone(self.fixed_item_embs)
            g = torch.sigmoid(self.W_g_backbone(torch.cat([self.h_id.weight, s_tilde_bb], dim=-1)))
            return self.h_id.weight + g * s_tilde_bb
        return self.h_id.weight  # [N+1, H]

    def encode_history(self, history):
        e_all = self.fused_item_vectors()

        bsz, seq_len = history.shape
        x = e_all[history] * self.emb_scale  # [B, L, H]

        positions = torch.arange(seq_len, device=history.device).unsqueeze(0).expand(bsz, seq_len)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        pad_mask = history.eq(0)
        h = self.backbone(x, pad_mask)  # [B, L, H] contextualized hidden states

        return h, pad_mask, e_all, x

    def forward_user_vector(self, history, delta_t, query_emb, return_diag=False):
        h, pad_mask, e_all, raw_hist = self.encode_history(history)
        bsz, seq_len, hid = h.shape

        # 핵심 변경점: query-history attention/게이트 전용 투영. Direct Semantic Path는
        # 이 q_attn을 전혀 쓰지 않고 raw query_emb(768-dim)를 그대로 사용한다(아래 score_against/
        # full_scores 참조) — 두 경로가 어떤 학습 투영도 공유하지 않는다.
        q_attn = self.W_attn(query_emb)  # [B, H]

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
        else:  # "hard"
            k_top = min(self.topk, seq_len)
            topk_vals, topk_idx = torch.topk(a, k=k_top, dim=1)

            hard = torch.zeros_like(a)
            hard.scatter_(1, topk_idx, torch.softmax(topk_vals / tau, dim=1))

            # straight-through: forward = hard(top-k) 선택, backward gradient = dense softmax 기준.
            alpha = hard.detach() + dense - dense.detach()

        u_act = torch.einsum("bl,blh->bh", alpha, v)

        diag = None
        if return_diag:
            # ★ 현재 이 dict에서 실제로 읽히는 값은 train.joint_and_act_losses()의 diag["u_act"]
            #   하나뿐이다(evaluate_verbose를 제거한 뒤). 나머지 진단 필드는 매 학습 배치마다
            #   계산만 되고 버려진다 — 전부 no_grad 안이라 결과·RNG에는 영향이 없지만, 비용은
            #   든다. 진단이 다시 필요해질 때를 대비해 계산 자체는 남겨둔다.
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

                # ---- Phase 2 negative-transfer diagnostic (read-only, additive) ----
                # sim(q_attn, h_j): pre-time-decay, pre-softmax attention compatibility, gathered/
                # maxed over valid (non-pad) positions only. Deliberately NOT `a` (time-decayed) or
                # `alpha` (post-softmax) -- see plan "Design decisions" #1: this must measure raw
                # query-history alignment, undistorted by recency weighting or per-instance softmax
                # normalization over a variable-length support.
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

        # h_n: 마지막 non-pad 위치의 은닉상태 (recency 표상)
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
            # M15와 동일한 게이트 구조 — 입력만 q(=W_p(query_emb))에서 q_attn(=W_attn(query_emb))로 교체.
            g_u = torch.sigmoid(self.W_f(torch.cat([h_n, u_act, q_attn], dim=-1)))

        u_star = g_u * h_n + (1.0 - g_u) * u_act
        if return_diag:
            diag["u_star_raw_norm"] = u_star.norm(dim=-1)  # captured BEFORE normalize below
        if self.normalize_u_star:
            u_star = F.normalize(u_star, dim=-1)

        if return_diag:
            diag["gate_mean"] = g_u.mean(dim=1)
            diag["gate_min"] = g_u.min(dim=1).values
            diag["gate_max"] = g_u.max(dim=1).values
            return u_star, query_emb, e_all, diag
        return u_star, query_emb, e_all

    def score_against(self, u_star, query_emb, item_ids, e_all):
        """behavioral_score = tau_beh * (u_star . e_ID_item);
        semantic_score = tau_sem * cosine(raw_query, raw_item_semantic) — 어떤 학습 투영도
        거치지 않은 raw E5 공간에서 계산 (W_p/W_attn 둘 다 미사용)."""
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
        """mode="query_only"는 use_dual_term/fusion_mode와 무관하게 항상 Direct Semantic Path만
        반환한다 — tau_sem*cosine(raw_query, raw_item)은 raw Query-only의 rank와 항상 정확히
        일치해야 하는 회귀검증 대상이므로, 이 정의가 어떤 ablation 스위치에도 좌우되지 않게 한다."""
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
    """train_eval_halo_sr_qsha.evaluate()와 완전히 동일 구조 (model.full_scores만 호출하므로
    모델 종류에 무관 — 미변경)."""
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
