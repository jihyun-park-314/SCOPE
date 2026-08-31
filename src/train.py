# -*- coding: utf-8 -*-
"""
T0/T1 재현 수정판(v2) -- 원본 model.py (구 train_eval_halo_sr_semantic_anchor.py) main()과 RNG 소비
순서를 정확히 일치시키기 위한 재작성.

v1(train_attention_residual.py)에서 발견된 RNG-궤적 이탈 3가지와 그 수정:

1. 원본은 학습 시작 전 `evaluate(model, valid_data, ..., mode="query_only")` 를 실행한다
   (epoch-0 sanity eval). 이 호출은 valid_data 전체(20,143개)에 대해 __getitem__ ->
   sample_negative() 를 호출해 전역 `random` 모듈 상태를 소비한다. v1은 이 단계를
   생략해 이후 모든 negative sampling이 원본과 다른 지점에서 시작됐다.
   -> v2는 원본과 동일한 함수(`model.evaluate`)를 동일한
      인자로, 동일한 위치(모델 생성 직후, train_loader 생성 전)에서 호출한다.

2. v1은 학습 시작 전 `ref_batch = collate_fn([train_data[i] for i in range(256)])` 를
   만들어 추가로 256개의 sample_negative 호출을 전역 random에서 소비했다.
   -> v2는 gradient-instrumentation용 reference batch를 학습 시작 전에 만들지 않는다
      (offline_diagnostics_v2.py에서 학습 종료 후에만 구성).

3. v1은 첫 배치 assert를 위해 `next(iter(train_loader))` 를 호출했다 -- DataLoader의
   RandomSampler는 매 `iter()` 호출마다 전역 torch RNG에서 1회 draw로 새 시드를 뽑아
   로컬 permutation을 생성하므로, 이 호출 자체가 실제 학습 루프가 만들 permutation과는
   다른 permutation을 소비/폐기하고 전역 torch RNG 위치도 앞으로 밀어버린다.
   -> v2는 act_raw assert를 실제 학습 루프의 epoch=1, batch_idx=0 배치 안에서, 그것도
      이미 계산된 단일 forward의 텐서(u_star/e_all/diag)만으로 수행한다(별도 forward,
      별도 iter() 호출 없음) -- history_ablation="u_act_only"에서는 정의상
      u_star == normalize(u_act) 이므로 act_raw_score(u_act,...)와
      sum(u_star*e_all[item])이 수학적으로 완전히 동일해야 한다.

학습 루프 자체는 원본과 동일하게 매 epoch (1) train_one_epoch, (2) evaluate(mode="full")
단 1회만 수행한다. attention-only 평가/ruin-rescue/gradient instrumentation은 전부
offline_diagnostics_v2.py로 옮겨 학습 궤적에 영향을 주지 않는다.
(※ offline_diagnostics_v2.py는 구 HALO 저장소의 학습 후 분석 스크립트로, SCOPE에는 포함돼
 있지 않다. 아래에서 이 이름이 나오는 곳은 전부 "여기서 하지 않는다"는 뜻이다.)

---
T1 확장 (gamma>0, checkpoint 보존 정책 추가): T0와 T1이 동일 batch 순서/동일 sampled
negative를 쓰도록(유일한 차이는 gamma*L_act) attention-only checkpoint 후보 선정용
평가도 `dataset.__getitem__`/`sample_negative()`를 전혀 거치지 않는 RNG-free 평가
경로(`evaluate_attention_only_no_rng`)로 구현한다 -- 이 경로는 애초에 "neg"를 쓰지
않는 full-catalog 랭킹이므로 negative sampling 자체가 불필요하며, 이를 우회함으로써
매 epoch 추가되는 이 평가가 전역 random 모듈 상태를 전혀 소비하지 않게 한다(joint
top-3 선정에 쓰이는 official_evaluate 호출 -- 이것도 원래 "neg"를 쓰지 않지만 원본
A5와의 RNG parity를 위해 원본과 동일하게 그대로 유지 -- 뒤에 이어지는 것과 무관하게
독립적으로 안전).
"""
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
    """history_ablation=None이면 A5_ARGS 기본값("u_act_only")을 쓴다 (T0/T1 하위호환).
    M2는 "h_n_only", M3는 "none"(learned gate)을 명시적으로 넘긴다."""
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
    """단일 forward에서 L_joint, L_act, 그리고 assert/instrumentation에 쓸 텐서를 반환."""
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
    """추가 forward 없이, 이미 계산된 u_star/e_all/diag['u_act']만으로 act_raw 공식을 검증한다.
    history_ablation="u_act_only" + normalize_u_star(기본 True)에서만 정의상
    u_star == F.normalize(u_act)이므로, act_raw_score(u_act, e_all, item)와
    sum(u_star * e_all[item], dim=-1)이 수학적으로 완전히 동일해야 한다(부동소수 오차 수준
    미만 -- eval-mode 전환도, model.full_scores() 재호출도 필요 없다). h_n_only/none(gated)
    에서는 이 항등식이 성립하지 않으므로 호출자가 history_ablation=="u_act_only"일 때만
    호출해야 한다."""
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
            f"-- history_ablation/normalize_u_star 설정이 예상과 다름. 학습 중단."
        )


def _build_batch_no_neg(dataset, indices, max_len):
    """dataset.__getitem__/sample_negative()를 전혀 거치지 않고(negative 없음) history/
    delta_t/query_emb/target만 직접 구성한다 -- attention-only/joint full-catalog 랭킹은
    애초에 "neg"를 쓰지 않으므로, 이 경로는 전역 random 모듈 상태를 0회 소비한다."""
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
    """scores[i, history[i,j]] = -1e9 (target 자신은 제외) -- Python 이중 루프 없이 scatter_로
    처리 (offline_diagnostics_v2.mask_seen_와 동일 로직; 순환 import를 피하려 여기 복제)."""
    scores[:, 0] = -1e9
    is_target = history.eq(target.view(-1, 1))
    seen_idx = torch.where(is_target, torch.zeros_like(history), history)
    scores.scatter_(1, seen_idx, torch.full_like(seen_idx, -1e9, dtype=scores.dtype))
    return scores


@torch.no_grad()
def evaluate_attention_only_no_rng(model, dataset, device, batch_size, max_len,
                                   ks=(10, 50), collect_gate_stats=False):
    """behavior-only 단독 랭킹(u_star 기준)을 계산하되, RNG를 전혀 소비하지 않는다(위
    _build_batch_no_neg 참조) -- 학습 중 매 epoch 호출해도 batch 순서/negative 샘플링 궤적에
    영향을 주지 않는다.

    u_star는 forward_user_vector()가 history_ablation 분기(h_n_only/none/u_act_only)에
    따라 이미 model-specific하게 계산하고 normalize_u_star=True(기본값)로 정규화까지 마친
    값이므로, history_ablation="u_act_only"에서는 정확히 기존 normalize(diag['u_act'])와
    동일하고(수치적으로 동등, 이 함수의 예전 구현과 bit-for-bit 동일한 랭킹을 낸다), "h_n_only"/
    "none"에서는 각각 h_n/gate 기반 behavior score로 자동 일반화된다 -- 모델별 분기 없이
    이 한 줄로 M2/M3/M4/M5 전체를 커버한다.

    collect_gate_stats=True (M3/history_ablation="none" 전용): model.W_f(gate linear layer)에
    forward hook을 걸어 이미 진행 중인 이 forward pass에서 g_u(=sigmoid(W_f(...)))를 추가 forward
    없이 그대로 캡처하고, epoch 전체 배치에 대한 gate 값의 mean/std/min/max를 반환한다."""
    was_training = model.training
    model.eval()
    n = len(dataset)
    ranks_all = []
    gate_vals = [] if collect_gate_stats else None
    captured = {}
    hook_handle = None
    if collect_gate_stats:
        def _hook(_module, _inp, out):    # forward hook의 고정 시그니처 (앞 두 개는 안 쓴다)
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
    """배치 구성(uid 대신 target/neg/history를 사용 -- uid는 별도로 넘기지 않으므로) 해시.
    첫 10배치 비교 검증에서 "동일 데이터가 동일 순서로 들어왔는가"를 확인하는 용도."""
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
            # 항등식 u_star == normalize(u_act)는 history_ablation="u_act_only"에서만 성립
            # (h_n_only/none에서는 u_star가 h_n/gate 기반이라 이 assert 자체가 성립하지 않음).
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
        print(f"[skip] 비유한 grad norm 으로 건너뛴 step: {n_skipped}")
    return total_loss / max(n_batches, 1)


def update_topk(topk_list, epoch, score, k=3):
    topk_list.append((score, epoch))
    topk_list.sort(key=lambda x: -x[0])
    return topk_list[:k]


def save_full_state(epoch, path, model, optimizer, best_joint, bad_count, joint_top3, attnonly_top3, log):
    """model/optimizer/RNG(python·numpy·torch·cuda)/early-stop counters/log를 전부 저장한다
    -- 이 파일만으로 학습을 정확히 이어서 재개할 수 있다 (save_ckpt()의 last.pt/epoch{N}.pt는
    평가·공유용 경량 체크포인트로 그대로 유지, resume에는 이 파일만 사용)."""
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
                     help="M2=h_n_only, M3/M6=none(learned gate), T0/T1(M4/M5)=u_act_only(기본값).")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2026)
    # ★ 예전에는 "amazon_books_queryA_card_22k/processed"가 하드코딩돼 있어서 cwd가 final/이
    # 아니면 조용히 못 찾았다. 이제 --dataset에서 SCOPE 표준 경로를 유도한다.
    ap.add_argument("--dataset", type=str, required=True,
                    help="데이터셋 키 (books / video_games / beauty). "
                         "--processed_dir/--embed_dir의 기본값을 이 값에서 유도한다.")
    ap.add_argument("--processed_dir", type=str, default=None,
                    help="미지정 시 data/preprocessed/{dataset}/processed")
    ap.add_argument("--embed_dir", type=str, default=None,
                    help="미지정 시 data/preprocessed/{dataset}/embeddings")
    ap.add_argument("--out_dir", type=str, required=True)
    # 학습 예산은 전 데이터셋 공통 200 epochs / patience 20으로 통일한다.
    # (예전 기본값은 100/10이라 드라이버 스크립트가 매번 --epochs 200 --patience 20을
    #  넘겨야 했고, 안 넘기면 조용히 다른 예산으로 학습됐다.)
    ap.add_argument("--epochs", type=int, default=A5_ARGS["epochs"])
    ap.add_argument("--patience", type=int, default=A5_ARGS["patience"])
    # 배치 크기를 실행 인자로 연다 -- baseline(MAPS)이 batch 72를 쓰므로 같은 배치에서
    # 비교하려면 필요하다. 생략하면 A5_ARGS 기본값(128)이라 기존 실행과 완전히 동일하다.
    ap.add_argument("--batch_size", type=int, default=A5_ARGS["batch_size"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verify_only_batches", type=int, default=0,
                     help="0이면 정식 학습. >0이면 그 개수만큼 배치만 돌리고 digest/loss를 저장 후 종료"
                          "(전체 재학습 전 원본과의 첫 N배치 일치 검증용).")
    ap.add_argument("--verify_out", type=str, default=None)
    ap.add_argument("--resume_from", type=str, default=None,
                     help="full_state_last.pt 경로. 지정 시 model/optimizer/RNG/early-stop 상태를 "
                          "전부 복원하고 epoch-0 sanity eval을 건너뛴 채 마지막 완료 epoch+1부터 이어서 "
                          "학습한다 (save_ckpt()가 만드는 last.pt/epoch{N}.pt로는 resume 불가 -- "
                          "반드시 save_full_state()가 만든 파일을 넘길 것).")
    ap.add_argument("--skip_nonfinite_step", action="store_true",
                    help="grad norm 이 inf/nan 인 step 을 건너뛴다. 기본 off 는 기존 동작 유지.")
    ap.add_argument("--sdpa_math", action="store_true",
                    help="scaled_dot_product_attention 을 math 커널로 강제한다. nn.MultiheadAttention 은 "
                         "need_weights=False 일 때 mem-efficient 융합 커널로 디스패치되는데, 이 커널의 "
                         "backward 가 패딩이 많은 마스크에서 폭발적인 그래디언트를 내어 NaN 을 유발한다 "
                         "(math 대비 grad 88.14 vs 0.028 실측). 기본 off 는 기존 동작 유지.")
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
        print("[SDPA] math 커널 강제 (flash/mem-efficient 비활성)")

    NO_AUX_CONDITIONS = {"T0", "M2", "M3"}
    if args.condition in NO_AUX_CONDITIONS:
        assert args.gamma == 0.0, f"{args.condition}은 gamma=0(L_joint만)이어야 함 (Auxiliary BPR 없음)"

    # M6 = A4/Original Full(gated) 구조 + Attention-specific auxiliary BPR. M3(gamma=0)이 그대로
    # 통제군이 되도록, 구조 관련 인자는 M3와 동일해야 하고 gamma만 >0이어야 한다. gamma=0으로
    # 잘못 넘기면 M3 재학습이 되어버리므로(중복 실험) 명시적으로 막는다.
    if args.condition == "M6":
        assert args.gamma > 0.0, "M6은 gamma>0(Auxiliary BPR)이어야 함 -- gamma=0은 M3와 동일 조건"
        assert args.history_ablation == "none", (
            f"M6은 A4/Original Full 구조(learned gate)를 유지해야 하므로 history_ablation='none' "
            f"필수 (받은 값: {args.history_ablation!r})")
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

    # map_location="cpu": RNG state 텐서(get_rng_state/get_rng_state_all)는 반드시 CPU ByteTensor여야
    # set_rng_state()가 받아들인다 -- map_location=device로 로드하면 이 텐서까지 GPU로 옮겨져
    # "RNG state must be a torch.ByteTensor" 에러가 난다. optimizer.load_state_dict()는 각 state
    # 텐서를 해당 파라미터의 device로 자동 캐스팅하므로 optimizer state는 cpu 로드 후에도 안전하다.
    resume_state = (torch.load(args.resume_from, map_location="cpu", weights_only=False)
                    if args.resume_from else None)   # torch>=2.6: weights_only 기본 True 이면
                                                     # full_state 의 numpy RNG 상태를 못 읽는다

    if resume_state is not None:
        assert args.verify_only_batches == 0, "--resume_from과 --verify_only_batches는 함께 쓸 수 없음"
        model.load_state_dict(resume_state["model_state_dict"])
        print(f"[Resume] loaded model/optimizer/RNG from {args.resume_from} "
              f"(last completed epoch={resume_state['epoch']})")
    else:
        # ---- 원본과 동일: epoch-0 sanity eval (RNG parity 핵심 -- train_loader 생성 전에 위치) ----
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

        # ---- 원본과 동일: evaluate(mode="full") 단 1회만 (attention-only/ruin-rescue/gradient는
        # offline_diagnostics_v2.py로 이동 -- 학습 중 추가 valid_data 순회로 RNG 궤적이 흔들리지
        # 않게 한다) ----
        valid_result = official_evaluate(model, valid_data, device, A5_ARGS["eval_batch_size"], mode="full")
        # RNG-free (dataset.__getitem__/sample_negative 미사용) -- T0/T1 batch 순서·negative
        # 샘플링 궤적에 전혀 영향을 주지 않는다 (모듈 docstring 참조).
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
