# -*- coding: utf-8 -*-
"""랭킹 지표 계산 (HR@k / NDCG@k / MRR).

원본은 final/ablation_report_newcard.py에 있었고 train.py(구 train_attention_residual_v2.py)가
거기서 metrics_from_ranks만 import했다. 그런데 그 파일은 newcard 3-실험 리포트 전용 스크립트로,
import 시점에 runs_newcard_ablation/report/를 mkdir하는 모듈 레벨 부작용이 있었다
(ablation_report_newcard.py:26). 학습 스크립트가 리포트용 출력 디렉터리를 만들 이유가 없으므로
순수 함수만 이 파일로 떼어냈다 — 함수 본문은 원본과 동일(무수정 이식).

ranks는 1-based(정답 아이템의 순위, 1이 최상위).
"""
import math


def metrics_from_ranks(ranks, ks=(10, 50)):
    n = len(ranks)
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    mrr = 0.0
    for r in ranks:
        mrr += 1.0 / r
        for k in ks:
            if r <= k:
                hits[k] += 1
                ndcgs[k] += 1.0 / math.log2(r + 1)
    out = {"n_eval": n, "MRR": mrr / n}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / n
        out[f"NDCG@{k}"] = ndcgs[k] / n
    return out
