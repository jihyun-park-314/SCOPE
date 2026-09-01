# -*- coding: utf-8 -*-
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
