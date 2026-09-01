import hashlib
import json
import re

import pandas as pd

def norm_text(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())

def sha1_16(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()[:16]

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def kcore_filter(df: pd.DataFrame, ku: int, ki: int):
    n_iter = 0
    while True:
        n_iter += 1
        uc = df["user_id"].value_counts()
        ic = df["parent_asin"].value_counts()
        keep = df["user_id"].isin(uc[uc >= ku].index) & \
               df["parent_asin"].isin(ic[ic >= ki].index)
        if keep.all():
            return df, n_iter
        df = df[keep]
