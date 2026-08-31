"""
SCOPE 파이프라인 공용 헬퍼 — 여러 스크립트에 글자 그대로 복제돼 있던 함수들을 한곳으로 모은 것.

여기 있는 4개는 전부 복제본끼리 **동작이 완전히 동일**했던 것만 옮겼다(로직 무수정):
  · norm_text / sha1_16 : preprocessing.py <-> semantic_card.py (바이트 동일)
                          이 둘이 만드는 review_hash가 manifest와 카드 제외 키를 잇는 접점이라,
                          두 곳에 복제돼 있으면 한쪽만 고쳐져 조용히 어긋날 수 있었다.
  · load_jsonl          : model.py <-> build_embeddings.py
                          model.py 쪽(빈 줄 skip)을 채택 — 파이프라인이 쓰는 jsonl에는 빈 줄이
                          없으므로 두 구현의 결과는 동일하고, 빈 줄에서 죽지만 않는다.
  · kcore_filter        : preprocessing.py <-> review2query.py
                          반복 조건·종료 조건이 동일했고, 반환값만 (df, n_iter) vs df로 달랐다.
                          여기서는 (df, n_iter)로 통일하고, 횟수가 필요 없는 쪽은 [0]만 쓴다.

safe_text는 여기 없다 — prepare_dataset.py 쪽은 개행을 공백으로 치환하고
build_embeddings.py 쪽은 strip만 한다. 이름만 같고 하는 일이 다르므로 합치면 산출물이 바뀐다.
"""
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
    """샘플 내 item>=ki AND user>=ku를 더 이상 걸러질 행이 없을 때까지 반복.
    반환: (필터된 df, 수렴까지의 반복 횟수)."""
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
