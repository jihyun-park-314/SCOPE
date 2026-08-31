"""
[01] 데이터 로드 (Books 대응판) — 메모리 안전 버전.

★ Video_Games판과의 차이:
  · Books 리뷰는 ~2,900만 행이라 `.to_pandas()`가 메모리를 터뜨린다.
    -> HF datasets의 Arrow 스트리밍 기반 `Dataset.to_parquet()`으로
       판다스를 거치지 않고 곧바로 디스크에 기록한다.
  · 메타(~440만 행)도 동일하게 필요한 컬럼만 select 후 스트리밍 기록.
    Books 메타에는 author/subtitle 관련 정보가 details에 있을 수 있어 keep 목록 확장.

실행:  python src/download_data.py --category Books
산출:  data/raw/{category}_reviews.parquet, data/raw/{category}_meta.parquet
       (SCOPE 정리판: 원본 parquet은 전부 data/raw/에 모으고, preprocessing.py 이후의
        모든 중간 산출물은 data/preprocessed/에 모은다 — 원본과 산출물이 섞이지 않게)
디스크: 리뷰 원본 캐시 포함 ~50GB+ 여유 권장 (HF_HOME으로 캐시 위치 변경 가능)
"""
import argparse
import os
from datasets import load_dataset
from config import CFG, raw_meta_path, raw_reviews_path


def main():
    ap = argparse.ArgumentParser()
    # ★ 여기서의 --category는 데이터셋 키(books)가 아니라 Amazon-Reviews-2023의 원본
    # 카테고리 이름이다 — 파일명이 그대로 data/raw/{category}_*.parquet가 되고,
    # config.DATASETS[*]["source_category"]가 이 이름을 가리킨다.
    ap.add_argument("--category", required=True,
                    help="Amazon-Reviews-2023의 원본 카테고리 이름 "
                         "(예: Books, Video_Games, Beauty_and_Personal_Care)")
    args = ap.parse_args()

    os.makedirs(CFG.raw_dir, exist_ok=True)
    cat = args.category
    reviews_out, meta_out = raw_reviews_path(cat), raw_meta_path(cat)
    print(f"[01] category={cat}  ->  {CFG.raw_dir}/")

    # ---------- 리뷰 ----------
    print(f"[01] downloading raw_review_{cat} (large: streaming write) ...")
    rev = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                       f"raw_review_{cat}", split="full", trust_remote_code=True)
    keep_r = [c for c in ("user_id", "parent_asin", "rating", "title", "text",
                          "timestamp", "helpful_vote", "verified_purchase")
              if c in rev.column_names]
    rev = rev.select_columns(keep_r)
    rev.to_parquet(reviews_out)   # Arrow -> parquet 직행
    print(f"      reviews: {rev.num_rows:,} rows -> {reviews_out}")

    # ---------- 메타 ----------
    print(f"[01] downloading raw_meta_{cat} ...")
    meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                        f"raw_meta_{cat}", split="full", trust_remote_code=True)
    keep_m = [c for c in ("parent_asin", "title", "subtitle", "author",
                          "features", "description", "categories",
                          "average_rating", "rating_number", "price", "store", "details")
              if c in meta.column_names]
    meta = meta.select_columns(keep_m)

    # 중첩(dict/list) 컬럼은 parquet 호환을 위해 문자열로 평탄화
    def _flatten(ex):
        for k in ("author", "details", "features", "description", "categories"):
            if k in ex and ex[k] is not None and not isinstance(ex[k], (str, float, int)):
                ex[k] = str(ex[k])[:800]
        return ex
    meta = meta.map(_flatten, desc="flatten nested meta")
    meta.to_parquet(meta_out)
    print(f"      meta   : {meta.num_rows:,} rows -> {meta_out}")


if __name__ == "__main__":
    main()
