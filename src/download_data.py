import argparse
import os
from datasets import load_dataset
from config import CFG, raw_meta_path, raw_reviews_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True,
                    help="source category name in Amazon Reviews 2023 "
                         "(e.g. Books, Video_Games, Beauty_and_Personal_Care)")
    args = ap.parse_args()

    os.makedirs(CFG.raw_dir, exist_ok=True)
    cat = args.category
    reviews_out, meta_out = raw_reviews_path(cat), raw_meta_path(cat)
    print(f"[01] category={cat}  ->  {CFG.raw_dir}/")

    print(f"[01] downloading raw_review_{cat} (large: streaming write) ...")
    rev = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                       f"raw_review_{cat}", split="full", trust_remote_code=True)
    keep_r = [c for c in ("user_id", "parent_asin", "rating", "title", "text",
                          "timestamp", "helpful_vote", "verified_purchase")
              if c in rev.column_names]
    rev = rev.select_columns(keep_r)
    rev.to_parquet(reviews_out)
    print(f"      reviews: {rev.num_rows:,} rows -> {reviews_out}")

    print(f"[01] downloading raw_meta_{cat} ...")
    meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                        f"raw_meta_{cat}", split="full", trust_remote_code=True)
    keep_m = [c for c in ("parent_asin", "title", "subtitle", "author",
                          "features", "description", "categories",
                          "average_rating", "rating_number", "price", "store", "details")
              if c in meta.column_names]
    meta = meta.select_columns(keep_m)

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
