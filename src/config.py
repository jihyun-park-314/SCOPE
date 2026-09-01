import os
from dataclasses import dataclass

@dataclass
class Config:

    raw_dir: str = "data/raw"
    preprocessed_dir: str = "data/preprocessed"
    result_dir: str = "results"

    sample_seed: int = 42
    max_seq_len: int = 200
    incore_item: int = 5
    incore_user: int = 5
    sample_priority_min_rating: float = 5.0
    sample_priority_min_textlen: int = 100
    sample_users_pool: int = 27_000
    target_final_users: int = 20_000

    card_pool_cap: int = 2000
    max_reviews_per_item: int = 8

    exclude_query_review_from_card: bool = True
    exclude_scope: str = "eval"

    ollama_urls: str = "http://localhost:11434,http://localhost:11435"
    ollama_model: str = "gemma4:26b"
    query_max_new_tokens: int = 220
    card_max_new_tokens: int = 300
    gemma_batch: int = 32

CFG = Config()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(REPO_ROOT, "prompts")

for _f in ("raw_dir", "preprocessed_dir", "result_dir"):
    setattr(CFG, _f, os.path.join(REPO_ROOT, getattr(CFG, _f)))

AMAZON_META_FIELDS = ("author", "categories", "features", "description", "details")

DATASETS = {
    "books": dict(source_category="Books", domain="book",
                  canonical_n=19748,
                  canonical_sha256="42756d159027d65c3c51cc89fb780c10381796de663713b66c066556ed6a7cd3",
                  gamma=0.01, meta_fields=AMAZON_META_FIELDS),
    "video_games": dict(source_category="Video_Games", domain="video game",
                        canonical_n=None, canonical_sha256=None,
                        gamma=0.5, meta_fields=AMAZON_META_FIELDS),
    "beauty": dict(source_category="Beauty_and_Personal_Care", domain="beauty product",
                   canonical_n=None, canonical_sha256=None,
                   gamma=0.5, meta_fields=AMAZON_META_FIELDS),
}

def canonical_eval_id(instances) -> str:
    import hashlib
    lines = ["\t".join([o["user_id"], o["target_item_id"], str(o["target_timestamp"]),
                        ",".join(o["history_item_ids"])]) for o in instances]
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()

def _cfg(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(
            f"[config] unregistered --dataset '{dataset}'. "
            f"available: {sorted(DATASETS)} — for a new dataset, add it to DATASETS in config.py and "
            f"create the three prompt files for its domain under prompts/.")
    return DATASETS[dataset]

def domain_of(dataset: str) -> str:
    return _cfg(dataset)["domain"]

def canonical_n(dataset: str):
    return _cfg(dataset).get("canonical_n")

def canonical_sha256(dataset: str):
    return _cfg(dataset).get("canonical_sha256")

def meta_fields_of(dataset: str) -> tuple:
    return _cfg(dataset)["meta_fields"]

def raw_reviews_path(source_category: str) -> str:
    return os.path.join(CFG.raw_dir, f"{source_category}_reviews.parquet")

def raw_meta_path(source_category: str) -> str:
    return os.path.join(CFG.raw_dir, f"{source_category}_meta.parquet")

def raw_paths(dataset: str) -> tuple:
    sc = _cfg(dataset)["source_category"]
    return raw_reviews_path(sc), raw_meta_path(sc)

def dataset_root(dataset: str) -> str:
    _cfg(dataset)
    return os.path.join(CFG.preprocessed_dir, dataset)

def _p(dataset: str, name: str) -> str:
    return os.path.join(dataset_root(dataset), name)

def interactions_path(dataset: str) -> str:      return _p(dataset, "interactions.pkl")
def interactions_raw_path(dataset: str) -> str:  return _p(dataset, "interactions_raw.pkl")
def sample_path(dataset: str) -> str:            return _p(dataset, "sample.parquet")
def manifest_path(dataset: str) -> str:          return _p(dataset, "split_manifest.json")
def queries_path(dataset: str) -> str:           return _p(dataset, "queries.parquet")
def leak_json_path(dataset: str) -> str:         return _p(dataset, "leak_dropped_uids.json")
def processed_dir(dataset: str) -> str:          return _p(dataset, "processed")
def embed_dir(dataset: str) -> str:              return _p(dataset, "embeddings")

def cards_path(dataset: str) -> str:
    return _p(dataset, "cards.jsonl" if CFG.exclude_query_review_from_card else "cards_noexcl.jsonl")

def stats_path(dataset: str, kind: str) -> str:
    return os.path.join(CFG.result_dir, f"{kind}_stats_{dataset}.json")

def scan_cache_path(reviews_path_: str, tag: str) -> str:
    base = os.path.basename(reviews_path_)
    return os.path.join(CFG.preprocessed_dir, "_cache", f"{base}.{tag}.cache.parquet")
