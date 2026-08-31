"""
SCOPE(HALO-SR) 전역 설정 — Amazon-Reviews-2023 기반 파이프라인 공통 참조 파일.
src/의 6개 스크립트(download_data, preprocessing, review2query, semantic_card,
prepare_dataset, build_embeddings)가 이 파일 하나를 참조한다.

[디렉터리 레이아웃 — SCOPE 정리판]
  <repo>/data/raw/           download_data.py 산출물 ({cat}_reviews.parquet, {cat}_meta.parquet)
  <repo>/data/preprocessed/{dataset}/  그 데이터셋의 모든 산출물
                             (interactions.pkl, split_manifest.json, queries.parquet,
                              cards.jsonl, processed/, embeddings/)
  <repo>/results/            통계 JSON
  <repo>/prompts/            query_prompt_{domain}.txt / card_prompt_{domain}.txt /
                             item_instruction_{domain}.txt
경로는 이 파일 위치(<repo>/src/config.py) 기준 절대경로로 고정되므로, cwd가 src/든
repo root든 항상 같은 파일을 가리킨다.

[--dataset 하나로 모든 경로 유도]
과거 config에는 card_reviews_path/card_meta_path가 "data/Video_Games_reviews.parquet"로
하드코딩돼 있어서, Books를 돌리면서 --reviews/--meta를 깜빡하면 조용히 Video_Games 리뷰를
카드 소스로 써버리는 사고가 났다. 이제는 아래 DATASETS 레지스트리가 --dataset("books")에서
원본 카테고리("Books")와 도메인("book")을 유도한다. 스크립트는 --reviews/--meta/--domain을
생략해도 항상 올바른 파일을 집는다.

[Books 스케일 주의]
Books는 전체 카테고리 중 최대 규모(리뷰 ~2,900만, 아이템 ~440만)라서 Video_Games용
기본값을 그대로 쓰면 메모리/학습시간/LLM 비용이 폭증한다. 아래 손잡이로 규모를 통제한다:
  · incore_user / incore_item : 샘플 내 k-core 강화
  · sample_users_pool         : 유저 오버샘플 풀 크기
  · max_reviews_per_item      : 카드 생성용 리뷰 수 축소
  · card_pool_cap             : 아이템당 리뷰 후보 풀 상한

[여기 없는 것]
모델 구조/학습 하이퍼파라미터(hidden_dim, lr, epochs, patience, dropout ...)는 이 파일이 아니라
train.py의 A5_ARGS와 각 스크립트의 argparse 기본값이 단일 출처다. 예전에는 같은 이름의 필드가
여기에도 있었지만 아무도 읽지 않아, config를 고쳐도 학습이 안 바뀌는 함정이었다.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    """파이프라인 스크립트가 실제로 읽는 값만 남긴다 — 여기 있는 필드는 전부 `CFG.<name>`으로
    참조되는 곳이 있다. 참조처 없는 값을 두면 '고쳤는데 안 바뀌는' 설정이 되므로 넣지 않는다."""

    # ---------- 경로 (아래에서 repo root 기준 절대경로로 고정) ----------
    raw_dir: str = "data/raw"                     # download_data.py 산출물 (원본 parquet)
    preprocessed_dir: str = "data/preprocessed"   # preprocessing.py 이후의 모든 중간 산출물
    result_dir: str = "results"                   # 통계 JSON

    # ---------- preprocessing.py: 유저 샘플링 + in-sample k-core (prompt3.md §1.2) ----------
    sample_seed: int = 42
    max_seq_len: int = 200
    # 전체 10/10-core는 144,082명 전체 기준 보장이라, 유저를 5,000명만 뽑으면 아이템
    # 밀도 보장이 깨진다(실측: 아이템당 1.49건, 74.4%가 1회 등장) — 샘플 안에서 재적용 필요.
    incore_item: int = 5                       # 샘플 내 아이템 최소 등장 수
    incore_user: int = 5                       # 샘플 내 유저 최소 상호작용 수 (Video_Games 실측값과 일치)
    sample_priority_min_rating: float = 5.0    # 우선순위 유저 풀: 이 이상 rating만 "좋은 리뷰"
    sample_priority_min_textlen: int = 100     # 우선순위 유저 풀: 이 이상 글자수만 "좋은 리뷰"
    # 유저 오버샘플 풀 크기 — in-sample k-core 재수렴 후 최종 인원은 이보다 줄어든다
    # (Video_Games 실측: 27,000 -> 약 22,900). target_final_users 근처로 맞추려면 카테고리별로
    # 실행 결과를 보고 조정해야 한다(밀도가 카테고리마다 달라 고정 배율이 없음).
    sample_users_pool: int = 27_000
    target_final_users: int = 20_000           # 참고용 목표치(로그 출력에만 사용, 강제 아님)

    # ---------- semantic_card.py: 카드 소스 (샘플과 분리, prompt3.md §2) ----------
    # k-core는 시퀀스 모델링용 필터일 뿐 — "이 책이 어떤 책인가"를 설명하는 카드는
    # 샘플 밖 유저의 리뷰를 포함한 원본 전체에서 만든다(누수 아님, §0-(2)).
    # ★ 카드 소스 원본 경로는 DATASETS[--dataset]["source_category"]에서 유도한다.
    # 굳이 다른 파일을 쓰고 싶을 때만 CLI --reviews/--meta로 명시할 것.
    card_pool_cap: int = 2000            # 아이템당 리뷰 후보 풀 상한(메모리 방어, min-heap으로 상위 유지)
    max_reviews_per_item: int = 8        # 카드 1장에 넣는 리뷰 수 (Books: 12 -> 8, 생성 비용 절감)

    # ---------- semantic_card.py: 카드 누수 제어 (prompt3.md §0-(4)) ----------
    exclude_query_review_from_card: bool = True
    exclude_scope: str = "eval"          # valid + test 타깃의 소스 리뷰(R_ui)만 제외

    # ---------- Ollama (review2query.py[3] / semantic_card.py[4] 공통, src/ollama_client.py) ----------
    # 서버/모델은 두 단계가 같은 값을 쓴다. 각 스크립트의 --ollama_urls/--model 기본값이 여기다.
    ollama_urls: str = "http://localhost:11434,http://localhost:11435"
    ollama_model: str = "gemma4:26b"
    # 생성 길이만 단계별로 다르다 — 쿼리는 한 줄, 카드는 4개 필드(<=120 words)다.
    query_max_new_tokens: int = 220      # 구 HALO 프롬프트 검증 기본값과 동일 (쿼리가 잘리지 않게)
    card_max_new_tokens: int = 300
    gemma_batch: int = 32                # semantic_card: N개마다 cards.jsonl에 flush(체크포인트)


CFG = Config()

# ---- 경로 고정: repo root(= 이 파일의 부모의 부모) 기준 절대경로 ----
# cwd가 src/든 repo root든 항상 같은 실제 디렉터리를 가리키게 한다.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(REPO_ROOT, "prompts")

for _f in ("raw_dir", "preprocessed_dir", "result_dir"):
    setattr(CFG, _f, os.path.join(REPO_ROOT, getattr(CFG, _f)))


# ---------------------------------------------------------------------------
# 데이터셋 레지스트리 — SCOPE의 단일 진실 공급원.
#
# 모든 스크립트가 `--dataset books` 하나만 받고, 원본 경로·프롬프트 도메인·산출물 경로·
# 평가 모집단을 전부 여기서 유도한다. 예전에는 `--category Books_sample22k`라는 긴 실험
# 식별자가 모든 파일명에 반복해서 박혔고(processed_Books_sample22k_queryA.parquet 등),
# 원본 경로는 config에 하드코딩돼 있었다.
#
# 새 데이터셋 추가: 여기에 한 줄 등록 + prompts/에 그 도메인의 3개 파일
# (query_prompt_/card_prompt_/item_instruction_{domain}.txt)을 만들면 된다.
#
#   source_category : data/raw/{source_category}_{reviews,meta}.parquet  (download_data.py --category)
#   domain          : prompts/*_{domain}.txt 의 슬러그 (공백은 밑줄로 치환)
#   canonical_n     : test.py --leak_drop 후 기대되는 평가 인스턴스 수 (불일치 시 중단).
#                     None이면 개수 검증을 생략한다.
#   gamma           : 그 데이터셋에서 채택된 auxiliary loss 계수. 데이터셋마다 다르므로
#                     train.py --gamma로 반드시 명시해야 한다.
#   meta_fields     : 카드 프롬프트의 [METADATA] 자리에 넣을 메타 parquet 컬럼과 그 출력 순서
#                     (semantic_card._fmt_meta). 예전에는 이 목록이 코드에 박혀 있어, 메타
#                     스키마가 다른 카테고리를 추가하면 없는 컬럼이 조용히 건너뛰어지고
#                     그 아이템만 빈약한 메타로 카드가 만들어졌다. 등록 시점에 드러나도록 옮겼다.
#
# 학습 예산은 세 데이터셋 공통으로 200 epochs / patience 20이며, train.py의 기본값
# (A5_ARGS)이 이미 그 값이라 따로 넘기지 않아도 된다.
# ---------------------------------------------------------------------------
# Amazon-Reviews-2023는 카테고리가 달라도 메타 스키마가 같아서 세 데이터셋이 같은 목록을 쓴다.
# title/subtitle이 빠진 것은 의도적이다 — title은 쿼리-타깃 표면 일치 경로를 만들고 subtitle은
# 91.4%가 판형/발행 문자열이다(semantic_card._fmt_meta 주석 참조).
AMAZON_META_FIELDS = ("author", "categories", "features", "description", "details")

DATASETS = {
    "books": dict(source_category="Books", domain="book",
                  canonical_n=19748, gamma=0.01, meta_fields=AMAZON_META_FIELDS),
    "video_games": dict(source_category="Video_Games", domain="video game",
                        canonical_n=22761, gamma=0.5, meta_fields=AMAZON_META_FIELDS),
    "beauty": dict(source_category="Beauty_and_Personal_Care", domain="beauty product",
                   canonical_n=23280, gamma=0.5, meta_fields=AMAZON_META_FIELDS),
}


def _cfg(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(
            f"[config] 등록되지 않은 --dataset '{dataset}'. "
            f"사용 가능: {sorted(DATASETS)} — 새 데이터셋이면 config.py의 DATASETS에 "
            f"등록하고 prompts/에 그 도메인의 3개 파일을 만드세요.")
    return DATASETS[dataset]


def domain_of(dataset: str) -> str:
    """프롬프트 도메인 명사 (prompts/*_{domain}.txt)."""
    return _cfg(dataset)["domain"]


def canonical_n(dataset: str):
    """leak-drop 후 기대 평가 인스턴스 수. None이면 검증 생략."""
    return _cfg(dataset).get("canonical_n")


def meta_fields_of(dataset: str) -> tuple:
    """카드 프롬프트 [METADATA]에 넣을 메타 컬럼 목록. 튜플 순서가 곧 출력 순서다.
    average_rating/rating_number는 형식이 달라("AvgRating: x (n ratings)")
    semantic_card._fmt_meta가 별도로 처리하므로 여기 넣지 않는다."""
    return _cfg(dataset)["meta_fields"]


# ---- data/raw : download_data.py 산출물 (원본 parquet) ----

def raw_reviews_path(source_category: str) -> str:
    return os.path.join(CFG.raw_dir, f"{source_category}_reviews.parquet")


def raw_meta_path(source_category: str) -> str:
    return os.path.join(CFG.raw_dir, f"{source_category}_meta.parquet")


def raw_paths(dataset: str) -> tuple:
    """(리뷰, 메타) 원본 경로. --reviews/--meta를 생략해도 항상 올바른 카테고리를 집는다 —
    예전에는 config에 Video_Games 경로가 하드코딩돼 있어서 Books를 돌리며 인자를 빠뜨리면
    조용히 Video_Games 리뷰가 카드 소스로 들어갔다."""
    sc = _cfg(dataset)["source_category"]
    return raw_reviews_path(sc), raw_meta_path(sc)


# ---- data/preprocessed/{dataset}/ : 파이프라인 산출물 ----
#
#   interactions.pkl      preprocessing.py — u2id/i2id/시퀀스/split (학습용)
#   interactions_raw.pkl  preprocessing.py — 필터 전, 진단용
#   sample.parquet        preprocessing.py — review2query.py --fixed_input 입력
#   split_manifest.json   preprocessing.py — split의 단일 진실 공급원 (immutable)
#   queries.parquet       review2query.py  — query 컬럼이 채워진 인터랙션
#   cards.jsonl           semantic_card.py — 아이템 시맨틱 카드
#   leak_dropped_uids.json  test.py --leak_drop 용 (구버전 산출물에만 필요)
#   processed/            prepare_dataset.py 산출물
#   embeddings/           build_embeddings.py 산출물

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
    """카드 파일. R_ui 제외를 끄면 파일명이 달라져 서로 다른 기준의 카드가 섞이지 않는다."""
    return _p(dataset, "cards.jsonl" if CFG.exclude_query_review_from_card else "cards_noexcl.jsonl")


def stats_path(dataset: str, kind: str) -> str:
    """results/{kind}_stats_{dataset}.json"""
    return os.path.join(CFG.result_dir, f"{kind}_stats_{dataset}.json")


def scan_cache_path(reviews_path_: str, tag: str) -> str:
    """무거운 전체 스캔 결과 캐시. data/raw는 다운로드 원본만 두는 곳이므로 캐시는
    data/preprocessed/_cache/에 원본 파일명을 접두사로 붙여 저장한다."""
    base = os.path.basename(reviews_path_)
    return os.path.join(CFG.preprocessed_dir, "_cache", f"{base}.{tag}.cache.parquet")
