# SCOPE — HALO-SR 최종 파이프라인 (Amazon-Reviews-2023)

리뷰 → 검색쿼리 / 아이템 시맨틱 카드 → query-aware 시퀀스 추천 데이터셋 → 동결 텍스트
임베딩까지의 6단계 파이프라인. `HALO/` 저장소에 흩어져 있던 구버전 스크립트
(`02x_kcore_sample_no_query.py`, `02e_sample_to_pkl.py`, `03_semantic_cards.py`,
`final/prepare_from_existing_query_parquet.py`, `final/build_halo_lite_embeddings.py`)를
`src/` 6개로 정리한 것으로, **로직은 그대로 두고 경로·도메인 해석만 일원화**했다.

## 디렉터리

```
SCOPE/
├── src/                    데이터 파이프라인 6단계 + 학습/평가 + config/utils
├── prompts/                도메인별 텍스트 — 코드에 문자열을 박지 않는다
├── data/
│   ├── raw/                download_data.py 산출물 — 원본 parquet만
│   └── preprocessed/
│       ├── books/          ← 데이터셋 하나가 폴더 하나
│       ├── video_games/
│       └── beauty/
└── results/                통계 JSON
```

데이터셋 폴더 안은 세 도메인이 전부 동일한 이름을 쓴다:

```
data/preprocessed/{dataset}/
├── interactions.pkl        preprocessing.py — u2id/i2id/시퀀스/split (학습용)
├── interactions_raw.pkl    preprocessing.py — 필터 전, 진단용
├── sample.parquet          preprocessing.py — review2query.py --fixed_input 입력
├── split_manifest.json     preprocessing.py — split의 단일 진실 공급원 (immutable)
├── queries.parquet         review2query.py  — query 컬럼이 채워진 인터랙션
├── cards.jsonl             semantic_card.py — 아이템 시맨틱 카드
├── leak_dropped_uids.json  test.py --leak_drop 용
├── processed/              prepare_dataset.py 산출물
└── embeddings/             build_embeddings.py 산출물
```

원본(`data/raw/`)과 산출물(`data/preprocessed/`)은 물리적으로 분리돼 있다. 무거운 전체 스캔
캐시는 `data/preprocessed/_cache/`에 떨어지므로 `data/raw/`는 다운로드한 그대로 유지된다.

## prompts/ — 도메인별로 달라지는 것은 전부 여기에

도메인마다 달라져야 하는 문자열은 코드에 박지 않고 `prompts/{종류}_{domain}.txt`로 관리한다.
새 도메인을 추가하려면 아래 3개 파일이 필요하다.

| 파일 | 쓰는 곳 | 내용 |
|---|---|---|
| `query_prompt_{domain}.txt` | `review2query.py` | 리뷰 -> 1인칭 검색쿼리 리라이트 프롬프트 |
| `card_prompt_{domain}.txt` | `semantic_card.py` | 아이템 시맨틱 카드 생성 프롬프트 (필드 정의) |
| `item_instruction_{domain}.txt` | `build_embeddings.py` | E5 인코딩 직전 아이템 텍스트 앞에 붙는 instruction |

`item_instruction_`이 `HALO/e5_prefix_fix`가 지적한 그 문구다 — 예전에는
`"Represent this **book** ..."`이 도메인과 무관하게 코드에 고정돼 있어서 Video_Games/Beauty
임베딩이 오염됐다. 파일로 뺐고, **3개 도메인 모두 기존 코드/산출물이 쓰던 문자열과
byte-identical**임을 실측 확인했으므로 동봉된 카드·임베딩은 변하지 않는다.

### 폴백 카드에는 파일이 없다

LLM이 빈 응답을 준 아이템에 쓸 카드는 `card_prompt_{domain}.txt`의 **필드명에서 생성**한다
(`semantic_card.build_fallback_card`) — Beauty는 `Product/Customer/Experience/Context`,
Books·Video_Games는 `Genre/Audience/Style/Context`이므로 각각 그 필드에 `unknown`이 붙는다.

예전에는 이게 `prompts/fallback_card_{domain}.txt` 3개 파일에 따로 있었고, 필드명이
`card_prompt_`와 어긋나면(Beauty에 `Genre:`를 쓰는 식) 실패한 아이템만 스키마가 다른 카드가
섞였다. 그래서 로드 시점에 두 파일을 대조하는 검사까지 필요했다. 필드명을 프롬프트에서 바로
가져오면 **어긋남 자체가 구조적으로 불가능**해지므로 파일 3개와 대조 검사를 모두 없앴다.

구 파일은 값에 도메인 문구를 섞어 썼지만(`Audience: general readers` /
`Context: general reading`) 지금은 전 필드가 `unknown`이다. 이 카드가 쓰이는 상황은 정의상 그
아이템에 대해 아무것도 모르는 경우이고, **동봉된 세 데이터셋 67,752장 중 이 경로를 탄 카드는
0장**이라 산출물에는 아무 영향이 없다.

`semantic_card.py::_fmt_meta()`가 쓰는 메타 필드 목록은 `prompts/`가 아니라
**`config.DATASETS[{ds}]["meta_fields"]`**에 있다. 축이 다르기 때문이다 — 이건 프롬프트에 들어가는
문구가 아니라 "메타 parquet의 어떤 컬럼을 읽을지"라서, 도메인 명사(`book`)가 아니라
`source_category`(메타 스키마)에 종속된다. 세 데이터셋은 Amazon-Reviews-2023 스키마가 같아
`AMAZON_META_FIELDS` 하나를 공유한다:

```python
AMAZON_META_FIELDS = ("author", "categories", "features", "description", "details")
```

튜플 순서가 곧 `[METADATA]` 블록의 출력 순서다. `average_rating`/`rating_number`는 형식이 달라
(`AvgRating: 4.3 (5124 ratings)`) `_fmt_meta`가 따로 처리하므로 이 목록에 없다.

예전에는 이 목록이 `_fmt_meta()` 안에 박혀 있었다. 메타 컬럼이 다른 카테고리를 추가하면
`row.get(k)`가 조용히 `None`을 반환해 그 컬럼이 카드 입력에서 통째로 빠졌고, 아무 신호도 없었다.
이제 `semantic_card.py`가 실행 시 선언된 컬럼이 실제 parquet에 다 있는지 확인하고 없으면 경고한다
(카테고리에 따라 실제로 없을 수 있는 컬럼이라 중단하지는 않는다).

## `--dataset` 하나로 끝난다

모든 스크립트가 `--dataset books|video_games|beauty` 하나만 받고, 원본 경로·프롬프트 도메인·
산출물 경로·평가 모집단을 전부 `src/config.py`의 `DATASETS` 레지스트리에서 유도한다.

| dataset | source_category (data/raw/) | domain (prompts/) | canonical_n | gamma | meta_fields |
|---|---|---|---|---|---|
| `books` | `Books` | `book` | 19,748 | 0.01 | `AMAZON_META_FIELDS` |
| `video_games` | `Video_Games` | `video game` | 22,761 | 0.5 | `AMAZON_META_FIELDS` |
| `beauty` | `Beauty_and_Personal_Care` | `beauty product` | 23,280 | 0.5 | `AMAZON_META_FIELDS` |

학습 예산은 세 데이터셋 공통 **200 epochs / patience 20**이고, `train.py`의 기본값이 이미
그 값이라 따로 넘기지 않아도 된다. `gamma`만 데이터셋마다 달라 항상 명시해야 한다.

새 데이터셋 추가: `DATASETS`에 한 줄 등록(`meta_fields` 포함) + `prompts/`에 그 도메인의 3개 파일
(`query_prompt_`/`card_prompt_`/`item_instruction_{domain}.txt`)을 만들면 된다.
등록되지 않은 키를 주면 폴백 없이 즉시 에러다.

## 데이터 파이프라인 6단계

```bash
# [1] 원본 다운로드 -> data/raw/Books_{reviews,meta}.parquet
#     ※ 여기서의 --category는 dataset 키가 아니라 Amazon-Reviews-2023의 원본 카테고리 이름
python3 src/download_data.py --category Books

# [2] dedup -> priority 유저 풀 -> 샘플링 -> in-sample k-core -> split -> manifest 고정
python3 src/preprocessing.py --dataset books

# [3] 리뷰 -> 검색쿼리 (Ollama/gemma)
python3 src/review2query.py --dataset books \
    --fixed_input data/preprocessed/books/sample.parquet \
    --ollama-urls http://localhost:11434,http://localhost:11435

# [4] 아이템 시맨틱 카드 (원본 전체에서 생성, valid/test 타깃 리뷰 R_ui는 제외)
python3 src/semantic_card.py --dataset books

# [5] 학습/평가용 데이터셋 변환 (split은 manifest에서만 가져옴)
#     ※ 여기서 5-core가 한 번 더 돈다 — normalize_columns가 query가 빈 행을 버리기 때문이다.
#       그 결과로 manifest 타깃 행까지 떨어지면 [split][WARN]이 뜨고 인스턴스가 줄어든다(--skip_kcore로 끔).
python3 src/prepare_dataset.py --dataset books

# [6] 동결 텍스트 인코더 임베딩
python3 src/build_embeddings.py --dataset books
```

[2]~[6]은 `--dataset`만 바꾸면 다른 도메인에서 그대로 동작한다. 개별 경로가 필요하면
`--reviews`/`--meta`/`--input`/`--out` 등으로 언제든 덮어쓸 수 있고, 명시한 값이 항상 우선한다.

## 학습 / 평가

```
src/train.py     학습 엔트리포인트   (구 final/train_attention_residual_v2.py)
src/test.py      full-catalog 평가   (구 final/eval_m7_test.py)
src/model.py     모델·데이터셋·evaluate (구 final/train_eval_halo_sr_semantic_anchor.py)
                 ※ 라이브러리 전용 — 직접 실행하지 않는다(엔트리포인트는 train.py/test.py)
src/metrics.py   HR@k / NDCG@k / MRR  (구 final/ablation_report_newcard.py에서 추출)
src/utils.py     스크립트 간 복제돼 있던 공용 헬퍼 (norm_text/sha1_16/load_jsonl/kcore_filter)
src/ollama_client.py  Ollama 호출 계층 — review2query[3]/semantic_card[4]가 공유
```

`train.py`가 v2인 이유는 원본 `model.py`의 `main()`과 **RNG 소비 궤적을 정확히 일치**시키기 위한
재작성이기 때문이다(epoch-0 sanity eval 복원, ref_batch 제거, `next(iter())` 제거). T0와 T1이
동일 batch 순서·동일 sampled negative·동일 dropout을 쓰므로 두 결과의 차이는 `gamma*L_act`
효과로만 해석된다.

`metrics.py`는 원래 `ablation_report_newcard.py`에서 import했는데, 그 파일은 import 시점에
`runs_newcard_ablation/report/`를 mkdir하는 모듈 레벨 부작용이 있었다(26행). 학습 스크립트가
리포트용 디렉터리를 만들 이유가 없어 순수 함수만 떼어냈다(본문 무수정).

### 실행

```bash
# Books — T0(gamma=0)와 T1(gamma=0.01) 짝. 같은 시드로 동시에 돌리면 batch 순서·negative가 같다.
# epochs/patience는 기본값이 200/20이라 생략 가능하다.
python3 src/train.py --dataset books --condition T1 --gamma 0.01 \
    --history_ablation u_act_only --batch_size 128 \
    --seed 2026 --device cuda --sdpa_math --skip_nonfinite_step \
    --out_dir runs/books_T1_seed2026

# Video_Games / Beauty — gamma만 다르다 (0.5)
python3 src/train.py --dataset video_games --condition T1 --gamma 0.5 \
    --history_ablation u_act_only --batch_size 128 \
    --seed 2026 --device cuda --sdpa_math --skip_nonfinite_step \
    --out_dir runs/vg_T1_seed2026

# 평가 (여러 run_dir을 한 번에 주면 마지막에 요약표가 나온다)
python3 src/test.py runs/books_T1_seed2026 --dataset books \
    --leak_drop --sdpa_math --device cuda:0
```

### 학습 예산 200 / 20

`train.py`의 `A5_ARGS` 기본값이 예전에는 **100 epochs / patience 10**이라, 드라이버 스크립트가
매번 `--epochs 200 --patience 20`을 넘겨야 했고 안 넘기면 조용히 다른 예산으로 학습됐다.
이제 기본값 자체가 200/20이고, 학습 예산의 단일 출처는 `train.py`의 `A5_ARGS` 하나다
(예전에는 `model.py`의 레거시 `main()`에도 별도 기본값이 있어 어느 쪽이 진짜인지 헷갈렸다 —
 그 `main()`은 train.py가 대체한 뒤로 호출된 적이 없어 제거했다).

⚠ **동봉된 Books 체크포인트는 patience=10으로 학습된 것이다.** 4개 시드 전부 best epoch
이후 정확히 10 epoch만에 멈춘 것을 `train_log.json`으로 확인했다:

```
seed2026 best=111 last=121   seed2027 best= 83 last= 93
seed2028 best= 91 last=101   seed2029 best= 92 last=102     (전부 last-best=10)
```

따라서 지금 설정으로 Books를 재학습하면 더 오래 돌고 다른 epoch가 선택될 수 있다 —
동봉 체크포인트의 수치를 그대로 쓰려면 재학습하지 말 것. Video_Games/Beauty는 원래 20이었으므로
변화 없다.

`--dataset`만 주면 `--processed_dir`/`--embed_dir`/leak 파일이 전부 유도된다. 예전에는
`"amazon_books_queryA_card_22k/processed"`가 하드코딩돼 있어 cwd가 `final/`이 아니면 조용히
못 찾았고, `test.py`는 **run 디렉터리 이름에 "books"/"beauty"/"vg"가 들어있는지로 도메인을
추측**했다 — 디렉터리 이름만 바꿔도 다른 도메인 데이터로 채점될 수 있는 구조였다.

### `--leak_drop`

`{dataset}/leak_dropped_uids.json`의 uid를 제외해 canonical 모집단으로 평가하고,
`config.DATASETS`의 `canonical_n`과 대조 검증한다(불일치 시 중단).

이 목록은 **`preprocessing.py` 7단계 `drop_unseen_targets`가 제외하는 유저와 동일하다** —
Books에서 양쪽 다 정확히 같은 395명임을 실측 확인했다(차집합 0/0). 즉 manifest 기반으로 새로
만든 데이터셋에는 애초에 이들이 없어 이 단계가 no-op가 되고, manifest 도입 이전 산출물
(동봉된 `dataset_*/`)에만 실제로 작동한다.

### 포팅 검증 (실측)

옮긴 코드가 원본과 동일한 수치를 내는지 실제 체크포인트로 확인했다.

**Books** — `scope_full_books22k_checkpoints/` 4시드, `--leak_drop` (20,143 → 19,748):

| 지표 | src/test.py 재현 | 체크포인트 README 기록 |
|---|---|---|
| HR@10 | 0.1442 ± 0.0055 | 0.1442 ± 0.0054 |
| NDCG@10 | 0.0887 ± 0.0034 | 0.0887 ± 0.0034 |
| MRR | 0.0799 ± 0.0029 | 0.0799 ± 0.0029 |

평균 3개 모두 정확히 일치(HR@10 표준편차 0.0055 vs 0.0054는 4자리 반올림값으로 재계산한 차이).

**Video_Games** — `runs_m7_e5fix/video_games_T1_g0.5_seed*` 4시드, `--leak_drop`
(22,772 → 22,761). 기존 `test_result_leakdrop.json`과 seed별 직접 대조:

| seed | epoch | HR@10 | NDCG@10 | MRR | 기존 기록과 |
|---|---|---|---|---|---|
| 2026 | 115 | 0.1746 | 0.1049 | 0.0931 | 일치 |
| 2027 | 137 | 0.1767 | 0.1071 | 0.0953 | 일치 |
| 2028 | 82 | 0.1761 | 0.1060 | 0.0940 | 일치 |
| 2029 | 90 | 0.1749 | 0.1047 | 0.0928 | 일치 |

**Beauty** — `runs_m7_e5fix/beauty_T1_g0.5_seed*` 4시드, `--leak_drop` (23,283 → 23,280):

| seed | epoch | HR@10 | NDCG@10 | MRR | 기존 기록과 |
|---|---|---|---|---|---|
| 2026 | 65 | 0.2239 | 0.1310 | 0.1145 | 일치 |
| 2027 | 91 | 0.2220 | 0.1294 | 0.1127 | 일치 |
| 2028 | 69 | 0.2265 | 0.1324 | 0.1157 | 일치 |
| 2029 | 77 | 0.2199 | 0.1289 | 0.1128 | 일치 |

VG·Beauty 모두 선택된 epoch까지 4시드 전부 기존 `test_result_leakdrop.json`과 동일하다.

## 동봉된 산출물

세 데이터셋 모두 **이미 만들어져 있으므로 [1]~[6]을 다시 돌릴 필요가 없다.** 아래 파일들은
`HALO/`에서 복사해 왔고(원본 유지), 10개 전부 원본과 **바이트 동일**함을 `cmp`로 확인했다.
`split_manifest.json` 3개만 `--from_pkl`로 새로 만든 것이다.

| `data/preprocessed/{ds}/` | books | video_games | beauty |
|---|---|---|---|
| `interactions.pkl` | `HALO/data/processed_Books_sample22k.pkl` | `HALO/data/processed_Video_Games_sample20k_priority.pkl` | `HALO/data/processed_Beauty_and_Personal_Care_sample20k.pkl` |
| `queries.parquet` | `HALO/data/processed_sample22k_queryA_5core.parquet` | `HALO/data/processed_..._priority_queryA.parquet` | `HALO/data/processed_..._sample20k_queryA.parquet` |
| `cards.jsonl` | `HALO/e5_prefix_fix/data/cards/cards_Books_sample22k_excl.jsonl` | 〃 `..._Video_Games_..._excl.jsonl` | 〃 `..._Beauty_..._excl.jsonl` |
| `processed/` | `HALO/final/amazon_books_queryA_card_22k/processed/` | `HALO/final/amazon_video_games_queryA/processed/` | `HALO/final/amazon_beauty_queryA/processed/` |
| `embeddings/` | 〃 `/halo_lite_newcard/` | 〃 `/halo_lite_e5fix/` | 〃 `/halo_lite_e5fix/` |
| `leak_dropped_uids.json` | `HALO/final/reports/22k_books_problem_validation/configs/` | 〃 `/video_games_validation/configs/` | 〃 `/beauty_validation/configs/` |
| `split_manifest.json` | `--from_pkl` 복원 (valid=test=19,754) | 〃 (23,585) | 〃 (23,885) |

규모: books 유저 20,143 / 아이템 30,291 · video_games 23,723 / 11,462 · beauty manifest 23,885.

**★ 임베딩 선택 주의**: VG와 Beauty는 `halo_lite`가 아니라 **`halo_lite_e5fix`**를 가져왔다.
원래 `halo_lite/`는 `"Represent this **book** ..."` instruction으로 인코딩된 오염본이다
(아래 8번 버그의 실제 피해). 직접 확인:

```
amazon_video_games_queryA/halo_lite/item_texts.json[1]       -> "Represent this book ..."       ✗
amazon_video_games_queryA/halo_lite_e5fix/item_texts.json[1] -> "Represent this video game ..." ✓
```

Books는 `halo_lite_newcard`가 정본이고, 이것이 `cards.jsonl`과 짝이라는 것은 `item_texts.json`의
카드 본문을 카드 jsonl과 직접 대조해 확인했다(2,000개 표본 전건 일치). 구버전 Books 카드는
`books/cards_legacy_v1.jsonl`로 보존만 해뒀다(미사용, 대조용).

## 원래 코드에서 고친 것 (로직 불변, 연결만 복구)

파이프라인이 실제로 끊겨 있던 지점과 조용히 틀릴 수 있던 지점만 손봤다. 샘플링·k-core·
split·leakage 제어·카드 선정(select8)·프롬프트 문구는 **한 줄도 바꾸지 않았다.**

1. **`review2query.py`가 프롬프트를 못 찾던 문제** — `PROMPT_DIR`이 `src/prompts`를 가리키고
   있었다(실제 위치는 repo root `prompts/`). `semantic_card.py`는 이미 root 기준이라 두
   스크립트의 기준이 서로 달랐다. 이제 둘 다 `config.PROMPT_DIR` 하나를 쓴다.

2. **Amazon-C4 유저 우선 포함 로직 삭제** — `load_c4_users()`가 이 저장소에 없는
   `02c_preprocess_c4` 모듈을 무조건 import해 `ModuleNotFoundError`로 즉시 죽었고,
   `--skip_c4_priority`를 붙여야만 실행됐다. 이 기능은 Books 평가 커버리지용 최적화였을 뿐
   파이프라인 본류가 아니므로, 우회 플래그를 남기는 대신 **관련 코드를 전부 제거**했다:
   `load_c4_users()` / `C4_USERS_CACHE` / `--skip_c4_priority` / `importlib` import /
   `select_sample_users()`의 C4 우선 분기 / `CFG.c4_category`·`c4_val_n` / 캐시 파일.
   유저 선정은 이제 k-core 생존 풀에서 고정 시드 무작위 추출 하나뿐이다.

   단, `is_c4_user` **컬럼 자체는 남아있다** — 동봉된
   `books/queries.parquet`에 실제 True 값이 4,261건 기록돼 있어서,
   이 컬럼을 못 읽게 하면 기존 데이터의 정보가 사라진다. `prepare_dataset.py`의 읽기 경로만
   유지했고(없는 입력에는 False로 채움), **이 컬럼을 새로 만드는 코드는 없다.**

3. **`split_manifest`가 존재한 적이 없던 문제** — 현재 Books 산출물은 `src/preprocessing.py`가
   아니라 구버전 스크립트로 만들어졌고, manifest는 그 경로에 없었다. manifest는 8단계에서
   pkl(`P`)만 보고 `build_manifest()`로 만들어지므로, **재샘플링 없이 복원**할 수 있다:

   ```bash
   python3 src/preprocessing.py --dataset books --from_pkl
   ```

   `--from_pkl`은 1~7단계를 통째로 건너뛰고 기존 pkl에서 manifest만 뽑는다. 동봉된
   `split_manifest_Books_sample22k.json`이 이 명령으로 만든 것이다.
   **기존 산출물을 쓸 때 `--from_pkl` 없이 재실행하면 안 된다** — 시드가 같아도 코드 경로가
   달라 유저 집합이 바뀔 수 있다.

4. **manifest 없을 때의 에러 메시지** — `semantic_card.py` / `prepare_dataset.py`가 그냥
   `FileNotFoundError`를 던지는 대신, `--from_pkl` 복원 명령을 안내하고 멈춘다.
   (조용히 pkl 재계산으로 폴백하는 쪽이 훨씬 위험하다: 22k Books에서 이 어긋남으로 valid/test
   타깃 788건이 카드 제외에서 빠졌고 그중 112건이 실제로 카드에 leak됐다.)

5. **파일명/경로 단순화 (`--category` -> `--dataset`)** — 예전에는 긴 실험 식별자가 모든
   파일명에 반복해서 박혀 있었다(`processed_Beauty_and_Personal_Care_sample20k_queryA.parquet`).
   이제 데이터셋 하나가 폴더 하나이고, 폴더 안의 파일명은 세 도메인이 전부 동일하다.
   재배치 후 Books 4시드 + Beauty 1시드를 다시 평가해 **수치가 한 자리도 변하지 않음**을
   확인했다.

6. **`preprocessing.py --out_dir`이 동작하지 않던 문제** — `data_out_dir`/`result_out_dir`을
   계산만 하고 저장 경로에는 `CFG.data_dir`을 쓰고 있었다. 이제 실제로 배선돼 있다.

7. **다른 카테고리의 원본을 조용히 집던 문제** — `config.py`에
   `card_reviews_path = "data/Video_Games_reviews.parquet"`가 하드코딩돼 있어, Books를 돌리며
   `--reviews/--meta`를 빠뜨리면 Video_Games 리뷰가 카드 소스로 들어갔다. 이제
   `config.DATASETS`가 `--dataset`에서 원본 경로를 유도하고, 등록되지 않은 키면 폴백 대신
   즉시 에러를 낸다.

8. **도메인 하드코딩 2건 -> prompts/ 로 분리** — `build_embeddings.py`의 아이템 instruction과
   `semantic_card.py`의 fallback card. 자세한 내용은 위 "prompts/" 절 참고.

9. **`build_embeddings.py`의 `"book"` 하드코딩** — 아이템 instruction
   (`"Represent this book ..."`)과 빈 아이템 폴백(`"general book recommendation"`)이 도메인과
   무관하게 고정이었다(`HALO/e5_prefix_fix/README.md`가 지적한 것과 동일 버그). 이제
   `prompts/item_instruction_{domain}.txt`에서 읽는다.
   **3개 instruction 파일 모두 동봉된 임베딩을 만든 문자열과 byte-identical임을 실측 확인했다**
   (`item_texts.json[1]`의 첫 문단과 직접 대조: book / video_game / beauty_product 전부 True).
   따라서 재실행해도 동일한 텍스트가 인코딩되며, 동봉된 `embeddings/`를 재생성할 이유가 없다.
   Books는 하드코딩 문구가 우연히 정답이었고, Video_Games/Beauty는 이 버그의 실제 피해자였다.

## 동봉된 dataset과 manifest의 알려진 불일치 (788건)

복원한 manifest와 동봉된 `books/processed/`를 대조하면 **788건**이 어긋난다.
이는 코드 주석이 경고하던 바로 그 수치이며, 실측으로 원인이 전부 특정됐다:

```
manifest      valid 19,754 / test 19,754   (pkl과 일치)
동봉 processed valid 20,143 / test 20,143   (= 전체 유저 수)
어긋난 키      395 + 393 = 788,  관련 유저 395명
  └ 389명: preprocessing.py 7단계 drop_unseen_targets가 평가에서 제외한 유저
           (valid/test 타깃 아이템이 train 카탈로그에 한 번도 없는 경우)
  └   6명: 구 leave-last-out과 manifest의 valid/test 경계가 실제로 다름
           (test instance 2건이 manifest의 valid 키와 겹침)
```

원인은 동봉된 `processed/`가 **manifest 도입 이전**에 만들어졌기 때문이다 — 그 폴더의
`preprocess_config.json`에도 `"split": "leave-last-out"`(독립 재계산)이라고 남아있다.
현재 `src/prepare_dataset.py`는 split을 manifest에서만 가져오므로 이 경로가 재발하지 않는다.

**따라서 [5]를 다시 돌리면 동봉된 dataset과 인스턴스 수가 달라진다**(20,143 → 19,754 근처).
지금 학습/평가에 쓰고 있는 것은 동봉된 쪽이므로, 논문 수치를 유지하려면 동봉된
`processed/` + `embeddings/`를 그대로 쓰고 [5]/[6]을 재실행하지 말 것.

## 알려진 재현성 caveat

`semantic_card.py::_fmt_meta()`는 현재 메타데이터에서 `title`/`subtitle`을 **제외**한다
(title은 쿼리-타깃 표면 일치 경로를 만들고, subtitle의 91.4%가 판형/발행 문자열이라서).
동봉된 `books/cards.jsonl`은 **이 변경 이전**, 즉 title/subtitle이 포함된
입력으로 생성된 것이다. 따라서 지금 [4]를 다시 돌리면 동봉된 카드와 동일한 파일이 나오지
않는다. 논문 결과와 짝이 되는 것은 동봉된 파일 쪽이다.

## 미사용 코드 정리 (로직 불변)

파이프라인이 실제로 실행하는 경로는 한 줄도 바꾸지 않고, **아무도 호출하지 않던 코드만** 제거했다.
구코드와 신코드로 같은 계산을 돌려 40개 항목(3개 `history_ablation` × full_scores/u_star/
score_against/evaluate/attention-only/손실 + 공용 헬퍼 + k-core 9시나리오)이 전부 동일함을
`halo` 컨테이너에서 실측 대조했다.

| 대상 | 제거 사유 |
|---|---|
| `config.py` 필드 47개 (66 → 19) | `CFG.<name>`으로 읽는 코드가 없었다. 모델/학습 하이퍼파라미터는 `train.py`의 `A5_ARGS`와 각 argparse가 실제 출처이므로, 같은 이름이 config에도 있으면 "고쳤는데 안 바뀌는" 함정이 된다 |
| `config.Config.run_name()` | 존재하지 않는 `self.category`를 참조 — 호출하면 `AttributeError`. 호출처 0 |
| `config.EPOCHS` / `PATIENCE` | 참조처 0 (실제 값은 `A5_ARGS`) |
| `model.main()` + `model.train_one_epoch()` | `train.py`가 대체한 레거시 엔트리포인트. RNG 소비 궤적이 달라 결과도 재현되지 않았다 |
| `model.compute_loss/bpr_loss/bce_loss` | `model.train_one_epoch()` 전용이었다. `train.py`는 `joint_and_act_losses()`에서 같은 식을 직접 계산한다 |
| `model.evaluate_verbose()` (148행) | 호출처 0 |
| `SASRecBackbone.forward_incremental()` | 호출처 0 (docstring도 "호출되지 않는다"고 명시). 파라미터가 없어 체크포인트 호환에 영향 없음 |
| 미사용 import·지역변수·파라미터 | `train.py`의 `json`, `test.py`의 `infer_paths(run_dir)`·`best`·`picked_epoch`, `model.py`의 `nonpad`·`top_idx`·`bsz` |
| `sys.path.insert(0, dirname(__file__))` 4곳 | `python3 src/x.py`로 실행하면 `sys.path[0]`이 이미 `src/`다 (나머지 4개 스크립트는 원래부터 이 줄 없이 동작했다) |

중복 정의는 `src/utils.py`로 모았다 — `norm_text`/`sha1_16`(preprocessing ↔ semantic_card),
`load_jsonl`(model ↔ build_embeddings), `kcore_filter`(preprocessing ↔ review2query). 특히 앞의
둘은 manifest의 `review_hash`와 카드 R_ui 제외 키를 잇는 접점이라, 복제본이 남아 있으면 한쪽만
고쳐져 조용히 어긋날 수 있었다.

**`safe_text`는 일부러 합치지 않았다** — `prepare_dataset.py` 쪽은 개행을 공백으로 치환하고
`build_embeddings.py` 쪽은 `strip`만 한다. 이름만 같고 하는 일이 달라서 합치면 산출물이 바뀐다.

`model.py`의 `objective` 생성자 인자는 손실 함수가 사라진 뒤에도 남겨뒀다 — `train.py`의
`save_ckpt`가 `A5_ARGS`를 그대로 체크포인트 `args`에 저장하고 `test.py`가 그 `args`로 모델을
재구성하므로, 기존 체크포인트를 읽으려면 생성자 시그니처가 유지돼야 한다.

## Ollama 호출은 한 곳에서 (`src/ollama_client.py`)

[3] 쿼리 생성과 [4] 카드 생성은 하는 일이 다르지만 Ollama를 쓰는 방식은 같아야 한다. 예전에는
각자 구현이라 아래가 전부 달랐고, `semantic_card.py`가 HTTP 헬퍼 하나를 얻으려고 이웃 단계인
`review2query.py`를 import하고 있었다.

| | 예전 [3] 쿼리 | 예전 [4] 카드 | 지금 (공통) |
|---|---|---|---|
| 재시도 | 3회 (2s/4s 백오프) | **없음** | 3회 |
| 동시성 | `len(urls) × --requests_per_server` | `len(urls)` 고정 | `len(urls) × --requests_per_server` |
| URL 파싱 | strip + 빈 값 제거 | `split(",")`만 | `parse_urls()` |
| 진행 표시 | tqdm | 청크 후 print 1줄 | tqdm |
| 서버/모델 지정 | `--ollama_urls` / `--model` | **CLI 없음**(config만) | 양쪽 다 CLI, 기본값은 `CFG` |

단계마다 달라야 하는 것은 그대로 분리돼 있다 — 프롬프트 문안, **`max_new_tokens`(쿼리 220 /
카드 300)**, 응답 후처리(`clean_query`/`clean_card`), 체크포인트 단위. 두 토큰 예산은
`CFG.query_max_new_tokens` / `CFG.card_max_new_tokens`로 나란히 둔다.

`--ollama-urls`(하이픈)는 별칭으로 계속 받는다 — 기존 실행 명령이 그대로 동작한다.

**동작 변화 1건**: 카드 생성이 재시도를 갖게 됐다. 예전에는 요청 하나가 실패하면 예외가
`ex.map()`에서 올라와 그 배치 32개가 통째로 죽었다(카드가 300토큰으로 더 긴 생성이라 타임아웃
위험은 더 큰데 보호는 더 약했다). 성공 경로의 응답은 바뀌지 않는다.
