# SCOPE

**Semantic–Contextual Orchestration of query-conditioned Personalization and Explicit Relevance**

Code and derived resources for the paper
*"Dual Grounding of Natural-Language Queries in Item Semantics and Behavioral Histories for
Sequential Recommendation."*

SCOPE is a query-conditioned sequential recommender built on the principle of *relevance-preserving
personalization*: direct query–item relevance is retained as an explicit term in the ranking score
rather than being absorbed into a personalized latent representation. An LLM reorganizes item
metadata and selected reviews **offline** into query-independent semantic cards, which a frozen text
encoder embeds once and matches directly against the current query. The same query representation is
projected into the collaborative space to retrieve position-wise contextual states from the user's
sequential trajectory. The semantic and behavioral scores are combined only at final ranking, so
**online inference requires no LLM call**.

Experiments cover three domains of Amazon Reviews 2023: **Books**, **Beauty & Personal Care**, and
**Video Games**.

---

## 1. What this repository provides

This repository provides the complete codebase and fixed experimental artifacts for the Books domain,
which is used for the detailed mechanism analyses in the paper — the query-similarity analysis and
the wrong-query control. The same implementation supports Beauty & Personal Care and Video Games;
these domains can be rebuilt locally from the public Amazon Reviews 2023 corpus using the provided
pipeline (Section 8.3).

Three categories of material are distinguished, because they differ in how they are obtained:

| Category | Where | Contents |
|---|---|---|
| **Source code** | this repository | pipeline, model, training and evaluation scripts, prompts, configurations |
| **Fixed experimental artifacts** | GitHub Release `v1.0.0` | the frozen files that define the reported Books experiments (Section 4) |
| **Raw corpus** | not redistributed | Amazon Reviews 2023, obtained from the official source (Section 3) |

The LLM-generated Books artifacts — surrogate queries and item semantic cards — are released, so
**reproducing the reported Books results requires no LLM server and no query or card regeneration**.

---

## 2. Repository structure

```
SCOPE/
├── run.sh                       Pipeline driver: stage selection, prerequisite checks, overwrite guards
├── requirements.txt
├── src/
│   ├── config.py                Single source of truth: dataset registry and path derivation
│   ├── download_data.py         [1] Download raw Amazon Reviews 2023 parquet files
│   ├── preprocessing.py         [2] Deduplication, user sampling, 5-core, chronological split, manifest
│   ├── review2query.py          [3] Review → first-person natural-language query (LLM)
│   ├── semantic_card.py         [4] Item metadata + selected reviews → semantic card (LLM)
│   ├── prepare_dataset.py       [5] Query parquet + manifest → training schema
│   ├── build_embeddings.py      [6] Frozen text-encoder embeddings for items and queries
│   ├── train.py                 Training entry point
│   ├── test.py                  Full-catalog evaluation entry point
│   ├── model.py                 Model, dataset, and evaluate(); library module, not run directly
│   ├── metrics.py               HR@k, NDCG@k, MRR
│   ├── ollama_client.py         Shared LLM request layer for stages [3] and [4]
│   └── utils.py                 Shared helpers (text normalization, review hashing, k-core, JSONL I/O)
├── prompts/                     Domain-specific prompt and instruction texts (Section 6)
├── scripts/
│   ├── fetch_release_assets.sh  Download and unpack the fixed artifacts
│   ├── make_release_assets.sh   Rebuild the release archive from a local working copy
│   └── collect_results.py       Runs → results/ (Section 9)
├── data/
│   ├── raw/                     Downloaded corpus (empty in a fresh clone)
│   └── preprocessed/{dataset}/  All per-dataset artifacts and derived outputs
├── results/                     Reported Books numbers, per seed and aggregated (Section 9)
└── runs/                        Training runs and evaluation results (created at run time)
```

The per-dataset layout is identical across domains; the release populates `books/`, and the other two
domains fill in when their artifacts are built locally.

```
data/preprocessed/books/
├── queries.parquet          [3] interactions with the generated query column     released
├── cards.jsonl              [4] item semantic cards                              released
├── split_manifest.json      [2] immutable train/valid/test split                 released
├── leak_dropped_uids.json       canonical evaluation population                  released
├── processed/               [5] training and evaluation instances                released
├── interactions.pkl         [2] user/item maps, sequences                        not distributed
├── sample.parquet           [2] input to stage [3]                               not distributed
└── embeddings/              [6] frozen encoder embeddings                        regenerated
```

---

## 3. Data

The raw Amazon Reviews 2023 corpus is **not redistributed**. It is publicly available from the
McAuley Lab at <https://amazon-reviews-2023.github.io/> and is downloaded with the script included
here:

```bash
python src/download_data.py --category Books
python src/download_data.py --category Beauty_and_Personal_Care
python src/download_data.py --category Video_Games
```

This writes `data/raw/{category}_reviews.parquet` and `data/raw/{category}_meta.parquet`; the
download is large (tens of GB). It is needed to build the Beauty & Personal Care and Video Games
datasets or to rebuild Books from scratch, and is not needed to reproduce the reported Books results
from the released artifacts.

The three domains are registered in `src/config.py`:

| `--dataset` | Source category | Prompt domain | `λ_aux` |
|---|---|---|---|
| `books` | `Books` | `book` | 0.01 |
| `beauty` | `Beauty_and_Personal_Care` | `beauty product` | 0.5 |
| `video_games` | `Video_Games` | `video game` | 0.5 |

Every script takes a single `--dataset` argument and derives the source category, prompt domain,
output paths, evaluation population, and `λ_aux` from this registry. `λ_aux` is the auxiliary-loss
coefficient of the training objective; it is domain-specific, injected automatically by `run.sh`, and
exposed in `train.py` as `--gamma`.

---

## 4. Fixed experimental artifacts

These files are the frozen experimental snapshot of the reported Books experiments. They are
distributed as a GitHub Release `v1.0.0` asset and unpack into `data/preprocessed/books/`.

| File | Produced by | Role |
|---|---|---|
| `queries.parquet` | stage [3], LLM | The surrogate natural-language query attached to every interaction; the input stage [5] consumes |
| `cards.jsonl` | stage [4], LLM | The item semantic cards |
| `split_manifest.json` | stage [2] | The immutable chronological train/validation/test split, and the single source of truth for it |
| `leak_dropped_uids.json` | — | User ids excluded to form the canonical evaluation population |
| `processed/` | stage [5] | The training and evaluation instances the reported runs consumed (271,012 / 20,143 / 20,143) |

Everything else is derived from them:

```
queries.parquet + split_manifest.json  ──[5] prepare_dataset.py──▶  processed/
processed/ + cards.jsonl               ──[6] build_embeddings.py─▶  embeddings/
```

Stage [5] is deterministic given these inputs: re-running it on the released `queries.parquet`
reproduces the same instances the reported runs used. `processed/` is shipped as well, so a
reproduction may either rebuild it and confirm the match or start directly from it.

Download and package:

```bash
SCOPE_REPO=<owner>/<repo> bash scripts/fetch_release_assets.sh   # download and unpack
bash scripts/make_release_assets.sh books                        # inverse: pack + SHA256SUMS
```

---

## 5. Requirements and environment

```bash
pip install -r requirements.txt
```

`torch` should be installed following the instructions at <https://pytorch.org> for the local CUDA
version. `run.sh` checks only the modules required by the stages actually being run.

| Task | Requirements |
|---|---|
| Training and evaluation | CUDA GPU, `torch`, `numpy`, `tqdm` |
| Stage [5] dataset preparation | `polars`, `pyarrow` |
| Stage [6] embeddings | `sentence-transformers`, GPU |
| Stages [3][4] query and card generation | an Ollama server, `requests`, `pandas`, `pyarrow` |
| Stage [1] corpus download | `datasets` |

**Text encoder.** Stage [6] uses the frozen `intfloat/e5-base-v2` encoder (downloaded automatically
from the Hugging Face Hub) for both item cards and queries. It is never updated during
recommendation training.

**LLM.** Stages [3] and [4] are the only components that call an LLM, and both run offline. Requests
go through `src/ollama_client.py` to one or more Ollama servers; the default model is set by
`CFG.ollama_model` in `src/config.py`, and server addresses are given by `--ollama-urls`. `run.sh`
verifies server availability before such a stage begins.

---

## 6. Prompts

All domain-dependent text is kept in `prompts/` rather than in the source code. Three files are
required per domain, where `{domain}` is the slug from the dataset registry (`book`,
`beauty_product`, `video_game`):

| File | Used by | Content |
|---|---|---|
| `query_prompt_{domain}.txt` | `review2query.py` | Instruction that rewrites a review into a first-person query, with a `{review}` slot |
| `card_prompt_{domain}.txt` | `semantic_card.py` | Semantic-card schema, grounding rules, and length constraint, with `{meta}` and `{reviews}` slots |
| `item_instruction_{domain}.txt` | `build_embeddings.py` | Instruction prefix prepended to item text before encoding |

The fallback card used when the LLM returns an empty response is derived from the field names in
`card_prompt_{domain}.txt`, so the card schema cannot drift between the prompt and the fallback. The
metadata columns supplied to card construction are declared per dataset as `meta_fields` in
`src/config.py`.

---

## 7. Pipeline

Six data stages, followed by training and evaluation. The stage numbers are the aliases accepted by
`run.sh --stages`.

| Stage | Alias | Script | Description |
|---|---|---|---|
| Download | `1` | `download_data.py` | Retrieve the raw corpus |
| Preprocessing | `2` | `preprocessing.py` | Deduplication, user sampling, in-sample 5-core re-convergence, chronological split, manifest freezing |
| Query preparation | `3` | `review2query.py` | Review → first-person natural-language query (LLM) |
| Card preparation | `4` | `semantic_card.py` | Item metadata and selected reviews → semantic card (LLM), with validation/test target reviews excluded |
| Dataset preparation | `5` | `prepare_dataset.py` | Convert to the training schema, joining the split from the manifest only |
| Embeddings | `6` | `build_embeddings.py` | Encode item cards and queries with the frozen encoder |
| Training | `train` | `train.py` | Train one run per seed |
| Evaluation | `test` | `test.py` | Full-catalog ranking evaluation |

Stage [2] fixes the split and no later stage recomputes it: stage [3] only fills a `query` column on
rows already determined by stage [2], stage [4] consults the manifest to exclude reviews associated
with held-out targets from an item's candidate review pool, and stage [5] joins the split from the
manifest. Semantic cards are therefore query-independent, and the behavioral path is the only
query-conditioned component.

Aliases `1-6`, `2-6`, `data`, and `all` expand to the corresponding stage groups, and stages combine
with commas (`--stages 2-6,train,test`).

---

## 8. Reproducing the reported results

### 8.1 Exact reproduction from the fixed Books artifacts

```bash
pip install -r requirements.txt
SCOPE_REPO=<owner>/<repo> bash scripts/fetch_release_assets.sh

bash run.sh --dataset books --stages 5,6,train,test
```

This rebuilds the training schema from `queries.parquet` and the split manifest, encodes items and
queries with the frozen encoder, trains four seeds, and evaluates every run under full-catalog
ranking on the canonical evaluation population. Completed seeds are skipped on re-invocation, so an
interrupted run resumes by repeating the command. `--dry-run` prints the commands without executing
them; `--deterministic` pins BLAS thread counts to one; `--stages 6,train,test` starts from the
released `processed/` instead of rebuilding it.

The canonical Books evaluation population — 19,748 users after `leak_dropped_uids.json` is applied —
is verified by a SHA-256 fingerprint covering each instance's user, target, and history. Stage [5]
reports it on completion, and `test.py --leak_drop` checks it again before evaluating:

```
[canonical] Expected canonical books evaluation population: 19,748
[canonical] Obtained: 19,748  (test instances 19,753 - leak-dropped users 395)
[canonical] sha256   expected=42756d159027d65c3c51cc89fb780c10381796de663713b66c066556ed6a7cd3
[canonical]          obtained=42756d159027d65c3c51cc89fb780c10381796de663713b66c066556ed6a7cd3
[canonical] PASS
```

### 8.2 Running the stages individually

```bash
# [5] training schema  (also the step that reports the canonical fingerprint)
python src/prepare_dataset.py --dataset books

# [6] frozen encoder embeddings
python src/build_embeddings.py --dataset books

# training (epochs and patience default to 200 and 20)
python src/train.py --dataset books --condition T1 --gamma 0.01 \
    --history_ablation u_act_only --batch_size 128 --seed 2026 \
    --device cuda --sdpa_math --skip_nonfinite_step \
    --out_dir runs/books_T1_seed2026

# evaluation; passing several run directories prints a per-seed summary table
python src/test.py runs/books_T1_seed202{6,7,8,9} --dataset books \
    --leak_drop --sdpa_math --device cuda:0

# collect the finished runs into results/ (Section 9)
python scripts/collect_results.py --datasets books
```

`--leak_drop` restricts evaluation to the canonical population defined by `leak_dropped_uids.json`
and is required for comparison with the reported numbers. `test.py` reports HR@k, NDCG@k, and MRR for
the cutoffs given by `--ks` (default `10,50`); the ranking-depth comparison in the paper uses
`--ks 1,10,20`.

### 8.3 End-to-end procedural reproduction for the other domains

For Beauty & Personal Care and Video Games, the released pipeline reproduces the complete
data-construction and training procedure from the public corpus. Because user sampling and LLM-based
query/card generation are stochastic, newly generated artifacts may differ from the fixed
experimental snapshots used for the reported results. Exact artifact-level reproduction is therefore
provided for Books, while the other domains support end-to-end procedural reproduction.

Stages [1]–[5] build a domain's artifacts from scratch. They need the raw corpus and, for [3] and
[4], a reachable Ollama server.

```bash
python src/download_data.py --category Beauty_and_Personal_Care
python src/preprocessing.py --dataset beauty
python src/review2query.py --dataset beauty \
    --fixed_input data/preprocessed/beauty/sample.parquet \
    --ollama_urls http://localhost:11434
python src/semantic_card.py --dataset beauty

python src/prepare_dataset.py --dataset beauty --out_dir <new-dir> --allow_new_population

bash run.sh --dataset beauty --stages 6,train,test --work-dir <new-dir>
```

The same sequence applies to `books` and `video_games` with the corresponding `--category`.
`--allow_new_population` tells stage [5] that the evaluation population is a newly built one rather
than a fixed snapshot; `run.sh` will not overwrite existing artifacts without `--force`, and
`--work-dir <dir>` keeps new outputs separate while reading inputs from the standard paths.

### 8.4 Ablation variants

| Paper variant | Command |
|---|---|
| (b) without behavioral path | reported by `test.py` on every run as the `query_only` result |
| (c) without query-conditioned retrieval | `train.py --history_ablation h_n_only` |
| (d) semantic card replaced by raw text | `build_embeddings.py --no_card`, then train on the resulting embeddings |

---

## 9. Outputs

| Path | Written by | Contents |
|---|---|---|
| `results/sample_stats_{dataset}.json` | stage [2] | Preprocessing statistics: filtering, sampling, k-core convergence, split |
| `results/card_stats_{dataset}.json` | stage [4] | Card-generation statistics: coverage, exclusions, review-pool sizes |
| `runs/{dataset}_{condition}_seed{n}/train_log.json` | `train.py` | Per-epoch training and validation trace |
| `runs/{dataset}_{condition}_seed{n}/checkpoint_manifest.json` | `train.py` | Selected checkpoint and run configuration |
| `runs/{dataset}_{condition}_seed{n}/test_result.json` | `test.py` | Full-catalog test metrics, evaluated population size, and the resolved configuration |
| `results/books/seed_{n}.json` | `collect_results.py` | Selected epoch and test metrics of one run: HR@10, NDCG@10, MRR |
| `results/paper_results.json` | `collect_results.py` | The same numbers as mean and standard deviation over the four seeds |

`runs/` and the stage statistics in `results/` are created at run time and are not version-controlled.
The two files written by `collect_results.py` are tracked, so a reproduction can be compared against
the reported Books numbers without rerunning anything.

---

## 10. Configuration and seeds

| Setting | Location |
|---|---|
| Training seeds (2026, 2027, 2028, 2029) | `SEEDS` in `run.sh`; `--seeds` overrides; `train.py --seed` for a single run |
| User-sampling seed (42) | `CFG.sample_seed` in `src/config.py` |
| Dataset-preparation seed (42) | `prepare_dataset.py --seed` |
| Model and optimization hyperparameters | `A5_ARGS` in `src/train.py` (hidden dimension, maximum history length, blocks, heads, dropout, learning rate, weight decay, batch size, epochs, patience) |
| `λ_aux` per domain | `DATASETS` in `src/config.py`; injected by `run.sh`, overridable with `--gamma` |
| Paths, sampling parameters, card-construction parameters, LLM settings | `Config` and `DATASETS` in `src/config.py` |

`A5_ARGS` is the single source of truth for the training budget and model configuration; it is stored
in each checkpoint and read back by `test.py` when reconstructing the model, so evaluation cannot
silently diverge from the configuration used at training time. Training is seeded but not bitwise
deterministic across BLAS thread configurations; `--deterministic` removes that source of drift by
pinning `OMP`, `MKL`, and `OpenBLAS` thread counts to one.

---

## 11. Reproducibility scope

- **Books — exact artifact-level reproduction.** The fixed snapshot in Section 4 reproduces the
  reported Books experiments without regenerating queries or semantic cards, and the canonical
  evaluation population is verified by fingerprint before evaluation.
- **Beauty & Personal Care and Video Games — end-to-end procedural reproduction.** The same code and
  configurations build both domains from the public corpus (Section 8.3).
- **Split integrity.** The manifest from stage [2] is the only definition of the validation and test
  targets; later stages join it rather than recomputing it, and stage [5] never drops a row the
  manifest names as a held-out target.
- **Scope of the implementation.** This repository focuses on the SCOPE implementation and its data
  pipeline. Baseline implementations follow their respective original sources; the repository provides
  the SCOPE-side evaluation outputs required for the comparisons reported in the paper.

---

## 12. Citation

```bibtex
@article{scope2026,
  title   = {Dual Grounding of Natural-Language Queries in Item Semantics and
             Behavioral Histories for Sequential Recommendation},
  year    = {2026},
  note    = {Under review}
}
```

---

## 13. Terms of use

The license for this code will be stated upon publication.

The artifacts distributed with the release are derived from Amazon Reviews 2023 and remain subject to
the terms of that dataset. The raw review corpus itself is not redistributed here.
