#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY="${PY:-python}"
DATASETS_ARG="all"
SEEDS="2026 2027 2028 2029"
STAGES="train,test"
OUT_ROOT="runs"
WORK_DIR=""
DEVICE="cuda:0"
CONDITION="T1"
HISTORY_ABLATION="u_act_only"
BATCH_SIZE=128
EPOCHS=""
PATIENCE=""
GAMMA_OVERRIDE=""
OLLAMA_URLS=""
LEAK_DROP=1
FORCE=0
DRY=0
DETERMINISTIC=0

usage() {
  cat <<'USAGE'
SCOPE pipeline driver.

  bash run.sh                                   train -> evaluate with the released artifacts
  bash run.sh --dataset books --stages test
  bash run.sh --dataset books --stages 5,6,train,test
  bash run.sh --stages 1-6,train,test --force --work-dir rebuild
  bash run.sh --dry-run                         print the commands without running them

Options
  --dataset <name|all>        books | beauty | video_games | all
  --seeds "<a b c d>"         training seeds (default: 2026 2027 2028 2029)
  --stages <list>             1..6 | train | test | data | all, comma-separated, ranges allowed
  --out-root <dir>            where run directories are written (default: runs)
  --work-dir <dir>            write new data outputs here, read inputs from the standard paths
  --device <dev>              cuda | cuda:0 | cpu
  --condition <T0|T1>         training condition (default: T1)
  --history-ablation <mode>   u_act_only | h_n_only | none
  --batch-size / --epochs / --patience / --gamma
  --ollama-urls <urls>        Ollama servers for stages [3] and [4]
  --no-leak-drop              evaluate on the full test population instead of the canonical one
  --force                     allow overwriting existing artifacts
  --deterministic             pin BLAS thread counts to one
  --python <bin>              python executable to use
USAGE
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)      DATASETS_ARG="$2"; shift 2;;
    --seeds)        SEEDS="$2"; shift 2;;
    --stages)       STAGES="$2"; shift 2;;
    --out-root)     OUT_ROOT="$2"; shift 2;;
    --work-dir)     WORK_DIR="$2"; shift 2;;
    --device)       DEVICE="$2"; shift 2;;
    --condition)    CONDITION="$2"; shift 2;;
    --history-ablation) HISTORY_ABLATION="$2"; shift 2;;
    --batch-size)   BATCH_SIZE="$2"; shift 2;;
    --epochs)       EPOCHS="$2"; shift 2;;
    --patience)     PATIENCE="$2"; shift 2;;
    --gamma)        GAMMA_OVERRIDE="$2"; shift 2;;
    --ollama-urls)  OLLAMA_URLS="$2"; shift 2;;
    --no-leak-drop) LEAK_DROP=0; shift;;
    --force)        FORCE=1; shift;;
    --dry-run)      DRY=1; shift;;
    --deterministic) DETERMINISTIC=1; shift;;
    --python)       PY="$2"; shift 2;;
    -h|--help)      usage;;
    *) echo "unknown argument: $1  (see --help)" >&2; exit 2;;
  esac
done

c_r=$'\033[31m'; c_g=$'\033[32m'; c_y=$'\033[33m'; c_b=$'\033[1m'; c_0=$'\033[0m'
log()  { printf '%s\n' "${c_b}[run]${c_0} $*"; }
warn() { printf '%s\n' "${c_y}[run][warn]${c_0} $*" >&2; }
die()  { printf '%s\n' "${c_r}[run][abort]${c_0} $*" >&2; exit 1; }
ok()   { printf '%s\n' "${c_g}[run]${c_0} $*"; }

run() {
  printf '%s\n' "      \$ $*"
  [[ $DRY -eq 1 ]] && return 0
  "$@"
}

expand_stages() {
  local out=() tok
  IFS=',' read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do
    case "$tok" in
      1-6|data) out+=(download preprocess query card dataset embed);;
      2-6)      out+=(preprocess query card dataset embed);;
      all)      out+=(download preprocess query card dataset embed train test);;
      1) out+=(download);; 2) out+=(preprocess);; 3) out+=(query);;
      4) out+=(card);;     5) out+=(dataset);;    6) out+=(embed);;
      download|preprocess|query|card|dataset|embed|train|test) out+=("$tok");;
      *) die "unknown stage '$tok' (1-6 / download preprocess query card dataset embed train test)";;
    esac
  done
  printf '%s\n' "${out[@]}"
}
mapfile -t STAGE_LIST < <(expand_stages "$STAGES")
has_stage() { local s; for s in "${STAGE_LIST[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

cfg() { "$PY" -c "
import sys; sys.path.insert(0, 'src')
from config import DATASETS, CFG
d = DATASETS['$1']
print(d.get('$2', CFG.__dict__.get('$2', '')))
" 2>/dev/null; }

ALL_DS=$("$PY" -c "import sys;sys.path.insert(0,'src');from config import DATASETS;print(' '.join(DATASETS))") \
  || die "cannot read config.py — check that this runs from the SCOPE root and that \$PY($PY) is correct."
if [[ "$DATASETS_ARG" == "all" ]]; then DS_LIST=($ALL_DS); else IFS=',' read -ra DS_LIST <<< "$DATASETS_ARG"; fi
for ds in "${DS_LIST[@]}"; do
  [[ " $ALL_DS " == *" $ds "* ]] || die "unregistered dataset '$ds' (available: $ALL_DS)"
done

need_py_mod() {
  "$PY" -c "import $1" 2>/dev/null || die "python module '$1' not found ($PY). See requirements.txt."
}
ollama_alive() {
  local urls="$1" u
  IFS=',' read -ra arr <<< "$urls"
  for u in "${arr[@]}"; do
    u="$(echo "$u" | xargs)"
    curl -s -m 3 "$u/api/tags" >/dev/null 2>&1 || return 1
  done
  return 0
}
guard_overwrite() {
  [[ -e "$1" ]] || return 0
  [[ $FORCE -eq 1 ]] && { warn "overwriting (--force): $1"; return 0; }
  die "already exists: $1
       $2
       pass --force to rebuild anyway, or --work-dir <dir> to keep the released files intact."
}

log "repo root  : $REPO"
log "datasets   : ${DS_LIST[*]}"
log "stages     : ${STAGE_LIST[*]}"
log "seeds      : $SEEDS"
log "device     : $DEVICE   |  out-root: $OUT_ROOT   |  work-dir: ${WORK_DIR:-(standard paths)}"
[[ $DRY -eq 1 ]] && warn "--dry-run: printing commands without running them"

if [[ $DETERMINISTIC -eq 1 ]]; then
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  log "determinism: OMP/MKL/OpenBLAS threads=1 (removes run-to-run drift from multi-threaded BLAS)"
fi

out_of() { [[ -n "$WORK_DIR" ]] && echo "$WORK_DIR/$1" || echo "data/preprocessed/$1"; }
in_of() {
  local ds="$1" f="$2"
  if [[ -n "$WORK_DIR" && -e "$WORK_DIR/$ds/$f" ]]; then echo "$WORK_DIR/$ds/$f"
  else echo "data/preprocessed/$ds/$f"; fi
}

stage_download() {
  local ds="$1" cat; cat="$(cfg "$ds" source_category)"
  need_py_mod datasets
  if [[ -f "data/raw/${cat}_reviews.parquet" && $FORCE -eq 0 ]]; then
    ok "  [1] skip — data/raw/${cat}_reviews.parquet already present"; return
  fi
  warn "  [1] downloading the ${cat} corpus: tens of GB, may take hours"
  run "$PY" src/download_data.py --category "$cat"
}

stage_preprocess() {
  local ds="$1" cat; cat="$(cfg "$ds" source_category)"
  need_py_mod pandas
  [[ -f "data/raw/${cat}_reviews.parquet" ]] || die "  [2] corpus not found: data/raw/${cat}_reviews.parquet
       fetch it with --stages download, or restore only the manifest if the pkl already exists:
         $PY src/preprocessing.py --dataset $ds --from_pkl"
  local out; out="$(out_of "$ds")"
  guard_overwrite "$out/interactions.pkl" \
    "re-running stage [2] draws the user sample again, so it may no longer match an existing
       queries.parquet / cards.jsonl / embeddings set.
       To restore only the manifest:  $PY src/preprocessing.py --dataset $ds --from_pkl"
  local extra=(); [[ -n "$WORK_DIR" ]] && extra=(--out_dir "$out")
  run "$PY" src/preprocessing.py --dataset "$ds" "${extra[@]}"
}

stage_query() {
  local ds="$1" urls="${OLLAMA_URLS:-$(cfg "$ds" ollama_urls)}"
  need_py_mod pandas
  local sample out; sample="$(in_of "$ds" sample.parquet)"; out="$(out_of "$ds")"
  [[ -f "$sample" ]] || die "  [3] sample.parquet not found: $sample  (output of preprocessing.py [2])
       it is not part of the released artifacts — rebuild from stage [2]."
  ollama_alive "$urls" || die "  [3] Ollama is not responding: $urls
       start the server first, or pass its address with --ollama-urls.
       Checking here avoids failing on the first request after a full corpus scan."
  run "$PY" src/review2query.py --dataset "$ds" \
      --fixed_input "$sample" --out "$out/queries.parquet" --ollama_urls "$urls"
}

stage_card() {
  local ds="$1" cat urls; cat="$(cfg "$ds" source_category)"; urls="${OLLAMA_URLS:-$(cfg "$ds" ollama_urls)}"
  need_py_mod pyarrow
  [[ -f "data/raw/${cat}_reviews.parquet" ]] || die "  [4] raw reviews not found: data/raw/${cat}_reviews.parquet
       cards are built from the full corpus, not from the sample."
  local cards="data/preprocessed/$ds/cards.jsonl"
  if [[ -f "$cards" ]]; then
    warn "  [4] $cards already present — generation resumes per asin and leaves existing cards
             untouched (a no-op when complete). Delete the file to regenerate from scratch."
  fi
  ollama_alive "$urls" || die "  [4] Ollama is not responding: $urls"
  run "$PY" src/semantic_card.py --dataset "$ds" --ollama_urls "$urls"
}

stage_dataset() {
  local ds="$1"
  need_py_mod polars
  local q mf out; q="$(in_of "$ds" queries.parquet)"; mf="$(in_of "$ds" split_manifest.json)"; out="$(out_of "$ds")"
  [[ -f "$q" ]]  || die "  [5] queries.parquet not found: $q  (output of stage [3])"
  [[ -f "$mf" ]] || die "  [5] split_manifest.json not found: $mf
       restore it without re-sampling:  $PY src/preprocessing.py --dataset $ds --from_pkl"
  guard_overwrite "$out/processed" \
    "the released processed/ holds the instances used for the reported training and evaluation.
       Rebuilding reproduces them (stage [5] verifies the canonical fingerprint), so there is no need
       to overwrite: use --work-dir <dir> to rebuild elsewhere, or --stages 6,train,test to skip [5]."
  local extra=(); [[ -n "$WORK_DIR" ]] && extra=(--input "$q" --manifest "$mf" --out_dir "$out")
  run "$PY" src/prepare_dataset.py --dataset "$ds" "${extra[@]}"
}

stage_embed() {
  local ds="$1"
  need_py_mod sentence_transformers
  local proc out; proc="$(in_of "$ds" processed)"; out="$(out_of "$ds")"
  [[ -d "$proc" ]] || die "  [6] processed/ not found: $proc  (output of stage [5])"
  guard_overwrite "$out/embeddings" "embeddings/ is regenerated from processed/ and cards.jsonl."
  local extra=(); [[ -n "$WORK_DIR" ]] && extra=(--data_dir "$proc" --out_dir "$out/embeddings")
  run "$PY" src/build_embeddings.py --dataset "$ds" "${extra[@]}"
}

run_dir_of() { echo "$OUT_ROOT/${1}_${CONDITION}_seed${2}"; }

stage_train() {
  local ds="$1" gamma seed d
  gamma="${GAMMA_OVERRIDE:-$(cfg "$ds" gamma)}"
  [[ -n "$gamma" ]] || die "  [train] could not read gamma for $ds from config.DATASETS"
  need_py_mod torch
  local emb proc; emb="$(in_of "$ds" embeddings)"; proc="$(in_of "$ds" processed)"
  [[ -f "$emb/item_embs.npy" ]] || die "  [train] item_embs.npy not found: $emb  (output of stage [6])"
  [[ -f "$proc/train_sequences.jsonl" ]] || die "  [train] train_sequences.jsonl not found: $proc  (output of stage [5])"
  log "  [train] $ds  gamma=$gamma (from config.DATASETS)  condition=$CONDITION  ablation=$HISTORY_ABLATION"
  for seed in $SEEDS; do
    d="$(run_dir_of "$ds" "$seed")"
    if [[ -f "$d/checkpoint_manifest.json" && $FORCE -eq 0 ]]; then
      ok "    skip seed $seed — already complete ($d). To continue: train.py --resume_from $d/full_state_last.pt"
      continue
    fi
    local extra=()
    [[ -n "$EPOCHS"   ]] && extra+=(--epochs "$EPOCHS")
    [[ -n "$PATIENCE" ]] && extra+=(--patience "$PATIENCE")
    [[ -n "$WORK_DIR" ]] && extra+=(--processed_dir "$proc" --embed_dir "$emb")
    run "$PY" src/train.py --dataset "$ds" --condition "$CONDITION" --gamma "$gamma" \
        --history_ablation "$HISTORY_ABLATION" --batch_size "$BATCH_SIZE" --seed "$seed" \
        --device "$DEVICE" --sdpa_math --skip_nonfinite_step --out_dir "$d" "${extra[@]}"
  done
}

stage_test() {
  local ds="$1" seed d dirs=()
  need_py_mod torch
  for seed in $SEEDS; do
    d="$(run_dir_of "$ds" "$seed")"
    [[ -d "$d" ]] && dirs+=("$d") || warn "  [test] run directory missing, skipping: $d"
  done
  [[ ${#dirs[@]} -gt 0 ]] || { warn "  [test] $ds — no runs to evaluate (run --stages train first)"; return; }
  local extra=()
  if [[ $LEAK_DROP -eq 1 ]]; then
    if [[ -f "data/preprocessed/$ds/leak_dropped_uids.json" ]]; then
      extra+=(--leak_drop)
      log "  [test] --leak_drop enabled (checked against canonical_n=$(cfg "$ds" canonical_n))"
    else
      warn "  [test] no leak_dropped_uids.json, so --leak_drop is skipped — evaluating the full test
             population, which is not the canonical population of the reported numbers."
    fi
  fi
  [[ -n "$WORK_DIR" ]] && extra+=(--processed_dir "$(in_of "$ds" processed)" --embed_dir "$(in_of "$ds" embeddings)")
  run "$PY" src/test.py "${dirs[@]}" --dataset "$ds" --device "$DEVICE" --sdpa_math "${extra[@]}"
}

START=$(date +%s)
for ds in "${DS_LIST[@]}"; do
  echo
  log "${c_b}===== $ds =====${c_0}"
  for st in "${STAGE_LIST[@]}"; do
    log "  --- $st ---"
    "stage_$st" "$ds"
  done
done
echo
ok "done (${SECONDS}s).  results: $OUT_ROOT/*/test_result.json"
[[ $DETERMINISTIC -eq 0 && $DRY -eq 0 ]] && has_stage train && \
  warn "add --deterministic when run-to-run reproducibility matters (multi-threaded BLAS makes
        train_loss drift slightly between runs)"
exit 0
