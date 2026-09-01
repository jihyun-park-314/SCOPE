#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SCOPE_REPO="${SCOPE_REPO:-OWNER/REPO}"
TAG="${TAG:-v1.0.0}"

if [[ "$SCOPE_REPO" == "OWNER/REPO" ]]; then
  echo "[abort] Set the repository first:" >&2
  echo "        SCOPE_REPO=owner/scope bash scripts/fetch_release_assets.sh" >&2
  echo "       (or edit the SCOPE_REPO default in this script)" >&2
  exit 2
fi

if [[ $# -gt 0 ]]; then DS_LIST=("$@"); else DS_LIST=(books); fi

base="https://github.com/$SCOPE_REPO/releases/download/$TAG"

for ds in "${DS_LIST[@]}"; do
  asset="scope-$TAG-$ds.tar.gz"
  target="data/preprocessed/$ds/queries.parquet"
  if [[ -f "$target" ]]; then
    echo "[skip] $ds — already present ($target)"
    continue
  fi
  echo "[get ] $base/$asset"
  curl -fL --progress-bar -o "/tmp/$asset" "$base/$asset"
  tar -xzf "/tmp/$asset" -C "$REPO_DIR"
  rm -f "/tmp/$asset"
  echo "[ok  ] data/preprocessed/$ds/"
done

echo
echo "Next:"
echo "  bash run.sh --dataset ${DS_LIST[0]} --stages 5,6,train,test"
