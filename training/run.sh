#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_DIR/hafvillai/.cache/huggingface/hub}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_DIR/hafvillai/.cache/triton}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$REPO_DIR/hafvillai/.cache/tilelang}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_DIR/hafvillai/.cache/torch_extensions}"

ACTION="${1:-}"
case "$ACTION" in
    prepare|train) shift ;;
    *) echo "Использование: bash training/run.sh prepare|train --config ..." >&2; exit 2 ;;
esac
exec "$REPO_DIR/.venv/bin/python" -u -m "training.$ACTION" "$@"
