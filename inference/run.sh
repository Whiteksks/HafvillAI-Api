#!/usr/bin/env bash
set -euo pipefail
INFERENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$INFERENCE_DIR/.." && pwd)"
PYTHON_BIN="${HAFVILLAI_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN. See inference/README.md." >&2; exit 1; }
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$INFERENCE_DIR/.cache/triton}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$INFERENCE_DIR/.cache/tilelang}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$INFERENCE_DIR/.cache/torch_extensions}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
cd "$PROJECT_DIR"
ACTION="${1:-serve}"
if (( $# )); then shift; fi
case "$ACTION" in
    serve) exec "$PYTHON_BIN" -u -m inference.api "$@" ;;
    test) exec "$PYTHON_BIN" -m unittest discover -s "$INFERENCE_DIR/tests" -v "$@" ;;
    smoke) exec "$PYTHON_BIN" -m inference.smoke "$@" ;;
    prepare|verify) exec "$PYTHON_BIN" -m inference.prepare_artifacts "$ACTION" "$@" ;;
    bundle) exec "$PYTHON_BIN" -m inference.bundle "$@" ;;
    *) echo "Usage: bash inference/run.sh [serve|test|smoke|prepare|verify|bundle]" >&2; exit 2 ;;
esac
