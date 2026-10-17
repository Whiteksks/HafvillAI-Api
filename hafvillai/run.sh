#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$APP_DIR/../.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo "Не найден Python: $PYTHON_BIN" >&2; exit 1; }
export HF_HUB_CACHE="${HF_HUB_CACHE:-$APP_DIR/.cache/huggingface/hub}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$APP_DIR/.cache/triton}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$APP_DIR/.cache/tilelang}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$APP_DIR/.cache/torch_extensions}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ "${1:-}" == "compare" ]]; then shift; fi
cd "$APP_DIR/.."
exec "$PYTHON_BIN" -u -m hafvillai.compare "$@"
