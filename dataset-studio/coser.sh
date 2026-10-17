#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/dataset-studio${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$REPO_DIR/hafvillai/.cache/huggingface/hub}"
ACTION="${1:-check}"
if (( $# )); then shift; fi
case "$ACTION" in
    download) exec .venv/bin/python -u dataset-studio/coser_download.py "$@" ;;
    catalog) exec .venv/bin/python -u dataset-studio/coser_catalog.py "$@" ;;
    prepare|prepare-sft)
        .venv/bin/python -u dataset-studio/coser_catalog.py
        .venv/bin/python -u dataset-studio/coser_sft.py build "$@"
        exec bash training/run.sh prepare --config training/configs/coser_sft.json --offline
        ;;
    check|check-sft) exec .venv/bin/python -u dataset-studio/coser_sft.py check "$@" ;;
    smoke|smoke-long)
        .venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Smoke требует CUDA GPU. Для проверки данных используй coser.sh check.")
device = torch.cuda.get_device_properties(0)
required = 223904  # Измеренный запрос текущего MIMO backward для этой модели.
if device.shared_memory_per_block_optin < required:
    raise SystemExit(f"{device.name}: shared memory {device.shared_memory_per_block_optin} < {required} байт. "
                     "Текущий MIMO backward на этой карте не прошёл; используй совместимый GPU (проверена H200).")
PY
        STAMP="$(.venv/bin/python -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')"
        exec bash training/run.sh train --config training/configs/coser_sft.json --offline \
            --output-dir "outputs/coser-check-$STAMP" "--$ACTION" "$@"
        ;;
    baseline)
        exec bash hafvillai/run.sh compare --config training/configs/coser_sft.json \
            --suite --cases-file dataset-studio/data/coser/sft-v1/validation_cases.json \
            --mode base --decode both --max-new-tokens 512 --verify --audit --heldout-eval 8 "$@"
        ;;
    *) echo "Использование: bash dataset-studio/coser.sh download|catalog|prepare|check|smoke|smoke-long|baseline" >&2; exit 2 ;;
esac
