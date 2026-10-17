#!/usr/bin/env bash
# Run inside tmux: verification followed by two independent full SFT jobs.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
configs=("$@")
if (( ${#configs[@]} == 0 )); then
    configs=(training/configs/coser_sft.json training/configs/coser_subconsciousness.json)
fi
config_value() {
    .venv/bin/python - "$1" "$2" <<'PY'
import sys
from training.config import load_config
c = load_config(sys.argv[1])
print(c.path('output_dir') if sys.argv[2] == 'output_dir' else c.role)
PY
}
for config in "${configs[@]}"; do
    config_value "$config" role >/dev/null
done
mkdir -p outputs/logs
exec 9>outputs/pipeline.lock
flock -n 9 || { echo 'Другой pipeline уже работает' >&2; exit 1; }
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
STAGE=starting
status() { printf '%s\n%s\n%s\n' "$1" "$STAGE" "$(date -u +%FT%TZ)" > outputs/pipeline-status.txt; }
trap 'result=$?; if (( result == 0 )); then status completed; else status "failed:$result"; fi' EXIT

STAGE=gpu-regression
status running
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_DIR/hafvillai/.cache/triton}"
.venv/bin/python -m unittest discover -s training/tests -v \
    2>&1 | tee "outputs/logs/gpu-regression-$RUN_ID.log"

for mode in smoke smoke-long; do
    for config in "${configs[@]}"; do
        role="$(config_value "$config" role)"
        STAGE="$role/$mode"
        status running
        bash training/run.sh train --config "$config" --offline \
            "--$mode" --output-dir "outputs/verification-$RUN_ID/$role" \
            2>&1 | tee "outputs/logs/$role-$mode-$RUN_ID.log"
    done
done

for config in "${configs[@]}"; do
    role="$(config_value "$config" role)"
    output_dir="$(config_value "$config" output_dir)"
    STAGE="$role/train"
    status running
    args=()
    if [[ -f "$output_dir/completed.json" ]]; then
        echo "$role уже завершён"
        continue
    elif [[ -f "$output_dir/latest.json" ]]; then
        args+=(--resume latest)
    fi
    bash training/run.sh train --config "$config" --offline "${args[@]}" \
        2>&1 | tee -a "outputs/logs/$role-train.log"
done
STAGE=done
