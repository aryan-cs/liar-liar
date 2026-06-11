#!/usr/bin/env bash
# Recalibrated pipeline. Launch detached:
#   setsid nohup bash scripts/run_recal.sh > logs/run_recal.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
source .venv/bin/activate

for stage in stage_recal stage2_recal stage3_recal; do
  echo "[run_recal] starting $stage at $(date -Is)"
  python -u "scripts/${stage}.py" 2>&1 | tee -a "logs/${stage}.log"
  echo "[run_recal] finished $stage at $(date -Is)"
done
echo "[run_recal] ALL RECAL STAGES COMPLETE at $(date -Is)"
