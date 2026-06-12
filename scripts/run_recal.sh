#!/usr/bin/env bash
# Recalibrated pipeline, GPU side. Launch detached:
#   setsid nohup bash scripts/run_recal.sh > logs/run_recal.log 2>&1 < /dev/null &
# Runs stage 0 (data + weights, idempotent) through stage 3, then the
# faithfulness probe. Mac-side analysis (figures, tables, paper numbers) is
# scripts/analyze.sh, which fetches these results and runs stage4_recal.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
source .venv/bin/activate

for stage in stage0_setup stage_recal stage2_recal stage3_recal probe_faithfulness; do
  echo "[run_recal] starting $stage at $(date -Is)"
  python -u "scripts/${stage}.py" 2>&1 | tee -a "logs/${stage}.log"
  echo "[run_recal] finished $stage at $(date -Is)"
done
echo "[run_recal] ALL RECAL STAGES COMPLETE at $(date -Is)"
