#!/usr/bin/env bash
# Launch the full pipeline on the H200. Designed for nohup:
#   nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="$PWD/.hf_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

source .venv/bin/activate

for stage in stage0_setup stage1_vectors stage2_headline stage3_probes; do
  marker_dir=""
  case "$stage" in
    stage0_setup)    marker_dir="artifacts/stage0" ;;
    stage1_vectors)  marker_dir="artifacts/stage1" ;;
    stage2_headline) marker_dir="results/stage2" ;;
    stage3_probes)   marker_dir="results/stage3" ;;
  esac
  if [ -f "$marker_dir/DONE" ]; then
    echo "[run_all] $stage already done, skipping"
    continue
  fi
  echo "[run_all] starting $stage at $(date -Is)"
  python -u "scripts/${stage}.py" 2>&1 | tee -a "logs/${stage}.log"
  echo "[run_all] finished $stage at $(date -Is)"
done

echo "[run_all] ALL STAGES COMPLETE at $(date -Is)"
