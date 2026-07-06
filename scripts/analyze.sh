#!/usr/bin/env bash
# Mac-side analysis: fetch results from the GPU host, regenerate the summary,
# figures, tables, and paper number macros, then rebuild the paper PDF.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/fetch_results.sh
.venv/bin/python scripts/stage4_recal.py
.venv/bin/python scripts/stage6_multi.py
bash scripts/build_paper.sh
