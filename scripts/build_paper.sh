#!/usr/bin/env bash
# Compile the paper and publish the PDF to docs/paper.pdf (the always-current copy).
set -euo pipefail
cd "$(dirname "$0")/.."
TECTONIC="$(command -v tectonic || echo /opt/homebrew/bin/tectonic)"
(cd docs/paper && "$TECTONIC" main.tex)
cp docs/paper/main.pdf docs/paper.pdf
echo "paper built -> docs/paper.pdf"
