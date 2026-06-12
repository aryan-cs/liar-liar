#!/usr/bin/env bash
# Compile the paper and publish the PDF to docs/paper.pdf (the always-current copy).
set -euo pipefail
cd "$(dirname "$0")/.."
(cd docs/paper && /opt/homebrew/bin/tectonic main.tex)
cp docs/paper/main.pdf docs/paper.pdf
echo "paper built -> docs/paper.pdf"
