#!/usr/bin/env bash
# Compile the paper to docs/paper.pdf (served by GitHub Pages from /docs at
# https://aryan-cs.github.io/liar-liar/paper.pdf).
set -euo pipefail
cd "$(dirname "$0")/.."
TECTONIC="$(command -v tectonic || echo /opt/homebrew/bin/tectonic)"
(cd docs && "$TECTONIC" main.tex)
mv docs/main.pdf docs/paper.pdf
echo "paper built -> docs/paper.pdf"
