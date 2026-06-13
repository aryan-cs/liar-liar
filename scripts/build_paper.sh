#!/usr/bin/env bash
# Compile the paper to docs/main.pdf (served by GitHub Pages from /docs at
# https://aryan-cs.github.io/liar-liar/main.pdf).
set -euo pipefail
cd "$(dirname "$0")/.."
TECTONIC="$(command -v tectonic || echo /opt/homebrew/bin/tectonic)"
(cd docs && "$TECTONIC" main.tex)
echo "paper built -> docs/main.pdf"
