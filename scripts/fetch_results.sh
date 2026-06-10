#!/usr/bin/env bash
# Fetch lightweight result artifacts from the H200 into the Mac repo.
# Heavy caches (.hf_cache, model weights) stay remote.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=uiuc-h200
RDIR=/home/aryang9/sandbox/liar-liar

mkdir -p artifacts/stage1 results/stage2 results/stage3

scp -q "$REMOTE:$RDIR/artifacts/stage1/{config.json,sweep.json,tokensets.json,capture_ids.json,certificates.json}" artifacts/stage1/ 2>/dev/null || true
scp -q "$REMOTE:$RDIR/artifacts/stage1/vectors.pt" artifacts/stage1/ 2>/dev/null || true
scp -q "$REMOTE:$RDIR/results/stage2/"*.jsonl results/stage2/ 2>/dev/null || true
scp -q "$REMOTE:$RDIR/results/stage3/"*.jsonl results/stage3/ 2>/dev/null || true
scp -q "$REMOTE:$RDIR/results/stage3/lens.pt" results/stage3/ 2>/dev/null || true
scp -q "$REMOTE:$RDIR/data/paraphrases.json" results/stage3/paraphrases.json 2>/dev/null || true

echo "fetched:"
ls -la artifacts/stage1 results/stage2 results/stage3 2>/dev/null | grep -v "^total\|^d" | awk '{print $NF, "("$5" bytes)"}' | grep -v "^(" | head -40
