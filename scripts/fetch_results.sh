#!/usr/bin/env bash
# Fetch lightweight result artifacts from the H200 into the Mac repo.
# Heavy caches (.hf_cache, model weights) stay remote.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=uiuc-h200
RDIR=/home/aryang9/sandbox/liar-liar

mkdir -p artifacts/recal results/recal/dec results/recal/mm

# Recalibrated run (current pipeline)
for f in config.json calibration.json tokensets.json capture_ids.json certificates.json vectors.pt; do
  scp -q "$REMOTE:$RDIR/artifacts/recal/$f" artifacts/recal/ 2>/dev/null || true
done
scp -q "$REMOTE:$RDIR/results/recal/baseline.jsonl" results/recal/ 2>/dev/null || true
scp -q "$REMOTE:$RDIR/results/recal/para_baseline.jsonl" results/recal/ 2>/dev/null || true
scp -q "$REMOTE:$RDIR/results/recal/lens.pt" results/recal/ 2>/dev/null || true
for fam in dec mm; do
  scp -q "$REMOTE:$RDIR/results/recal/$fam/"*.jsonl "results/recal/$fam/" 2>/dev/null || true
done
scp -q "$REMOTE:$RDIR/data/paraphrases.json" results/recal/paraphrases.json 2>/dev/null || true

echo "fetched recal artifacts:"
find artifacts/recal results/recal -type f 2>/dev/null | sort | while read -r p; do
  printf '  %s (%s bytes)\n' "$p" "$(stat -f%z "$p" 2>/dev/null || stat -c%s "$p")"
done
