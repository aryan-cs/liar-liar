#!/usr/bin/env bash
# Fetch lightweight result artifacts from the GPU host into the Mac repo.
# Heavy caches (.hf_cache, model weights) stay remote.
# Reports every transfer; exits nonzero if any required artifact is missing.
set -uo pipefail
cd "$(dirname "$0")/.."

REMOTE="${LIAR_REMOTE:-uiuc-h200}"
RDIR="${LIAR_REMOTE_DIR:-/home/aryang9/sandbox/liar-liar}"

mkdir -p artifacts/recal artifacts/stage1 results/recal/dec results/recal/mm

missing_required=0

fetch() {  # fetch <required|optional> <remote-path> <local-dir>
  local kind="$1" rpath="$2" ldir="$3"
  if scp -q "$REMOTE:$RDIR/$rpath" "$ldir/" 2>/dev/null; then
    printf '  ok       %s\n' "$rpath"
  else
    printf '  MISSING  %s (%s)\n' "$rpath" "$kind"
    [ "$kind" = required ] && missing_required=1
  fi
}

echo "fetching from $REMOTE:$RDIR"
for f in config.json calibration.json tokensets.json capture_ids.json certificates.json vectors.pt; do
  fetch required "artifacts/recal/$f" artifacts/recal
done
# legacy stage1 artifacts: provenance of the naive operating point (Section 5.1)
for f in config.json vectors.pt; do
  fetch optional "artifacts/stage1/$f" artifacts/stage1
done
fetch required "results/recal/baseline.jsonl" results/recal
fetch optional "results/recal/para_baseline.jsonl" results/recal
fetch optional "results/recal/lens.pt" results/recal
fetch optional "results/recal/probe.json" results/recal
for fam in dec mm; do
  for f in $(ssh -o BatchMode=yes "$REMOTE" "ls $RDIR/results/recal/$fam/" 2>/dev/null); do
    fetch required "results/recal/$fam/$f" "results/recal/$fam"
  done
done

if [ "$missing_required" -ne 0 ]; then
  echo "fetch INCOMPLETE: required artifacts missing (see above)" >&2
  exit 1
fi
echo "fetch complete"
