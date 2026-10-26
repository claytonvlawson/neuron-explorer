#!/usr/bin/env bash
# push_layer.sh -- publish one finished layer's artifacts to object storage.
#
#   RCLONE_REMOTE=r2:neuron-explorer-vault sweep/push_layer.sh q35-4b-L22 22
#   RCLONE_REMOTE=r2:... sweep/push_layer.sh q35-4b-L22 22 --full     # also the top-k logprob archive
#
# Runs from the repo root on the pod. Writes, per layer, under $RCLONE_REMOTE/slim/L<layer>/:
#   <run>.jsonl.zst      explorer-only rows (no per-token top-k harvest), zstd -19
#   <run>.meta.json      the run's meta, so the explorer discovers it
#   q35-4b_L<layer>.npz  per-neuron extremes + down columns (47 MB; live pushes and lens need it)
#   MANIFEST.json        rows, bytes, md5 of every file, plus the pod's own row count
# With --full it also writes full/L<layer>/<run>.jsonl.zst, the complete sweep including top_ids/top_lps.
#
# It is a no-op for a layer already published with a matching MANIFEST (idempotent on re-run), because
# a pod can die after the sweep but before the push and re-running the sweep costs GPU hours.
set -uo pipefail

RUN="${1:?usage: push_layer.sh <run-name> <layer> [--full]}"
LAYER="${2:?usage: push_layer.sh <run-name> <layer> [--full]}"
FULL=""; [[ "${3:-}" == "--full" ]] && FULL=1
REMOTE="${RCLONE_REMOTE:?set RCLONE_REMOTE, e.g. r2:neuron-explorer-vault}"
RCLONE="rclone"
# point rclone at a non-default config with RCLONE_CONFIG=<path> (rclone reads it itself)

JSONL="runs/${RUN}.jsonl"
META="runs/${RUN}.meta.json"
STATS="stats/q35-4b_L${LAYER}.npz"
SLIM="runs/${RUN}.slim.jsonl"
SLIM_META="runs/${RUN}.slim.meta.json"
STAGE=".push/L${LAYER}"

for f in "$JSONL" "$META"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f -- nothing to push" >&2; exit 1; }
done

if $RCLONE lsf "$REMOTE/slim/L${LAYER}/MANIFEST.json" >/dev/null 2>&1; then
  echo "L${LAYER}: already published ($REMOTE/slim/L${LAYER}/MANIFEST.json); re-run with"
  echo "        RCLONE_REMOTE=$REMOTE FORCE=1 $0 $RUN $LAYER to overwrite"
  [[ "${FORCE:-}" == "1" ]] || exit 0
fi

mkdir -p "$STAGE"
command -v zstd >/dev/null || { echo "ERROR: zstd not installed (apt-get install -y zstd)" >&2; exit 1; }
command -v "$RCLONE" >/dev/null || { echo "ERROR: rclone not installed on the pod" >&2; exit 1; }

echo "L${LAYER}: slim export"
python sweep/export_slim.py "$JSONL" --out "$SLIM" || { echo "ERROR: export_slim failed" >&2; exit 1; }
[[ -f "$SLIM_META" ]] || cp "$META" "$SLIM_META"
cp "$META" "$STAGE/${RUN}.meta.json"

echo "L${LAYER}: compressing"
zstd -19 -T0 -f -q "$SLIM" -o "$STAGE/${RUN}.jsonl.zst" || exit 1
[[ -f "$STATS" ]] && cp "$STATS" "$STAGE/" || echo "WARNING: no $STATS (live pushes will 501 for this layer)"
if [[ -n "$FULL" ]]; then
  mkdir -p "$STAGE/full"
  zstd -19 -T0 -f -q "$JSONL" -o "$STAGE/full/${RUN}.jsonl.zst" || exit 1
fi

# MANIFEST: rows counted on the pod, plus an md5 of every staged file. This is what pull_layer.sh checks.
ROWS=$(wc -l < "$SLIM" | tr -d ' ')
{
  echo "{"
  echo "  \"run\": \"$RUN\", \"layer\": $LAYER, \"rows\": $ROWS,"
  echo "  \"design_name\": $(python -c "import json;print(json.dumps(json.load(open('$META')).get('design_name')))"),"
  echo "  \"slim_mb\": $(du -m "$SLIM" | cut -f1),"
  echo "  \"full_included\": ${FULL:-0},"
  echo "  \"files\": {"
  first=1
  for f in "$STAGE"/*.zst "$STAGE"/*.npz "$STAGE"/*.meta.json "$STAGE"/full/*.zst; do
    [[ -f "$f" ]] || continue
    [[ $first -eq 1 ]] || echo ","
    first=0
    printf '    "%s": {"bytes": %s, "md5": "%s"}' "$(basename "$f")" "$(wc -c < "$f" | tr -d ' ')" "$(md5sum "$f" | cut -d' ' -f1)"
  done
  echo ""
  echo "  }"
  echo "}"
} > "$STAGE/MANIFEST.json"
python -c "import json;json.load(open('$STAGE/MANIFEST.json'))" || { echo "ERROR: bad MANIFEST" >&2; exit 1; }

echo "L${LAYER}: uploading $STAGE -> $REMOTE"
$RCLONE copy "$STAGE/" "$REMOTE/slim/L${LAYER}/" --checksum --transfers 4 --retries 5 || exit 1
if [[ -n "$FULL" ]]; then
  $RCLONE copy "$STAGE/full/" "$REMOTE/full/L${LAYER}/" --checksum --transfers 2 --retries 5 || exit 1
fi

echo "L${LAYER}: verifying remote"
$RCLONE check "$STAGE/" "$REMOTE/slim/L${LAYER}/" --one-way || exit 1
echo "L${LAYER}: published ($ROWS rows, $(du -sh "$STAGE" | cut -f1) staged)"
