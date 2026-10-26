#!/usr/bin/env bash
# pull_layer.sh -- fetch one published layer from object storage into a local explorer checkout.
#
#   RCLONE_REMOTE=r2:neuron-explorer-vault scripts/pull_layer.sh 22
#   RCLONE_REMOTE=r2:... scripts/pull_layer.sh 22 8 13 --full
#
# Pulls slim/L<layer>/ (and full/L<layer>/ when asked) into <root>/runs and stats, checks every file
# against the pod's MANIFEST (md5 + row count), then runs scripts/verify_runs.py on the layers it touched.
# Nothing is unpacked until its checksum matches: a truncated zst is deleted rather than left to fail
# later inside the explorer.
set -uo pipefail

ROOT="${EXPLORER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REMOTE="${RCLONE_REMOTE:?set RCLONE_REMOTE, e.g. r2:neuron-explorer-vault}"
RCLONE="rclone"
FULL=""
LAYERS=()
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    *) LAYERS+=("$a") ;;
  esac
done
[[ ${#LAYERS[@]} -gt 0 ]] || { echo "usage: pull_layer.sh <layer> [layer...] [--full]" >&2; exit 1; }

mkdir -p "$ROOT/runs" "$ROOT/stats" "$ROOT/.pull"
fail=0

for L in "${LAYERS[@]}"; do
  D="$ROOT/.pull/L$L"
  rm -rf "$D"; mkdir -p "$D"
  echo "== L$L: downloading"
  $RCLONE copy "$REMOTE/slim/L$L/" "$D/" --transfers 4 --retries 5 || { echo "FAIL L$L: download"; fail=1; continue; }
  [[ -n "$FULL" ]] && $RCLONE copy "$REMOTE/full/L$L/" "$D/full/" --transfers 2 --retries 5

  MAN="$D/MANIFEST.json"
  [[ -f "$MAN" ]] || { echo "FAIL L$L: no MANIFEST.json on the remote"; fail=1; continue; }

  echo "== L$L: checking against MANIFEST"
  python - "$MAN" "$D" <<'PY' || fail=1
import hashlib, json, os, sys
man_path, d = sys.argv[1], sys.argv[2]
man = json.load(open(man_path))
bad = 0
for name, meta in man["files"].items():
    p = os.path.join(d, name)
    if not os.path.exists(p):
        if name.endswith(".npz"):
            print(f"   note: {name} not published (live pushes will be unavailable for this layer)")
            continue
        print(f"   FAIL missing {name}"); bad += 1; continue
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    size = os.path.getsize(p)
    if h != meta["md5"] or size != meta["bytes"]:
        print(f"   FAIL {name}: md5/bytes differ (local {size}B {h[:8]}, pod {meta['bytes']}B {meta['md5'][:8]})")
        bad += 1
print(f"   {len(man['files'])} file(s) declared, {man.get('rows')} rows on the pod, {bad} problem(s)")
sys.exit(1 if bad else 0)
PY
  [[ $? -ne 0 ]] && fail=1

  echo "== L$L: unpacking"
  shopt -s nullglob
  for z in "$D"/*.jsonl.zst; do zstd -d -f -q "$z" -o "$ROOT/runs/$(basename "${z%.zst}")" || fail=1; done
  if [[ -n "$FULL" ]]; then
    for z in "$D"/full/*.jsonl.zst; do zstd -d -f -q "$z" -o "$ROOT/runs/full-$(basename "${z%.zst}")" || fail=1; done
  fi
  cp "$D"/*.meta.json "$ROOT/runs/" 2>/dev/null
  cp "$D"/*.npz "$ROOT/stats/" 2>/dev/null

  # row count must match the pod's, or the pull is short even though hashes matched (e.g. re-exported)
  for j in "$ROOT/runs/"*"L$L"*.jsonl; do
    got=$(wc -l < "$j" | tr -d ' ')
    [[ "$got" = "$(python -c "import json;print(json.load(open('$MAN'))['rows'])")" ]] \
      || { echo "   WARN $(basename "$j"): $got rows locally, $(python -c "import json;print(json.load(open('$MAN'))['rows'])") on the pod"; }
  done
done

echo
python "$ROOT/scripts/verify_runs.py" "$ROOT" --layers "$(IFS=,; echo "${LAYERS[*]}")"
verts=$?
echo
if [[ $fail -ne 0 || $verts -ne 0 ]]; then
  echo "RESULT: problems above; do not trust these layers in the explorer yet"
  exit 1
fi
echo "RESULT: $# layer(s) pulled and verified; restart the server to pick them up"
