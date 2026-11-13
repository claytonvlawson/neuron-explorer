#!/usr/bin/env python3
"""Pull the public advisor's replayable exchanges off the Jetson into this repo.

    uv run python scripts/pull_taps.py                # only what is new since the last pull
    uv run python scripts/pull_taps.py --all          # every tap the Jetson still keeps
    uv run python scripts/pull_taps.py --dry-run      # show what would land, write nothing

Why: the explorer can read a question typed into its own panel, but the advisor answers
strangers all day. The Jetson already keeps each of those exchanges (the two messages that
went to the model, the answer, the model's name) in `logs/taps.jsonl`, served by `GET /taps`
behind a bearer token that is not the widget's chat token. This copies them to
`explorer/cache/advisor_taps/<id>.json`, which is gitignored because the questions belong to
other people, and which is the input to the replay pass (step 5).

Config, in `neuron-explorer/.env.local` (0600, gitignored):

    ADVISOR_TAP_URL=https://jetson.tail155e26.ts.net/parent
    ADVISOR_TAP_TOKEN=<32-byte hex from /home/snowbeach/advisor/advisor.env>

Exit code 0 = pulled (or nothing new), 2 = not configured, 3 = the server refused or failed.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(HERE, ".env.local")
OUT_DIR = os.path.join(HERE, "explorer", "cache", "advisor_taps")
MAX_PAGES = 20


def load_env(path=ENV_FILE):
    """KEY=VALUE lines, no export, no substitution. Missing file is not an error."""
    env = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def newest_on_disk(out_dir):
    newest = 0.0
    try:
        names = os.listdir(out_dir)
    except FileNotFoundError:
        return newest
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
                newest = max(newest, float(json.load(fh).get("ts") or 0))
        except (OSError, ValueError, TypeError):
            continue
    return newest


def fetch(url, token, since, limit, timeout=30):
    full = "%s/taps?since=%r&limit=%d" % (url.rstrip("/"), since, limit)
    req = urllib.request.Request(full, headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        print("the Jetson said %d: %s" % (exc.code, body), file=sys.stderr)
        if exc.code == 401:
            print("check ADVISOR_TAP_TOKEN in %s against ADVISOR_TAP_TOKEN in "
                  "/home/snowbeach/advisor/advisor.env" % ENV_FILE, file=sys.stderr)
        sys.exit(3)
    except urllib.error.URLError as exc:
        print("could not reach %s: %s" % (full, exc.reason), file=sys.stderr)
        sys.exit(3)


def write_tap(out_dir, rec):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s.json" % rec["id"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser(description="Pull the public advisor's taps for local replay.")
    ap.add_argument("--all", action="store_true", help="start from the oldest tap the Jetson keeps")
    ap.add_argument("--since", type=float, default=None, help="unix timestamp to start after")
    ap.add_argument("--limit", type=int, default=50, help="records per request (server caps it)")
    ap.add_argument("--dry-run", action="store_true", help="print what is new, write nothing")
    ap.add_argument("--out", default=OUT_DIR, help="where taps land (default: explorer/cache/advisor_taps)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    env = load_env()
    url = args.url or os.environ.get("ADVISOR_TAP_URL") or env.get("ADVISOR_TAP_URL", "")
    token = args.token or os.environ.get("ADVISOR_TAP_TOKEN") or env.get("ADVISOR_TAP_TOKEN", "")
    if not url or not token:
        print("set ADVISOR_TAP_URL and ADVISOR_TAP_TOKEN in %s" % ENV_FILE, file=sys.stderr)
        return 2

    since = 0.0 if args.all else (args.since if args.since is not None else newest_on_disk(args.out))
    print("out    %s" % args.out)
    print("since  %s" % ("the beginning" if since <= 0 else
                         "%s (%.3f)" % (time.strftime("%Y-%m-%d %H:%M", time.localtime(since)), since)))

    landed = []
    for _ in range(MAX_PAGES):
        body = fetch(url, token, since, args.limit)
        taps = body.get("taps") or []
        for rec in taps:
            if not rec.get("id"):
                continue
            if not args.dry_run:
                write_tap(args.out, rec)
            landed.append(rec)
        if not taps:
            break
        advanced = float(body.get("since") or since)
        if advanced <= since:
            break
        since = advanced

    if not landed:
        print("no new taps")
        return 0

    for rec in landed:
        size = len(json.dumps(rec, ensure_ascii=False)) // 1024
        question = (rec.get("question") or "").replace("\n", " ")[:64]
        print("%s  %s  %-22s %2d KB  %s" % (
            rec.get("id", "?"),
            time.strftime("%m-%d %H:%M", time.localtime(float(rec.get("ts") or 0))),
            rec.get("model") or "?",
            size,
            question,
        ))
    print("\n%d tap%s %s" % (len(landed), "" if len(landed) == 1 else "s",
                             "would land" if args.dry_run else "written"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
