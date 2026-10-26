"""End-to-end check for the money lens and the asleep screen. Asserts, does not print-and-hope.

    uv run python scripts/verify_money.py                     # lens against the running server + the newest screen
    uv run python scripts/verify_money.py --screen-only       # just re-derive the screen's criteria from its JSON
    EXPLORER_URL=http://127.0.0.1:8791 uv run python scripts/verify_money.py

The lens half streams a real read of pasted text (no network), so it works offline. The screen half re-derives every
criterion from the numbers the report already printed: if a row says 3/5, the three must be the ones its own
numbers earn. Exits 1 on the first failed check.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n_ok = [0]

DOC = ("The company recorded a net loss of $41.2 million for the year. "
       "Customer demand weakened in Europe and the order backlog fell. "
       "We depend on three suppliers for our only manufacturing line. "
       "A fire at that facility would halt shipments indefinitely.")


def check(cond, label, detail=""):
    print(("  ok   " if cond else "  FAIL ") + label + (f"  {detail}" if detail else ""))
    n_ok[0] += 1 if cond else 0
    if not cond:
        sys.exit(1)


def info(label, detail=""):
    print("  info " + label + (f"  {detail}" if detail else ""))


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def stream(path, body, limit=6000):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if len(events) > limit:
                    break
    return events


def lens(quick=True):
    st, s = call("/api/money/status")
    check(st == 200 and s.get("person") == "Money", "the Money person is saved", f"{s.get('person')}")
    check(len(s.get("pushes") or []) == 4, "four pushes in the set", str(len(s.get("pushes") or [])))
    check(s.get("money_tokens", 0) > 100, "money lexicon resolves to real tokens", str(s.get("money_tokens")))
    check(s.get("neutral_tokens", 0) > 20, "neutral control lexicon resolves too", str(s.get("neutral_tokens")))

    st, e = call("/api/money/read", {"slug": "no-such-person", "text": DOC})
    check(st == 400, "an unknown person slug is rejected", f"HTTP {st}")
    st, e = call("/api/money/read", {"slug": "money"})
    check(st == 400, "a request with no document is rejected", f"HTTP {st}")

    t0 = time.time()
    ev = stream("/api/money/read", {"text": DOC, "top": 2, "max_new": 32})
    secs = time.time() - t0
    stages = [e.get("stage") for e in ev]
    check("doc" in stages, "the document is read")
    check("scored" in stages, "every sentence is scored")
    check("done" in stages, "the stream reaches done, not a truncated tail")
    check(not any(x == "error" for x in stages), "no error events")

    doc = next(e for e in ev if e["stage"] == "doc")
    check(doc["sentences"] >= 3, "sentences found", str(doc["sentences"]))
    scored = next(e for e in ev if e["stage"] == "scored")["ranked"]
    check(len(scored) == doc["sentences"], "one score per sentence", f"{len(scored)}/{doc['sentences']}")
    check(all(len(r["money"]) == 3 and len(r["neutral"]) == 3 for r in scored),
          "each sentence measured on all three rows")
    check(all(r["amp"] == round(r["d_money"] - r["d_neutral"], 3) for r in scored),
          "amp is exactly d_money minus d_neutral, as documented")
    check([r["amp"] for r in scored] == sorted((r["amp"] for r in scored), reverse=True),
          "the ranking is sorted by amplification, not by order of appearance")
    check(any(abs(r["d_money"]) > 0.01 for r in scored), "the person moves the money mass somewhere",
          f"max |d_money| {max(abs(r['d_money']) for r in scored):.2f}")
    check(any(r["d_money"] != r["d_neutral"] for r in scored), "it is not just a global sharpening")

    rew = [e["rec"] for e in ev if e["stage"] == "rewrite"]
    check(1 <= len(rew) <= 2, "only the requested number of sentences is rewritten", str(len(rew)))
    check(all(len(r["rewrites"]) == 3 for r in rew), "three rewrites per sentence (person / baseline / anti)")
    check(all(r["rewrites"][0].strip() for r in rew), "the person row produced text")
    check(all(r["rewrites"][0] != r["rewrites"][1] for r in rew), "the person rewrite differs from the baseline")
    info("density", "; ".join(f"{r['density']}/100 vs {r['baseline_density']} baseline" for r in rew))

    done = next(e for e in ev if e["stage"] == "done")
    path = done.get("report") or ""
    check(bool(path) and os.path.exists(path), "the report was written to disk", path)
    md = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    check("## Ranking" in md, "the report carries its ranking section")
    check("investment advice" in md.lower() and "not a forecast" in md.lower(),
          "the report says what it is not")
    check("| # | amp | d(money)" in md, "every ranked sentence is a cited row in the report table")
    info("timing", f"{secs:.1f}s for {doc['sentences']} sentences scored, {len(rew)} rewritten")


def screen():
    files = sorted(glob.glob(os.path.join(ROOT, "reports", "screen-*.json")))
    check(bool(files), "a screen report exists", files[-1] if files else "none")
    d = json.load(open(files[-1]))
    hits = d["hits"]
    check(len(hits) > 0, "the screen found names", str(len(hits)))
    check(all(h["score"] >= 3 for h in hits), "every listed name meets the threshold")
    check(all(h["coverage"] is None or h["coverage"] <= d["args"]["max_coverage"] for h in hits),
          "every listed name is inside the coverage ceiling")

    for h in hits:
        c = h["criteria"]
        # re-derive each criterion from the numbers printed next to it, not from the stored booleans
        want = {
            "revenue_growing": h["growth"] is not None and h["growth"] >= 5.0,
            "price_asleep": h["span"] <= 0.25 and h["last"] <= 0.7 * h["hi52"],
            "cheap_vs_self": bool(h["ps_now"] and h["ps_median"] and h["ps_now"] <= 0.7 * h["ps_median"]),
            "volume_waking": bool(h["vol_ratio"] and h["vol_ratio"] >= 1.3 and h["chg60"] is not None
                                and abs(h["chg60"]) <= 10),
        }
        bad = [k for k, v in want.items() if c.get(k) is not v]
        check(not bad, f"{h['symbol']}: its criteria are what its own numbers say", ",".join(bad))
        if h["insider_net"] is not None:
            check(c["insider_buying"] == (h["insider_net"] > 0), f"{h['symbol']}: insider flag matches its net")
        if h["ps_median"] is not None:
            check(h["ps_years"] >= 2, f"{h['symbol']}: P/S history rests on at least 2 years", str(h["ps_years"]))
    info("screened", f"{d['meta']['universe']} universe → {d['meta']['measured']} measured → {len(hits)} hits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-only", action="store_true")
    a = ap.parse_args()
    if not a.screen_only:
        print("money lens (live server)")
        lens()
    print("asleep screen (report on disk)")
    screen()
    print(f"all checks passed ({n_ok[0]})")
