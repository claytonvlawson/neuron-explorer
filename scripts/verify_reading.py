"""Check the live reading: the per-token neuron firings streamed while the local model answers.

    uv run python scripts/verify_reading.py                  # offline: merge rules + one synthetic token
    uv run python scripts/verify_reading.py --live           # asks the running explorer a question
    uv run python scripts/verify_reading.py --live --max-new 40 --question "What is language deprivation?"

Why this exists: a reading is only worth showing if it is of the token you think it is. The failure modes are
all silent, so they are asserted instead: a reading whose buffer leaks into the next token, a reading that
arrives after the answer instead of during it, a firing list scored against the wrong calibration, a run id
the reading endpoint cannot find. Offline needs torch and `explorer/cache/act_scale.npz`; the live check needs
the model loaded, so it is opt-in.

Exit code 0 = every assertion held. 1 = the first failed one (printed) or a skipped prerequisite.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "explorer"))
sys.path.insert(0, os.path.join(HERE, "sweep"))

BASE = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
ok = [0]


def check(cond, label, detail=""):
    print(("  ok   " if cond else "  FAIL ") + label + (f"  {detail}" if detail else ""))
    ok[0] += 1 if cond else 0
    if not cond:
        sys.exit(1)


def offline():
    from runtime import Runtime, merge_reads

    print("merge: strongest |score| across layers, from each layer's shortlist")
    per = {3: [(10, 2.0, 8.0), (11, -6.5, -30.0)], 22: [(7, 4.0, 9.0)], 9: [(1, 0.5, 1.0)]}
    out = merge_reads(per, k=2)
    check([r["n"] for r in out] == [11, 7], "|score| decides, both signs", str(out))
    check(out[0]["layer"] == 3 and out[0]["s"] == -6.5, "the negative side is kept signed")
    check(len(merge_reads(per, k=99)) == 4, "k larger than the pool returns the pool")
    check(merge_reads({}, k=12) == [], "no layers, no rows")

    try:
        import torch
    except Exception:
        check(False, "torch is missing: run the offline check inside the project venv")
    from runtime import CACHE
    scale = os.path.join(CACHE, "act_scale.npz")
    if not os.path.exists(scale):
        check(False, f"{scale} is missing: pull a sweep's calibration before checking the score scale")

    print("one token: record, flush, and prove the buffers do not leak")
    rt = Runtime(os.path.join(HERE, "stats"))
    import numpy as np
    n_layers = int(np.load(scale)["pos"].shape[0])  # the calibration is [layer, neuron]; no model needed to know L
    rt._reading = {"layers": set(range(n_layers)), "per_layer": 4, "k": 12, "lens": False, "out": {}, "resid": {}}
    h = torch.zeros(1, 1, 9216)  # one token at the residual width the lens indexes assume
    h[0, 0, 1234], h[0, 0, 77], h[0, 0, 500] = 9.5, -7.2, 0.3
    rt._record_read(22, h)
    check(22 in rt._reading["out"] and len(rt._reading["out"][22]) == 4, "each layer keeps its shortlist")
    ev = rt._flush_read(0, 5)
    check(ev["pos"] == 0 and bool(ev["read"]), "a reading has a position and rows", str(ev["read"][:1]))
    check(all({"layer", "n", "s", "a", "w"} <= set(r) for r in ev["read"]), "a row names the neuron, its score, its words")
    check(ev["read"][0]["layer"] == 22, "the strongest synthetic activation is the first row")
    check(isinstance(ev.get("tok"), str) and bool(ev["tok"]), "the reading names the token it produced", repr(ev.get("tok")))
    check(rt._reading["out"] == {} and rt._reading["resid"] == {}, "buffers are cleared between tokens")
    ev2 = rt._flush_read(1, 6)
    check(ev2["read"] == [], "a flush with no forward pass behind it invents nothing")
    rt._reading = None
    check(rt._flush_read(2) is None, "no reading when no reading was asked for")


def stream(path, body, limit=2000):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    events, t0 = [], time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            events.append(json.loads(line[6:]))
            if events[-1].get("stage") in ("done", "error") or len(events) >= limit:
                break
    return events, time.time() - t0


def live(question, max_new):
    print(f"live: {BASE}, {max_new} new tokens max")
    events, secs = stream("/api/advisor/ask", {"question": question, "max_new": max_new, "live": 1})
    stages = [e.get("stage") for e in events]
    check("error" not in stages, "the ask did not error", str([e for e in events if e.get("stage") == "error"][:1]))
    check(stages[0] == "retrieval", "retrieval comes first", str(stages[:3]))
    first = events[0]
    check(bool(first.get("run")) and first.get("live") is True, "the run id and live flag are on the first event")
    reads = [e for e in events if e.get("stage") == "read"]
    done = next(e for e in events if e.get("stage") == "done")
    check(bool(reads), "readings arrived", f"{len(reads)} of {done.get('gen_tokens')} generated tokens")
    check(stages.index("read") < stages.index("done"), "a reading arrives during the answer, not after it")
    check([r["pos"] for r in reads] == list(range(len(reads))), "positions are 0..n-1 in stream order")
    check(all(r.get("read") for r in reads), "every token has at least one firing")
    rows = [x for r in reads for x in r["read"]]
    check(all({"layer", "n", "s", "a", "w", "name"} <= set(x) for x in rows),
          "every firing names its neuron, score and words")
    check(all(isinstance(x["n"], int) and x["layer"] >= 0 and x["n"] >= 0 for x in rows),
          "every firing addresses a real layer and neuron")
    check(len(reads) == done.get("gen_tokens"), "one reading per generated token",
          f"{len(reads)} reads, {done.get('gen_tokens')} tokens")
    check(done.get("scored") == len(reads), "done reports the number of positions it actually read")
    check(done.get("live") is True, "done says this was a live ask")
    if reads and reads[-1].get("lens"):
        layers = [l for l, _ in reads[-1]["lens"]]
        check(layers == sorted(layers) and len(layers) > 1, "the depth strip is ordered and covers layers", str(len(layers)))

    got = json.load(urllib.request.urlopen(f"{BASE}/api/advisor/reading?run={done['run']}", timeout=30))
    check(got.get("ready") is True, "the reading endpoint finds the run")
    check(got.get("n") == len(reads), "the endpoint has every captured position", f"{got.get('n')} vs {len(reads)}")
    one = json.load(urllib.request.urlopen(f"{BASE}/api/advisor/reading?run={done['run']}&i=0", timeout=30))
    check(bool(one.get("reading", {}).get("read")), "asking for one position returns that reading")
    print(f"  ok   {len(reads)} readings in {secs:.1f}s, {secs / max(1, len(reads)):.2f}s/token end to end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="ask the running explorer instead of checking offline")
    ap.add_argument("--question", default="Why does language deprivation matter for deaf children?")
    ap.add_argument("--max-new", type=int, default=60)
    a = ap.parse_args()
    if a.live:
        live(a.question, a.max_new)
    else:
        offline()
    print(f"{ok[0]} assertions held")


if __name__ == "__main__":
    main()
