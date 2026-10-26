"""End-to-end check for persons and per-token firing against a running explorer.

    uv run python scripts/verify_person.py            # reads http://127.0.0.1:8791
    EXPLORER_URL=http://127.0.0.1:8791 uv run python scripts/verify_person.py

Writes a throwaway person, talks to it, streams its firing, then deletes it. Exits 1 on the first failed check.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
SLUG = "verify-person"
ok = [0]


def check(cond, label, detail=""):
    print(("  ok   " if cond else "  FAIL ") + label + (f"  {detail}" if detail else ""))
    ok[0] += 1 if cond else 0
    if not cond:
        sys.exit(1)


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method or ("POST" if data else "GET"),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def stream(path, body, limit=400):
    """Collect SSE events until the stream closes or `limit` events have arrived."""
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    events, t0 = [], time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            events.append(json.loads(line[6:]))
            if len(events) >= limit or events[-1].get("row") is None:
                break
    return events, time.time() - t0


print(f"explorer at {BASE}")
# the throwaway person: two real neurons, one add and one mul, so both modes are exercised
PUSHES = [{"layer": 22, "n": 2742, "dose": 1.0, "mode": "add"},
          {"layer": 31, "n": 7843, "dose": 1.5, "mode": "mul"}]

print("\n1. server and model")
st = call("/api/status")
check(st.get("status") == "ready", "model is ready", json.dumps(st))
check(call("/api/index")["arch"]["n_layers"] == 32, "architecture loads")

print("\n2. persons store")
before = call("/api/persons")["persons"]
saved = call("/api/person", {"name": "Verify Person", "slug": SLUG, "notes": "throwaway", "pushes": PUSHES})
check(saved["slug"] == SLUG and len(saved["pushes"]) == 2, "saved a person", saved["slug"])
check(os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "persons", SLUG + ".json")),
      "person file on disk")
check(any(p["slug"] == SLUG for p in call("/api/persons")["persons"]), "person appears in the list")
try:
    call("/api/person", {"name": "bad", "pushes": [{"layer": 99, "n": 1, "dose": 1, "mode": "add"}]})
    check(False, "a layer outside the model is rejected")
except urllib.error.HTTPError as e:
    check(e.code == 400, "a layer outside the model is rejected", f"HTTP {e.code}")
try:
    call("/api/person", {"name": "bad", "pushes": [{"layer": 1, "n": 1, "dose": 900, "mode": "add"}]})
    check(False, "an absurd dose is rejected")
except urllib.error.HTTPError as e:
    check(e.code == 400, "an absurd dose is rejected", f"HTTP {e.code}")

print("\n3. talking to a person (streaming text + firing)")
events, dt = stream("/api/person/chat", {"slug": SLUG, "prompt": "Who are you?", "max_new": 24, "fire_rows": [0]})
toks = [e for e in events if e.get("t")]
fires = [e for e in events if e.get("fire")]
rows = sorted({e["row"] for e in toks})
check(rows == [0, 1, 2], "three rows stream", f"rows {rows}")
check(len(toks) > 20, "tokens arrive", f"{len(toks)} events in {dt:.1f}s")
check(not [e for e in events if e.get("error")], "no error events")
text = {r: "".join(e["t"] for e in toks if e["row"] == r) for r in (0, 1, 2)}
check(all(text[r].strip() for r in (0, 1, 2)), "every row has text", f"{[text[r][:24] for r in (0, 1, 2)]}")
check(text[0] != text[1], "the person differs from the unsteered baseline")

print("\n4. firing")
check(len(fires) > 0, "fire events arrive", f"{len(fires)} fire events")
layers = {f["fire"]["layer"] for f in fires}
check(len(layers) == 32, "every layer reports firing", f"{len(layers)} layers")
pos = sorted({f["fire"]["pos"] for f in fires})
check(pos == list(range(min(len(pos), 24))), "positions are per generated token", f"pos 0..{pos[-1]}")
check(all(len(f["fire"]["u"]) == 12 for f in fires), "12 neurons per layer per token")
check(all(f["row"] == 0 for f in fires), "only the person row fires")
check(all(isinstance(u[0], int) and 0 <= u[0] < 9216 for f in fires for u in f["fire"]["u"]), "neuron indices in range")
# firing is measured before the push, so the dose never shows up as activation: a typical max is 1.0 and the
# strongest neurons in the sweep's calibration sit well under 10
scores = [abs(u[1]) for f in fires for u in f["fire"]["u"]]
check(max(scores) < 25, "scores stay in the calibrated range", f"max |score| {max(scores):.1f}")
pushed = {(p["layer"], p["n"]) for p in PUSHES}
others = [u for f in fires for u in f["fire"]["u"] if (f["fire"]["layer"], u[0]) not in pushed]
check(len(others) > 0.5 * len(scores), "the map shows recruited neurons, not just the pushes",
      f"{len(others)}/{len(scores)} entries outside the pushed set")
print(f"  info  L22·2742 fires naturally on "
      f"{sum(1 for f in fires if f['fire']['layer'] == 22 and any(u[0] == 2742 for u in f['fire']['u']))}/{len(pos)} tokens")

print("\n5. single-neuron steering also fires now")
events, dt = stream("/api/steer", {"layer": 22, "n": 2742, "dose": 0.8, "mode": "add", "prompt": "Who are you?", "max_new": 12})
check([e for e in events if e.get("t")], "tokens stream", f"{dt:.1f}s")
check([e for e in events if e.get("fire")], "fire events stream")

print("\n6. cleanup")
call("/api/person?slug=" + SLUG, method="DELETE")
check(not any(p["slug"] == SLUG for p in call("/api/persons")["persons"]), "throwaway person deleted")
check(len(call("/api/persons")["persons"]) == len(before), "list is back to where it started")

print(f"\nall checks passed ({ok[0]})")
