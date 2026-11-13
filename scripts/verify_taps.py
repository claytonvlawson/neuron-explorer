#!/usr/bin/env python3
"""Check the tap in `parent_advisor_server.py`: the writer, the reader, and the auth split.

    uv run python scripts/verify_taps.py

Runs the real server module against the Mac's advisor mirror with `llm_router` stubbed, so no
network and no OpenRouter spend. What it asserts, in order: `/taps` refuses a missing token, a
wrong one, and the chat token; an empty tap file is an empty list (not a 500); a chat writes one
record with both messages, the chunk text, the model name, the question and the visitor's cleaned
answer; `since` advances and filters; a bad `since` or `limit` is a 400; `limit` takes the newest;
the trim keeps the last N; a crisis turn keeps metadata only; a torn last line is skipped; the
unknown path is a 404.

Needs `~/vault-advisor-cloud` (the module) and `~/neuron-explorer/advisor-prod` (index + vault).
Exit code 0 = every check held, 1 = the first failure, 2 = a missing prerequisite.
"""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUD = os.environ.get("VAULT_ADVISOR_CLOUD", os.path.expanduser("~/vault-advisor-cloud"))
INDEX = os.path.join(HERE, "advisor-prod", "bm25.pkl")
VAULT = os.path.join(HERE, "advisor-prod", "vault")
PORT = int(os.environ.get("VERIFY_TAP_PORT", "8021"))

for path, what in ((CLOUD, "the server module"), (INDEX, "the BM25 index"), (VAULT, "the vault mirror")):
    if not os.path.exists(path):
        print("missing %s at %s" % (what, path), file=sys.stderr)
        sys.exit(2)

TMP = tempfile.mkdtemp(prefix="tap-check-")
TAPS = os.path.join(TMP, "taps.jsonl")
os.environ.update({
    "ADVISOR_HOST": "127.0.0.1",
    "ADVISOR_PORT": str(PORT),
    "ADVISOR_TOKEN": "chat-token",
    "ADVISOR_TAP_TOKEN": "tap-token",
    "ADVISOR_LOG_DIR": TMP,
    "ADVISOR_TAPS": TAPS,
    "ADVISOR_INDEX": INDEX,
    "ADVISOR_VAULT": VAULT,
    "ADVISOR_TAP_MAX_LIMIT": "5",
    "ADVISOR_TAP_TRIM_BYTES": "200",
    "ADVISOR_TAP_KEEP": "2",
})

sys.path.insert(0, CLOUD)
from http.server import ThreadingHTTPServer  # noqa: E402

import parent_advisor_server as srv  # noqa: E402
import llm_router  # noqa: E402

calls = {"n": 0}


def fake_generate(messages):
    calls["n"] += 1
    n = calls["n"]
    return {
        "text": "Answer %d with an em dash \u2014 and a Source: leak.md line." % n,
        "served_by": "free",
        "model": "test/model-%d" % n,
        "finish_reason": "stop",
        "error": None,
    }


llm_router.generate_with_fallback = fake_generate

server = ThreadingHTTPServer(("127.0.0.1", PORT), srv.Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:%d" % PORT
ok = [0]


def check(cond, label, detail=""):
    print(("  ok   " if cond else "  FAIL ") + label + ("" if not detail else "  %s" % (detail,)))
    if not cond:
        server.shutdown()
        sys.exit(1)
    ok[0] += 1


def req(path, method="GET", body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    if token:
        r.add_header("Authorization", "Bearer " + token)
    if data:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def tap(path, **kw):
    return req(path, token="tap-token", **kw)


print("auth")
check(req("/taps", token=None)[0] == 401, "no token is a 401")
check(req("/taps", token="wrong")[0] == 401, "a wrong token is a 401")
check(req("/taps", token="chat-token")[0] == 401, "the chat token cannot read taps")
code, body = tap("/taps?since=0")
check(code == 200 and body["count"] == 0, "an empty tap file is an empty list", body)

print("the writer")
code, chat = req("/chat", "POST", {"question": "what is language deprivation"}, token="chat-token")
check(code == 200 and chat["answer"], "chat answers through the stub", chat.get("id"))
check("\u2014" not in chat["answer"], "the stored answer has no em dash")
check("Source: leak.md" not in chat["answer"], "the stored answer has no source leak")

with open(TAPS) as fh:
    recs = [json.loads(line) for line in fh if line.strip()]
check(len(recs) == 1, "one tap was written", len(recs))
rec = recs[0]
check(len(rec["messages"]) == 2 and rec["messages"][0]["role"] == "system", "the tap keeps both messages")
check("Context:" in rec["messages"][1]["content"], "the tap keeps the retrieved chunk text")
check(rec["question"] == "what is language deprivation", "the tap keeps the question")
check(rec["model"] == "test/model-1" and rec["served_by"] == "free", "the tap names the model")
check(rec["answer"] == chat["answer"], "the tap answer is the visitor's answer")
check(bool(rec["sources"]) and isinstance(rec["sources"], list), "the tap carries sources")
check(rec["crisis"] is False and rec["location"] is None, "a normal turn is not marked crisis")

print("the reader")
code, body = tap("/taps?since=0")
check(code == 200 and body["count"] == 1, "since=0 returns the record")
check(abs(body["since"] - rec["ts"]) < 1e-6, "since advances to the newest ts")
code, body = tap("/taps?since=%r" % rec["ts"])
check(code == 200 and body["count"] == 0, "since filters the record it names")
check(tap("/taps?since=yesterday")[0] == 400, "a bad since is a 400")
check(tap("/taps?limit=all")[0] == 400, "a bad limit is a 400")

print("limit and trim")
time.sleep(0.01)
code, chat2 = req("/chat", "POST", {"question": "how many deaf children are born each year"}, token="chat-token")
check(code == 200 and chat2["answer"], "a second chat answers")
code, body = tap("/taps?since=0&limit=1")
check(code == 200 and body["count"] == 1 and body["taps"][0]["model"] == "test/model-2",
      "limit returns the newest", body.get("count"))
check(sum(1 for _ in open(TAPS)) == 2, "the trim kept the last two")

print("crisis and damage")
code, crisis = req("/chat", "POST", {"question": "I want to kill myself tonight"}, token="chat-token")
check(code == 200 and crisis["answer"], "a crisis chat still answers")
with open(TAPS) as fh:
    last = [json.loads(line) for line in fh if line.strip()][-1]
check(set(last) == {"id", "ts", "model", "served_by", "crisis"} and last["crisis"] is True,
      "a crisis tap keeps metadata only", sorted(last))

with open(TAPS, "a") as fh:
    fh.write('{"id": "torn", "ts": 1')  # what a crash mid-write leaves
code, body = tap("/taps?since=0&limit=5")
check(code == 200 and body["count"] >= 1, "a torn last line is skipped, not a 500")
check(req("/nope")[0] == 404, "an unknown path is a 404")
check(req("/health")[0] == 200, "health still answers")

server.shutdown()
print("\n%d checks passed" % ok[0])
