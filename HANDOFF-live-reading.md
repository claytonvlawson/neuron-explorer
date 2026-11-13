# Handoff: the live reading

The advisor panel used to stream an answer and then spend a second pass scoring every position, so the map lit
up only after the writing stopped. This change reads the model out of the decode pass that is already running:
one SSE event per token, before that token's text, so the map shows the state that produced the word.

Honest framing, unchanged from the design: the public bot at learnwithclay.ai answers on an OpenRouter model,
whose neurons are unobservable. What this panel shows is the same pipeline answered by Qwen3.5-4B on this
machine, so the firings belong to the words on screen. Anything read off a production answer is a replay of
text, not a reading of the model that wrote it, and has to be labelled that way. See step 5.

Nothing is committed. `git status`: seven modified files (`.gitignore` is one), five new
(`scripts/verify_reading.py`, `scripts/pull_taps.py`, `scripts/verify_taps.py`, `HANDOFF-replay.md`,
this file). The tap also touched `~/vault-advisor-cloud` and the Jetson; see step 4.

## What changed

| file | change |
| --- | --- |
| `explorer/runtime.py` | `merge_reads` (pure), the reading state, `_record_read`, `_read_words`, `_logit_lens`, `_flush_read`, and `generate_messages(..., read=callback)` |
| `explorer/server.py` | `live` flag on `/api/advisor/ask` (on by default), one `read` event per token, `GET /api/advisor/reading` |
| `explorer/static/app.js` | the panel consumes `read`, paints the map and the firing list at 8 Hz, uses recorded readings for hover, click, the "all" button and tab switches in live mode, and drives the depth strip and scrubber |
| `explorer/static/index.html` | the depth strip, the position readout and the follow/hold scrubber under the answer |
| `explorer/static/style.css` | the strip (three columns, all 32 layers), the slider, and an 11px stage line that no longer wraps the buttons |
| `scripts/verify_reading.py` | offline assertions plus the live stream check |
| `scripts/pull_taps.py` | pulls the public advisor's replayable exchanges from the Jetson into `explorer/cache/advisor_taps/` |
| `scripts/verify_taps.py` | the tap writer and `/taps` reader, against the real server module with the model stubbed |
| `.gitignore` | `.env.local` (the tap token) and the tap cache |
| `README.md` | the script row and the check list |

## The event contract

One per emitted token, before that token's text delta:

```json
{"stage":"read","run":"a1b2c3d4e5f6","pos":0,"tok":" The",
 "read":[{"layer":22,"n":8412,"s":3.41,"a":11.2,"w":["money","cost","price","pay"],"name":"money lens"}],
 "lens":[[0,"The"],[1,"The"],[2,"A"],[31,"The"]]}
```

- `pos` is the index into the generated tokens, 0 based, the same index `done.tokens` uses.
- `read` is the global top `EXPLORER_READ_K` (12) by `|s|`, from each layer's top `EXPLORER_READ_PER_LAYER`
  (4). `s` is the activation divided by that neuron's calibrated extreme from `act_scale.npz`, the same
  scale `/api/fires` uses, so 1.0 means "as strong as this neuron usually gets". `a` is raw activation.
- `w` is the steer lens (four words, up or down by the sign of `s`); `name` is `names.json` when present.
  Names are snapshotted once per ask so a rename in another tab cannot rewrite a reading already shown.
- `lens` is the logit lens, top-1 next token at every captured layer. One batched matmul over the captured
  residuals. `EXPLORER_READ_LENS=0` skips it, which is the first thing to try on a slow host.
- The reading arrives **before** the token text, so the map updates as the word appears, not after it. The
  first reading is the state after the last prompt token: the model having read the retrieved passages,
  before its first word.
- `done` in live mode carries `live: true`, `run`, `scored` (positions read), `tokens`, `pos`,
  `answer_from: 0`, `prompt_tokens`, `gen_tokens`, `answer`, `sources`, `citations`, `crisis`.

`GET /api/advisor/reading?run=<id>` returns `{ready, run, n, tokens, question, readings}` with every position
captured so far; add `&i=<pos>` for one. It is readable while the stream is open, which is what a scrubber
needs. The last 8 runs are kept in memory (`READS_KEEP`), about 0.5 MB each.

## How it works

`_hook` already saw every layer's MLP input during the prefill and every decode step. `_record_read` takes the
last position, scores it against `act_scale.npz`, and keeps that layer's top 4. `_resid_hook` keeps the
residual leaving each layer at the same position. When the next token is chosen,
`generate_messages` calls `_flush_read(pos, tok)`, which merges the 32 shortlists into one global top 12,
attaches lens words, runs one `lm_head` matmul over the 32 residuals for the depth strip, and clears the
buffers. The callback then emits the event.

Costs per token, none of which is a second forward pass:

- 32 `topk` over 9,216 activations, about 1.2M comparisons.
- one `[32, 2560] @ [2560, 248320]` matmul, about 4.1e10 FLOP, only when `READ_LENS` is on.
- 48 lens-word lookups and one vocab lookup.

The `read` callback runs inside the generation lock on the generation thread, so it must stay cheap and must
not block. `emit` is `call_soon_threadsafe`, so it does not. If the callback ever raises, the reading is
disabled for that answer with one line on stderr and the answer continues; observation never costs the answer.

## The depth strip and the scrubber

Under the answer, one line per layer: the logit-lens token at the position being read, in three columns so all 32
layers fit in the 380px panel (143px tall). Layers that already agree with the deepest one are bright and the rest
are dim, so the answer visibly settles with depth. With `EXPLORER_READ_LENS=0` the strip says there is no lens
rather than inventing words.

The slider and the arrow keys move `pos`. Every move paints from `advReads`, the readings this tab already
received: no model call, no `/api/infer/view`, and no fetch unless the position predates the tab (then
`/api/advisor/reading?run=<id>&i=<pos>`, cached after). **follow** paints each new reading as it arrives; **hold**
freezes the position and says "held at N" while the stage line keeps counting tokens. Clicking a token pins it,
clicking it again resumes follow, and hovering previews a token's reading and then returns to the pinned or followed
one. A deliberate move cancels a hover preview that has not fired yet, so a click cannot be repainted by the pointer.

## How to see it

    cd ~/neuron-explorer
    uv run python -m explorer.server          # model loads on the first ask
    open http://127.0.0.1:8791/               # advisor tab

Ask a question. The stage line counts tokens and firings while the answer writes, the map and the firing list
update about 8 times a second, and hovering or clicking a token chip shows that token's recorded reading
instead of the dense cube. Under the answer the depth strip shows the logit lens at the current position; drag the
slider or press Left/Right to scrub, and switch follow/hold to freeze a position while the answer keeps writing.
The old behaviour is still there:

    {"question": "...", "max_new": 260, "live": 0}

That streams the text, then scores the window into the dense cube the run tab reads. Same stages as before
this change, verified.

## How to check it

    uv run python scripts/verify_reading.py             # 12 assertions, no model
    uv run python scripts/verify_reading.py --live      # 16 more, against a running server
    uv run python scripts/verify_reading.py --live --question "..." --max-new 40

The offline check covers the merge rules and one synthetic token through the real recording path, against the
real `act_scale.npz`, `steer_index.npz` and `vocab.json` (248,077 entries). The live check covers ordering
(a reading before `done`), one reading per generated token, positions `0..n-1`, names and words present, the
depth strip ordered, `done.scored` matching, and the endpoint finding the run.

**Now verified on the real weights (2026-11-12).** The earlier "9.7 MB of about 8 GB" was `du` not following
symlinks: both shards are complete in the shared blob store (`~/.cache/huggingface/hub/blobs/ee/` and
`.../45/`, 9.34 GB total), the model loads in about 10 s, and all 16 live assertions pass on it (60 tokens,
28.8 s warm). The stub work below still stands for the `live: 0` path and for the transport. The latency
measurements and the replay are in `HANDOFF-replay.md`.

The strip and scrubber were checked in Chromium against the same stub, with the stub slowed to 150 ms per token so
the stream could be held mid-answer: 32 layer rows in order, no horizontal overflow (360px of content in 360px), a
position readout that matches the slider, ArrowLeft moving exactly one token, hold frozen at 1 while the count
reached 7 and the slider's max grew, follow jumping to the newest position, hover previewing and leaving restoring
the followed one, "whole answer" clearing the strip, the collapse returning 143px to the firing list, and **zero
requests to `/api/infer/view`** across all of it. One real bug came out of that: a queued 90 ms hover preview could
repaint the position right after a click, so a deliberate move now cancels the pending preview.

## One deviation from the plan

The plan said live mode would still populate `_infer` so the existing map works. It does not. `_infer` holds a
dense `[layer, position, neuron]` cube, about 1.2 MB per position and 300 MB for a 260-token answer, and
building it is exactly the second pass this change removes. Instead the panel paints from the recorded sparse
readings, which is also better to click: a token shows what was captured when it was written, instantly.

What that costs: in live mode, `/api/infer/view` has nothing new to serve, and the run tab's dense viewer is
not the source of truth for the current answer. The panel therefore does not call it: hover and click use
`advNearest` and the reading endpoint, the "all" button merges the captured readings, and switching back to the
advisor tab repaints the last reading instead of fetching a cube that belongs to an older ask. If a token is
missing from this tab's cache (page opened mid-answer), the click falls back to
`/api/advisor/reading`. `live: 0` is unchanged and still fills the cube.

## What remains

### 3. The panel's depth strip and scrubber (done)

The strip, the slider, the arrow keys and the follow/hold toggle are described under "The depth strip and the
scrubber". Acceptance, as checked in the browser:

- hovering any token shows its reading;
- arrowing moves one token at a time;
- nothing calls `/api/infer/view` while a live answer is current;
- the strip is legible at 32 rows on a laptop (three columns, 143px).

One item is still open on the real weights: whether 8 Hz is the right paint rate when the decode pass is slower than
the stub. If it is not, the throttle is one constant (`advMapAt > 120`).

### 4. The tap (public answers) (done, transport)

The point: after a public answer, the explorer can replay that exact question and context locally. Without it,
the reading only covers questions asked in the explorer's own panel.

It lives on the Jetson, not a droplet. `parent_advisor_server.py` runs as `parent-advisor.service` under the
`snowbeach` user, listens on `127.0.0.1:8001`, and Tailscale Funnel publishes it at
`https://jetson.tail155e26.ts.net/parent`, which is the URL the Vercel proxy already calls. So the tap is a
local append plus a read-only GET on that server: no extra store, no third party, no new host.

What it writes, in `answer_question(..., tap=True)` after the model returns and after the answer is cleaned:

```json
{"id": "f9905ef9ca81", "ts": 1794503035.877, "model": "openai/gpt-4.1-nano", "served_by": "free",
 "finish_reason": "stop", "question": "...", "messages": [...], "answer": "...", "sources": [...],
 "crisis": false, "location": null}
```

- `messages` is the two-message prompt that actually went to the model: system prompt plus the retrieved chunk
  text and the question. `qa_log.jsonl` already keeps the question, the answer and the source paths, but not
  the chunk text, so it cannot be replayed. This can.
- `answer` is the cleaned text, the one the visitor read, so a replay reads the same tokens it paints.
- `model` is the OpenRouter model that wrote it (`openai/gpt-4.1-nano` on the free tier today); `served_by` is
  `free` or `paid`. The replay label has to name these, not just say "an OpenRouter model".
- A crisis turn stores only `{id, ts, model, served_by, crisis}`. The wording is not kept.
- A failed generation writes nothing: there is no answer to read.
- `/health/deep` asks a fixed probe question with `tap` off, so uptime checks never appear in the list.

`GET /taps?since=<ts>&limit=<n>` returns `{taps, since, count}` with the records newer than `since`, oldest
first, and `since` advanced to the newest one returned. Its bearer is `ADVISOR_TAP_TOKEN`, separate from the
widget's `ADVISOR_TOKEN`; unset, it falls back to the chat token. The reader is a different client on a
different machine, so it gets its own revocable credential.

The file is `logs/taps.jsonl`, 14 KB per record today, 12.9 KB of which is the prompt. Above 50 MB,
`_trim_taps` keeps the newest 500. The visitor response is unchanged.

On the Mac, `scripts/pull_taps.py` copies new records to `explorer/cache/advisor_taps/<id>.json`, which is
gitignored because the questions belong to other people:

```sh
uv run python scripts/pull_taps.py             # new since the newest file on disk
uv run python scripts/pull_taps.py --all       # everything the Jetson still keeps
uv run python scripts/pull_taps.py --dry-run
```

It reads `ADVISOR_TAP_URL` and `ADVISOR_TAP_TOKEN` from `neuron-explorer/.env.local` (0600, gitignored).

Verified end to end on 2026-11-12: a question posted to `https://learnwithclay.ai/api/parent/chat` was answered
by `openai/gpt-4.1-nano`, tapped at 14.6 KB with both messages (3.7 KB system, 9.5 KB context plus question),
and pulled to `explorer/cache/advisor_taps/f9905ef9ca81.json`. The reader token gets `200` in 47 ms; the chat
token gets `401`. Twenty-eight offline checks (`scripts/verify_taps.py`, against the real module
and the real vault, with `llm_router` stubbed) cover the auth split, `since`, `limit`, clamping,
crisis redaction, a torn line, trimming and the 404.

Remaining for step 5, and one consequence to know before building it: a real tap's prompt is about 3,200
tokens and `ADV_WINDOW` is 512, so the `live: 0` window cannot cover it. A replay either reads the answer
positions only (teacher-force the prompt, capture per decode step, which is what `live: 1` already does) or
accepts a partial window. Answer-only is the honest reading anyway: the words on screen are the model's, and
the context is something it read.

### 5. Replay (a reading of the text, labelled) (in progress, see `HANDOFF-replay.md`)

`explorer/cache/advisor_taps/<id>.json` gives `messages` and `answer`. Replay is the existing `live: 0` pass
re-shaped into readings: teacher-force `messages` plus the answer tokens, score the window, convert each
position's scores to the same `read` shape, and stream them. Cap at `ADV_WINDOW` (512). Label the view
"written by <served_by>; read by Qwen3.5-4B", and never let the label be optional.

Acceptance: a tap round-trips from the droplet to a replay in the panel; the label names the model that wrote
the answer; the replay says "replay" everywhere a live answer says "live".

### 6. Counterfactual re-ask

From any reading, re-ask the same prompt with the top neuron clamped or pushed by a stated dose, then diff the
next few tokens. `steer_multi` already does the push, and the reading hooks measure before the push, so the
dose stays an instrument setting and not something the model did. Acceptance: two answers, one diff, the dose
and the neuron recorded with the comparison.

## Costs to measure

The reading adds no forward pass, but it is not free. Measure with the real weights:

    EXPLORER_READ_LENS=0  vs  EXPLORER_READ_LENS=1, same question, same max_new

Time to `done` is in the panel's stage line. If the lens dominates on the host (the Jetson, or CPU), keep it
off by default there and let the strip load from `/api/advisor/reading` on demand. `EXPLORER_READ_LAYERS` takes
a comma list (for example `0,8,16,24,31`) and cuts the matmul by that factor.

Other numbers, measured on the stub, so transport only: seven readings in 34 ms warm, about 2 KB per event. A
cold first ask in a fresh process paid 1.2 to 6.8 s of warm-up in my runs, before any reading, and that is the
advisor import and BM25 load, not this change.

## Failure modes and loose ends

- **Privacy.** `_reads` holds the last 8 questions and answers in memory on the local machine, and
  `/api/advisor/reading` serves them to any local page. The server's host and origin middleware keeps other
  sites out, but do not expose this endpoint beyond the machine as it stands. The tap is a second copy of other
  people's questions: it sits in `~/neuron-explorer/explorer/cache/advisor_taps/` (gitignored), and the richer
  record on the Jetson now holds the retrieved chunk text too, not just the source paths. Say so in the
  advisor's privacy copy before the replay ships.
- **Stale vocab.** While testing with a stub I hit an `IndexError` at `server.py:590`: `vocab[k]` assumes
  `cache/vocab.json` covers every id in the lens index. A rebuilt or truncated vocab makes `/api/infer/view`
  return 500. One-line hardening, if you want it: skip ids outside `range(len(vocab))` in that comprehension.
- **One reader at a time.** The reading runs inside the generation lock, like every other generation. A second
  ask queues rather than interleaving, which is the existing behaviour.
- **The name is `live`, not `--replay`.** The design said a `--replay` flag; the panel talks HTTP, so the flag
  became the `live` body field, default on, `live: false` for the old path. Keep the word "replay" for step 5,
  where it means a reading of text.
- **End state.** `llm_router.LLM_BASE_URL` already points the production answer at any OpenAI-compatible
  server. Point it at a local vLLM or mlx server and the public answer becomes observable directly, and the
  shadow and the tap both stop being necessary. That is the better end state if the droplet ever gets a GPU.

## Files and anchors

- `explorer/runtime.py`: `merge_reads` 39, `_record_read` 179, `_read_words` 197, `_logit_lens` 209,
  `_flush_read` 224, `_resid_hook` 265, `generate_messages` 417.
- `explorer/server.py`: `_reads` 1226, `_live_flag` 1230, `advisor_ask` 1250, `on_read` 1281,
  `advisor_reading` 1366.
- `explorer/static/app.js`: `syncLeftMap` 223, `advDepthStrip` 1187, `advPaint` 1202, `advPaintAll` 1216,
  `advNearest` 1222, `advSyncFollow` 1230, `advSeek` 1235, `advHold` 1250, the follow toggle 1257, the arrow
  keys 1267, the depth toggle 1275, the `read` branch 1323, the live `done` branch 1343, the "all" button 1382,
  hover 1398, the click pin 1412.
- `explorer/static/index.html`: the depth box 87, the strip 93, the follow/hold scrubber 94.
- `explorer/static/style.css`: the strip 342 to 353, the scrubber 354 to 356.
- `scripts/verify_reading.py`: offline `offline()`, live `live()`.
- `scripts/pull_taps.py`: `load_env`, `newest_on_disk`, `fetch`, `write_tap`, `main`.
- `scripts/verify_taps.py`: one `check` per property, in four groups (auth, writer, reader, trim).
- `~/vault-advisor-cloud/parent_advisor_server.py`: `TAPS` and `TAP_TOKEN` 90 and 93, `_append_tap` 231,
  `_trim_taps` 248, `answer_question(..., tap=)` 855, `_handle_taps` 929, `_authorized_tap` 1006.
