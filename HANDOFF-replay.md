# Handoff: the replay (step 5)

A reading of a public answer. The advisor at learnwithclay.ai answers on `openai/gpt-4.1-nano` through
OpenRouter, whose neurons cannot be seen. This panel takes the exact prompt and answer the visitor got, feeds
both through Qwen3.5-4B on this machine, and shows what fired for each word the visitor read. It is a reading
of text, not a recording of the model that wrote it, and the panel says so on screen for the whole answer.

Steps 1 to 4 (the live reading, the depth strip, the tap) are in `HANDOFF-live-reading.md`. This file covers
what the weights unblocked: the real model, the measured costs, and the replay engine and panel.

Nothing is committed. `git status` in `~/neuron-explorer`: seven modified files, five new (this file,
`HANDOFF-live-reading.md`, `scripts/verify_reading.py`, `scripts/pull_taps.py`, `scripts/verify_taps.py`).

## The weights were never missing

The previous handoff said "9.7 MB of about 8 GB". That was `du` not following symlinks. Hugging Face stores the
shards once in a shared blob store:

```
snapshots/851bf6e.../model.safetensors-00001-of-00002.safetensors
  -> models--Qwen--Qwen3.5-4B/blobs/26a93f...
  -> hub/blobs/ee/ee0b8779...       5,329,398,688 bytes
snapshots/851bf6e.../model.safetensors-00002-of-00002.safetensors
  -> models--Qwen--Qwen3.5-4B/blobs/cb544b...
  -> hub/blobs/45/459e5ad6...       3,990,429,408 bytes
```

Both are complete, the safetensors headers parse (738 tensors, BF16), and the tree is 9.34 GB. `du -sh` on the
model directory reports 9.7 MB because the directory itself holds symlinks. Check with `ls -lL` or
`os.path.getsize`, never `du` on a snapshot directory. There is nothing to download. Free disk is about 10 GiB
with the weights already counted, so a second copy would leave roughly 1 GiB; do not start one.

Load: `Loading weights: 723/723` then `/api/status` reports `ready` about 10 s after process start on a 16 GB
M4. The 16 live assertions in `scripts/verify_reading.py --live` pass on the real model.

## What is implemented

| file | change |
| --- | --- |
| `explorer/runtime.py` | `replay_messages(messages, answer, cancel, read)`: prefill the recorded prompt, then feed the recorded answer one token at a time. The reading for token i is captured after prompt plus tokens 0..i-1, which is the state `generate_messages` reads before emitting token i. Nothing is sampled and EOS does not stop it. |
| `explorer/server.py` | `GET /api/advisor/taps` lists `explorer/cache/advisor_taps/*.json` newest first. `POST /api/advisor/replay {id}` is an SSE stream with the same `read` event shape as a live ask. |
| `explorer/static/app.js` | `advReset`, `advStream` (one SSE loop for ask and replay), `advisorAsk`, `advisorReplay`, `advTapsLoad`. The replay is treated as a sparse reading: the map, strip, scrubber, hover and click work unchanged, and `/api/infer/view` is never called. |
| `explorer/static/index.html` | the tap menu and replay button under the ask bar, and `#adv-note`, the label above the sources. Assets bumped to `style.css?v=69`, `app.js?v=109`. |
| `explorer/static/style.css` | `#adv-tap-bar`, `#adv-tap`, `#adv-read`, `#adv-note`. |

### The replay event contract

`retrieval`: `{stage, run, replay: true, live: false, question, model, served_by, sources, titles: [], crisis, ms: 0}`
`prompt`: `{stage, prompt_tokens, replay: true}`
`read`: the live shape (`pos`, `tok`, `read`, `lens`) plus `replay: true` and `forced: true`
`token`: `{stage, t}` per answer token
`done`: `{stage, run, replay: true, live: false, answer, sources, citations: [], tokens, pos, answer_from: 0,
scored, prompt_tokens, gen_tokens, model, served_by}`
`error`: `{stage, error}`

`forced: true` is the honest bit: this token was not chosen by the model being read. It is in the JSON but the
panel does not need to render it.

## Measured on the real model

Fixed question "What is language deprivation?", `max_new 60`, prompt 2,345 tokens, warm runs after one
throwaway ask:

| lens | warm runs (ms/token) | median |
| --- | --- | --- |
| `EXPLORER_READ_LENS=0` | 383, 392, 398, 404 | about 392 |
| `EXPLORER_READ_LENS=1` | 436, 440, 452 | about 440 |

The lens costs about 48 ms per token, roughly 12 percent. A cold first ask is much slower (724 ms per token in
the first lens-on run); the first ask after a restart is always the slow one (27.4 s to first read against 11.3 s
warm). Decode is about 2.3 tokens per second, so the 8 Hz paint throttle coalesces 3 or 4 readings and is fine
as it stands.

Replay of tap `f9905ef9ca81` ("how early should a deaf baby start learning sign language", written by
`openai/gpt-4.1-nano`, 601 characters):

- 3,199 prompt tokens, 105 forced tokens, 105 readings at positions 0 to 104;
- every read has `forced: true`, a `tok`, and a lens row;
- the replayed answer is byte identical to the tap's answer;
- 31.7 s to the first reading (prefill), 77.4 s total, about 435 ms per answer token;
- `done` reports `replay: true, live: false, scored: 105`.

At 3,200 prompt tokens and 105 answer tokens, a tap replay is a minute and a half cold and about a minute warm.
That is the cost of reading a public answer honestly with this setup, and the panel should keep showing the
stage line so the wait is legible.

## Bugs found on the way

- `yield text[sent:]` where `sent` is a string raised `TypeError: slice indices must be integers` after the
  first reading. It is `text[len(sent):]`, the same expression `generate_messages` uses.
- An edit replaced the `def token_units(self, ids):` line and left its body inside `replay_messages`. Restored;
  both functions are separate again (`runtime.py` 468 and 518).
- The first lens on/off comparison said the lens was cheaper than no lens. It was cold-start noise, not a
  result. Warm runs reverse it. Any future timing here needs a throwaway ask first.

## What remains, in order

1. **Browser check of the replay panel.** The server path is verified end to end with `curl`; the UI has only
   been syntax checked (`node --check`). Acceptance: the menu lists both taps; clicking replay shows the note
   naming `openai/gpt-4.1-nano` and Qwen3.5-4B; the stage line says `replay`, never `live`; readings arrive and
   the strip fills all 32 layers; the scrubber, hover and click work; the answer text matches the tap; zero
   requests to `/api/infer/view`.
2. **`scripts/verify_replay.py`.** Same shape as the other verify scripts. Assert: the list endpoint returns
   the local taps; a replay's first event is `retrieval` with `replay: true` and the tap's model; one `read` per
   answer token; positions are `0..n-1`; every read is `forced` and has a `tok`; `done.answer` equals the tap's
   `answer`; `done.live` is false; a bad id is a 400 and an unknown id a 404; a crisis tap has no `messages` and
   returns 409 instead of replaying.
3. Rows for `verify_replay.py` in `README.md`, and mark step 5 done in `HANDOFF-live-reading.md`.
4. Refresh the tap menu after a pull. Today `advTapsLoad` runs once at page load; a reload is enough, but the
   panel should offer it.
5. Step 6, the counterfactual re-ask: same prompt, top neuron clamped or pushed, diff the next tokens.
6. Optional speed work. Transformers warns that `causal_conv1d` and `flash-linear-attention` are missing and
   the reference PyTorch kernels are "much slower". Both are CUDA oriented, so do not expect an MPS win from
   installing them; measure with one ask before changing the environment.

## How to run it

```sh
cd ~/neuron-explorer
.venv/bin/python explorer/server.py          # about 10 s to ready, then /api/status
.venv/bin/python scripts/pull_taps.py        # newest public answers from the Jetson
curl -s localhost:8791/api/advisor/taps | .venv/bin/python -m json.tool

# the replay stream, one event per line
curl -N -X POST localhost:8791/api/advisor/replay \
  -H 'content-type: application/json' -d '{"id":"f9905ef9ca81"}'

.venv/bin/python scripts/verify_reading.py --live
.venv/bin/python scripts/verify_taps.py
```

The server that was running when this was written is pid 79128, up with `/api/status` ready. It already has the
Python changes; only static files changed after it started, and those are read off disk, so a reload picks them
up.

## Gotchas

- One generation lock. A replay blocks asks and a live ask blocks a replay, same as any generation.
- `_reads` keeps the last 8 runs; a replay can evict a live run from `/api/advisor/reading`. That is existing
  behaviour, not new.
- The tap id is checked against `[0-9a-f]{6,32}` and the file path is built from it, so no traversal.
- `explorer/advisor.py` imports the cloud module with a temporary `ADVISOR_LOG_DIR` and a stubbed index load.
  The `Tapping replayable exchanges to /var/folders/...` line at explorer startup is expected; it is not the
  Jetson file.
- Tap files are other people's questions, with the retrieved chunk text they were answered from. The cache is
  gitignored. The advisor's privacy copy still needs the richer-record disclosure before this ships publicly.
- The word "live" must not appear anywhere in a replay's labels. The note and the stage line are the honest
  surface; keep them.

## Files and anchors

- `explorer/runtime.py`: `replay_messages` 468, `token_units` 518.
- `explorer/server.py`: `ADV_TAPS` 1385, `advisor_taps` 1389, `advisor_replay` 1414.
- `explorer/static/app.js`: `advReset` 1283, `advStream` 1297, `advisorAsk` 1397, `advisorReplay` 1411,
  `advTapsLoad` 1424, the button 1439, the init call 1487.
- `explorer/static/index.html`: the tap bar 83, the note 88, asset versions 7 and 164.
- `explorer/static/style.css`: `#adv-tap-bar` 334, `#adv-note` 341.
- `scripts/pull_taps.py`, `scripts/verify_taps.py`: the tap transport and its checks.
- Scratch, not in the repo: `replay_once.py <id>` and `time_ask.py "<question>" 60` under
  `/private/var/folders/5p/cj0g1b_j2tn2c8mbpl2l07_w0000gs/T/opencode/`.
