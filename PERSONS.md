# Persons and live firing

A **person** is a named set of neuron pushes you talk to as one voice. The explorer streams its answer, the
unsteered baseline and the same set reversed, and lights the map with what fires token by token while it talks.

## The shape

    persons/doctor-strange.json

    {
      "slug": "doctor-strange",
      "name": "Doctor Strange",
      "notes": "why these neurons, at these doses",
      "pushes": [
        {"layer": 22, "n": 2742, "dose": 0.5, "mode": "add"},
        {"layer": 22, "n": 5344, "dose": 0.3, "mode": "add"},
        {"layer": 22, "n": 4957, "dose": 0.3, "mode": "add"}
      ],
      "created": "2026-10-26T09:51:31",
      "updated": "2026-10-26T09:51:31"
    }

`mode` is `add` (dose in units of the median residual norm leaving the layer, the same unit as the sweep's
`--dose-unit rel`) or `mul` (a multiplier on the neuron's own activation). `dose` may be negative on `add`: that
is the neuron pushed the other way. Up to 24 pushes per person.

## Talking to one

In the explorer, **Chat** tab, person bar at the top:

- `+ neuron` adds the open neuron at the dose and mode currently set in the composer line below.
- a chip opens its neuron (click the label) or removes it (click `×`).
- type a name, `save`; the `load…` select lists every person on disk.
- with any chips present, `run` applies the whole set at once. The three columns become the person, the
  baseline, and the anti-person.

Composing is kept in `localStorage`, so a reload does not lose an unsaved set.

Without the UI:

    curl -sN -X POST 127.0.0.1:8791/api/person/chat -H 'Content-Type: application/json' \
      -d '{"slug":"doctor-strange","prompt":"Who are you?","max_new":120,"fire_rows":[0]}'

Endpoints:

    GET    /api/persons                 list, with the directory and the 24-push cap
    POST   /api/person                  {name, notes?, slug?, pushes} upsert by slug, validated
    DELETE /api/person?slug=…           remove a file
    POST   /api/person/chat             SSE: push the set, stream text and firing

Validation is strict on purpose: layers and neuron indices must exist in the model, `mode` must be `add` or
`mul`, an `add` dose must be within −5…5 and a `mul` dose within 0…8. A bad person is a 400, not a silently
broken rollout.

## Live firing

Every generated token reports the strongest activations in every layer, scored the way the **fires** view
scores them: activation over the neuron's typical max on the positive side, over its typical min on the
negative side (`explorer/cache/act_scale.npz`, 10k tokens of FineWeb-Edu). The client fades the last few
tokens into each other so the map glows instead of strobing, and the line under the chips names the top three
for the current token.

    {"row": 0, "fire": {"r": 0, "pos": 12, "layer": 22, "u": [[9210, -5.4, -12.1], …]}}

`u` is 12 `[neuron, score, activation]` triples per layer, so 384 entries and about 6 KB of JSON per token.

Two decisions worth knowing:

- **Measured before the layer's own push.** The dose you add is an instrument setting, not something the model
  did. Firing at L22 is the natural activation at L22, so the marks show what the person *recruits*; the pushes
  appear only when they also fire on their own. (Measured after the push instead, the three pushed neurons
  dominate every token at +35, +29, +17 and the view says nothing.)
- **Row 0 only by default.** `fire_rows` takes any of 0 (person), 1 (baseline), 2 (anti-person). All three at
  once triples the payload for a picture nobody reads.

`/api/steer` (the single-neuron chat) streams the same firing events, so the map lights up there too. Pass
`"fire": 0` to turn it off.

## Measured on this Mac (MPS, Qwen3.5-4B)

    first token            0.7 to 2.1 s warm, 22 s cold
    decode                 about 9 tok/s across all three rows
    firing payload         about 6 KB per token, 384 entries
    top-k cost             a 12-of-9216 topk per layer per token, inside the existing forward pass

## What a person is not

Pushes compound. The same three L22 neurons at 0.8/0.6/0.6 collapse into "extranet life alien" loops; at
0.5/0.3/0.3 the persona holds across prompts and still answers a real question. Every dose you add is spent
against one coherent window of about 0.5 to 0.8 residual norms for the whole set, so a person with many
neurons needs small doses, and the anti-person row is the honest control: if row 2 reads like row 0, the set
is not doing the work you think it is.

## Verification

    uv run python scripts/verify_person.py

24 checks: the persons store (save, list, on-disk file, rejection of a bad layer and an absurd dose), three
rows streaming, the person differing from baseline, fire events for every layer on every token, scores inside
the calibrated range, the map showing recruited neurons rather than the pushes, firing on the plain
`/api/steer` path, and cleanup. In the browser (Playwright, system Chrome): the bar renders, Doctor Strange
loads as three chips, the answer streams in three columns with the person's neurons lighting the map (1,301
marks at peak, mixed signs, layers 7 to 27), and no console errors.

## Not done

- The Claude agent (`explorer/agent_mcp.py`) does not know about persons yet: it can steer a set, but cannot
  save or load one by name.
- No per-token firing on the advisor tab; that path still scores once at the end.
- No delete or rename in the UI: `save` overwrites by slug, removing a file is `curl -X DELETE`.
