"""Check that a pulled sweep is complete enough for the explorer, per layer.

    python scripts/verify_runs.py ~/neuron-explorer                  # all layers
    python scripts/verify_runs.py ~/neuron-explorer --layers 22,8

Why this exists: the pipeline's one known data-loss mode is a partial pull (research_log.md, "Captures
lost and rebuilt"): the runs came back without every output directory the meta named, and nothing
noticed until the pods were gone. `meta.json` does not record the neuron range or the control count, so
completeness is checked structurally instead: every unit must have a row for every (prompt, dose) pair in
the design, every baseline prompt must exist, and every layer must have its stats npz.

Exit code 0 = every layer passes. 1 = at least one layer is incomplete (the rows it names are printed).
"""
import argparse
import collections
import glob
import json
import os
import sys

REQUIRED = ("kind", "neuron", "ref_neuron", "prompt", "dose", "rel_push", "text", "finish", "n_gen",
            "token_ids", "logprob")


def check_run(meta_path):
    meta = json.load(open(meta_path))
    layer = meta["layer"]
    jsonl = meta_path.replace(".meta.json", ".jsonl")
    errs, notes = [], []

    if not os.path.exists(jsonl):
        return layer, [f"jsonl missing: {os.path.basename(jsonl)}"], []
    if meta.get("design_name") == "calib":
        return layer, [], ["design_name=calib (the explorer skips these)"]

    design = {(p, round(float(d), 4)) for p, d in meta["design"]}
    prompts = set(meta["prompts"])
    seen = collections.defaultdict(set)
    units, base, n, missing_key, bad_layer, no_tokens = set(), set(), 0, None, 0, 0

    with open(jsonl) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if missing_key is None:
                miss = [k for k in REQUIRED if k not in r]
                if miss:
                    missing_key = f"line {i} lacks {miss}"
            if r.get("layer") != layer:
                bad_layer += 1
            if not r.get("token_ids"):
                no_tokens += 1
            if r["kind"] == "baseline":
                base.add(r["prompt"])
            else:
                units.add((r["kind"], r["neuron"]))
                seen[(r["kind"], r["neuron"])].add((r["prompt"], round(float(r["dose"]), 4)))

    if missing_key:
        errs.append(missing_key + " -- the explorer's ranking/rollout views will 500 on this layer")
    if bad_layer:
        errs.append(f"{bad_layer} rows have a layer other than {layer} (merged from the wrong run?)")
    if no_tokens:
        errs.append(f"{no_tokens} rows have no token_ids (distinct-4-gram scoring will fail)")
    if not units:
        errs.append("no steered units -- baselines only (run never started or was cut at the first flush)")
    missing_base = prompts - base
    if missing_base:
        errs.append(f"baselines missing for prompts: {sorted(missing_base)} (ranking needs them to strip common words)")

    short = {u: sorted(design - s) for u, s in seen.items() if design - s}
    if short:
        sample = list(short.items())[:3]
        errs.append(f"{len(short)}/{len(units)} units missing (prompt, dose) rows, e.g. "
                    + "; ".join(f"{k[0]}-{k[1]}: {len(v)} missing" for k, v in sample))

    n_rand = sum(1 for k, _ in units if k == "random")
    if n_rand == 0:
        errs.append("no random-direction controls -- null_pct will be None for every unit")
    else:
        notes.append(f"{n_rand} controls")

    notes.append(f"{len(units)} units, {n} rows, design {len(design)} per unit")
    return layer, errs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--layers", help="comma-separated; default every meta in runs/")
    a = ap.parse_args()

    metas = sorted(glob.glob(os.path.join(a.root, "runs", "*.meta.json")))
    if not metas:
        sys.exit(f"no runs/*.meta.json under {a.root} -- the explorer will show an empty map")
    want = {int(x) for x in a.layers.split(",")} if a.layers else None

    bad, seen_layers = 0, set()
    for mp in metas:
        try:
            layer, errs, notes = check_run(mp)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"L?  {os.path.basename(mp)}: unreadable meta ({e!r})")
            bad += 1
            continue
        if want is not None and layer not in want:
            continue
        seen_layers.add(layer)
        stats = os.path.join(a.root, "stats", f"q35-4b_L{layer}.npz")
        if not os.path.exists(stats):
            errs.append(f"stats missing: {os.path.basename(stats)} (live pushes/lens need the down columns)")
        ok = "OK  " if not errs else "FAIL"
        print(f"{ok} L{layer:<3} {os.path.basename(mp)}  " + (", ".join(notes) if notes else ""))
        for e in errs:
            print(f"       - {e}")
            bad += 1

    if want is not None:
        absent = sorted(want - seen_layers)
        if absent:
            print(f"FAIL layers with no run at all: {absent}")
            bad += 1
    print(f"\n{len(seen_layers)} layer(s) checked, {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
