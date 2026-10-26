"""Write an explorer-only view of a sweep run: same lines, without the per-step generation harvest.

A sweep row carries the text plus, for every generated token, its chosen logprob and the top-5
alternatives (`logprob`, `top_ids`, `top_lps`). `top_ids`/`top_lps` are the bulk of the file (a 200-token
row serializes to ~21 KB) and nothing reads them: `server.py:load_layer` keeps only
(kind, neuron, prompt, dose, rel_push, text, finish, n_gen), `rank.compute_units` additionally needs
`token_ids` (distinct-4-gram ratios) and `logprob` (mean chosen logprob per dose), and that is the whole list.
Keep `logprob`; drop the alternates.

    python sweep/export_slim.py runs/q35-4b-L22.jsonl --out runs/q35-4b-L22.slim.jsonl

Writes <out> and <out with .meta.json>, so the explorer discovers the run exactly like the original.
Rows stay in the same order and count, so line i in the slim file is line i in the full one: keep the
full jsonl if you want the alternates back (merge_runs.py and capture row alignment both assume full).
"""
import argparse
import json
import os

KEEP = ("layer", "kind", "neuron", "ref_neuron", "prompt", "dose", "rel_push", "text", "finish", "n_gen",
        "token_ids", "logprob")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", help="default: <input>.slim.jsonl")
    ap.add_argument("--keep", action="append", help="extra field to keep (repeatable)")
    a = ap.parse_args()

    keep = KEEP + tuple(a.keep or ())
    out = a.out or a.jsonl[:-6] + ".slim.jsonl"
    n = n_in = n_out = 0
    with open(a.jsonl) as f, open(out, "w") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            missing = [k for k in ("kind", "neuron", "prompt", "dose", "text", "token_ids") if k not in r]
            if missing:
                raise SystemExit(f"line {n + 1}: missing {missing} -- not a sweep row?")
            slim = {k: r[k] for k in keep if k in r}
            g.write(json.dumps(slim, ensure_ascii=False) + "\n")
            n += 1
            n_in += len(line.encode())
            n_out += len(json.dumps(slim, ensure_ascii=False)) + 1

    meta = a.jsonl[:-6] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as f:
            m = json.load(f)
        m["slim_of"] = os.path.basename(a.jsonl)
        m["dropped"] = ["logprob", "top_ids", "top_lps"]
        with open(out[:-6] + ".meta.json", "w") as f:
            json.dump(m, f, indent=1)
    else:
        print(f"WARNING: no {meta} -- the explorer will not discover {out}")

    print(f"{n} rows | {n_in / 1e6:.2f} MB -> {n_out / 1e6:.2f} MB "
          f"({n_in / max(n_out, 1):.1f}x smaller) -> {out}")


if __name__ == "__main__":
    main()
