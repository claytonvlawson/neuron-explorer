"""Turn a steer lens into a list of steerable nodes, with a null to threshold against.

`sweep/steer_lens.py` measures every neuron's next-token effect at one dose, but it has no control, so "this neuron
matters" has no meaning: at dose 0.8 the median KL at L0 is 10.3 nats for every neuron in the layer. This adds the
control it lacks by pushing 256 RANDOM directions of the same norm at the same point and keeping the KL distribution
as the null, then marking a neuron steerable when its KL clears that null's `--q` percentile.

The neuron push at down_proj's input with value x adds x * W_down[:, n] to the layer's output, and x is
dose * (median residual norm / ||W_down[:, n]||), so the added vector has norm dose * median residual norm. The
control pushes a unit random vector scaled to that same norm at the layer OUTPUT, which is the same residual
location downstream of the push. Equal norm, meaningless direction.

    python sweep/steer_nodes.py --lens stats/steer_lens --layers 0-31 --controls 256 --out stats/steer_nodes

Writes `<out>.csv.gz` (one row per layer-neuron, with the top words it raises and lowers) and `<out>_report.json`
(per layer: null median and p95, count and share steerable, and the KL medians).

A node here is steerable in the narrow, measured sense: pushing it at this dose changes the next-token distribution
through the rest of the network more than a random push of the same size. It says nothing about any downstream
behaviour, and nothing about a sign, a mood or a persona. Behaviour needs rollouts.
"""
import argparse
import csv
import gzip
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

CHAT = ["Who are you?", "Write whatever you like."]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--lens", default="stats/steer_lens")
    ap.add_argument("--layers", default="0-31")
    ap.add_argument("--dose", type=float, default=0.8, help="must match the dose the lens was run at")
    ap.add_argument("--controls", type=int, default=256)
    ap.add_argument("--q", type=float, default=95.0, help="null percentile a node must clear")
    ap.add_argument("--top", type=int, default=6, help="words kept per node in the csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="stats/steer_nodes")
    a = ap.parse_args()
    lo, _, hi = a.layers.partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))

    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = model.model.language_model
    ctxs = [tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, enable_thinking=False,
                                    tokenize=True) for c in CHAT] + [tok(p).input_ids for p in PLAIN]
    ctxs = [list(c["input_ids"] if hasattr(c, "keys") else c) for c in ctxs]
    C, T = len(ctxs), max(map(len, ctxs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([[pad] * (T - len(c)) + c for c in ctxs], device="cuda")
    mask = torch.tensor([[0] * (T - len(c)) + [1] * len(c) for c in ctxs], device="cuda")
    posn = (mask.cumsum(-1) - 1).clamp(min=0)

    # control push: (rows, hidden) added to the layer output, one row per context per random direction
    ctrl = {"vec": None, "layer": None}
    resid = {}

    def ctrl_hook(i):
        def hook(mod, inp, out):
            if ctrl["vec"] is None or ctrl["layer"] != i:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h = h + ctrl["vec"].to(h.dtype)[:, None, :]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    for i, layer in enumerate(lm.layers):
        layer.register_forward_hook(ctrl_hook(i))
        layer.register_forward_hook(lambda m, inp, out, i=i: resid.__setitem__(
            i, (out[0] if isinstance(out, tuple) else out)))

    def last_logprobs(input_ids, attn, pos):
        h = lm(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(h).float(), -1)

    with torch.inference_mode():
        base = last_logprobs(ids, mask, posn)
        norms = {i: r[mask.bool()].float().norm(dim=-1).median().item() for i, r in resid.items()}
        kl_ctrl = {}
        rng = np.random.default_rng(a.seed)
        for l in layers:
            dirs = rng.standard_normal((a.controls, model.config.text_config.hidden_size)).astype(np.float32)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            vec = torch.tensor(dirs, device="cuda") * (a.dose * norms[l])  # [K, hidden]
            rep = vec.repeat_interleave(C, dim=0)  # [K*C, hidden]
            ctrl.update(layer=l, vec=rep)
            lp = last_logprobs(ids.repeat(a.controls, 1), mask.repeat(a.controls, 1),
                               posn.repeat(a.controls, 1)).view(a.controls, C, -1)
            ctrl["vec"] = None
            kl = (lp.exp() * (lp - base[None])).sum(-1).mean(1)
            kl_ctrl[l] = kl.cpu().numpy()
            print(f"L{l}: control KL p50 {np.median(kl_ctrl[l]):.3f} p{a.q:g} "
                  f"{np.percentile(kl_ctrl[l], a.q):.3f}", flush=True)

    report = {"model": a.model, "dose": a.dose, "controls": a.controls, "q": a.q, "layers": {}}
    n_total = n_steer = 0
    # csv.writer, not hand-joined lines: a token can decode to a quote or a carriage return, which splits a row and
    # silently drops it (9,741 of 294,912 rows vanished that way).
    with gzip.open(f"{a.out}.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "neuron", "kl_up", "kl_down", "steerable", "null_p95", "ratio", "top_up", "top_down"])
        for l in layers:
            f = f"{a.lens}_L{l}.npz"
            if not os.path.exists(f):
                print(f"missing {f}, skipped", flush=True)
                continue
            z = np.load(f)
            up, down = z["up_kl"].astype(np.float32), z["down_kl"].astype(np.float32)
            up_tok, down_tok = z["up_tok"], z["down_tok"]
            thr = float(np.percentile(kl_ctrl[l], a.q))
            best = np.maximum(up, down)
            steer = best > thr
            def words(row):
                """Token id to a printable label. Some vocabulary entries decode to control characters or NUL,
                which breaks a CSV row and reads as a parser fault downstream."""
                out = []
                for t in row[: a.top]:
                    w = "".join(ch if ch.isprintable() else "_" for ch in tok.decode([int(t)])).strip()
                    out.append(w.replace(",", ";").replace("\n", " ") or "_")
                return " ".join(out)
            for n in range(len(up)):
                w.writerow([l, n, f"{up[n]:.4f}", f"{down[n]:.4f}", int(steer[n]), f"{thr:.4f}",
                            f"{best[n] / max(thr, 1e-9):.2f}", words(up_tok[n]), words(down_tok[n])])
            n_l, n_s = int(len(up)), int(steer.sum())
            n_total += n_l
            n_steer += n_s
            report["layers"][str(l)] = {"neurons": n_l, "steerable": n_s, "share": round(n_s / n_l, 4),
                                        "null_p50": round(float(np.median(kl_ctrl[l])), 4),
                                        "null_p95": round(thr, 4),
                                        "kl_up_p50": round(float(np.median(up)), 4),
                                        "kl_down_p50": round(float(np.median(down)), 4),
                                        "kl_up_max": round(float(up.max()), 4),
                                        "kl_down_max": round(float(down.max()), 4)}
            print(f"L{l}: {n_s}/{n_l} steerable ({n_s / n_l:.1%}), null p95 {thr:.3f}, "
                  f"median KL up {np.median(up):.2f} down {np.median(down):.2f}", flush=True)

    report["total"] = {"neurons": n_total, "steerable": n_steer, "share": round(n_steer / max(n_total, 1), 4)}
    json.dump(report, open(f"{a.out}_report.json", "w"), indent=1)
    print(json.dumps(report["total"]), flush=True)


if __name__ == "__main__":
    main()
