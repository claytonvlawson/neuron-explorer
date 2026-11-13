"""Rank every MLP neuron of every layer against the twelve emotion directions.

A push on neuron n of layer L adds `add * W_down_L[:, n]` to the residual stream. The mood probe scores a residual
by projecting it on unit emotion directions and dividing by the spread of neutral text at that layer, so the score
a push can buy is proportional to cos(W_down_L[:, n], dir_e) at that layer. This fits the directions at every layer
(same corpus and same common-mode correction as explorer/mood.py) and writes one cos matrix per layer, which ranks
all 9,216 neurons for each of the 12 emotions without a single rollout.

    python sweep/mood_rank.py --model Qwen/Qwen3.5-4B --out mood_rank
    python sweep/mood_rank.py --layers 20,22,24 --show 10

Writes mood_rank/L<layer>.npz with
    cos            [12, neurons] float16, cosine of the neuron's down column with each emotion direction
    z_per_dose     [12, neurons] float16, z gained by a +1.0 relative dose (dose x median residual norm)
    dirs, mu, sd, mean_neu, spec_mean, spec_sd   the probe at this layer (same layout as explorer/cache/mood_probe.npz)
    resid_norm_p50 median residual norm leaving the layer
    accuracy       held-out sentence accuracy of the twelve-way fit at this layer
and prints the top neurons per emotion.

Ranking is an upper bound on what the push does at the moment of the push: it ignores what later layers do to the
perturbation. Confirm a candidate with a real steered rollout before believing it, which is what sweep/steer_sweep.py
is for.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

SKIP = 2  # leading tokens of a sentence sit on a position sink, as in explorer/mood.py
N_TEST = 4


def load_corpus():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "explorer", "mood.py")
    spec = importlib.util.spec_from_file_location("_mood_corpus", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module level only defines the corpus, no model needed
    return mod.CORPUS, mod.EMOTIONS, mod.VA


def split(corpus):
    train, test = {}, {}
    for k, v in corpus.items():
        n = len(v)
        te = list(range(n - N_TEST, n)) if k != "neutral" else list(range(n - 6, n))
        train[k] = [s for i, s in enumerate(v) if i not in te]
        test[k] = [v[i] for i in te]
    return train, test


def fit_layer(res, names):
    """Directions, neutral scale and held-out accuracy for one layer, from that layer's residuals."""
    tr, te = res["train"], res["test"]
    neu_tok = np.concatenate(tr["neutral"])
    mu = neu_tok.mean(0)
    neu_mean = np.mean([t.mean(0) for t in tr["neutral"]], 0)
    dirs = []
    for e in names:
        d = np.mean([t.mean(0) for t in tr[e]], 0) - neu_mean
        dirs.append(d / (np.linalg.norm(d) + 1e-9))
    D = np.stack(dirs).astype(np.float32)
    neu_p = (neu_tok - mu) @ D.T
    sd = neu_p.std(0) + 1e-6
    mean_neu = neu_p.mean(0)
    neu_z = (neu_p - mean_neu) / sd
    neu_s = neu_z - neu_z.mean(1, keepdims=True)
    spec_mean, spec_sd = neu_s.mean(0), neu_s.std(0) + 1e-6

    def z(tokens):
        zz = (((tokens - mu) @ D.T) - mean_neu) / sd
        return ((zz - zz.mean(-1, keepdims=True)) - spec_mean) / spec_sd

    ok = tot = 0
    conf = np.zeros((len(names), len(names) + 1), int)
    for i, e in enumerate(names):
        for t in te[e]:
            zz = z(t).mean(0)
            pred = int(zz.argmax()) if zz.max() > 1.0 else len(names)
            conf[i, pred] += 1
            ok += pred == i
            tot += 1
    alltok = np.concatenate([t for v in tr.values() for t in v])
    return dict(D=D, mu=mu.astype(np.float32), sd=sd.astype(np.float32), mean_neu=mean_neu.astype(np.float32),
                spec_mean=spec_mean.astype(np.float32), spec_sd=spec_sd.astype(np.float32),
                accuracy=ok / max(tot, 1), confusion=conf,
                resid_norm_p50=float(np.median(np.linalg.norm(alltok, axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layers", default="all", help='"all" or e.g. 20,22,24')
    ap.add_argument("--out", default="mood_rank")
    ap.add_argument("--show", type=int, default=12, help="top neurons printed per emotion per layer")
    ap.add_argument("--show-layers", default="", help="layers to print, default the middle and last")
    a = ap.parse_args()

    corpus, names, _va = load_corpus()
    train, test = split(corpus)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = model.model.language_model
    dev = next(model.parameters()).device
    n_layers = len(lm.layers)
    layers = list(range(n_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    show_layers = [int(x) for x in a.show_layers.split(",")] if a.show_layers else [layers[len(layers) // 2], layers[-1]]
    os.makedirs(a.out, exist_ok=True)

    report = {"model": a.model, "emotions": names, "layers": {}}
    # One forward per sentence, all requested layers kept, so 32 layers cost 236 forwards not 7,552.
    res = {l: {"train": {}, "test": {}} for l in layers}
    t0 = time.time()
    n_sent = 0
    with torch.inference_mode():
        for sp, data in (("train", train), ("test", test)):
            for k, sents in data.items():
                for s in sents:
                    ids = tok(s, return_tensors="pt").input_ids.to(dev)
                    hs = lm(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
                    n_sent += 1
                    for l in layers:
                        t = hs[l + 1][0].float().cpu().numpy()
                        res[l][sp].setdefault(k, []).append(t[SKIP:] if t.shape[0] > SKIP + 1 else t)
    print(f"{n_sent} sentences x {len(layers)} layers in {time.time() - t0:.1f}s", flush=True)

    for layer in layers:
        f = fit_layer(res[layer], names)
        w_down = lm.layers[layer].mlp.down_proj.weight.detach().float().T.cpu().numpy()  # [neurons, hidden]
        wn = w_down / (np.linalg.norm(w_down, axis=1, keepdims=True) + 1e-9)
        cos = (wn @ f["D"].T).T.astype(np.float16)  # [12, neurons]
        # A push of `add` along w_down changes the raw projection by add * cos; the probe divides that by the emotion's
        # neutral spread sd, then the common-mode-corrected score divides by spec_sd. Both factors belong here or the
        # number is in raw residual units mislabelled as z (which read as +60 per unit dose).
        z_per_dose = (cos.astype(np.float32) * f["resid_norm_p50"]
                      / f["sd"][:, None] / f["spec_sd"][:, None])
        # What the probe shows is z minus the mean of the twelve, so the push a neuron buys on a specific emotion is
        # its projection minus the average of its projections: without this, the neurons that lift every emotion at
        # once (the common emotional component) top all twelve lists and the ranking says nothing.
        z_spec = (z_per_dose - z_per_dose.mean(0, keepdims=True)).astype(np.float16)
        np.savez(os.path.join(a.out, f"L{layer}.npz"), cos=cos, z_per_dose=z_per_dose.astype(np.float16),
                 z_spec=z_spec, dirs=f["D"], mu=f["mu"],
                 sd=f["sd"], mean_neu=f["mean_neu"], spec_mean=f["spec_mean"], spec_sd=f["spec_sd"],
                 resid_norm_p50=f["resid_norm_p50"], layer=layer, emotions=np.array(names),
                 accuracy=f["accuracy"])
        report["layers"][str(layer)] = {"accuracy": round(f["accuracy"], 3),
                                        "resid_norm_p50": round(f["resid_norm_p50"], 1),
                                        "top_spec_z": {e: round(float(z_spec[i].max()), 3) for i, e in enumerate(names)},
                                        "top_cos": {e: round(float(cos[i].max()), 3) for i, e in enumerate(names)}}
        print(f"L{layer}: held-out acc {f['accuracy']:.2f}  resid p50 {f['resid_norm_p50']:.0f}  "
              f"best spec z {z_spec.max():+.3f} ({names[int(z_spec.max(1).argmax())]})", flush=True)
        if layer in show_layers:
            for i, e in enumerate(names):
                top = np.argsort(-z_spec[i])[: a.show]
                print(f"  {e:11s} " + " ".join(f"N{int(n)}({z_spec[i, n]:+.2f})" for n in top), flush=True)

    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"wrote {a.out}/ ({len(layers)} layers)")


if __name__ == "__main__":
    main()
