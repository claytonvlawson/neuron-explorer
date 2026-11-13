"""Group every neuron by the emotion it favours, from the per-layer cos matrices of sweep/mood_rank.py.

Each neuron of each layer is assigned to the emotion whose contrast direction its down column pushes hardest on
(argmax of z_spec, the same quantity the probe displays). Writes

    groups.csv.gz     one row per neuron: layer, neuron, emotion, z_per_dose, cos, z of the runner-up emotion
    summary.csv       layer x emotion: how many neurons chose it, mean and max z_per_dose
    report.json       totals, per-emotion totals across layers, and the share of neurons whose top two emotions are
                      within 0.5 z of each other (an ambiguous assignment, which is most of them at depth)

    python sweep/group_neurons.py --dir reports/mood-rank/npz --out reports/mood-rank

Reading it: a group is "the emotion this neuron's own down column leans to", which is a static property of the
weights. It is not a claim that pushing the neuron produces that emotion in the text, which needs a steered rollout.
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of L<layer>.npz from sweep/mood_rank.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ambiguous", type=float, default=0.5, help="z gap below which an assignment is called ambiguous")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.dir, "L*.npz")), key=lambda p: int(os.path.basename(p)[1:-4]))
    assert files, f"no L<layer>.npz under {a.dir}"
    summary, amb_total, n_total = [], 0, 0
    per_emotion = {}

    with gzip.open(os.path.join(a.out, "groups.csv.gz"), "wt", newline="") as fh:
        fh.write("layer,neuron,emotion,z_per_dose,cos,runner_up,gap\n")
        for f in files:
            z = np.load(f)
            layer, em = int(z["layer"]), [str(x) for x in z["emotions"]]
            spec = z["z_spec"].astype(np.float32)
            cos = z["cos"].astype(np.float32)
            order = np.argsort(-spec, axis=0)  # [12, neurons], best first
            best, second = order[0], order[1]
            n = spec.shape[1]
            n_total += n
            for k in range(n):
                e = int(best[k])
                z1 = float(spec[e, k])
                z2 = float(spec[second[k], k])
                gap = z1 - z2
                amb_total += gap < a.ambiguous
                fh.write(f"{layer},{k},{em[e]},{z1:.3f},{float(cos[e, k]):.3f},{em[int(second[k])]},{gap:.3f}\n")
                per_emotion[em[e]] = per_emotion.get(em[e], 0) + 1
            for i, e in enumerate(em):
                sel = best == i
                summary.append({"layer": layer, "emotion": e, "neurons": int(sel.sum()),
                                "share": round(float(sel.mean()), 4),
                                "z_mean": round(float(spec[i][sel].mean()), 3) if sel.any() else None,
                                "z_max": round(float(spec[i].max()), 3)})
            print(f"L{layer}: largest group {em[int(np.bincount(best, minlength=len(em)).argmax())]} "
                  f"({np.bincount(best, minlength=len(em)).max()}/{n})", flush=True)

    with open(os.path.join(a.out, "summary.csv"), "w") as fh:
        fh.write("layer,emotion,neurons,share,z_mean,z_max\n")
        for r in summary:
            fh.write(f"{r['layer']},{r['emotion']},{r['neurons']},{r['share']},{r['z_mean']},{r['z_max']}\n")

    report = {"neurons": n_total, "layers": len(files), "per_emotion": per_emotion,
              "ambiguous_share": round(amb_total / n_total, 4), "ambiguous_gap_below": a.ambiguous}
    json.dump(report, open(os.path.join(a.out, "group_report.json"), "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("neurons", "layers", "ambiguous_share")}), flush=True)
    print("per emotion:", json.dumps(per_emotion), flush=True)


if __name__ == "__main__":
    main()
