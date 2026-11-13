# 5090 gate and the mood node ranking, 2026-11-13

Pod: `83xxuzm03okqr2`, RTX 5090 32 GB, compute cap 12.0 (sm_120), driver 595.91.07, 32 cores, 186 GB host RAM,
50 GB container disk, no network volume. Direct SSH on `root@213.173.103.92:44218`; the `ssh.runpod.io` proxy for
this pod refused port 2002 and then hung, so every command here went over the direct address.
Source pinned at `d51716ff5a3d75a9efb32fee4311f96fd1ba2622` plus the three commits that follow it.

## Verdicts

1. **The 5090 works.** `setup_pod.sh` passes its gate and both equivalences in `check_equivalence.py` hold. sm_120
   is not a blocker for this repo at the pinned wheel commit.
2. **The 5090 buys no throughput.** 530 gen tok/s on `neuron_stats`, against 533 to 625 measured on a 4090. The
   sweep is engine-core bound, exactly as the runbook says, so the 5090's advantage is 32 GB of VRAM and stock,
   not speed.
3. **All 294,912 neurons are now ranked for 12 emotions and grouped by emotion**, analytically, with no rollouts.
   The grouping is a static property of the weights. No steered rollout has confirmed any of it yet (see below).

## Gate 1: setup, and the exact versions

Whole setup finished inside ten minutes of launch (`setup.log` last write 16:43:41 UTC), against the 28 minutes
measured on a 4090 with a cold volume. Gate line as printed:

    vllm 0.1.dev21064+g4506f935b | transformers 5.23.0 | torch 2.13.0+cu132 | fla ok | cuda NVIDIA GeForce RTX 5090

`fla ok` is the coin flip, and it landed: `import fla.ops.gated_delta_rule` and the Triton kernels compile on sm_120.
Steering kernel warmup compiled 3 variants in 470 ms; the steering monitor kernel in 125 ms. The capture path
allocated its CUDA-graph-safe buffer (6.2 MiB) and registered `_declarative_steering` with no fallback.
Engine init took 137.67 s, 33.7 s of it compilation.

Version drift from the 4090 run to note: `transformers` 5.23.0 here against 5.22.2 recorded in the runbook.
`vllm` and `torch` are identical.

## Gate 2: check_equivalence

    L22, prompt 'Who are you?', 48 greedy tokens. agree = shared prefix length HF vs vLLM
      baseline       agree  48/48
      n2742 +10      agree  48/48
      n2742 -2       agree  48/48
      n17   +10      agree  48/48
      n17   -2       agree  48/48
      n5000 +10      agree  17/48     <- near tie, the case the runbook calls benign
      n5000 -2       agree  48/48

    capture: requested all 16 prompt positions; got [0..15]
    recomputed vs HF neuron activations: relative error per position
      0.0179 0.0237 0.0094 0.0100 0.0109 0.0135 0.0131 0.0136 0.0143 0.0398 0.0194 0.0142 0.0179 0.0213 0.0226 0.0151
    same top neuron at 100% of positions

Baseline agreement 48/48 means the two engines are not drifting apart at all on this card; six of seven steered
cases are exact and the seventh splits at token 17, which is the near-tie signature, not a wrong push. Capture
recomputes neuron activations at 1 to 4 percent relative error with the correct top neuron in every position.

`causal_conv1d` is not installed in the pod venv, so HF falls back to its reference PyTorch implementation
(`causal_conv1d_fn`), which is "much slower" by transformers' own warning. Matters only for the HF reference and
for `mood_rank`'s sentence passes, not for the vLLM sweep.

## The ranking: `sweep/mood_rank.py`

A push on neuron `n` of layer `L` adds `add * W_down_L[:, n]` to the residual stream, and the mood probe scores a
residual as `z_e = ((r - mu) . dir_e)/sd_e`, common-mode corrected. So the score a push buys is proportional to
`cos(W_down[:, n], dir_e)` at that layer, and every neuron can be ranked without generating anything:

    z_per_dose_e[n] = cos_e[n] * resid_p50 / sd_e / spec_sd_e

Directions are fitted per layer from the same 188-sentence corpus and the same split as `explorer/mood.py`. Cost on
the 5090: **188 sentences x 32 layers in 4.2 s**, then a 9216 x 2560 projection per layer. The whole run is under
90 seconds including model load.

Held-out twelve-way sentence accuracy per layer (chance 0.083): L0 0.23, L8 0.50, L12 0.50, L16 0.58, **L17 0.625
(best)**, L20 0.604, L24 0.50, L28 0.44, L31 0.31. The probe is strongest around L17 to L20 and degrades at both
ends, and L31 sits on a different footing entirely because `hidden_states[l+1]` there is post-norm (its residual
norm p50 is 156 against 12 at L17).

Top nodes per emotion at L17, the best-fitting layer (z per unit relative dose):

    joy         N789(+6.34) N3748(+4.35) N7611(+2.81) N3625(+2.71) N6055(+2.33)
    sadness     N6177(+4.10) N8935(+3.18) N9030(+2.73) N4888(+2.59) N4172(+2.25)
    anger       N8623(+6.35) N5409(+4.77) N8391(+4.30) N222(+4.13) N4888(+3.68)
    fear        N4172(+3.14) N4888(+3.12) N3924(+2.84) N6976(+2.52) N222(+2.48)
    tenderness  N8612(+2.90) N6550(+2.69) N2689(+2.34) N2639(+2.30) N2545(+2.07)
    hope        N5194(+6.93) N3839(+3.53) N7611(+3.52) N6018(+3.01) N4291(+2.99)
    concern     N5777(+2.06) N4888(+1.96) N4501(+1.95) N4172(+1.87) N8264(+1.84)
    calm        N6554(+3.11) N0(+3.04) N7017(+2.92) N7996(+2.84) N7478(+2.68)
    curiosity   N5777(+2.43) N8098(+2.37) N730(+2.34) N7265(+2.14) N3382(+2.06)
    confidence  N8204(+3.16) N8655(+2.36) N5510(+2.33) N6216(+2.05) N3495(+1.95)
    doubt       N5907(+2.94) N5777(+2.46) N3302(+2.29) N3058(+2.18) N8098(+2.05)
    urgency     N4291(+3.25) N5777(+3.07) N3763(+2.32) N5907(+2.11) N4501(+2.06)

At L20 the leaders move (anger N8619 +7.24, calm N8730 +7.71, joy N7560 +3.32), so the node that carries an
emotion is layer-specific and the full 32-layer table matters. `reports/mood-rank/nodes_top64.csv` holds the top 64
per emotion per layer (24,576 rows).

## Every neuron grouped by emotion: `sweep/group_neurons.py`

Each of the 294,912 (layer, neuron) pairs is assigned to the emotion whose contrast direction its down column
favours most, with the runner-up and the gap recorded.

    groups.csv.gz   294,912 rows: layer, neuron, emotion, z_per_dose, cos, runner_up, gap
    summary.csv     32 layers x 12 emotions: group size, share, mean and max z_per_dose
    group_report.json

    group sizes across all layers   anger 50,797 | hope 31,145 | calm 29,672 | tenderness 29,627 | urgency 25,293
                                    joy 23,738 | fear 22,111 | curiosity 21,470 | sadness 18,028 | doubt 17,341
                                    confidence 15,605 | concern 10,085
    clear leans only (gap >= 0.5)   54,332 of 294,912 = 18.4 percent
                                    anger 17,338 | hope 6,414 | tenderness 5,819 | calm 5,693 | urgency 3,590
                                    joy 3,582 | fear 3,422 | curiosity 2,950 | sadness 1,981 | doubt 1,560
                                    confidence 1,154 | concern 829

**81.6 percent of neurons are ambiguous**: their top two emotions are within 0.5 z of each other. That is the honest
headline. The mood directions are close together in a 2560-dimensional space, and most MLP columns sit near the
shared emotional component rather than cleanly on one emotion. A usable node list is the 54,332 clear leans, and
within those, the per-layer maxima in `summary.csv`.

Anger is over-assigned relative to the others (17 percent of all neurons, 32 percent of clear leans). Treat that as
a property of the fitted direction, not as evidence that the model has five times more anger circuitry: the anger
corpus is the highest-arousal negative set, and its direction is the closest of the twelve to the generic
high-activation axis. Any emotion-to-node claim should be read as "this node leans anger more than the other eleven".

## Pitfalls found here (both cost a rerun)

- **Ranking on raw cosine puts the same neurons on top of every emotion.** The first pass returned N498, N65 and
  N356 as the leader for nine of twelve emotions: that is the common emotional component, the exact artefact
  `mood.py` corrects for at scoring time. Rank on `z_spec`, the projection minus that neuron's own mean across the
  twelve, or the table says nothing.
- **`z_per_dose` must divide by the emotion's `sd` as well as `spec_sd`.** Dividing by `spec_sd` alone leaves the
  number in raw residual units wearing a z label, and single neurons then appeared to buy +60 z per unit dose.
  The corrected range is +0.5 to +8, and at the coherent dose window (0.5 to 0.8) one neuron buys roughly 1 to 3 z.

## What is not done

No steered rollout has been run against any neuron in this table. The ranking is an upper bound at the moment of
the push: it ignores what layers 18 to 31 do with the perturbation, and the mood probe's own held-out accuracy is
0.625, so eleven of twelve calls are right at best on a clean sentence. Confirming a candidate is the sweep's job:

    python sweep/steer_sweep.py --stats stats/q35-4b_L17.npz --dose-unit rel --neurons 8623,5194,789 \
        --design main_rel --random-controls 64 --run q35-4b-L17-moodcheck --capture-layers ""

Then score the generated text with the probe at L17 and compare each neuron's pushed rows against the 64 random
directions. Until that runs, every emotion-to-node line above is a hypothesis with a number attached.

## Ops notes for the next pod

- `scp` does not work through `ssh.runpod.io` (PTY proxy), but it works fine against the direct pod IP and port:
  `scp -P 44218 -i ~/.ssh/id_ed25519 root@<ip>:/workspace/neuron-explorer/mood_rank ./`. That is how the 25 MB of
  npz got home without an object store or a volume.
- Disk: setup plus weights plus the repo came to 25 GB of a 50 GB overlay, leaving room for `mood_rank` (25 MB)
  and a small sweep, but not for captures. `--capture-layers ""` if a real sweep runs on this pod.
- A pod at idle on this shape still bills. Confirm in the console that `83xxuzm03okqr2` is stopped when done.
