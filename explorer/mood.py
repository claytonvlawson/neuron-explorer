"""Emotion and mood probe.

A linear probe on the residual stream of the local model. Directions are difference-of-means over a small
hand-written corpus (sentences that carry one emotion each, plus neutral factual sentences), taken at one layer
chosen by held-out accuracy. At generation time the residual at the position that predicts the next token is
projected on each direction and standardised against the neutral corpus, so a score of 0 is neutral prose and
+3 is three neutral standard deviations of that emotion.

What this measures: how strongly the model's internal state at that word resembles the state it is in when it
reads or writes text of that emotion. It is a register detector. It is not evidence that the model feels anything,
and the UI says so.
"""
import json
import os

import numpy as np
import torch

EMOTIONS = ["joy", "sadness", "anger", "fear", "tenderness", "hope", "concern", "calm", "curiosity",
            "confidence", "doubt", "urgency"]
# (valence, arousal) in [-1, 1], used only to turn the twelve scores into one mood point
VA = {"joy": (0.9, 0.6), "sadness": (-0.8, -0.5), "anger": (-0.7, 0.8), "fear": (-0.8, 0.7),
      "tenderness": (0.8, -0.3), "hope": (0.6, 0.2), "concern": (-0.4, 0.3), "calm": (0.4, -0.8),
      "curiosity": (0.4, 0.4), "confidence": (0.5, 0.3), "doubt": (-0.3, -0.1), "urgency": (-0.2, 0.9)}

CORPUS = {
    "joy": [
        "I am so happy right now that I could laugh out loud.",
        "She burst into delighted laughter when she saw the surprise party.",
        "What wonderful news, we got the grant and everyone is celebrating!",
        "He grinned from ear to ear as his daughter signed her first full sentence.",
        "It was the best day of my life, pure joy from morning to night.",
        "The children cheered and danced around the room with glee.",
        "I'm thrilled to tell you that your child is thriving and making great progress!",
        "We hugged each other and cried happy tears at the finish line.",
        "That is fantastic, congratulations, you should be incredibly proud!",
        "The whole family was overjoyed when the baby finally said her name.",
        "Sunshine, music, and friends: I felt delighted and alive.",
        "Oh, I love it, this is exactly what I hoped for, thank you so much!",
        "They shouted with excitement and jumped up and down.",
        "What a joyful, lovely surprise to hear your voice today.",
    ],
    "sadness": [
        "I miss him terribly and the house feels empty without him.",
        "She wept quietly at the funeral, unable to speak.",
        "It is heartbreaking to watch a child go years without being understood.",
        "He sat alone in the dark, feeling a deep and heavy sorrow.",
        "The loss left her grieving for months and nothing could console her.",
        "Tears ran down my face as I read the last letter.",
        "It is a tragedy that so many children are denied language in their early years.",
        "A sad silence fell over the room after the news.",
        "I feel hollow and miserable, as if all the color has drained from my life.",
        "They mourned the friend they had lost far too soon.",
        "The little boy cried himself to sleep, lonely and cut off from everyone.",
        "Sorry to say, there is no way to get those lost years back, and that is painful.",
        "She felt a crushing sadness and could not stop sobbing.",
        "What a sorrowful, bleak and lonely winter that was.",
    ],
    "anger": [
        "I am furious that they ignored every warning and did nothing.",
        "He slammed his fist on the table and shouted in rage.",
        "It is outrageous and unacceptable that families are told to wait and see.",
        "She glared at him, seething with resentment over the betrayal.",
        "How dare they lie to us about this for years!",
        "The crowd erupted in angry protest against the decision.",
        "I'm sick and tired of being dismissed, and I am done being polite about it.",
        "He was livid, his jaw clenched and his voice shaking with fury.",
        "That is infuriating, a disgraceful failure by people who should have known better.",
        "She snapped at them, bitter and irritated by the endless excuses.",
        "The injustice made my blood boil.",
        "Stop lying to me, this is completely unfair and you know it!",
        "They cursed the officials who had denied access to interpreters.",
        "Rage built in his chest until he could hardly think.",
    ],
    "fear": [
        "I was terrified, my heart pounding as footsteps came closer in the dark.",
        "She trembled with dread at what the doctor might say.",
        "He froze in panic when he realized the door was locked behind him.",
        "A cold fear crept over me, and I could hardly breathe.",
        "The child hid under the blanket, frightened of the thunder.",
        "I'm scared that it is already too late and the window has closed.",
        "They lived in constant anxiety that something awful would happen.",
        "Horror washed over her as the water kept rising.",
        "His hands shook; he was afraid to open the letter.",
        "What if we lose everything, what if nobody comes to help us?",
        "The dark corridor filled him with creeping terror.",
        "She screamed and ran, overwhelmed by fright.",
        "It is frightening to think how quickly the critical period passes.",
        "Nervous and alarmed, he checked the locks again and again.",
    ],
    "tenderness": [
        "She gently stroked the baby's hair and whispered that she was loved.",
        "I hold you close, and I am so grateful you are in my life.",
        "He tucked the blanket around his sleeping daughter with a soft smile.",
        "You are doing a loving, caring job, and your child feels that warmth every day.",
        "The grandmother cradled the newborn with infinite tenderness.",
        "Sweet and kind, he held her hand and stayed by her bedside.",
        "My dear, you are not alone, I'm here with you and I care about you.",
        "They shared a quiet, affectionate embrace.",
        "She kissed his forehead softly and sang him to sleep.",
        "With warmth and compassion, the teacher knelt down to the child's level.",
        "Your love is the most precious gift a child can have.",
        "He looked at his wife with deep affection and gratitude.",
        "Be gentle with yourself, you are doing the best you can, and that is enough.",
        "The puppy curled up in her arms, warm and trusting.",
    ],
    "hope": [
        "I believe things will get better, and tomorrow holds real promise.",
        "There is every reason to be optimistic, because early signing works.",
        "She looked toward the future with new hope in her heart.",
        "It is not too late, and many children make remarkable progress with the right support.",
        "We kept going because we trusted that a brighter day was coming.",
        "He dreamed of a world where every deaf child grows up with a full language.",
        "Hopefully this is the start of something wonderful, and I am looking forward to it.",
        "With patience and the right access, there is a real path forward.",
        "The new program gives families a chance they never had before.",
        "I am confident that change is possible, and I'm holding on to that.",
        "Every small step is a sign that recovery is within reach.",
        "She prayed and wished for a miracle, and for the first time felt encouraged.",
        "The future looks bright; the opportunities ahead are enormous.",
        "Don't give up, a better outcome is still possible.",
    ],
    "concern": [
        "I'm worried about how long your child has gone without access to language.",
        "It is troubling that the delay may affect later reading and thinking.",
        "She frowned with concern as she studied the test results.",
        "We should be careful here, because the risks are serious and easy to overlook.",
        "He felt uneasy about the situation and could not shake the worry.",
        "I am concerned that waiting any longer could make things harder for your family.",
        "The doctor looked grave and spoke carefully, aware of how much was at stake.",
        "That is a real problem, and it deserves your attention and a prompt conversation with a specialist.",
        "It worries me that nobody has told you what the research actually shows.",
        "Please take this seriously; the consequences of delay can be lasting.",
        "She fretted over every detail, anxious about her son's future.",
        "There are some warning signs here that I do not want to minimize.",
        "I'm sorry to be the one to say it, but the situation is more difficult than it looks.",
        "He worried quietly, afraid of making the wrong choice for his child.",
    ],
    "calm": [
        "The lake was still and quiet, and I felt completely at peace.",
        "She took a slow breath and let her shoulders relax.",
        "Everything is under control, so there's no need to rush.",
        "A gentle breeze moved through the trees as he sat in serene silence.",
        "Take your time; there is no hurry, and we can go through it one step at a time.",
        "The evening was tranquil, with soft light and the hush of the sea.",
        "He felt relaxed and settled, with nothing weighing on his mind.",
        "Let's slow down, breathe, and look at this calmly.",
        "The room was peaceful, and the tea was warm in her hands.",
        "She spoke in a steady, soothing voice that put everyone at ease.",
        "Nothing needs to be decided today; rest, and we will come back to it.",
        "The garden was quiet and still, a place of gentle stillness.",
        "I feel centered, balanced, and unhurried.",
        "It is fine, everything is okay, simply rest.",
    ],
    "curiosity": [
        "I wonder how the brain learns a language when it has never heard one.",
        "She leaned in, fascinated, eager to find out what would happen next.",
        "Why does this happen, and what is actually going on underneath?",
        "He was intrigued by the strange pattern and began to investigate.",
        "That's interesting, I would love to learn more about how it works.",
        "The child poked and prodded the machine, wanting to see how it worked.",
        "What would happen if we tried it a different way?",
        "I am curious about what the evidence says and where the studies disagree.",
        "She asked question after question, hungry to understand.",
        "How do sign languages organize grammar in space, and who first described it?",
        "He explored the old library, drawn to every unfamiliar book.",
        "It is a fascinating puzzle, and I can't wait to dig into it.",
        "Let's explore this idea and see what we discover.",
        "Curious and wide-eyed, the toddler studied her own hands.",
    ],
    "confidence": [
        "I know exactly what to do, and I will get it done.",
        "The evidence is clear and consistent: early access to language works.",
        "She stood tall and spoke with complete certainty.",
        "There is no doubt about it, this approach is proven and reliable.",
        "He walked into the room with a strong, self-assured stride.",
        "Absolutely, this is the right decision, and I stand behind it.",
        "Decades of research agree, and the conclusion is firm.",
        "We will succeed; I have no hesitation whatsoever.",
        "Trust me on this, I have done it hundreds of times and it works.",
        "The facts are unambiguous: children need language from birth.",
        "She was proud and sure of herself, ready for anything.",
        "That is definitely correct, and the data confirm it beyond question.",
        "He answered immediately and without a flicker of uncertainty.",
        "Rest assured, this is well established and I can state it plainly.",
    ],
    "doubt": [
        "I'm not sure that is right, and I could easily be wrong.",
        "Maybe, perhaps, it's hard to say; the evidence is mixed.",
        "She hesitated, unsure whether to believe him.",
        "It might be the case, but I honestly don't know.",
        "He wondered whether he had made a terrible mistake.",
        "The results are inconclusive, so we cannot be certain either way.",
        "I'm skeptical; something about this doesn't add up.",
        "Possibly, though it depends, and there are many unknowns.",
        "She second-guessed herself again and again.",
        "I can't tell for sure, and I would be guessing if I said more.",
        "Perhaps it works for some children, but frankly nobody really knows.",
        "He doubted every word and questioned the whole story.",
        "It's unclear, uncertain, and honestly a bit murky.",
        "I'm torn, I could see it going either way.",
    ],
    "urgency": [
        "We need to act right now, because every day counts!",
        "Hurry, there is no time to lose!",
        "This cannot wait; start immediately and do not delay.",
        "The deadline is tonight, so move fast!",
        "Emergency: call for help at once!",
        "Every month of delay matters, so please begin as soon as you possibly can.",
        "She ran to the door, desperate to get there before it closed.",
        "Quick, now, before it is too late!",
        "It is critical that you get an evaluation immediately and don't put it off.",
        "The clock is ticking, and we have to move today.",
        "He urged them to act at once, with no more waiting.",
        "Don't wait for the next appointment, call them this morning.",
        "Time is running out, and the window is closing fast.",
        "Right away, without delay, drop everything and go!",
    ],
    "neutral": [
        "The meeting is scheduled for Tuesday at three o'clock in the main conference room.",
        "Water boils at one hundred degrees Celsius at sea level.",
        "The report contains four sections and an appendix with tables.",
        "The library is located on the second floor of the building.",
        "Paris is the capital of France and sits on the river Seine.",
        "The function takes two arguments and returns their sum.",
        "Please find the attached document with the updated figures.",
        "The train departs from platform six and arrives at nine fifteen.",
        "A triangle has three sides and the angles add up to one hundred eighty degrees.",
        "The committee reviewed the proposal and recorded the results in the minutes.",
        "The software update includes several changes to the settings menu.",
        "Oak trees lose their leaves in the autumn and grow new ones in the spring.",
        "The store opens at nine in the morning and closes at six in the evening.",
        "Language is a system of symbols used for communication among people.",
        "The study enrolled two hundred participants across three sites.",
        "Mix the flour and sugar, then add the eggs and stir until smooth.",
        "The bridge is four hundred meters long and was completed in 1998.",
        "The form must be submitted to the office before the end of the month.",
        "Sign languages have their own grammar, vocabulary, and regional variation.",
        "The average annual rainfall in the region is about eight hundred millimeters.",
    ],
}

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "mood_probe.npz")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "mood_probe.json")
SKIP = 2  # leading tokens of a sentence that are mostly a position sink, left out of every statistic
N_TEST = 4


def split():
    train, test = {}, {}
    for k, v in CORPUS.items():
        n = len(v)
        te = list(range(n - N_TEST, n)) if k != "neutral" else list(range(n - 6, n))
        train[k] = [s for i, s in enumerate(v) if i not in te]
        test[k] = [v[i] for i in te]
    return train, test


def sentence_residuals(rt, text, layers):
    """{layer: [T, D] float32} residual after each requested layer, over a plain (non-chat) sentence."""
    ids = rt.tok(text, return_tensors="pt").input_ids.to(rt.dev)
    with torch.inference_mode():
        hs = rt.lm(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
    return {l: hs[l + 1][0].float().cpu().numpy() for l in layers}


def build(rt, layers=(8, 12, 16, 20, 24, 28), progress=None):
    """Fit the probe: per candidate layer, directions from the train split, accuracy on the held-out split;
    keep the best layer. Needs the live model (takes its lock)."""
    rt.ensure_loaded()
    train, test = split()
    names = EMOTIONS
    res = {}  # layer -> {split -> {emotion -> [ (mean_vec, tokens[T,D]) ]}}
    with rt.lock:
        for si, (sp, data) in enumerate((("train", train), ("test", test))):
            for k, sents in data.items():
                for s in sents:
                    r = sentence_residuals(rt, s, layers)
                    for l in layers:
                        t = r[l][SKIP:] if r[l].shape[0] > SKIP + 1 else r[l]
                        res.setdefault(l, {}).setdefault(sp, {}).setdefault(k, []).append(t)
                if progress:
                    progress(f"{sp}:{k}")
    best, report = None, {"layers": {}, "emotions": names}
    for l in layers:
        tr, te = res[l]["train"], res[l]["test"]
        neu_tok = np.concatenate(tr["neutral"])
        mu = neu_tok.mean(0)
        dirs = []
        for e in names:
            m = np.mean([t.mean(0) for t in tr[e]], 0)
            d = m - np.mean([t.mean(0) for t in tr["neutral"]], 0)
            dirs.append(d / (np.linalg.norm(d) + 1e-9))
        D = np.stack(dirs)  # [E, dim]
        neu_p = (neu_tok - mu) @ D.T  # neutral token projections: the scale
        sd = neu_p.std(0) + 1e-6
        mean_neu = neu_p.mean(0)
        # The twelve projections share a large common part (emotional text, or any text at this depth of context,
        # lifts them all). What tells emotions apart is each score minus the mean of the twelve, in units of how
        # much that varies on neutral text. That is the number the UI shows.
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
                pred = int(zz.argmax()) if zz.max() > 1.0 else len(names)  # last column: none
                conf[i, pred] += 1
                ok += pred == i
                tot += 1
        neu_ok = sum(z(t).mean(0).max() <= 1.0 for t in te["neutral"])
        # token-level: do held-out emotional sentences light their own emotion on the final token?
        last_ok = sum(int(z(t[-1:]).argmax()) == i for i, e in enumerate(names) for t in te[e])
        acc = ok / tot
        report["layers"][str(l)] = {"sentence_acc": round(acc, 3), "last_token_acc": round(last_ok / tot, 3),
                                    "neutral_quiet": f"{neu_ok}/{len(te['neutral'])}"}
        if best is None or acc > best[0]:
            best = (acc, l, D, mu, sd, mean_neu, conf, spec_mean, spec_sd)
    acc, l, D, mu, sd, mean_neu, conf, spec_mean, spec_sd = best
    report.update(layer=l, held_out_accuracy=round(acc, 3), chance=round(1 / len(names), 3),
                  n_test=int(conf.sum()), confusion=conf.tolist(),
                  per_emotion={e: f"{conf[i, i]}/{conf[i].sum()}" for i, e in enumerate(names)})
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    np.savez(PATH, layer=l, dirs=D.astype(np.float32), mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             mean_neu=mean_neu.astype(np.float32),
             spec_mean=spec_mean.astype(np.float32), spec_sd=spec_sd.astype(np.float32))
    json.dump(report, open(REPORT, "w"), indent=1)
    return report


def load(rt):
    """Put the saved probe on the runtime's device. Returns the report dict or None."""
    if not os.path.exists(PATH):
        rt.mood = None
        return None
    z = np.load(PATH)
    dev = rt.dev if getattr(rt, "dev", None) else "cpu"
    rt.mood = {"layer": int(z["layer"]), "dirs": torch.tensor(z["dirs"], device=dev),
               "mu": torch.tensor(z["mu"], device=dev), "sd": torch.tensor(z["sd"], device=dev),
               "mean_neu": torch.tensor(z["mean_neu"], device=dev),
               "spec_mean": torch.tensor(z["spec_mean"], device=dev), "spec_sd": torch.tensor(z["spec_sd"], device=dev),
               "names": EMOTIONS}
    return json.load(open(REPORT)) if os.path.exists(REPORT) else {"layer": rt.mood["layer"]}


def score_vec(rt, resid):
    """Distinctive z per emotion for one residual vector [D]: each emotion's score minus the mean of all twelve,
standardised on neutral text. Around 0 is nothing in particular; above about 2 is a clear lean."""
    m = rt.mood
    r = resid.to(m["dirs"].device).float()
    zz = (((r - m["mu"]) @ m["dirs"].T) - m["mean_neu"]) / m["sd"]
    return ((zz - zz.mean()) - m["spec_mean"]) / m["spec_sd"]


def va(z):
    """One (valence, arousal) point from positive evidence only, so the point sits at the origin when quiet."""
    w = np.clip(np.asarray(z, float), 0, None)
    if w.sum() < 1e-6:
        return 0.0, 0.0
    w = w ** 2
    w = w / w.sum() * min(1.0, np.sqrt(w.sum()) / 4)
    return (float(sum(w[i] * VA[e][0] for i, e in enumerate(EMOTIONS))),
            float(sum(w[i] * VA[e][1] for i, e in enumerate(EMOTIONS))))


def score_text(rt, text):
    """Mean-pooled z per emotion over a plain text (skipping the first tokens). For the user's own message."""
    if getattr(rt, "mood", None) is None:
        return None
    r = sentence_residuals(rt, text, [rt.mood["layer"]])[rt.mood["layer"]]
    r = r[SKIP:] if r.shape[0] > SKIP + 1 else r
    z = []
    for row in r:
        z.append(score_vec(rt, torch.tensor(row)).cpu().numpy())
    return np.mean(z, 0)
