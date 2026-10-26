"""The money lens: read a real document, sentence by sentence, and measure which sentences the Money person
rewrites into the language of money.

This is a reading instrument, not a forecaster. It does not use the person's own completions as analysis: it takes
text that somebody else wrote (a 10-K's risk factors, or headlines you paste in), runs the same three rows the chat
does (person / unsteered baseline / the same pushes reversed), and reports two measured things per sentence:

  mass    the log-probability the row's next-token distribution puts on a fixed money lexicon, in nats. One forward
          pass per row, no sampling, so it is a number without a lucky completion behind it.
  rewrite one greedy completion of "rewrite this sentence in the language of money" per row, for the sentences the
          numbers say are worth reading, kept verbatim so you can disagree with it.

The amplification of a sentence is `mass(person) - mass(baseline)` minus the same difference for a neutral lexicon
of function words, so a person that merely sharpens every distribution does not look like a person that money-ises
one sentence in particular. Nothing here says a sentence is bullish, bearish or true.

Env:
  EDGAR_UA   contact string sent to SEC (default below). SEC requires a real one.
"""
import html
import json
import os
import re
import time
import urllib.request
from html.parser import HTMLParser

EDGAR_UA = os.environ.get("EDGAR_UA", "neuron-explorer money lens clayton@learnwithclay.ai")
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "cache", "money_docs")

# The instrument's vocabulary. Not "all money words": a fixed list, so the same number means the same thing twice.
MONEY_WORDS = """
money cash cost costs price prices profit profits market markets revenue revenues dollar dollars pay paid buying
buy sell sold trade trading trader invest investor investment value worth economy economic finance financial coin
currency capital fund funds debt loan tax taxes income wage wages salary rich wealth budget account bank business
sale sales fee fees charge charges earn earning earnings spend spending expense expenses payment customers
customer demand supply margin margins pricing valuation premium discount return returns yield yields credit
""".split()

NEUTRAL_WORDS = "the a an of to and in it is that for this with as on at by from".split()


# ---------- the document ----------

class _Text(HTMLParser):
    """Visible text only, paragraph breaks kept: 10-K HTML is 90% tables and inline XBRL, neither of which reads."""

    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "td"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "ix:header"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "ix:header") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def to_text(markup):
    p = _Text()
    p.feed(markup)
    text = html.unescape("".join(p.out))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text)


def sentences(text, min_words=8, max_words=45):
    """Sentence split plus a length window: a sentence the lens can read in one greedy breath, long enough to carry
    a claim. Abbreviations are the usual trap, so a split needs a capital or a quote after the stop."""
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'“])", text)
    out = []
    for s in parts:
        s = s.strip()
        n = len(s.split())
        if min_words <= n <= max_words:
            out.append(s)
    return out


def _get(url, data=None, tries=3):
    for i in range(tries):
        req = urllib.request.Request(url, data=data, headers={
            "User-Agent": EDGAR_UA, "Accept-Encoding": "identity", "Host": url.split("/")[2]})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(0.4 * (i + 1))


def cik_for(symbol):
    """symbol -> zero-padded CIK, from the SEC's own ticker map."""
    os.makedirs(DOCS, exist_ok=True)
    cache = os.path.join(DOCS, "_tickers.json")
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 86400:
        raw = open(cache, encoding="utf-8").read()
    else:
        raw = _get("https://www.sec.gov/files/company_tickers.json")
        open(cache, "w", encoding="utf-8").write(raw)
    for row in json.loads(raw).values():
        if row["ticker"].upper() == symbol.upper():
            return str(row["cik_str"]).zfill(10)
    raise LookupError(f"no CIK for {symbol}")


def latest_filing(cik, form="10-K"):
    sub = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    rec = sub["filings"]["recent"]
    for i, f in enumerate(rec["form"]):
        if f == form:
            acc = rec["accessionNumber"][i]
            return {"form": form, "accession": acc, "filed": rec["filingDate"][i],
                    "doc": rec["primaryDocument"][i], "cik": cik,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{rec['primaryDocument'][i]}"}
    raise LookupError(f"no {form} filing for CIK {cik}")


def _section(text, start_pat, end_pat, min_chars=1500):
    """The real section, not its table-of-contents entry or a later cross-reference to it. Every heading hit is
    tried: the first one that runs to an explicit end heading and carries real text wins, else the longest body."""
    hits = list(re.finditer(start_pat, text, re.IGNORECASE))
    cands = []
    for m in hits:
        rest = text[m.end():]
        e = re.search(end_pat, rest, re.IGNORECASE)
        cands.append((m, rest[: e.start()] if e else rest, e is not None))
    for m, body, closed in cands:
        if closed and len(body) >= min_chars:
            return m, body
    best = max(cands, key=lambda c: len(c[1]), default=(None, ""))
    return best[0], best[1]


def risk_factors(symbol, max_sentences=24):
    """The 10-K's Item 1A, from EDGAR, with the filing it came from. Returns (sentences, citation)."""
    cik = cik_for(symbol)
    f = latest_filing(cik)
    cache = os.path.join(DOCS, f"{symbol.upper()}-{f['accession']}.txt")
    if os.path.exists(cache):
        text = open(cache, encoding="utf-8").read()
    else:
        text = to_text(_get(f["url"]))
        os.makedirs(DOCS, exist_ok=True)
        open(cache, "w", encoding="utf-8").write(text)
    m, body = _section(text, r"item\s*1a\.?\s*risk\s+factors", r"item\s*1b\.?")
    if body.strip():
        body = body[:60000]
    else:  # some filers render the heading as an image; fall back to the document head rather than pretend
        body = text[:20000]
    sents = sentences(body)
    citation = {"symbol": symbol.upper(), "form": f["form"], "filed": f["filed"], "accession": f["accession"],
                "url": f["url"], "section": "Item 1A Risk Factors" if m else "document text (Item 1A not found)"}
    return sents[:max_sentences], citation


def from_text(text, max_sentences=24):
    return sentences(to_text(text) if "<" in text[:2000] else text)[:max_sentences], {"source": "pasted text"}


def from_file(path, max_sentences=24):
    return from_text(open(path, encoding="utf-8").read(), max_sentences)


# ---------- the instrument ----------

REWRITE_PROMPT = ("Rewrite this sentence in the language of money. One sentence, same facts, no advice.\n\n"
                  "Sentence: %s\n\nRewrite:")


class Lens:
    """Three rows per sentence, one set of pushes: the person, an unsteered baseline, the same set reversed."""

    def __init__(self, rt, pushes):
        self.rt = rt
        self.pushes = [(p["layer"], p["n"], p["dose"], p["mode"]) for p in pushes]

    # token ids for a word list, in the forms a tokenizer will actually use mid-sentence
    def lex(self, words):
        ids, forms = set(), {}
        for w in words:
            for form in {w, " " + w, w.capitalize(), " " + w.capitalize()}:
                got = self.rt.tok.encode(form, add_special_tokens=False)
                if len(got) == 1:  # a multi-token word would bias the mass towards its first piece
                    ids.add(int(got[0]))
                    forms.setdefault(int(got[0]), form.strip())
        return sorted(ids), forms

    def _rows(self, sentence):
        msgs = [{"role": "user", "content": REWRITE_PROMPT % sentence}]
        return [msgs, msgs, msgs]

    def score(self, sents):
        """One forward pass per sentence for all three rows, two lexicons measured at once."""
        money_ids, _ = self.lex(MONEY_WORDS)
        neutral_ids, _ = self.lex(NEUTRAL_WORDS)
        sets = {"money": money_ids, "neutral": neutral_ids}
        out = []
        for i, s in enumerate(sents):
            rows = self.rt.score_multi(self.pushes, self._rows(s), lex_sets=sets)
            rec = {"i": i, "text": s,
                   "money": [r["mass"]["money"] for r in rows],
                   "neutral": [r["mass"]["neutral"] for r in rows],
                   "next": [r["top"][0][0] if r["top"] else "" for r in rows],
                   "money_top": [[w for w, _ in r["lex"]["money"][:3]] for r in rows]}
            rec["d_money"] = round(rec["money"][0] - rec["money"][1], 3)
            rec["d_neutral"] = round(rec["neutral"][0] - rec["neutral"][1], 3)
            rec["amp"] = round(rec["d_money"] - rec["d_neutral"], 3)
            out.append(rec)
        return out

    def rewrite(self, sentence, max_new=64, fire_rows=(0,), cancel=None):
        """Greedy rewrites for one sentence, streamed: ("tok", row, delta), ("fire", row, pos, layer, pairs)."""
        for ev in self.rt.steer_multi(self.pushes, self._rows(sentence), max_new,
                                      cancel=cancel, fire_rows=fire_rows):
            yield ev


def money_density(text):
    """Money words per 100 words of a rewrite, case-insensitive, listed so a reader can check the count."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    hits = [w for w in words if w.rstrip("s") in {m.rstrip("s") for m in MONEY_WORDS}]
    return round(100.0 * len(hits) / max(1, len(words)), 1), hits


def read_document(rt, pushes, sents, citation, top=6, max_new=64, cancel=None):
    """The whole reading, as events. Yields dicts the server forwards as SSE and the CLI prints.

    stage=doc      citation, how many sentences were read
    stage=scored   every sentence with its measured masses, ranked by amplification
    stage=start    the lens starts rewriting one sentence (i, text)
    stage=token    a delta of that rewrite (row, t)   row 0 person, 1 baseline, 2 anti
    stage=fire     what fired for the token just written (same shape the chat uses)
    stage=rewrite  that sentence's three rewrites, with money density and the words that tripped it
    stage=done     totals, and the ranking as it stands
    """
    lens = Lens(rt, pushes)
    t0 = time.perf_counter()
    yield {"stage": "doc", "citation": citation, "sentences": len(sents)}
    scored = lens.score(sents)
    for rec in scored:
        rec["share"] = round(100.0 * rec["amp"] / max(1e-6, abs(rec["amp"]) + abs(rec["d_neutral"])), 1)
    ranked = sorted(scored, key=lambda r: -r["amp"])
    yield {"stage": "scored", "ranked": ranked, "ms": round((time.perf_counter() - t0) * 1000)}

    picked = ranked[: max(0, int(top))]
    out = []
    for rec in picked:
        yield {"stage": "start", "i": rec["i"], "text": rec["text"]}
        texts, sent = ["", "", ""], ["", "", ""]
        for ev in lens.rewrite(rec["text"], max_new=max_new, fire_rows=(0,), cancel=cancel):
            kind = ev[0]
            if kind == "tok":
                _, row, delta = ev
                texts[row] += delta
                sent[row] = texts[row]
                yield {"stage": "token", "row": row, "t": delta}
            elif kind == "fire":
                _, row, pos, layer, pairs = ev
                yield {"stage": "fire", "fire": {"r": row, "pos": pos, "layer": layer, "u": pairs}}
            else:
                break
        dens, hits = money_density(texts[0])
        base_dens, _ = money_density(texts[1])
        rec.update(rewrites=texts, density=dens, baseline_density=base_dens, hits=hits[:12],
                   source_density=money_density(rec["text"])[0])
        out.append(rec)
        yield {"stage": "rewrite", "rec": rec}

    yield {"stage": "done", "ranked": ranked, "read": out, "citation": citation,
           "sentences": len(sents), "ms": round((time.perf_counter() - t0) * 1000)}


def report_markdown(ranked, read, citation, person, pushes, meta):
    """The durable artifact: every number with where it came from and what it does not mean."""
    L = []
    title = citation.get("symbol") or meta.get("source") or "document"
    L.append(f"# Money lens — {title}\n")
    L.append(f"Person: **{person}** ({len(pushes)} pushes: "
             + ", ".join(f"L{p['layer']}·{p['n']} {p['dose']:g}×" for p in pushes) + ")\n")
    L.append(f"Read {len(ranked)} sentences, rewrote the top {len(read)}.\n")
    src = citation.get("url") or citation.get("source", "")
    L.append(f"Source: {citation.get('form', 'text')} filed {citation.get('filed', '')} "
             f"— {citation.get('section', '')}\n{src}\n")
    L.append("""
## What this is, and what it is not

This is a reading instrument. It measures, per sentence, how much probability the Money person's
next-token distribution puts on a fixed 80-word money lexicon compared with the same model unsteered, and it
shows you the rewrites that difference is made of. `amp` is that difference minus the same difference for a
neutral lexicon of function words, in nats, so a person that merely sharpens every distribution does not score
as a person that money-ised one sentence in particular.

It is not a forecast, a rating, or investment advice, and the rewrites are not analysis: they are the persona's
prose, kept verbatim so you can see the register it imposes. `mass` moving is a fact about the model; whether a
risk factor is underpriced is not something this instrument can see.
""")
    L.append("## Ranking\n")
    L.append("| # | amp | d(money) | d(neutral) | source $/100 | rewrite $/100 | next token (person) | sentence |")
    L.append("|---|-----|----------|------------|--------------|---------------|--------------------|----------|\n")
    for k, r in enumerate(ranked):
        rew = f"{r['density']:.1f}" if "density" in r else "—"
        L.append(f"| {k + 1} | {r['amp']:+.2f} | {r['d_money']:+.2f} | {r['d_neutral']:+.2f} | "
                 f"{money_density(r['text'])[0]:.1f} | {rew} | `{r['next'][0]}` | {r['text'][:160]} |")
    L.append("\n## The rewrites the ranking asked for\n")
    for r in read:
        L.append(f"### Sentence {r['i'] + 1} · amp {r['amp']:+.2f}\n")
        L.append(f"**Original** ({r['source_density']:.1f} money words per 100)\n\n> {r['text']}\n")
        L.append(f"**Money** ({r['density']:.1f} per 100; {', '.join(r['hits'][:8]) or 'no money words'})\n\n"
                 f"> {r['rewrites'][0]}\n")
        L.append(f"**Baseline** ({r['baseline_density']:.1f} per 100)\n\n> {r['rewrites'][1]}\n")
        L.append(f"**Anti**\n\n> {r['rewrites'][2]}\n")
    return "\n".join(L)
