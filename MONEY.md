# The money lens, and the asleep screen

Two separate instruments, built on 2026-10-26 from the same starting point: the Money person existed as four
neuron pushes and was being asked to analyse the stock market, which it cannot do. Neither of these asks it to.
Both are on this machine, both cite their sources, and neither is investment advice.

## 1. The money lens — a reading instrument

`explorer/money_lens.py`, `POST /api/money/read`, and the **money** tab in the left panel.

It takes text somebody else wrote — a 10-K's Item 1A risk factors fetched from EDGAR, or anything you paste — and
reads it sentence by sentence with three rows in one batch: the Money person, the unsteered baseline, and the same
four pushes reversed. For every sentence it measures, in nats, the log-probability the row's next-token
distribution puts on a fixed 80-word money lexicon, and the same for a neutral lexicon of function words.

    amp = (mass_money[person] - mass_money[baseline]) - (mass_neutral[person] - mass_neutral[baseline])

The neutral term is the control: a person that just sharpens every distribution raises both masses and scores near
zero. Sentence ranking is by `amp`, and the panel rewrites only the top few, so the completion you read was chosen
by a number rather than by whoever asked first. The rewrites are the persona's prose, shown as evidence of the
register it imposes — the baseline and anti rows sit next to them for comparison.

Measured on the CRAI 10-K (filed 2026-02-26, 24 risk-factor sentences): scoring all 24 sentences took 42 s;
rewriting one took 55 s of the total. The strongest sentence was the hiring risk — the person turned
"recruit and retain qualified employee consultants" into "money-making consultants ... revenue-generating
capacity", money-word density 13.6 per 100 against 3.2 in the source and 6.5 in the baseline. Firing streamed on
the same connection: the pushed neurons (L26·3403, L29·510) show up in the readout along with neurons nobody
pushed (L29·4093, L31·4624).

What it is not: a forecast, a rating, or a reason to buy anything. `mass` moving is a fact about the model.
Whether a risk factor is priced in is not something this instrument can see, and the persona's sentences are not
analysis.

## 2. The asleep screen — public data, written-down criteria

`scripts/money_screen.py`. Run it:

    uv run python scripts/money_screen.py                  # whole funnel, writes reports/screen-<date>.md and .json
    uv run python scripts/money_screen.py --min-score 4    # stricter
    uv run python scripts/money_screen.py --max-coverage 1 # only names with at most one analyst

Universe, criteria, and the reasoning are in the report itself, so they can be argued with:

| # | Criterion | Test | Source |
|---|-----------|------|--------|
| 1 | revenue growing | latest annual revenue ≥ 5% above the prior year | SEC XBRL frames |
| 2 | price asleep | last close in the bottom quarter of the 52-week range and ≥30% below the high | Nasdaq quotes |
| 3 | cheap against itself | P/S ≤ 0.7 × the median of its own P/S at year ends, up to 5 years | Nasdaq + SEC |
| 4 | insider buying | net open-market Form 4 purchases (P offset by S) over 90 days > 0 | SEC EDGAR Form 4 |
| 5 | volume waking | 20-day volume ≥ 1.3 × the 100-day, 60-day price change inside ±10% | Nasdaq quotes |

Cost gates that are not criteria, and are printed as such: Form 4 costs a request per filing, so it is fetched
only for names fitting at least two of the other four; a name below that gate is reported as "not checked", never
as "no buying". A number in the table that came from a lagging source is labelled with that source.

The funnel for the 2026-10-26 run: 1,113 US-listed names at $150M–3B market cap, financials, REITs and
non-common equity excluded → 542 with revenue reported and not shrinking → 542 measured → 128 under the analyst
coverage ceiling (≤2 or unreported) → 13 at ≥3 of 5 criteria. The top name, NATR, meets 4 of 5: revenue +5.7%,
price 56% below its high, P/S 0.45 against its own 0.67 median, and $2.6M of open-market insider buying.

## Pitfalls that cost a run each

- **A sentinel put from a worker thread can overtake the last event.** `q.put_nowait(None)` next to
  `loop.call_soon_threadsafe(q.put_nowait, ev)` ends the stream one event early — `done` and the written report
  are both real, but the client never sees the event, and nothing raises. Every put, sentinel included, goes
  through the loop.
- **The SEC frames path needs the unit.** `/api/xbrl/frames/us-gaap/Revenues/USD/CY2025.json`; drop `USD` and it
  is a 404, which reads exactly like "this period has no data". Cache negatives only as "retry next time".
- **The current calendar year is not the latest complete one.** CY2026 holds ~4% of filers; treating it as "the"
  year drops almost the whole universe as missing revenue. `latest_year()` requires half the prior year's count.
- **Form 4 `primaryDocument` is often `xslF345X06/form4.xml`** — that path serves rendered HTML, whose
  transaction tags are tables, so a regex over it finds nothing and every company silently reports $0 bought. Use
  the raw filename without the xsl directory.
- **Price/sales against a company's own history is noise while it was pre-revenue.** A biotech at $200k of
  revenue reads as 900× and every later year looks cheap. Comparison years must be at least a tenth the size of
  the latest, and one year is not a history.
- **`analyst_coverage` returning `None` is not zero analysts.** It means the source has no record. The screen
  treats it as inside the ceiling and prints "unknown", so nobody mistakes silence for a fact.

## Verification

    uv run python scripts/verify_money.py              # lens against the live server + the newest screen report
    uv run python scripts/verify_money.py --screen-only

The lens half streams a real read of pasted text (no network) and asserts: 400s for a bad slug and for no document,
the full stage sequence including `done`, one score per sentence, `amp == d_money - d_neutral`, the ranking sorted
by `amp`, the three rewritten rows, the person row differing from the baseline, and the report on disk carrying its
ranking table and its "not investment advice" line.

The screen half re-derives every criterion from the numbers the report printed — a row claiming 3/5 must earn its
three from its own figures — and checks the coverage rule and the two-year floor on P/S history.

Browser side (Playwright, system Chrome, headless): open the page, click **money**, type `CRAI`, read. The panel
shows the citation and URL, 24 ranked sentences, two rewrite blocks each with person / baseline / anti and a
density line, a firing readout that advances per token (`token 37 · L31·4624 +12.8 · L29·4093 +11.9`), and zero
console errors. Screenshot: `~/.hermes/cache/scratch/money_lens_ui.png`.
