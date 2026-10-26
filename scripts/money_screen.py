"""Screen public data for stocks that look asleep rather than broken.

The five criteria are written down, thresholded, and each number in the output carries the source it came from, so
the screen can be argued with rather than believed. It sorts names; it does not rate them, and nothing here is
investment advice.

Criteria (all measured, none of them a hunch):

  1. revenue growing      latest annual revenue > prior annual revenue by at least 5%   SEC XBRL company frames
  2. price asleep         last close in the bottom quarter of its own 52-week range,
                          and at least 30% below the 52-week high                       Nasdaq historical quotes
  3. cheap against self   current price/sales <= 0.7 x the median price/sales at its own
                          fiscal year ends over the last six years                      Nasdaq prices + SEC frames
  4. insider buying       net open-market purchases (Form 4 code P) over the last 90
                          days are positive                                            SEC EDGAR Form 4
  5. volume waking up     20-day average volume >= 1.3 x the 100-day average, while the
                          60-day price change is within +/-10%                         Nasdaq historical quotes

Universe: US-listed, market cap $150M-$3B, financials excluded (P/S and revenue growth do not mean the same thing
for a bank), analyst coverage <= 2 or unreported (Yahoo Finance). Every stage's count is printed, so a name that
never reaches the report is accounted for.

Usage:
    uv run python scripts/money_screen.py                     # default funnel, writes reports/screen-<date>.md
    uv run python scripts/money_screen.py --max-cap 3e9 --min-score 3 --top 25
"""
import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "explorer", "cache", "screen")
REPORTS = os.path.join(ROOT, "reports")
UA_SEC = os.environ.get("EDGAR_UA", "neuron-explorer money screen clayton@learnwithclay.ai")
UA_NAS = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

_sec_lock = threading.Lock()
_sec_last = [0.0]


def _get(url, ua, tries=3, timeout=40):
    for i in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "application/json, text/html;q=0.9",
                                                   "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def sec_json(path):
    """SEC allows 10 requests a second; one lock and a floor between calls keeps us well inside it."""
    with _sec_lock:
        gap = 0.12 - (time.time() - _sec_last[0])
        if gap > 0:
            time.sleep(gap)
        _sec_last[0] = time.time()
    return json.loads(_get("https://data.sec.gov" + path, UA_SEC))


def nasdaq_json(path):
    return json.loads(_get("https://api.nasdaq.com" + path, UA_NAS))


def _cache_path(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, re.sub(r"[^A-Za-z0-9_.-]+", "_", name))


def cached(name, fn, ttl=6 * 3600):
    p = _cache_path(name)
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < ttl:
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    v = fn()
    tmp = p + f".{os.getpid()}.{threading.get_ident()}.tmp"  # a shared tmp name corrupts under the thread pool
    json.dump(v, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, p)
    return v


# ---------- sources ----------

def universe(min_cap, max_cap):
    """Nasdaq's own screener list: every listed stock with a market cap, which is the cheapest honest universe."""
    d = cached("nasdaq_screener.json", lambda: nasdaq_json(
        "/api/screener/stocks?tableonly=true&limit=25000&offset=0&download=true"), ttl=12 * 3600)
    rows = d["data"]["rows"]
    out = []
    for r in rows:
        try:
            cap = float(str(r.get("marketCap", "")).replace(",", ""))
        except ValueError:
            continue
        if not (min_cap <= cap <= max_cap):
            continue
        if (r.get("country") or "").strip() != "United States":
            continue
        if (r.get("sector") or "").strip() in ("Finance", "Real Estate"):
            continue  # price/sales and revenue growth do not describe a bank or a REIT
        name = (r.get("name") or "").lower()
        if any(w in name for w in ("preferred", "depositary", "warrant", " notes due", "rights", "%")):
            continue  # not common equity: the share count and the price are a different instrument
        if re.search(r"\bunits?\b", name):
            continue
        if not re.fullmatch(r"[A-Z]{1,5}", r.get("symbol", "")):  # warrants, units, preferreds
            continue
        out.append({"symbol": r["symbol"], "name": r.get("name") or r["symbol"], "cap": cap,
                    "sector": r.get("sector") or "", "industry": r.get("industry") or "",
                    "last": _num(r.get("lastsale")), "volume": _num(r.get("volume"))})
    return out


def _num(s):
    if s in (None, "", "N/A"):
        return None
    try:
        return float(str(s).replace(",", "").replace("$", "").replace("%", ""))
    except ValueError:
        return None


def cik_map():
    """symbol -> CIK, from the SEC's own ticker file."""
    d = cached("company_tickers.json", lambda: json.loads(_get(
        "https://www.sec.gov/files/company_tickers.json", UA_SEC)), ttl=24 * 3600)
    return {r["ticker"].upper(): str(r["cik_str"]).zfill(10) for r in d.values()}


def frames(concept, ccp, taxonomy="us-gaap", unit=None):
    """One calendar period of one XBRL concept for every filer at once: {cik: value}.

    The unit is part of the path and not optional (USD for money, shares for counts); leaving it out is a 404, not
    an empty result. An empty period is not cached: a missing period is exactly what a later run must retry.
    """
    unit = unit or ("shares" if taxonomy == "dei" else "USD")
    key = f"frames_{taxonomy}_{concept}_{unit}_{ccp}.json"
    p = _cache_path(key)
    d = None
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < 12 * 3600:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            d = None
    if d is None:
        try:
            d = sec_json(f"/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{ccp}.json")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            return {}
        json.dump(d, open(p + ".tmp", "w", encoding="utf-8"))
        os.replace(p + ".tmp", p)
    out = {}
    for row in d.get("data", []):
        out[str(row["cik"]).zfill(10)] = float(row["val"])
    return out


def revenue(cik, year):
    """Largest of the two revenue tags a filer may use. Different filers pick different ones."""
    a = frames("Revenues", f"CY{year}").get(cik)
    b = frames("RevenueFromContractWithCustomerExcludingAssessedTax", f"CY{year}").get(cik)
    vals = [v for v in (a, b) if v]
    return max(vals) if vals else None


def shares(cik, year):
    """Shares outstanding from the cover page instant closest to that year's Q3."""
    for q in ("Q3I", "Q2I", "Q4I", "Q1I"):
        v = frames("EntityCommonStockSharesOutstanding", f"CY{year}{q}", "dei").get(cik)
        if v:
            return v
    return None


def prices(symbol, years=5):
    """Daily closes and volumes from Nasdaq's historical endpoint: one call for the whole window."""
    to = date.today()
    frm = to - timedelta(days=int(365.25 * years))
    q = urllib.parse.urlencode({"assetclass": "stocks", "fromdate": frm.isoformat(), "todate": to.isoformat(),
                                "limit": int(1400 * years / 5) + 100})
    d = cached(f"px_{symbol}.json", lambda: nasdaq_json(f"/api/quote/{symbol}/historical?{q}"), ttl=12 * 3600)
    rows = ((d.get("data") or {}).get("tradesTable") or {}).get("rows") or []
    out = []
    for r in rows:
        try:
            out.append({"d": datetime.strptime(r["date"], "%m/%d/%Y").date(), "c": _num(r["close"]),
                        "v": _num(r["volume"])})
        except (ValueError, KeyError):
            continue
    return sorted([x for x in out if x["c"]], key=lambda x: x["d"])


def analyst_coverage(symbol):
    """Yahoo Finance, via yfinance if it is installed. None means the field was not reported, which is not the same
    as zero coverage: a name nobody writes about and a name the source has no record for both land here."""
    def fetch():
        try:
            import yfinance as yf
        except ImportError:
            return {"n": None}
        try:
            n = yf.Ticker(symbol).info.get("numberOfAnalystOpinions")
            return {"n": int(n) if n is not None else None}
        except Exception:
            return {"n": None}
    return cached(f"cov_{symbol}.json", fetch, ttl=12 * 3600).get("n")


def form4_net_buying(cik, days=90):
    """Net open-market purchases from Form 4 filings in the window. Returns (net_usd, [(date, insider, usd)]).

    Only transaction code P (open-market or private purchase) counts as buying, and only S (sale) offsets it;
    option exercises (M) and grants (A) are not purchases and are ignored, which is the difference between an
    insider choosing to put money in and an insider being paid in stock.
    """
    try:
        sub = cached(f"sub_{cik}.json", lambda: sec_json(f"/submissions/CIK{cik}.json"), ttl=12 * 3600)
    except Exception:
        return None, []
    rec = sub["filings"]["recent"]
    cutoff = date.today() - timedelta(days=days)
    hits = []
    for i, form in enumerate(rec["form"]):
        if form != "4":
            continue
        filed = rec["filingDate"][i]
        try:
            if datetime.strptime(filed, "%Y-%m-%d").date() < cutoff:
                continue
        except ValueError:
            continue
        hits.append((filed, rec["accessionNumber"][i], rec["primaryDocument"][i].split("/")[-1]))
    net, detail = 0.0, []
    for filed, acc, doc in hits[:12]:
        # primaryDocument is often "xslF345X06/form4.xml": that path serves rendered HTML, whose transaction tags
        # are HTML tables. The raw XML is the same filename without the xsl directory.
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
        try:
            xml = _get(url, UA_SEC)
            time.sleep(0.12)
        except Exception:
            continue
        for block in re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", xml, re.S):
            code = re.search(r"<transactionCode>\s*([A-Z])\s*</transactionCode>", block)
            sh = re.search(r"<transactionShares>\s*<value>\s*([\d.]+)", block)
            px = re.search(r"<transactionPricePerShare>\s*<value>\s*([\d.]+)", block)
            if not (code and sh):
                continue
            usd = float(sh.group(1)) * (float(px.group(1)) if px else 0.0)
            who = re.search(r"<rptOwnerName>\s*([^<]+)", xml)
            if code.group(1) == "P":
                net += usd
                detail.append((filed, (who.group(1).strip() if who else "insider"), round(usd)))
            elif code.group(1) == "S":
                net -= usd
    return round(net), detail


# ---------- the criteria ----------

def latest_year():
    """The most recent calendar year the SEC frames endpoint has substantially complete annual revenue for.

    The current year is never complete (only the early filers), and treating it as "the" year would silently drop
    95% of the universe as missing revenue. So a year counts only if its filer count is at least half the previous
    year's: the cutoff that separates "some have filed" from "this is the year".
    """
    today = date.today().year
    counts = {}
    for y in (today, today - 1, today - 2, today - 3):
        counts[y] = len(frames("Revenues", f"CY{y}")) + len(
            frames("RevenueFromContractWithCustomerExcludingAssessedTax", f"CY{y}"))
    for y in sorted(counts, reverse=True):
        prev = counts.get(y - 1) or 0
        if counts[y] and (y == min(counts) or counts[y] >= 0.5 * prev):
            return y
    raise SystemExit("no annual revenue frames available")


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def measure(u, cik, year_now, gate=2):
    """Everything the five criteria need for one name. Returns None when the price series is unusable."""
    px = prices(u["symbol"])
    if len(px) < 120:
        return None
    last, closes = px[-1]["c"], [p["c"] for p in px]
    window = closes[-252:] if len(closes) >= 252 else closes
    lo, hi = min(window), max(window)
    span = (last - lo) / (hi - lo) if hi > lo else 0.5

    vols = [p["v"] for p in px if p["v"]]
    v20 = sum(vols[-20:]) / 20 if len(vols) >= 20 else None
    v100 = sum(vols[-100:]) / 100 if len(vols) >= 100 else None
    chg60 = (closes[-1] / closes[-61] - 1) * 100 if len(closes) > 61 else None

    r_now, r_prev = revenue(cik, year_now), revenue(cik, year_now - 1)
    growth = (r_now / r_prev - 1) * 100 if (r_now and r_prev) else None

    # price/sales at each fiscal year end, from that year's own close and share count
    ps_hist = []
    for y in range(year_now - 5, year_now):
        rev, sh = revenue(cik, y), shares(cik, y)
        if not (rev and sh):
            continue
        # a pre-revenue year makes price/sales meaningless (a biotech at $200k of revenue reads as 900x), so a
        # comparison year must be at least a tenth the size of the latest one
        if r_now and rev < 0.1 * r_now:
            continue
        end = date(y, 12, 31)
        cand = [p for p in px if p["d"] <= end]
        if not cand:
            continue
        ps_hist.append(cand[-1]["c"] * sh / rev)
    ps_now = u["cap"] / r_now if r_now else None
    ps_med = median(ps_hist[-5:]) if len(ps_hist) >= 2 else None  # one year is not a history

    cov = analyst_coverage(u["symbol"])
    pre = [bool(growth is not None and growth >= 5.0), bool(span <= 0.25 and last <= 0.7 * hi),
           bool(ps_now and ps_med and ps_now <= 0.7 * ps_med),
           bool(v20 and v100 and v20 / v100 >= 1.3 and chg60 is not None and abs(chg60) <= 10)]
    # Form 4 costs up to a dozen requests per name, so it is fetched only for names already fitting at least
    # `gate` of the other four. That is a cost gate, not a criterion: a name excluded here is reported as such.
    net, buys = form4_net_buying(cik) if sum(pre) >= gate else (None, [])

    c = {
        "revenue_growing": bool(growth is not None and growth >= 5.0),
        "price_asleep": bool(span <= 0.25 and last <= 0.7 * hi),
        "cheap_vs_self": bool(ps_now and ps_med and ps_now <= 0.7 * ps_med),
        "insider_buying": (bool(net > 0) if net is not None else None),  # None = not checked (cost gate)
        "volume_waking": bool(v20 and v100 and v20 / v100 >= 1.3 and chg60 is not None and abs(chg60) <= 10),
    }
    return {"symbol": u["symbol"], "name": u["name"], "cap": u["cap"], "sector": u["sector"],
            "last": last, "lo52": lo, "hi52": hi, "span": round(span, 3), "off_high": round((last / hi - 1) * 100, 1),
            "vol_ratio": round(v20 / v100, 2) if (v20 and v100) else None,
            "chg60": round(chg60, 1) if chg60 is not None else None,
            "revenue": r_now, "revenue_prev": r_prev, "growth": round(growth, 1) if growth is not None else None,
            "ps_now": round(ps_now, 2) if ps_now else None,
            "ps_median": round(ps_med, 2) if ps_med else None,
            "ps_years": len(ps_hist),
            "insider_net": net, "insider_buys": buys[:4], "coverage": cov,
            "criteria": c, "score": sum(1 for v in c.values() if v)}


def run(args):
    print(f"universe: US-listed, market cap ${args.min_cap/1e6:.0f}M-${args.max_cap/1e9:.2f}B, financials excluded")
    uni = universe(args.min_cap, args.max_cap)
    print(f"  {len(uni)} names")
    ciks = cik_map()
    uni = [u for u in uni if u["symbol"] in ciks]
    print(f"  {len(uni)} with an SEC CIK")

    year_now = latest_year()
    # cheapest filter first: revenue must exist and be growing, using two frame calls rather than a call per name
    rev_now = {c: revenue(c, year_now) for c in set(ciks[u["symbol"]] for u in uni)}
    rev_prev = {c: revenue(c, year_now - 1) for c in set(ciks[u["symbol"]] for u in uni)}
    keep = []
    for u in uni:
        c = ciks[u["symbol"]]
        a, b = rev_now.get(c), rev_prev.get(c)
        if a and b and a / b - 1 >= 0.0:
            keep.append(u)
    print(f"  {len(keep)} with revenue reported for CY{year_now - 1} and CY{year_now} and not shrinking")

    random.seed(7)
    random.shuffle(keep)  # so a truncated run is not biased by listing order
    todo = keep[: args.max_names]
    print(f"measuring {len(todo)} (prices, filings, coverage)…")
    out, done = [], [0]
    lock = threading.Lock()

    def one(u):
        try:
            m = measure(u, ciks[u["symbol"]], year_now, args.gate)
        except Exception as e:
            m = None
            if os.environ.get("SCREEN_DEBUG"):
                import traceback
                print(f"    {u['symbol']}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"    {done[0]}/{len(todo)}")
            if m:
                out.append(m)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, todo))

    out.sort(key=lambda r: (-r["score"], -(r["ps_median"] / r["ps_now"] if r["ps_now"] and r["ps_median"] else 0)))
    covered = [r for r in out if r["coverage"] is None or r["coverage"] <= args.max_coverage]
    hits = [r for r in covered if r["score"] >= args.min_score]
    print(f"  {len(out)} measured, {len(covered)} within coverage, {len(hits)} at or above {args.min_score}/5")
    return out, hits, {"universe": len(uni), "with_cik": len(uni), "revenue_ok": len(keep),
                       "measured": len(out), "covered": len(covered), "hits": len(hits), "year": year_now}


def report(out, hits, meta, args):
    today = date.today().isoformat()
    L = [f"# Asleep screen — {today}\n",
         f"Universe: US-listed, market cap ${args.min_cap/1e6:.0f}M–${args.max_cap/1e9:.1f}B, "
         f"financials excluded, analyst coverage ≤{args.max_coverage} or unreported.\n",
         f"Funnel: {meta['universe']} in the universe → {meta['revenue_ok']} with revenue reported and not shrinking "
         f"→ {meta['measured']} measured → {meta['covered']} under the coverage ceiling "
         f"→ **{meta['hits']} meeting ≥{args.min_score} of 5 criteria**.\n"]
    L.append("""
## The criteria, so you can argue with them

| # | What | Test | Source |
|---|------|------|--------|
| 1 | revenue growing | latest annual revenue ≥ 5% above the prior year | SEC XBRL frames, `Revenues` or `RevenueFromContractWithCustomerExcludingAssessedTax` |
| 2 | price asleep | last close in the bottom quarter of the 52-week range, and ≥30% below the 52-week high | Nasdaq historical quotes |
| 3 | cheap against itself | price/sales ≤ 0.7 × the median price/sales at its own year ends, up to 5 years back | Nasdaq prices + SEC share counts and revenue |
| 4 | insider buying | net open-market purchases (Form 4 code P, offset by S) over 90 days > 0 | SEC EDGAR Form 4 |
| 5 | volume waking | 20-day average volume ≥ 1.3 × the 100-day average, with the 60-day price change inside ±10% | Nasdaq historical quotes |

What this screen cannot see: whether the revenue is real, whether the insider is right, whether the price is low
for a reason the filings state plainly, or anything at all that happens after the last filing. A name scoring 5/5
is a name whose public numbers fit a description, not a recommendation, and not investment advice.
""")
    L.append(f"\n## Meeting ≥{args.min_score} of 5\n")
    L.append("| score | symbol | price | off 52w high | rev growth | P/S now | P/S own median | vol 20/100 | 60d | insider 90d | coverage |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in hits[: args.top]:
        L.append(f"| {r['score']}/5 | **{r['symbol']}** | ${r['last']:.2f} | {r['off_high']:+.1f}% | "
                 f"{_pct(r['growth'])} | {_f(r['ps_now'])} | {_f(r['ps_median'])} | {_f(r['vol_ratio'])} | "
                 f"{_pct(r['chg60'])} | {_ins(r['insider_net'])} | {r['coverage'] if r['coverage'] is not None else 'unknown'} |")
    L.append(f"\n## Each name, and which criteria it meets\n")
    for r in hits[: args.top]:
        L.append(f"### {r['symbol']} — {r['name']} ({r['sector']}) · {r['score']}/5\n")
        L.append(f"- market cap ${r['cap']/1e6:,.0f}M · last ${r['last']:.2f} · 52-week ${r['lo52']:.2f}–${r['hi52']:.2f}")
        L.append(f"- revenue: ${_f(r['revenue'])} latest vs ${_f(r['revenue_prev'])} prior ({_pct(r['growth'])})")
        L.append(f"- price/sales: {_f(r['ps_now'])} now vs {_f(r['ps_median'])} median of {r['ps_years']} year ends")
        L.append(f"- price: {r['span']:.2f} of the way up its 52-week range, {r['off_high']:+.1f}% from the high")
        L.append(f"- volume: 20-day is {_f(r['vol_ratio'])}× the 100-day; 60-day change {_pct(r['chg60'])}")
        if r["insider_net"] is None:
            L.append("- insider open-market buying, 90 days: **not checked** — this name met fewer than the gate's "
                     "other criteria, and Form 4 costs a request per filing")
        else:
            buystr = "; ".join(f"{w} {d} ${v:,}" for d, w, v in r["insider_buys"]) or "none in the window"
            L.append(f"- insider open-market buying, 90 days: net ${r['insider_net']:,} — {buystr}")
        L.append("- criteria met: " + ", ".join(k for k, v in r["criteria"].items() if v) or "- none")
        L.append(f"- coverage: {r['coverage'] if r['coverage'] is not None else 'unreported by the source'}\n")
    L.append("""
## Sources

- Nasdaq screener, `api.nasdaq.com/api/screener/stocks`, and Nasdaq historical quotes
  `api.nasdaq.com/api/quote/<sym>/historical`: prices, volumes, market cap, sector.
- SEC XBRL frames, `data.sec.gov/api/xbrl/frames/...`: revenue, share counts, by CIK.
- SEC EDGAR submissions and Form 4 XML, `data.sec.gov/submissions/CIK##########.json` and
  `sec.gov/Archives/edgar/data/...`: insider transactions.
- Yahoo Finance `quoteSummary` through yfinance: analyst coverage, where reported.

Retrieved %s. Cached copies of every response are under `explorer/cache/screen/`.
""" % today)
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, f"screen-{today}.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    json_path = os.path.join(REPORTS, f"screen-{today}.json")
    json.dump({"generated": datetime.now().isoformat(), "meta": meta, "args": vars(args),
               "all": out, "hits": hits}, open(json_path, "w"), indent=1)
    return path, json_path


def _ins(v):
    return "not checked" if v is None else f"${v:,}"


def _f(v):
    return f"{v:,.2f}" if isinstance(v, (int, float)) else "—"


def _pct(v):
    return f"{v:+.1f}%" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cap", type=float, default=150e6)
    ap.add_argument("--max-cap", type=float, default=3e9)
    ap.add_argument("--max-names", type=int, default=260, help="names to measure in depth (prices, Form 4, coverage)")
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-coverage", type=int, default=2,
                    help="analyst coverage ceiling; names above it are outside the universe")
    ap.add_argument("--gate", type=int, default=2,
                    help="fetch Form 4 only for names meeting this many of the other four criteria")
    args = ap.parse_args()
    out, hits, meta = run(args)
    md, js = report(out, hits, meta, args)
    print(f"\n{len(hits)} names at ≥{args.min_score}/5 → {md}\n{js}")


if __name__ == "__main__":
    main()
