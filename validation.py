"""
validation.py — Person 3: hypothesis, red-team challenge and falsification.

MAIN FUNCTION (called by app.py, Person 4):

    run_validation(company_data,            # output of research.fetch_company_data()
                   anomalies=None,          # output of analysis.analyze_financials()  (optional)
                   official_rows=None,      # rows of hul_financial_evidence_dataset.csv style file
                   source_register=None,    # rows of hul_source_register.csv style file
                   peer_data=None,          # {"DABUR": fetch_company_data("Dabur"), ...}
                   llm=None,                # GeminiLLM() or None (template mode, no API key needed)
                   search_fn=None)          # optional counter-search function(query) -> [{title, url, excerpt}]
        -> dict   # the Insight Result (see INSIGHT SCHEMA at the bottom of this file)

THE RULE OF THIS MODULE: the LLM writes words, the CODE decides facts.
  1. Every number in the claim must exist in the evidence or in our own calculations.
  2. Every evidence item keeps its source URL and a source tier.
  3. One-offs, restructuring, base effects and source conflicts are checked in code.
  4. Evidence Strength comes from a fixed rubric, not from the LLM's opinion.
  5. If no angle survives the red team, status = "insufficient_evidence".

Run a test (no API key needed):
    python validation.py --data sample_hul_data.json \
        --official hul_financial_evidence_dataset.csv --register hul_source_register.csv
Add --llm to use Gemini 4 Argon (needs GEMINI_API_KEY).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

SCHEMA_VERSION = "1.0"

# Thresholds (one place, so the team can tune them and explain them to judges)
INVESTIGATE_SCORE = 1.5     # robust z-score needed before an angle is investigated
TOP_ANGLES = 3              # investigate at most this many angles
ONE_OFF_RATIO = 2.0         # a value > 2x (or < 0.5x) the median of its series is a one-off suspect
CONFLICT_PCT = 2.0          # two sources disagreeing by more than 2% is a conflict
GAP_PP = 5.0                # revenue-vs-profit growth gap that counts as meaningful (percentage points)

# Materiality: the smallest move that is worth telling a judge about, per test.
# Why we need this: if a company's history is very flat (debtor days 6, 6, 7, 6),
# then a 2-day move gives a huge z-score but is commercially nothing. So an angle
# must be BOTH unusual (score) AND material (move >= this) to be investigated.
MIN_DAYS_MOVE = 5.0         # working-capital style days
MIN_MARGIN_MOVE_PP = 1.5    # operating margin, percentage points year on year
MIN_CASH_RATIO_MOVE = 0.2   # operating cash flow / net profit, vs its prior median

# One-offs (exceptional gains/losses) are a P&L idea. We only run the "2x the median"
# test on these. Cash flow and working-capital days swing for normal business reasons
# (e.g. a jeweller buying gold), and calling that swing a "one-off" was killing the
# very signal the cash-conversion test is looking for.
ONE_OFF_METRICS = ("net_profit", "operating_profit", "revenue_from_operations")

RESTRUCTURING_WORDS = ("demerg", "merger", "acqui", "divest", "discontinued", "restat",
                       "one-off", "one off", "exceptional", "disposal")


# =============================================================================
# 1. SOURCE TIERS
# =============================================================================
EXCHANGE_REGULATOR_HOSTS = ("nseindia.com", "bseindia.com", "sebi.gov.in", "rbi.org.in",
                            "mospi.gov.in", "mca.gov.in")
AGGREGATOR_HOSTS = ("screener.in", "trendlyne.com", "stockanalysis.com", "stockssena.com",
                    "moneycontrol.com", "tickertape.in", "tijorifinance.com")
NEWS_HOSTS = ("yahoo.com", "economictimes", "livemint.com", "business-standard.com",
              "reuters.com", "bloomberg", "thehindubusinessline", "financialexpress",
              "ndtvprofit", "cnbctv18")

# Tier rank: lower = more trustworthy. "filing" = company / exchange / regulator document.
TIER_RANK = {"filing": 1, "aggregator": 2, "news": 3, "other": 4}


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def classify_tier(url: str, company_words: List[str], register_type: str = "") -> str:
    """Decide how trustworthy a source is. The source register (if given) wins."""
    rt = register_type.lower()
    if rt:
        if "official" in rt or "exchange" in rt or "regulator" in rt:
            return "filing"
        if "news" in rt:
            return "news"
    host = _host(url)
    if any(h in host for h in EXCHANGE_REGULATOR_HOSTS):
        return "filing"
    if any(h in host for h in AGGREGATOR_HOSTS):
        return "aggregator"
    if any(h in host for h in NEWS_HOSTS):
        return "news"
    # the company's own investor-relations site, e.g. hul.co.in, dabur.com
    if any(w in host.split(".")[0] or host.startswith(w + ".") for w in company_words):
        return "filing"
    return "other"


# =============================================================================
# 2. PERIOD + METRIC NORMALISATION (so Person 1 and Person 2 data line up)
# =============================================================================
def norm_period(p: str) -> str:
    """'FY2025-26' -> 'FY2026', 'Q3 FY2025-26 (December 2025)' -> 'Q3_FY2026', 'Q3_FY2026' stays."""
    p = str(p).strip()
    m = re.match(r"Q([1-4])[_\s]*FY(\d{4})(?:-(\d{2}))?", p, re.I)
    if m:
        q, y1, y2 = m.groups()
        return f"Q{q}_FY{int(y1) + 1 if y2 else int(y1)}"
    m = re.match(r"FY(\d{4})(?:-(\d{2}))?", p, re.I)
    if m:
        y1, y2 = m.groups()
        return f"FY{int(y1) + 1 if y2 else int(y1)}"
    return p


def period_key(p: str) -> Tuple[int, int]:
    m = re.match(r"Q([1-4])_FY(\d{4})", p)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.match(r"FY(\d{4})", p)
    if m:
        return int(m.group(1)), 5
    return 0, 0


def same_quarter_last_year(p: str) -> Optional[str]:
    m = re.match(r"Q([1-4])_FY(\d{4})", p)
    return f"Q{m.group(1)}_FY{int(m.group(2)) - 1}" if m else None


def next_period(p: str) -> str:
    m = re.match(r"Q([1-4])_FY(\d{4})", p)
    if m:
        q, fy = int(m.group(1)), int(m.group(2))
        return f"Q{q + 1}_FY{fy}" if q < 4 else f"Q1_FY{fy + 1}"
    m = re.match(r"FY(\d{4})", p)
    return f"FY{int(m.group(1)) + 1}" if m else "the next reported period"


# Person 2's metric names -> the same names research.py uses (longest match first)
OFFICIAL_METRIC_MAP = [
    ("profit after tax before exceptional", "net_profit_before_exceptional"),
    ("reported profit after tax", "net_profit"),
    ("underlying volume growth", "underlying_volume_growth_pct"),
    ("underlying sales growth", "underlying_sales_growth_pct"),
    ("cash from operations", "operating_cash_flow"),
    ("ebitda margin", "operating_margin_pct"),
    ("ebitda", "operating_profit"),
    ("turnover", "revenue_from_operations"),
]


def map_official_metric(name: str) -> str:
    low = name.lower().strip()
    for prefix, canon in OFFICIAL_METRIC_MAP:
        if low.startswith(prefix):
            return canon
    return re.sub(r"\W+", "_", low).strip("_")


def scope_of(text: str) -> str:
    low = text.lower()
    if "standalone" in low:
        return "standalone"
    if "consolidated" in low:
        return "consolidated"
    return "unspecified"


# =============================================================================
# 3. EVIDENCE POOL — every fact gets an ID, a source and a tier
# =============================================================================
@dataclass
class Evidence:
    id: str
    company: str
    metric: str
    period: str
    period_type: str
    value: Optional[float]
    unit: str
    source_name: str
    source_url: str
    tier: str
    scope: str
    quote: str = ""
    page: Optional[int] = None
    note: str = ""
    kind: str = "number"      # "number" or "text"
    role: str = ""            # supports / conflicts / context (set when grading)


USEFUL_TEXT_WORDS = ("demand", "growth", "volume", "price", "margin", "receivable", "inventory",
                     "channel", "commerce", "rural", "urban", "distribut", "input cost", "inflation")


def _company_words(company_data: Dict[str, Any]) -> List[str]:
    words = {company_data.get("company", ""), company_data.get("ticker", "")}
    try:  # reuse Person 1's alias table if research.py is importable
        from research import TRACK_C_TICKERS
        words |= {k for k, v in TRACK_C_TICKERS.items() if v == company_data.get("ticker")}
    except Exception:
        pass
    out = set()
    for w in words:
        for part in re.split(r"\W+", str(w).lower()):
            if len(part) >= 3 and part not in ("ltd", "limited", "india", "the", "and"):
                out.add(part)
    return sorted(out)


def build_evidence_pool(company_data: Dict[str, Any],
                        official_rows: Optional[List[Dict[str, Any]]] = None,
                        source_register: Optional[List[Dict[str, Any]]] = None
                        ) -> Tuple[List[Evidence], List[str]]:
    """Turns Person 1 + Person 2 data into one list of Evidence. Returns (pool, data_quality_flags)."""
    pool: List[Evidence] = []
    flags: List[str] = []
    cw = _company_words(company_data)
    reg = {r["source_id"]: r for r in (source_register or [])}
    ticker = company_data.get("ticker", company_data.get("company", ""))

    def add(**kw):
        pool.append(Evidence(id=f"E{len(pool) + 1}", **kw))

    # Person 1's own warnings (source failed, odd periods etc.) go straight to the user
    flags += company_data.get("data_quality_flags", [])

    # (a) Person 1 structured numbers
    mislabeled = 0
    not_comparable = set()
    for r in company_data.get("financial_records", []):
        # A 15 month year, or a year that ends in December, cannot be compared with
        # a normal April-March year. We keep it out of the pool fully, else a
        # year-on-year test would quietly compare apples and oranges.
        months = r.get("period_months", 3 if r.get("period_type") == "quarterly" else 12)
        if months not in (3, 12) or period_key(norm_period(r["period"])) == (0, 0):
            # print the length only when we really know it is odd (e.g. a 15 month
            # year); a 'Sep_2026' half-year column has no length on screener
            not_comparable.add(r["period"] + (f" ({months} months)" if months not in (3, 12) else ""))
            continue
        url = r.get("source_url", "")
        tier = classify_tier(url, cw)
        if tier != "filing" and "filing" in r.get("doc_title", "").lower():
            mislabeled += 1
        add(company=ticker, metric=r["metric"], period=norm_period(r["period"]),
            period_type=r.get("period_type", ""), value=r.get("value"), unit=r.get("unit", ""),
            source_name=f"{_host(url) or 'unknown source'} — {r.get('doc_title', '')}",
            source_url=url, tier=tier,
            scope="consolidated" if "/consolidated/" in url else scope_of(r.get("doc_title", "")),
            quote=r.get("quote", ""), page=r.get("page"))
        if r.get("value") is None:
            flags.append(f"Missing value: {r['metric']} {r['period']} (left empty, not guessed).")
    if not_comparable:
        flags.append(f"Left out of all comparisons (not a normal 12-month April-March year or a "
                     f"3-month quarter): {', '.join(sorted(not_comparable))}.")
    if mislabeled:
        flags.append(f"{mislabeled} records are titled 'Filing' but come from an aggregator "
                     f"({_host(company_data['financial_records'][0]['source_url'])}); treated as aggregator tier.")
    if company_data.get("financial_records") and not any(r.get("quote") for r in company_data["financial_records"]):
        flags.append("Structured records have no page/quote fields, so numbers cannot be quote-checked yet.")

    # (b) Person 2 official, source-linked numbers
    for r in official_rows or []:
        src = reg.get(r.get("source_id", ""), {})
        url = src.get("url", "")
        try:
            val = float(r["value"]) if str(r.get("value", "")).strip() != "" else None
        except ValueError:
            val = None
        unit = "percentage" if r.get("unit", "").strip() == "%" else r.get("unit", "")
        add(company=ticker, metric=map_official_metric(r["metric"]), period=norm_period(r["period"]),
            period_type="quarterly" if r.get("period_type", "").lower().startswith("q") else "annual",
            value=val, unit=unit,
            source_name=f"{src.get('source_name', r.get('source_id', ''))} [{r.get('source_id', '')}]",
            source_url=url, tier=classify_tier(url, cw, src.get("source_type", "")),
            scope=scope_of(r.get("scope_and_definition", "")),
            note=f"{r.get('scope_and_definition', '')} | {r.get('data_quality_note', '')}".strip(" |"))

    # (c) Person 1 text excerpts — keep only ones with real content
    dropped = 0
    for x in company_data.get("text_excerpts", []):
        text = x.get("excerpt", "")
        if len(text) < 80 or not any(w in text.lower() for w in USEFUL_TEXT_WORDS):
            dropped += 1
            continue
        url = x.get("source_url", "")
        add(company=ticker, metric="text", period="", period_type="", value=None, unit="",
            source_name=x.get("doc_title", ""), source_url=url, tier=classify_tier(url, cw),
            scope="unspecified", quote=text, kind="text")
    if dropped:
        flags.append(f"{dropped} search excerpts dropped: they describe a web page, not a fact "
                     f"(need real quotes from transcripts/presentations).")
    return pool, flags


# =============================================================================
# 4. CONTRADICTION DETECTOR — same metric, same period, different sources
# =============================================================================
def find_conflicts(pool: List[Evidence]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Evidence]] = {}
    for e in pool:
        if e.kind == "number" and e.value is not None and e.period:
            groups.setdefault((e.metric, e.period), []).append(e)
    conflicts = []
    for (metric, period), items in groups.items():
        if len({_host(e.source_url) + e.source_name for e in items}) < 2:
            continue
        vals = [e.value for e in items]
        lo, hi = min(vals), max(vals)
        if lo == 0 or (hi - lo) / abs(lo) * 100 <= CONFLICT_PCT:
            continue
        if items[0].unit == "percentage" and hi - lo <= 1.0:
            continue  # one source rounds 23.7% to 23% — rounding, not a conflict
        notes = " ".join(e.note.lower() for e in items)
        basis = (len({e.scope for e in items}) > 1
                 or any(w in notes for w in ("basis", "restat", "continuing", "discontinued", "definition")))
        conflicts.append({
            "metric": metric, "period": period,
            "values": [{"value": e.value, "source": e.source_name, "scope": e.scope, "evidence_id": e.id}
                       for e in items],
            "diff_pct": round((hi - lo) / abs(lo) * 100, 1),
            "type": "basis_or_definition_difference" if basis else "unexplained_conflict",
        })
    return conflicts


# =============================================================================
# 5. SERIES HELPERS + ONE-OFF DETECTION
# =============================================================================
def series(pool: List[Evidence], metric: str, period_type: str) -> Dict[str, Evidence]:
    """One value per period for a metric. Uses the single source that covers the most periods
    (so a series never mixes two sources/bases), and only fills gaps from others."""
    items = [e for e in pool if e.metric == metric and e.kind == "number"
             and e.value is not None and e.period_type == period_type]
    by_source: Dict[str, Dict[str, Evidence]] = {}
    for e in items:
        by_source.setdefault(e.source_name.split(" — ")[0], {})[e.period] = e
    if not by_source:
        return {}
    main = max(by_source.values(), key=len)
    return dict(sorted(main.items(), key=lambda kv: period_key(kv[0])))


def robust_z(latest: float, history: List[float]) -> float:
    """How unusual is `latest` versus its own history? Median/MAD version of a z-score,
    so one crazy quarter in the history does not hide everything else."""
    if len(history) < 2:
        return 0.0
    med = statistics.median(history)
    mad = statistics.median([abs(h - med) for h in history])
    scale = max(1.4826 * mad, 0.05 * abs(med), 1e-9)
    return (latest - med) / scale


def one_off_periods(pool: List[Evidence], metric: str, period_type: str) -> Dict[str, str]:
    """Periods where this metric looks like a one-off, with the reason."""
    out: Dict[str, str] = {}
    s = series(pool, metric, period_type)
    vals = {p: e.value for p, e in s.items()} if metric in ONE_OFF_METRICS else {}
    for p, v in vals.items():
        others = [x for q, x in vals.items() if q != p]
        if len(others) >= 3:
            med = statistics.median(others)
            # ratio only makes sense when both are positive; a loss vs a profit
            # gives a negative "ratio" which is meaningless (-1.9x was being flagged)
            if med <= 0:
                continue
            if v <= 0:   # a loss when the company normally makes a profit -- classic one-off sign
                out[p] = f"{metric} {p} = {v:,.0f} is a loss, while the median of other periods is {med:,.0f}"
                continue
            if v / med >= ONE_OFF_RATIO or v / med <= 1 / ONE_OFF_RATIO:
                out[p] = f"{metric} {p} = {v:,.0f} is {v / med:.1f}x the median of other periods ({med:,.0f})"
    for e in pool:  # notes from Person 2's official dataset
        if (e.metric == metric and e.period_type == period_type
                and any(w in e.note.lower() for w in RESTRUCTURING_WORDS)):
            reason = f"source note: {e.note.split('|')[0].strip()}"
            out[e.period] = (out[e.period] + "; " + reason) if e.period in out else reason
    return out


def value_of(pool: List[Evidence], metric: str, period: str) -> Optional[Evidence]:
    """Best-tier value for metric+period (filing beats aggregator beats news)."""
    c = [e for e in pool if e.metric == metric and e.period == period and e.value is not None]
    return min(c, key=lambda e: TIER_RANK.get(e.tier, 9)) if c else None


# =============================================================================
# 6. FALLBACK ANGLE DETECTOR (used only if Person 2's anomalies are not passed in)
#    Every test is generic: it works for any company with the same record schema.
# =============================================================================
def _pct(a: float, b: float) -> Optional[float]:
    return (a / b - 1) * 100 if b else None


def test_revenue_vs_profit(pool, profit_override: Optional[Dict[str, float]] = None,
                           base_revenue_override: Optional[Dict[str, float]] = None):
    rev = series(pool, "revenue_from_operations", "quarterly")
    np_ = series(pool, "net_profit", "quarterly")
    rows = []
    for p in rev:
        b = same_quarter_last_year(p)
        if b in rev and p in np_ and b in np_:
            base_rev = (base_revenue_override or {}).get(b, rev[b].value)
            prof = (profit_override or {}).get(p, np_[p].value)
            rg, pg = _pct(rev[p].value, base_rev), _pct(prof, np_[b].value)
            rows.append({"period": p, "base": b, "revenue_yoy_pct": round(rg, 1), "profit_yoy_pct": round(pg, 1),
                         "gap_pp": round(rg - pg, 1),
                         "evidence_ids": [rev[p].id, rev[b].id, np_[p].id, np_[b].id]})
    if len(rows) < 3:
        return None
    latest = rows[-1]
    # "Unusual vs own history" only makes sense if the history is NORMAL quarters.
    # A demerger-gain quarter (+120% profit) inside the history makes every later
    # quarter look ordinary. So history rows whose period or base is a detected
    # one-off are left out of the score -- they are still shown in `rows`.
    oneoff = one_off_periods(pool, "net_profit", "quarterly")
    hist_rows = [r for r in rows[:-1] if r["period"] not in oneoff and r["base"] not in oneoff]
    if len(hist_rows) < 2:      # too little clean history left; better than nothing
        hist_rows = rows[:-1]
    hist = [r["gap_pp"] for r in hist_rows]
    persistence = 0
    for r in reversed(rows):
        if r["gap_pp"] * latest["gap_pp"] > 0 and abs(r["gap_pp"]) >= GAP_PP:
            persistence += 1
        else:
            break
    direction = "revenue outgrew profit" if latest["gap_pp"] > 0 else "profit outgrew revenue"
    return {
        "test": "revenue_vs_profit", "title": f"Revenue vs profit growth divergence ({direction})",
        "latest_period": latest["period"], "score": round(abs(robust_z(latest["gap_pp"], hist)), 2),
        "history_left_out": sorted({r["period"] for r in rows[:-1]} - {r["period"] for r in hist_rows}),
        "headline_value": latest["gap_pp"], "persistence": persistence, "rows": rows,
        "material": abs(latest["gap_pp"]) >= GAP_PP, "min_move": GAP_PP, "yoy": True,
        "periods_used": [latest["period"], latest["base"]],
        "metrics_used": ["revenue_from_operations", "net_profit"],
        "evidence_ids": latest["evidence_ids"],
    }


def test_cash_conversion(pool, profit_override: Optional[Dict[str, float]] = None):
    ocf = series(pool, "operating_cash_flow", "annual")
    np_ = series(pool, "net_profit", "annual")
    rows = []
    for p in ocf:
        if p in np_ and np_[p].value:
            prof = (profit_override or {}).get(p, np_[p].value)
            rows.append({"period": p, "ocf": ocf[p].value, "net_profit": prof,
                         "ocf_to_profit": round(ocf[p].value / prof, 2), "evidence_ids": [ocf[p].id, np_[p].id]})
    if len(rows) < 3:
        return None
    latest, hist = rows[-1], [r["ocf_to_profit"] for r in rows[:-1]]
    med = statistics.median(hist)
    persistence = 0
    for r in reversed(rows):
        if (r["ocf_to_profit"] < med) == (latest["ocf_to_profit"] < med):
            persistence += 1
        else:
            break
    return {
        "test": "cash_conversion", "title": "Operating cash flow vs net profit (cash conversion)",
        "latest_period": latest["period"], "score": round(abs(robust_z(latest["ocf_to_profit"], hist)), 2),
        "headline_value": latest["ocf_to_profit"], "history_median": round(med, 2),
        "material": abs(latest["ocf_to_profit"] - med) >= MIN_CASH_RATIO_MOVE,
        "min_move": MIN_CASH_RATIO_MOVE, "yoy": False,
        "persistence": persistence, "rows": rows, "periods_used": [latest["period"]],
        "metrics_used": ["operating_cash_flow", "net_profit"], "evidence_ids": latest["evidence_ids"],
    }


def test_level_trend(pool, metric: str, period_type: str, title: str, min_move: float):
    """Latest value vs the median of its own history. Use this for ANNUAL data only:
    for quarters the history mixes seasons, so use test_yoy_change instead."""
    s = series(pool, metric, period_type)
    if len(s) < 4:
        return None
    periods, vals = list(s), [e.value for e in s.values()]
    hist = vals[:-1]
    med = statistics.median(hist)
    persistence = 0
    for v in reversed(vals):
        if (v > med) == (vals[-1] > med) and v != med:
            persistence += 1
        else:
            break
    return {
        "test": f"trend_{metric}", "title": title, "latest_period": periods[-1],
        "score": round(abs(robust_z(vals[-1], hist)), 2), "headline_value": vals[-1],
        "history_median": med, "persistence": persistence,
        "material": abs(vals[-1] - med) >= min_move, "min_move": min_move,
        "yoy": False, "unit": s[periods[-1]].unit,
        "rows": [{"period": p, "value": e.value, "evidence_ids": [e.id]} for p, e in s.items()],
        "periods_used": [periods[-1]], "metrics_used": [metric], "evidence_ids": [s[periods[-1]].id],
    }


def test_yoy_change(pool, metric: str, title: str, min_move: float):
    """Quarterly level metric (like margin %): change vs the SAME quarter last year,
    scored against the history of such changes. Comparing Q1 with Q1 only is what
    actually takes care of seasonality -- e.g. a festive quarter always has a better margin."""
    s = series(pool, metric, "quarterly")
    rows = []
    for p in s:
        b = same_quarter_last_year(p)
        if b in s:
            rows.append({"period": p, "base": b, "value": s[p].value, "base_value": s[b].value,
                         "change_pp": round(s[p].value - s[b].value, 1),
                         "evidence_ids": [s[p].id, s[b].id]})
    if len(rows) < 4:
        return None
    latest, hist = rows[-1], [r["change_pp"] for r in rows[:-1]]
    persistence = 0
    for r in reversed(rows):
        if r["change_pp"] * latest["change_pp"] > 0 and abs(r["change_pp"]) >= min_move:
            persistence += 1
        else:
            break
    return {
        "test": f"yoy_{metric}", "title": title, "latest_period": latest["period"],
        "score": round(abs(robust_z(latest["change_pp"], hist)), 2),
        "headline_value": latest["change_pp"], "history_median": round(statistics.median(hist), 1),
        "persistence": persistence, "material": abs(latest["change_pp"]) >= min_move,
        "min_move": min_move, "yoy": True, "unit": s[latest["period"]].unit, "rows": rows,
        "periods_used": [latest["period"], latest["base"]], "metrics_used": [metric],
        "evidence_ids": latest["evidence_ids"],
    }


def detect_angles(pool: List[Evidence]) -> List[Dict[str, Any]]:
    tests = [
        test_revenue_vs_profit(pool),
        test_cash_conversion(pool),
        test_level_trend(pool, "debtor_days", "annual", "Receivable (debtor) days vs own history", MIN_DAYS_MOVE),
        test_level_trend(pool, "inventory_days", "annual", "Inventory days vs own history", MIN_DAYS_MOVE),
        test_yoy_change(pool, "operating_margin_pct", "Operating margin vs same quarter last year",
                        MIN_MARGIN_MOVE_PP),
        test_level_trend(pool, "working_capital_days", "annual", "Working-capital days vs own history",
                         MIN_DAYS_MOVE),
    ]
    angles = [t for t in tests if t]
    for i, a in enumerate(sorted(angles, key=lambda a: -a["score"]), 1):
        a["id"] = f"A{i}"
    return sorted(angles, key=lambda a: -a["score"])


RETEST = {
    "revenue_vs_profit": lambda pool, **kw: test_revenue_vs_profit(pool, **kw),
    "cash_conversion": lambda pool, **kw: test_cash_conversion(pool, profit_override=kw.get("profit_override")),
}


# =============================================================================
# 7. RED TEAM — code checks that try to kill the angle
#    result: "survived" | "weakened" | "fatal" | "not_run"
# =============================================================================
def red_team(angle: Dict[str, Any], pool: List[Evidence], conflicts: List[Dict[str, Any]],
             peer_pools: Dict[str, List[Evidence]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    test = angle["test"]
    periods = set(angle["periods_used"])
    ptype = "quarterly" if any(p.startswith("Q") for p in periods) else "annual"

    # (1) One-offs / exceptional items in the periods the claim uses
    oneoffs = {}
    for m in angle["metrics_used"]:
        for p, why in one_off_periods(pool, m, ptype).items():
            if p in periods:
                oneoffs[(m, p)] = why
    # annual figures also inherit one-offs from their quarters (Q3_FY2026 -> FY2026)
    if ptype == "annual":
        for m in angle["metrics_used"]:
            for p, why in one_off_periods(pool, m, "quarterly").items():
                if "_FY" not in p:
                    continue
                fy = "FY" + p.split("_FY")[1]
                if fy in periods:
                    prev = oneoffs.get((m, fy))
                    oneoffs[(m, fy)] = (prev + "; " if prev else "") + f"contains quarter {p}: {why}"

    if not oneoffs:
        checks.append({"check": "one_off", "argument": "A one-off gain or loss could be driving this.",
                       "result": "survived", "detail": "No one-off or exceptional item found in the periods used."})
    else:
        # recompute with profit before exceptional items where an official figure exists
        override = {p: value_of(pool, "net_profit_before_exceptional", p).value
                    for (m, p) in oneoffs if m == "net_profit" and value_of(pool, "net_profit_before_exceptional", p)}
        detail = "; ".join(oneoffs.values())
        if override and test in RETEST:
            redo = RETEST[test](pool, profit_override=override)
            if redo and abs(redo["score"]) < INVESTIGATE_SCORE:
                checks.append({"check": "one_off", "argument": "A one-off gain or loss could be driving this.",
                               "result": "fatal",
                               "detail": f"{detail}. Recomputed with profit before exceptional items "
                                         f"({', '.join(f'{p}: {v:,.0f}' for p, v in override.items())}): "
                                         f"value becomes {redo['headline_value']} and unusualness score drops "
                                         f"from {angle['score']} to {redo['score']}. The pattern disappears."})
                angle["adjusted"] = redo
            else:
                checks.append({"check": "one_off", "argument": "A one-off gain or loss could be driving this.",
                               "result": "survived",
                               "detail": f"{detail}. Recomputed without it and the pattern still stands "
                                         f"(score {redo['score'] if redo else 'n/a'})."})
                if redo:
                    angle["adjusted"] = redo
        else:
            checks.append({"check": "one_off", "argument": "A one-off gain or loss could be driving this.",
                           "result": "fatal",
                           "detail": f"{detail}. No 'before exceptional items' figure available to recompute, "
                                     f"so the pattern cannot be separated from the one-off."})

    # (2) Base effect: is the comparison base itself unusual?
    base_hit = False
    for m in angle["metrics_used"]:
        for p in periods - {angle["latest_period"]}:
            if p in one_off_periods(pool, m, ptype):
                base_hit = True
                checks.append({"check": "base_effect", "argument": "The comparison period may itself be unusual.",
                               "result": "weakened", "detail": f"Base period {p} for {m} looks unusual."})
    if not base_hit:
        checks.append({"check": "base_effect", "argument": "The comparison period may itself be unusual.",
                       "result": "survived", "detail": "Base period values are in line with their own history."})

    # (3) Source conflicts / basis changes on the numbers we use — with a sensitivity test
    hits = [c for c in conflicts if c["metric"] in angle["metrics_used"] and c["period"] in periods]
    if not hits:
        checks.append({"check": "source_conflict", "argument": "Different sources may report different numbers.",
                       "result": "survived" if any(c for c in conflicts) or pool else "not_run",
                       "detail": "No conflicting values found for the periods and metrics used."})
    for c in hits:
        desc = ", ".join(f"{v['value']:,.0f} ({v['source'].split(' — ')[0]})" for v in c["values"])
        result, extra = "weakened", ""
        if test == "revenue_vs_profit" and c["metric"] == "revenue_from_operations" and c["period"] != angle["latest_period"]:
            alt = max(v["value"] for v in c["values"])     # worst case for the claim: highest base
            redo = test_revenue_vs_profit(pool, base_revenue_override={c["period"]: alt})
            if redo:
                still = abs(redo["headline_value"]) >= GAP_PP
                result = "survived" if still else "fatal"
                extra = (f" Sensitivity test: using the other base value {alt:,.0f}, the gap becomes "
                         f"{redo['headline_value']} pp — the claim {'still holds' if still else 'no longer holds'}.")
        checks.append({"check": "source_conflict", "argument": "Different sources may report different numbers.",
                       "result": result,
                       "detail": f"{c['metric']} {c['period']}: {desc} — {c['diff_pct']}% apart "
                                 f"({c['type'].replace('_', ' ')}).{extra}"})

    # (4) Restructuring (merger / demerger / divestment) anywhere in the data window
    notes = " ".join(e.note.lower() + " " + e.quote.lower() for e in pool
                     if e.metric in angle["metrics_used"] or e.kind == "text")
    labels = {r"demerg": "demerger", r"\bmerger": "merger", r"\bacqui": "acquisition",
              r"divest": "divestment", r"discontinued": "discontinued business", r"restat": "restated figures"}
    found = sorted({lab for pat, lab in labels.items() if re.search(pat, notes)})
    checks.append({"check": "restructuring",
                   "argument": "A merger, demerger or divestment may have changed what the numbers include.",
                   "result": "weakened" if found else "survived",
                   "detail": (f"Sources mention: {', '.join(found)}. Year-on-year comparisons across this event "
                              f"may compare different business perimeters.") if found
                   else "No restructuring mentioned in the collected sources."})

    # (5) Seasonality -- we only say "controlled" when the test really compared Q with same Q
    if ptype == "annual":
        season = ("survived", "Annual data: seasonality does not apply.")
    elif angle.get("yoy"):
        season = ("survived", "Controlled by design: quarters are compared with the same quarter last year.")
    else:
        season = ("weakened", "Latest quarter is compared with all quarters, so a seasonal swing is not ruled out.")
    checks.append({"check": "seasonality", "argument": "This could just be a seasonal pattern.",
                   "result": season[0], "detail": season[1]})

    # (6) Peer control
    if not peer_pools:
        peer = {"result": "not_run", "detail": "No peer data supplied, so a sector-wide explanation cannot be ruled out."}
    else:
        same, tested = [], []
        for name, ppool in peer_pools.items():
            pa = {a["test"]: a for a in detect_angles(ppool)}.get(test)
            if pa:
                tested.append(name)
                if pa["score"] >= INVESTIGATE_SCORE and (pa["headline_value"] > 0) == (angle["headline_value"] > 0):
                    same.append(name)
        if not tested:
            peer = {"result": "not_run", "detail": "Peers lacked the data for this test."}
        elif len(same) >= max(1, len(tested) / 2):
            peer = {"result": "sector_wide", "detail": f"Same pattern at {', '.join(same)} (of {', '.join(tested)})."}
        else:
            peer = {"result": "company_specific",
                    "detail": f"Pattern not seen at {', '.join(n for n in tested if n not in same)}."}
    checks.append({"check": "peer_control", "argument": "Peers may show the same pattern (sector story).",
                   "result": {"company_specific": "survived", "sector_wide": "fatal"}.get(peer["result"], "not_run"),
                   "detail": peer["detail"]})

    verdict = {"fatal": any(c["result"] == "fatal" for c in checks),
               "weakened": sum(c["result"] == "weakened" for c in checks)}
    return checks, peer, verdict


# =============================================================================
# 8. STRENGTH RUBRIC (fixed rules — explain these to the judges)
# =============================================================================
def grade(angle: Dict[str, Any], pool: List[Evidence], peer: Dict[str, Any],
          checks: List[Dict[str, Any]]) -> Tuple[str, List[str], List[Evidence]]:
    ids = set(angle["evidence_ids"])
    used = [e for e in pool if e.id in ids]
    # independent support: the same metric/period confirmed elsewhere, plus relevant text
    roles = {e.id: "supports" for e in used}
    for e in list(used):
        for o in pool:
            if o.metric == e.metric and o.period == e.period and o.id not in roles and o.value is not None:
                agree = e.value and abs(o.value - e.value) / abs(e.value) * 100 <= CONFLICT_PCT
                roles[o.id] = "supports" if agree else "conflicts"
                used.append(o)
    for e in pool:
        if e.kind == "text":
            roles[e.id] = "context"
            used.append(e)
    for e in used:
        e.role = roles[e.id]
    support = [e for e in used if roles[e.id] == "supports"]
    hosts = {_host(e.source_url) for e in support if e.source_url}
    tiers = {e.tier for e in support}
    weakened = sum(c["result"] == "weakened" for c in checks)
    reasons = [
        f"{len(hosts)} independent source(s) ({', '.join(sorted(hosts)) or 'none'}) across "
        f"{len(tiers)} type(s) ({', '.join(sorted(tiers))}). Needed for Strong: 3 sources, 2 types.",
        f"Pattern holds for {angle['persistence']} consecutive period(s). Needed for Strong: 3.",
        f"Peer control: {peer['result'].replace('_', ' ')}. Needed for Strong: company-specific.",
        f"{weakened} counterargument(s) weakened the finding.",
    ]
    if any(roles[e.id] == "conflicts" for e in used):
        reasons.append("Some sources disagree with the numbers used (see evidence rows marked 'conflicts').")
    if "filing" not in tiers:
        reasons.append("No company filing or exchange document confirms the key numbers yet.")
    strong = len(hosts) >= 3 and len(tiers) >= 2 and angle["persistence"] >= 3 and peer["result"] == "company_specific" and weakened == 0
    moderate = len(hosts) >= 2 and angle["persistence"] >= 2 and peer["result"] != "sector_wide" and weakened <= 1
    return ("Strong" if strong else "Moderate" if moderate else "Weak"), reasons, used


# =============================================================================
# 9. FALSIFICATION CONDITION (measurable threshold + a dated future result)
# =============================================================================
def falsification(angle: Dict[str, Any], company: str) -> str:
    nxt = next_period(angle["latest_period"]).replace("_", " ")
    t = angle["test"]
    if t == "revenue_vs_profit":
        return (f"This finding is wrong if {company}'s {nxt} results show net-profit growth (before exceptional "
                f"items) within {GAP_PP:.0f} percentage points of revenue growth, year on year.")
    if t == "cash_conversion":
        # direction matters: weak conversion is disproved by going back UP to the
        # median, strong conversion by falling back DOWN to it
        side = "at or above" if angle["headline_value"] < angle["history_median"] else "at or below"
        return (f"This finding is wrong if {nxt} operating cash flow / profit after tax (before exceptional items, "
                f"same consolidated basis) is {side} {angle['history_median']}, the prior median.")
    if t.startswith("yoy_"):
        return (f"This finding is wrong if {_metric_label(angle['metrics_used'][0])} in {nxt} is within "
                f"{angle['min_move']} percentage points of the same quarter last year.")
    return (f"This finding is wrong if {_metric_label(angle['metrics_used'][0])} in {nxt} returns to "
            f"{angle.get('history_median')} (its prior median) or beyond.")


def _metric_label(metric: str) -> str:
    """'operating_margin_pct' -> 'operating margin' (for sentences, not for code)."""
    return re.sub(r"_pct$", "", metric).replace("_", " ")


def _fmt(value: Any, unit: str) -> str:
    """Number + unit the way a person would write it: 27.0%, 125 days, 1,234 INR crore."""
    if not isinstance(value, (int, float)):
        return str(value)
    if unit == "percentage":
        return f"{value}%"
    if unit == "days":
        return f"{value:g} days"
    return f"{value:,}"


# =============================================================================
# 10. LLM LAYER (Gemini 4 Argon) — writes the words, never the facts
# =============================================================================
FREE_TIER_FALLBACKS = ["gemini-3.8-flash", "flash-lite"]   # "flash-lite" = newest Flash-Lite on the key


def _err_kind(e: Exception) -> str:
    m = str(e).lower()
    if "429" in m or "quota" in m or "resource_exhausted" in m or "rate" in m:
        return "quota"
    if "403" in m or "404" in m or "permission" in m or "billing" in m or "not found" in m or "free tier" in m:
        return "no_access"
    return "other"


class GeminiLLM:
    """Needs: pip install google-genai ; env var GEMINI_API_KEY.

    Free-tier friendly: tries `model` first, then falls back to free models
    (gemini-3.8-flash, then the newest Flash-Lite) if the key has no access or runs out of quota.
    Waits and retries on per-minute rate limits."""

    def __init__(self, model: str = "gemini-4-argon", api_key: Optional[str] = None,
                 temperature: float = 0.2, fallbacks: Optional[List[str]] = None, verbose: bool = True,
                 timeout_s: int = 60):
        from google import genai
        from google.genai import types
        self._types = types
        key = api_key or os.environ["GEMINI_API_KEY"]
        try:   # stop waiting after `timeout_s` seconds instead of hanging forever
            self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
        except Exception:
            self.client = genai.Client(api_key=key)
        self.temperature, self.verbose, self.timeout_s = temperature, verbose, timeout_s
        self.chain = self._resolve([model] + (FREE_TIER_FALLBACKS if fallbacks is None else fallbacks))
        self.model = self.chain[0] if self.chain else model
        self.calls = 0

    def _resolve(self, wanted: List[str]) -> List[str]:
        """Keep only models this key can see; turn 'flash-lite' into the newest Flash-Lite name."""
        try:
            names = [m.name.split("/")[-1] for m in self.client.models.list()]
        except Exception:
            return [w for w in wanted if w != "flash-lite"]
        out = []
        for w in wanted:
            if w == "flash-lite":
                lites = sorted(n for n in names if "flash-lite" in n and "preview" not in n)
                if lites:
                    out.append(lites[-1])
            elif w in names:
                out.append(w)
        return list(dict.fromkeys(out))

    def json(self, system: str, prompt: str) -> Dict[str, Any]:
        import time
        cfg = self._types.GenerateContentConfig(system_instruction=system, temperature=self.temperature,
                                                response_mime_type="application/json")
        last = None
        while self.chain:
            self.model = self.chain[0]
            for attempt in range(3):
                t0 = time.time()
                try:
                    self.calls += 1
                    if self.verbose:
                        print(f"    asking {self.model}... (up to {self.timeout_s}s)", flush=True)
                    resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                    if self.verbose:
                        print(f"    got reply in {time.time() - t0:.0f}s", flush=True)
                    return json.loads(resp.text)
                except json.JSONDecodeError as e:
                    last = e                          # bad JSON: just retry
                    if self.verbose:
                        print("    reply was not valid JSON, retrying...", flush=True)
                except Exception as e:
                    last, kind = e, _err_kind(e)
                    if self.verbose:
                        print(f"    error after {time.time() - t0:.0f}s: {str(e)[:100]}", flush=True)
                    if kind == "other" and attempt >= 1:
                        break                         # timeouts / network: don't keep waiting
                    if kind == "quota" and attempt < 2 and "per day" not in str(e).lower():
                        wait = 20 * (attempt + 1)     # free tier: a few requests per minute
                        if self.verbose:
                            print(f"    [rate limit on {self.model}] waiting {wait}s...", flush=True)
                        time.sleep(wait)
                        continue
                    if kind in ("quota", "no_access"):
                        break                         # move to the next model
            dropped = self.chain.pop(0)
            if self.chain and self.verbose:
                print(f"    [{dropped} unavailable: {str(last)[:80]}] switching to {self.chain[0]}", flush=True)
        raise RuntimeError(f"LLM call failed on all models: {last}")


HYPOTHESIS_SYSTEM = """You are a careful equity research analyst. You write ONE testable hypothesis
from the computed numbers and evidence you are given.
Rules:
- Use ONLY numbers that appear in the input. Do not compute new numbers. Do not use outside knowledge.
- Cite evidence IDs (E1, E2...) for every factual statement.
- Separate fact (what the numbers show) from interpretation (a possible reason). Mark the reason as a possibility.
- If the counterargument checks weakened the finding, say so honestly in the claim.
- No investment advice.
Return JSON: {"claim": "<one sentence, max 40 words>",
              "why_it_matters": ["<3 to 5 short bullets>"],
              "possible_reasons": ["<2 to 3 possible explanations, each marked 'possible'>"],
              "evidence_ids": ["E.."]}"""

REDTEAM_SYSTEM = """You are a sceptical red-team analyst. Your only job is to find reasons the claim could be WRONG.
You are given the claim, the numbers, and the checks already run in code.
Do NOT repeat checks that were already run. Suggest NEW alternative explanations and what evidence would test each.
Return JSON: {"extra_counterarguments": [{"argument": "...", "evidence_needed": "..."}],
              "counter_search_queries": ["<2 or 3 short web searches that could find evidence AGAINST the claim>"]}"""


_PERIOD_TOKENS = re.compile(r"\bQ[1-4]\b|\bFY\s?\d{2,4}(?:-\d{2})?\b|\b(?:19|20)\d{2}\b|\bE\d+\b", re.I)


def ungrounded_numbers(text: str, allowed: List[float]) -> List[str]:
    """Numbers in `text` that do not match any evidence value or computed value."""
    bad = []
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", _PERIOD_TOKENS.sub(" ", text)):
        try:
            n = float(tok.replace(",", "").rstrip("."))
        except ValueError:
            continue
        if n.is_integer() and 0 <= n <= 12:
            continue  # small counts like "3 quarters"
        if not any(abs(abs(n) - abs(a)) <= max(0.06, 0.005 * abs(a)) for a in allowed):
            bad.append(tok)
    return bad


def allowed_numbers(angle: Dict[str, Any], pool: List[Evidence]) -> List[float]:
    nums = [e.value for e in pool if e.value is not None]
    for a in (angle, angle.get("adjusted") or {}):
        for r in a.get("rows", []):
            nums += [v for k, v in r.items() if isinstance(v, (int, float)) and k != "evidence_ids"]
        for k in ("headline_value", "history_median", "score", "min_move"):
            if isinstance(a.get(k), (int, float)):
                nums.append(a[k])
    return nums


# =============================================================================
# 11. TEMPLATE TEXT (used when no LLM, or when the LLM output fails the grounding check)
# =============================================================================
def template_claim(angle: Dict[str, Any], company: str, checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    a = angle.get("adjusted") or angle
    t = angle["test"]
    weakened = [c["check"].replace("_", " ") for c in checks if c["result"] == "weakened"]
    caveat = f" (caveat: {', '.join(weakened)} not fully ruled out)" if weakened else ""
    if t == "revenue_vs_profit":
        r = a["rows"][-1]
        rv, pv = r["revenue_yoy_pct"], r["profit_yoy_pct"]
        period = r["period"].replace("_", " ")
        # say it the right way round -- earlier this said "sales outran profit"
        # even when profit had grown faster than sales
        if r["gap_pp"] > 0:
            moved = f"fell {abs(pv)}%" if pv < 0 else f"grew only {pv}%"
            claim = (f"In {period}, {company}'s revenue grew {rv}% year on year while net profit {moved}: "
                     f"a {r['gap_pp']} percentage-point gap{caveat}.")
            why = ["Sales growth is not reaching the bottom line.",
                   "Possible causes to check next: higher costs, lower other income, higher tax or "
                   "depreciation, or a mix shift."]
        else:
            moved = f"fell {abs(rv)}%" if rv < 0 else f"grew only {rv}%"
            claim = (f"In {period}, {company}'s net profit grew {pv}% year on year while revenue {moved}: "
                     f"profit outgrew sales by {abs(r['gap_pp'])} percentage points{caveat}.")
            why = ["Profit is growing faster than sales, so margins are expanding.",
                   "Possible causes to check next: softer input costs, price increases, higher other "
                   "income or lower tax -- some of these do not repeat."]
        why.append(f"The gap has the same direction for {angle['persistence']} consecutive quarter(s) "
                   f"of at least {GAP_PP:.0f} percentage points.")
    elif t == "cash_conversion":
        r = a["rows"][-1]
        claim = (f"In {r['period']}, {company}'s operating cash flow was {r['ocf_to_profit']}x net profit, versus a prior "
                 f"median of {a['history_median']}x{caveat}.")
        why = ["Profit that does not turn into cash can signal working-capital build-up."
               if r["ocf_to_profit"] < a["history_median"] else
               "Cash coming in much faster than profit can mean working capital was released "
               "(lower inventory, faster collections, or longer supplier credit).",
               "Check receivables, inventory and payables movements in the cash-flow statement."]
    elif t.startswith("yoy_"):
        r = a["rows"][-1]
        unit = angle.get("unit", "")
        claim = (f"{company}'s {_metric_label(angle['metrics_used'][0])} was {_fmt(r['value'], unit)} in "
                 f"{r['period'].replace('_', ' ')}, versus {_fmt(r['base_value'], unit)} in "
                 f"{r['base'].replace('_', ' ')}: a {r['change_pp']} percentage-point change year on year, "
                 f"against a typical change of {a['history_median']} points{caveat}.")
        why = ["Margin moving this much against the same quarter last year is not a seasonal effect.",
               "Possible causes to check next: input costs, pricing, product mix or ad spend."]
    else:
        unit = angle.get("unit", "")
        claim = (f"{company}'s {_metric_label(angle['metrics_used'][0])} was {_fmt(angle['headline_value'], unit)} "
                 f"in {angle['latest_period'].replace('_', ' ')}, versus a prior median of "
                 f"{_fmt(angle['history_median'], unit)}{caveat}.")
        why = ["A change in working-capital days can show channel stress or a change in trade terms."]
    return {"claim": claim, "why_it_matters": why, "possible_reasons": [], "evidence_ids": angle["evidence_ids"]}


# =============================================================================
# 12. MAIN ENTRY POINT
# =============================================================================
def run_validation(company_data: Dict[str, Any],
                   anomalies: Optional[List[Dict[str, Any]]] = None,
                   official_rows: Optional[List[Dict[str, Any]]] = None,
                   source_register: Optional[List[Dict[str, Any]]] = None,
                   peer_data: Optional[Dict[str, Dict[str, Any]]] = None,
                   llm: Any = None,
                   search_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    company = company_data.get("company", "")
    pool, flags = build_evidence_pool(company_data, official_rows, source_register)
    base = {"schema_version": SCHEMA_VERSION, "company": company, "ticker": company_data.get("ticker", ""),
            "angles_scored": [], "data_quality_flags": flags, "insufficient_reason": None,
            "generated_by": {"llm": getattr(llm, "model", None) or "none (template mode)"}}

    # Gate 1: nothing to work with. Say so plainly instead of "nothing survived".
    if not company_data.get("financial_records"):
        return _insufficient(base, "no_data",
                             f"No financial data could be collected for '{company}'. Check the name or ticker, "
                             f"or use offline mode if the live sites are blocked.", [], [])

    # Gate 2: not a consumer business. Our tests (volume vs price, inventory days, trade
    # receivables) are built for FMCG / retail; on a bank they would give nonsense.
    # is_track_c comes from NSE's own classification, so no company list is hard-coded.
    if company_data.get("is_track_c") is False:
        info = company_data.get("industry_info", {})
        label = " > ".join(v for v in (info.get("broad_sector"), info.get("industry")) if v)
        return _insufficient(base, "not_track_c",
                             f"{company} is classified by NSE as {label or 'a non-consumer business'}, not a "
                             f"Track C consumer, FMCG or retail company. The agent's tests are built for "
                             f"consumer businesses, so it does not force a finding here.", [], [])

    conflicts = find_conflicts(pool)
    for c in conflicts:
        flags.append(f"Source conflict: {c['metric']} {c['period']} differs by {c['diff_pct']}% "
                     f"({c['type'].replace('_', ' ')}).")
    peer_pools = {n: build_evidence_pool(d)[0] for n, d in (peer_data or {}).items()}

    angles = anomalies if anomalies else detect_angles(pool)
    limitations = [] if anomalies else ["Angles came from validation.py's built-in detector, not analysis.py."]
    # unusual (score) AND big enough to matter (material) -- see MIN_*_MOVE at the top
    candidates = [a for a in angles
                  if a.get("score", 0) >= INVESTIGATE_SCORE and a.get("material", True)][:TOP_ANGLES]
    for a in angles:
        if a.get("score", 0) >= INVESTIGATE_SCORE and not a.get("material", True):
            flags.append(f"Not investigated: '{a['title']}' is unusual for this company (score {a['score']}) "
                         f"but the move is smaller than {a['min_move']}, too small to matter.")

    base["angles_scored"] = [{"id": a["id"], "angle": a["title"], "score": a["score"]} for a in angles]

    rejected = []
    for angle in candidates:
        checks, peer, verdict = red_team(angle, pool, conflicts, peer_pools)
        if verdict["fatal"]:
            killer = next(c for c in checks if c["result"] == "fatal")
            rejected.append({"angle": angle["title"], "reason": f"{killer['check']}: {killer['detail']}"})
            continue

        strength, reasons, used = grade(angle, pool, peer, checks)
        text, extra_counter, counter_search = template_claim(angle, company, checks), [], []

        if llm is not None:
            try:
                slim = [{"id": e.id, "metric": e.metric, "period": e.period, "value": e.value, "unit": e.unit,
                         "tier": e.tier, "quote": e.quote[:300]} for e in used]
                payload = json.dumps({"company": company, "angle": angle["title"],
                                      "computed": (angle.get("adjusted") or angle)["rows"],
                                      "evidence": slim, "checks_run": checks}, default=str)
                talk = getattr(llm, "verbose", False)
                if talk:
                    print(f"  [LLM 1/2] writing the hypothesis for: {angle['title']}", flush=True)
                out = llm.json(HYPOTHESIS_SYSTEM, payload)
                bad = ungrounded_numbers(out.get("claim", "") + " " + " ".join(out.get("why_it_matters", [])),
                                         allowed_numbers(angle, pool))
                if bad:
                    flags.append(f"LLM claim rejected (numbers not in evidence: {', '.join(bad)}); used template text.")
                else:
                    text = out
                if talk:
                    print("  [LLM 2/2] red team: looking for reasons the claim is wrong", flush=True)
                rt = llm.json(REDTEAM_SYSTEM, json.dumps({"claim": text["claim"], "checks_run": checks}, default=str))
                extra_counter = rt.get("extra_counterarguments", [])
                if search_fn:
                    for q in rt.get("counter_search_queries", [])[:3]:
                        for hit in (search_fn(q) or [])[:3]:
                            counter_search.append({"query": q, **hit})
            except Exception as e:
                flags.append(f"LLM step failed ({e}); used template text.")

        for x in extra_counter:
            checks.append({"check": "llm_suggested", "argument": x.get("argument", ""), "result": "not_run",
                           "detail": f"Not verified in code. Evidence needed: {x.get('evidence_needed', '')}"})
        if peer["result"] == "not_run":
            limitations.append("Peer check not run: supply peer_data to rule out a sector-wide story.")
        if angle.get("history_left_out"):
            limitations.append(f"Unusualness score ignores one-off quarters in the history: "
                               f"{', '.join(angle['history_left_out'])} (still shown in the numbers table).")

        show = angle.get("adjusted") or angle
        return {**base, "status": "ok",
                "claim": text["claim"], "why_it_matters": text.get("why_it_matters", []),
                "possible_reasons": text.get("possible_reasons", []),
                "numbers": show["rows"],
                "timeline": [{"period": r["period"],
                              "note": ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("period", "base", "evidence_ids"))}
                             for r in show["rows"]],
                "evidence": [{"id": e.id, "fact": f"{e.metric} {e.period}".strip(), "value": e.value, "unit": e.unit,
                              "source_name": e.source_name, "source_url": e.source_url, "tier": e.tier,
                              "scope": e.scope, "role": e.role, "page": e.page, "quote": e.quote} for e in used],
                "counterarguments": checks, "counter_search": counter_search, "peer_check": peer,
                "falsification_condition": falsification(angle, company),
                "strength": strength, "strength_reasons": reasons,
                "rejected_angles": rejected, "limitations": limitations}

    return _insufficient(base, "nothing_survived",
                         "No finding survived the red-team checks. The agent does not force a story.",
                         rejected, limitations)


def _insufficient(base: Dict[str, Any], reason: str, claim: str,
                  rejected: List[Dict[str, Any]], limitations: List[str]) -> Dict[str, Any]:
    """One place to build the 'insufficient_evidence' result, so all three exits
    (no data / not Track C / nothing survived) have exactly the same schema."""
    return {**base, "status": "insufficient_evidence", "insufficient_reason": reason,
            "claim": claim,
            "why_it_matters": [], "numbers": [], "timeline": [], "evidence": [], "counterarguments": [],
            "peer_check": {"result": "not_run", "detail": ""}, "falsification_condition": "",
            "strength": "Weak", "strength_reasons": ["Nothing survived." if reason == "nothing_survived"
                                                     else "No finding attempted."],
            "rejected_angles": rejected, "limitations": limitations}


# =============================================================================
# 13. CLI TEST
# =============================================================================
def _read_csv(path: Optional[str]) -> Optional[List[Dict[str, str]]]:
    if not path:
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Person 3 validation test run")
    ap.add_argument("--data", default="sample_hul_data.json")
    ap.add_argument("--official", default=None)
    ap.add_argument("--register", default=None)
    ap.add_argument("--llm", action="store_true", help="use Gemini 4 Argon (needs GEMINI_API_KEY)")
    ap.add_argument("--out", default="insight_result.json")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    result = run_validation(data, official_rows=_read_csv(args.official),
                            source_register=_read_csv(args.register),
                            llm=GeminiLLM() if args.llm else None)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"STATUS:   {result['status']}\nSTRENGTH: {result['strength']}\nCLAIM:    {result['claim']}\n")
    print("Angles scored:", *[f"  {a['id']} {a['score']:>5}  {a['angle']}" for a in result["angles_scored"]], sep="\n")
    print("\nRejected by red team:", *[f"  - {r['angle']}\n      {r['reason']}" for r in result["rejected_angles"]] or ["  none"], sep="\n")
    print("\nCounterarguments:", *[f"  [{c['result']:>8}] {c['check']}: {c['detail']}" for c in result["counterarguments"]], sep="\n")
    print("\nFalsification:", result["falsification_condition"])
    print("\nStrength reasons:", *[f"  - {r}" for r in result["strength_reasons"]], sep="\n")
    print("\nData quality flags:", *[f"  - {x}" for x in result["data_quality_flags"]], sep="\n")
    print(f"\nFull result saved to {args.out}")

# =============================================================================
# INSIGHT SCHEMA (what Person 4 renders)
# -----------------------------------------------------------------------------
# status                  "ok" | "insufficient_evidence"
# insufficient_reason     None | "no_data" | "not_track_c" | "nothing_survived"   (added field)
# claim                   one sentence (for insufficient_evidence: the plain-English reason)
# why_it_matters          [str]
# numbers                 [{period, ...computed values, evidence_ids}]  -> charts
# timeline                [{period, note}]
# evidence                [{id, fact, value, unit, source_name, source_url, tier, scope, page, quote}]
# counterarguments        [{check, argument, result: survived|weakened|fatal|not_run, detail}]
# peer_check              {result: company_specific|sector_wide|not_run, detail}
# falsification_condition str
# strength                "Strong" | "Moderate" | "Weak"
# strength_reasons        [str]
# rejected_angles         [{angle, reason}]   <- show this: proves the agent drops weak stories
# angles_scored           [{id, angle, score}]
# data_quality_flags      [str]
# limitations             [str]
# =============================================================================
