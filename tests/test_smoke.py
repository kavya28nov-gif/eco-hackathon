"""
test_smoke.py
-------------
Runs the full pipeline OFFLINE on every company in data/cache/ and checks that
the output matches the insight schema (bottom of validation.py). No network,
no API key -- so this can be run 5 minutes before the finals also.

    pytest tests/                      # pass / fail
    python tests/test_smoke.py         # prints claim, strength, rejected angles per company
"""

import csv
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import research     # noqa: E402  (path has to be set first)
import validation   # noqa: E402

OFFICIAL_DIR = os.path.join(ROOT, "data", "official")

# key -> type it must have, in EVERY result (ok or insufficient_evidence)
REQUIRED = {
    "schema_version": str, "status": str, "claim": str, "why_it_matters": list,
    "numbers": list, "timeline": list, "evidence": list, "counterarguments": list,
    "peer_check": dict, "falsification_condition": str, "strength": str,
    "strength_reasons": list, "rejected_angles": list, "angles_scored": list,
    "data_quality_flags": list, "limitations": list,
}
EVIDENCE_KEYS = {"id", "fact", "value", "unit", "source_name", "source_url", "tier", "scope", "page", "quote"}
CHECK_RESULTS = {"survived", "weakened", "fatal", "not_run"}


def _read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_offline(ticker):
    """Same call app.py makes, but from the cache. Peers too, if they are cached."""
    data = research.load_cache(ticker)
    peers = {}
    for p in research.peers_for(data["ticker"]) if hasattr(research, "peers_for") else []:
        peer_data = research.load_cache(p)
        if peer_data:
            peers[p] = peer_data
    return data, validation.run_validation(
        data,
        official_rows=_read_csv(os.path.join(OFFICIAL_DIR, f"{data['ticker']}_evidence.csv")),
        source_register=_read_csv(os.path.join(OFFICIAL_DIR, f"{data['ticker']}_register.csv")),
        peer_data=peers or None)


def check_schema(insight):
    for key, typ in REQUIRED.items():
        assert key in insight, f"missing key: {key}"
        assert isinstance(insight[key], typ), f"{key} should be {typ.__name__}"
    assert insight["status"] in ("ok", "insufficient_evidence")
    assert insight["strength"] in ("Strong", "Moderate", "Weak")
    assert insight["claim"].strip(), "claim is empty"
    for c in insight["counterarguments"]:
        assert {"check", "argument", "result", "detail"} <= set(c)
        assert c["result"] in CHECK_RESULTS, c["result"]
    for e in insight["evidence"]:
        assert EVIDENCE_KEYS <= set(e), f"evidence missing keys: {EVIDENCE_KEYS - set(e)}"
        assert e["tier"] in validation.TIER_RANK, e["tier"]
        # rule 5: every evidence item keeps its source
        assert e["source_url"] or e["source_name"], f"evidence {e['id']} has no source"
    for r in insight["rejected_angles"]:
        assert {"angle", "reason"} <= set(r)
    if insight["status"] == "ok":
        assert insight["falsification_condition"].strip(), "ok result needs a falsification condition"
        assert insight["counterarguments"], "ok result needs counterarguments"
        assert insight["evidence"], "ok result needs evidence"


CACHED = research.cached_tickers()


@pytest.mark.parametrize("ticker", CACHED)
def test_schema_on_every_cached_company(ticker):
    _, insight = run_offline(ticker)
    check_schema(insight)


def test_there_is_something_to_test():
    assert len(CACHED) >= 3, "cache is nearly empty -- run the live fetch first"


def test_non_consumer_company_is_refused_cleanly():
    """A bank must give insufficient_evidence with a clear reason, not a fake finding."""
    fake_bank = {"company": "Some Bank", "ticker": "SOMEBANK", "is_track_c": False,
                 "industry_info": {"broad_sector": "Financial Services", "industry": "Private Sector Bank"},
                 "financial_records": [{"metric": "net_profit", "value": 1.0, "unit": "INR crore",
                                        "period": "FY2026", "period_type": "annual",
                                        "source_url": "https://www.screener.in/x/", "doc_title": "x"}],
                 "text_excerpts": []}
    insight = validation.run_validation(fake_bank)
    check_schema(insight)
    assert insight["status"] == "insufficient_evidence"
    assert insight["insufficient_reason"] == "not_track_c"
    assert "Financial Services" in insight["claim"]


def test_hul_demerger_gain_still_rejected():
    """Regression: the fake cash-conversion story from the ice cream demerger gain
    must keep getting rejected by the one-off check."""
    import json
    with open(os.path.join(ROOT, "data", "sample_hul_data.json"), encoding="utf-8") as f:
        data = json.load(f)
    insight = validation.run_validation(
        data, official_rows=_read_csv(os.path.join(OFFICIAL_DIR, "HINDUNILVR_evidence.csv")),
        source_register=_read_csv(os.path.join(OFFICIAL_DIR, "HINDUNILVR_register.csv")))
    check_schema(insight)
    killed = [r for r in insight["rejected_angles"] if "cash conversion" in r["angle"].lower()]
    assert killed and killed[0]["reason"].startswith("one_off")


def test_quarter_from_filing_date():
    # results filed in early August are about the June quarter = Q1 of the NEXT FY
    assert research._period_from_filing_date("04-Aug-2026 17:15:14") == "Q1_FY2027"
    assert research._period_from_filing_date("07-May-2026 20:10:39") == "Q4_FY2026"
    # filed 3 July: June quarter is not "over" by a week yet -> still the March quarter
    assert research._period_from_filing_date("03-Jul-2026 10:00:00") == "Q4_FY2026"


def test_volume_wording_and_decimals():
    pages = ["Results. India FMCG business reported 9.5% growth with an underlying Volume growth of 5%. "
             "Delivered 7% Underlying Sales Growth (USG), driven by 6% Underlying Volume Growth (UVG). "
             "Soap volumes declined by 3% in the quarter."]
    found = research._extract_growth(pages)
    assert found["underlying_volume_growth_pct"][0] == 5.0
    assert found["underlying_sales_growth_pct"][0] == 7.0
    # the quote must keep "9.5%" whole -- a decimal point is not a sentence end
    assert "9.5%" in found["underlying_volume_growth_pct"][2]
    assert research._extract_growth(["Soap volumes declined by 3% this quarter."])[
        "underlying_volume_growth_pct"][0] == -3.0


def test_segment_usg_is_not_used_as_company_figure():
    """Godrej regression: a segment 'USG ~47%' when consolidated revenue grew ~15%
    must not be used as the company's sales growth."""
    def rec(metric, period, value, unit="INR crore"):
        return {"metric": metric, "value": value, "unit": unit, "period": period, "period_type": "quarterly",
                "source_url": "https://www.screener.in/company/X/", "doc_title": "x"}
    data = {"company": "X", "ticker": "X", "text_excerpts": [], "financial_records": [
        rec("revenue_from_operations", "Q1_FY2026", 100.0), rec("revenue_from_operations", "Q1_FY2027", 115.0),
        rec("underlying_volume_growth_pct", "Q1_FY2027", 9.0, "percentage"),
        rec("underlying_sales_growth_pct", "Q1_FY2027", 47.0, "percentage")]}
    pool, _ = validation.build_evidence_pool(data)
    angle = validation.test_price_vs_volume(pool)
    assert angle["sales_basis"] == "reported revenue growth"
    assert angle["rows"][-1]["sales_growth_pct"] == 15.0
    assert angle["usg_rejected"]


if __name__ == "__main__":
    # human-readable summary, for showing results after each priority
    for t in CACHED:
        data, ins = run_offline(t)
        print(f"\n=== {t} ({data.get('company')}) -- {ins['status']}, {ins['strength']}")
        print(f"  claim: {ins['claim']}")
        if ins.get("falsification_condition"):
            print(f"  falsify: {ins['falsification_condition']}")
        print(f"  peer check: {ins['peer_check'].get('result')} -- {ins['peer_check'].get('detail', '')}")
        for r in ins["rejected_angles"]:
            print(f"  rejected: {r['angle']} -- {r['reason'][:160]}")
        for c in ins["counterarguments"]:
            if c["result"] in ("weakened", "fatal"):
                print(f"  [{c['result']}] {c['check']}: {c['detail'][:140]}")
