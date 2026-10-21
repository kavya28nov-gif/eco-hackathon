# analysis.py — Person 2: calculations for DISPLAY ONLY
#
# STATUS (read this first): this module does NOT feed validation.py.
# validation.py has its own angle detector (detect_angles), because every angle it
# investigates must carry evidence IDs, periods and metric names so the red team can
# recompute it (one-offs, base effect, peers). The anomalies below are only
# {title, detail, numbers, score} -- they have no evidence IDs, so passing them to
# run_validation(anomalies=...) would break the red-team checks. That is why app.py
# does not pass them.
#
# What app.py DOES use: the "metrics" dict, shown in the "Raw data" tab as a quick
# year-on-year table. The "anomalies" list is shown there too, marked as "not used
# for the finding".
#
# If Person 2 wants to feed validation later: return angles in the same shape as
# validation.test_revenue_vs_profit() returns (test, title, latest_period, score,
# headline_value, persistence, rows, periods_used, metrics_used, evidence_ids,
# material, direction), built on validation.build_evidence_pool() so the IDs match.
import pandas as pd
from typing import Dict, Any, List


def _series(df: pd.DataFrame, metric: str, period_type: str) -> pd.Series:
    sub = df[(df["metric"] == metric) & (df["period_type"] == period_type)]
    sub = sub.dropna(subset=["value"]).sort_values("period")
    return pd.Series(sub["value"].values, index=sub["period"].values)


def analyze_financials(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: output of research.fetch_company_data()
    Output:
      {
        "metrics": {name: {period: value}},
        "anomalies": [ {title, detail, numbers, score} ]  (sorted, highest score first)
      }
    """
    df = pd.DataFrame(data.get("financial_records", []))
    if df.empty:
        return {"metrics": {}, "anomalies": []}
    # Nestle's FY2024 was 15 months long (year-end change). It is labelled "FY2024", so
    # the label alone cannot catch it -- research.py's period_months can.
    if "period_months" in df:
        df = df[df["period_months"].isna() | df["period_months"].isin([3, 12])]

    metrics: Dict[str, Dict[str, float]] = {}
    anomalies: List[Dict[str, Any]] = []

    rev = _series(df, "revenue_from_operations", "annual")
    np_ = _series(df, "net_profit", "annual")
    ocf = _series(df, "operating_cash_flow", "annual")
    debtor = _series(df, "debtor_days", "annual")

    # only normal April-March years ("FY2025"), same rule as validation.py --
    # a 15 month year or a December year-end would give a fake growth number
    rev, np_, ocf, debtor = [s[s.index.str.match(r"^FY\d{4}$")] for s in (rev, np_, ocf, debtor)]

    if len(rev) > 1:
        metrics["revenue_growth_pct"] = (rev.pct_change() * 100).round(1).dropna().to_dict()
    if len(np_) > 1:
        metrics["net_profit_growth_pct"] = (np_.pct_change() * 100).round(1).dropna().to_dict()

    common = ocf.index.intersection(np_.index)
    if len(common) > 0:
        ratio = (ocf[common] / np_[common]).round(2)
        metrics["ocf_to_net_profit"] = ratio.to_dict()
        latest = common[-1]
        if ratio[latest] < 0.9:
            anomalies.append({
                "title": "Cash conversion fell below profit",
                "detail": f"Operating cash flow was {ratio[latest]:.2f}x net profit in {latest}. "
                          "Check for one-off gains inflating profit before reading this as weak cash quality.",
                "numbers": {"ocf": float(ocf[latest]), "net_profit": float(np_[latest])},
                "score": round(1 - float(ratio[latest]), 2),
            })

    if len(debtor) > 1:
        metrics["debtor_days"] = debtor.to_dict()
        change = float(debtor.iloc[-1] - debtor.iloc[0])
        if abs(change) >= 3:
            anomalies.append({
                "title": "Debtor days moved",
                "detail": f"Debtor days went from {debtor.iloc[0]:.0f} to {debtor.iloc[-1]:.0f} "
                          f"({debtor.index[0]} to {debtor.index[-1]}).",
                "numbers": debtor.to_dict(),
                "score": round(abs(change) / 10, 2),
            })

    # Quarterly net profit spikes (one-off detector)
    qnp = _series(df, "net_profit", "quarterly")
    if len(qnp) > 3:
        med = qnp.median()
        for p, v in qnp.items():
            if v > 1.5 * med:
                anomalies.append({
                    "title": f"Profit spike in {p}",
                    "detail": f"Net profit of {v:,.0f} cr is {v / med:.1f}x the median quarter. "
                              "Likely an exceptional item; use profit before exceptionals for trends.",
                    "numbers": {"value": float(v), "median": float(med)},
                    "score": round(v / med / 3, 2),
                })

    anomalies.sort(key=lambda a: a["score"], reverse=True)
    return {"metrics": metrics, "anomalies": anomalies}
