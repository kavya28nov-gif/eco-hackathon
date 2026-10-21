# app.py — Person 4: Streamlit UI + orchestration
# Run: streamlit run app.py
import csv
import os
import traceback

import pandas as pd
import plotly.express as px
import streamlit as st

import research
import analysis
import validation
import report

HERE = os.path.dirname(__file__)
OFFICIAL_DIR = os.path.join(HERE, "data", "official")

RESULT_ICON = {"survived": "✅", "weakened": "⚠️", "fatal": "❌", "not_run": "⏸️"}
BADGE = {"Strong": "green", "Moderate": "orange", "Weak": "red"}

st.set_page_config(page_title="NIFTY Insight Agent", layout="wide")
st.title("NIFTY Insight Agent")
st.caption("Track C: Consumer, FMCG & Retail — public data only, no investment advice.")

with st.sidebar:
    company = st.text_input("Company name", value="HUL")
    use_sample = st.checkbox("Use cached data (offline demo)", value=False,
                             help="Loads data/cache/<TICKER>.json for the company typed above, instead of "
                                  "scraping. Backup if live sites block us.")
    cached = research.cached_tickers()
    st.caption(f"Cached: {', '.join(cached) if cached else 'none yet'}")
    use_llm = st.checkbox("Use LLM wording (needs GEMINI_API_KEY)", value=bool(os.environ.get("GEMINI_API_KEY")))
    run = st.button("Run Research", type="primary", width="stretch")


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_data(name, offline):
    if offline:
        data = research.load_cache(name)
        if data is None:
            # no silent fallback to HUL here -- showing HUL when the judge typed
            # Britannia would be the worst possible mistake in a live demo
            raise FileNotFoundError(f"No cached data for '{name}'. Cached companies: "
                                    f"{', '.join(research.cached_tickers()) or 'none'}. "
                                    f"Untick offline mode to fetch it live.")
        return data
    return research.fetch_company_data(name)


def make_llm():
    if not use_llm:
        return None
    try:
        return validation.GeminiLLM(verbose=False)
    except Exception as e:
        st.warning(f"LLM not available ({type(e).__name__}); using template wording.")
        return None


if run:
    status = st.status("Running research pipeline...", expanded=True)
    try:
        status.write("1/4 Collecting public data...")
        data = load_data(company, use_sample)
        ticker = data.get("ticker", "")
        status.write(f"Got {data.get('record_count', 0)} financial records and "
                     f"{len(data.get('text_excerpts', []))} text excerpts for {ticker}.")
        if data.get("record_count", 0) == 0:
            status.write("⚠️ No financial records. Check the ticker or tick the offline option.")

        official = read_csv(os.path.join(OFFICIAL_DIR, f"{ticker}_evidence.csv"))
        register = read_csv(os.path.join(OFFICIAL_DIR, f"{ticker}_register.csv"))
        if official:
            status.write(f"Loaded {len(official)} official source-linked figures for cross-checking.")

        # peers: from the cache when offline, else fetched live (numbers only, in parallel)
        peer_list = data.get("peer_tickers") or research.peers_for(ticker)
        status.write(f"2/4 Fetching peers for the peer-control check: {', '.join(peer_list) or 'none found'}")
        if use_sample:
            peers = {p: research.load_cache(p) for p in peer_list}
            peers = {p: d for p, d in peers.items() if d}
            for p in peer_list:
                status.write(f"Peer {p}: {'from cache' if p in peers else 'not cached, skipped'}")
        else:
            peers = research.fetch_peers(peer_list, progress=status.write)

        status.write("3/4 Calculating metrics...")
        result = analysis.analyze_financials(data)

        status.write("4/4 Forming hypothesis and red-teaming it...")
        insight = validation.run_validation(data, official_rows=official, source_register=register,
                                            peer_data=peers or None, llm=make_llm())

        st.session_state["run"] = {"data": data, "analysis": result, "insight": insight}
        status.update(label="Done", state="complete", expanded=False)
    except Exception as e:
        status.update(label="Pipeline failed", state="error")
        st.error(f"{type(e).__name__}: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

if "run" in st.session_state:
    data = st.session_state["run"]["data"]
    insight = st.session_state["run"]["insight"]
    df = pd.DataFrame(data.get("financial_records", []))

    # --- Headline ---
    st.subheader(f"{data.get('company')} ({data.get('ticker')})")
    if insight.get("status") == "insufficient_evidence":
        st.warning({"not_track_c": "Not a Track C consumer company.",
                    "no_data": "No data could be collected."}.get(
            insight.get("insufficient_reason"), "No finding survived the red team. The agent does not force a story."))
    st.markdown(f"### {insight.get('claim', '')}")
    strength = insight.get("strength", "")
    st.markdown(f"Evidence strength: :{BADGE.get(strength, 'gray')}[**{strength}**]")
    for w in insight.get("why_it_matters", []):
        st.markdown(f"- {w}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Numbers", "Evidence", "Challenge", "Data quality", "Raw data"])

    with tab1:
        nums = pd.DataFrame(insight.get("numbers", []))
        if not nums.empty and "revenue_yoy_pct" in nums:
            long = nums.melt(id_vars="period", value_vars=["revenue_yoy_pct", "profit_yoy_pct"],
                             var_name="series", value_name="yoy_pct")
            st.plotly_chart(px.bar(long, x="period", y="yoy_pct", color="series", barmode="group",
                                   title="Revenue vs net profit growth, year on year (%)"), width="stretch")
        elif not nums.empty and "price_mix_pp" in nums:
            long = nums.melt(id_vars="period", value_vars=["volume_growth_pct", "price_mix_pp"],
                             var_name="part", value_name="pct")
            st.plotly_chart(px.bar(long, x="period", y="pct", color="part", barmode="relative",
                                   title="Sales growth split: volume vs price/mix (percentage points)"),
                            width="stretch")
        elif not nums.empty:
            # only numeric columns can go on the y-axis (rows can also carry text like 'sales_basis')
            numeric = [c for c in nums.columns if c not in ("period", "base", "evidence_ids")
                       and pd.api.types.is_numeric_dtype(nums[c])]
            ycol = numeric[-1]
            st.plotly_chart(px.line(nums, x="period", y=ycol, markers=True, title=ycol), width="stretch")
        if insight.get("timeline"):
            st.markdown("#### Timeline")
            st.dataframe(pd.DataFrame(insight["timeline"]), hide_index=True, width="stretch")
        if not df.empty:
            q = df[(df["period_type"] == "quarterly") &
                   (df["metric"].isin(["revenue_from_operations", "net_profit"]))].dropna(subset=["value"])
            if not q.empty:
                st.plotly_chart(px.line(q.sort_values("period"), x="period", y="value", color="metric", markers=True,
                                        title="Quarterly revenue and net profit (INR cr)"), width="stretch")

    with tab2:
        ev = pd.DataFrame(insight.get("evidence", []))
        if not ev.empty:
            cols = [c for c in ["id", "fact", "value", "unit", "role", "tier", "scope", "source_name", "source_url", "quote"]
                    if c in ev]
            st.dataframe(ev[cols], hide_index=True, width="stretch",
                         column_config={"source_url": st.column_config.LinkColumn("Source")})
        else:
            st.write("No evidence attached.")

    with tab3:
        st.markdown("#### Counterarguments (code checks that tried to kill the finding)")
        for c in insight.get("counterarguments", []):
            st.markdown(f"{RESULT_ICON.get(c['result'], '')} **{c['check'].replace('_', ' ')}** "
                        f"({c['result']}): {c['detail']}")
        peer = insight.get("peer_check", {})
        if peer:
            st.markdown(f"**Peer check:** {peer.get('result', '').replace('_', ' ')} — {peer.get('detail', '')}")
        if insight.get("falsification_condition"):
            st.markdown("#### Falsification condition")
            st.info(insight["falsification_condition"])
        st.markdown("#### Why this strength")
        for r in insight.get("strength_reasons", []):
            st.markdown(f"- {r}")
        if insight.get("rejected_angles"):
            st.markdown("#### Angles the agent dropped")
            for r in insight["rejected_angles"]:
                st.markdown(f"- ❌ **{r['angle']}** — {r['reason']}")

    with tab4:
        if insight.get("angles_scored"):
            st.markdown("#### Angles scored (how unusual vs own history)")
            st.dataframe(pd.DataFrame(insight["angles_scored"]), hide_index=True, width="stretch")
        st.markdown("#### Data quality flags")
        for f in insight.get("data_quality_flags", []):
            st.markdown(f"- {f}")
        st.markdown("#### Limitations")
        for l in insight.get("limitations", []):
            st.markdown(f"- {l}")
        st.caption(f"Wording generated by: {insight.get('generated_by', {}).get('llm', '')}")

    with tab5:
        st.dataframe(df, width="stretch", hide_index=True)

    pdf_bytes = report.build_pdf(data, insight)
    st.download_button("Download PDF report", data=pdf_bytes,
                       file_name=f"{data.get('ticker', 'report')}_insight_report.pdf",
                       mime="application/pdf", type="primary")
else:
    st.info("Enter a company and press Run Research. Tick the offline option if live sites are blocked.")
