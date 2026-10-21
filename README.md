# NIFTY Insight Agent

Enter a company name -> collect public data -> calculate -> hypothesis + challenge -> report (screen + PDF).

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Tick "Use cached data (offline demo)" in the sidebar if live scraping is blocked (demo backup).
It loads `data/cache/<TICKER>.json` for whatever company is typed. Every successful live run saves
there, so before the finals just run each likely company once with internet on.

## Test (offline, no API key)
```
pytest tests/                 # schema check on every cached company + regressions
python tests/test_smoke.py    # claim / strength / rejected angles per cached company
```

## Modules and contracts
| File | Owner | Function | Returns |
|---|---|---|---|
| research.py | Person 1 | `fetch_company_data(name)` | `{company, ticker, sector, record_count, financial_records[], text_excerpts[]}` + added: `industry_info, is_track_c, data_quality_flags, fetched_at` |
| research.py | Person 1 | `load_cache(name)` / `cached_tickers()` | same dict from `data/cache/`, or None |
| analysis.py | Person 2 | `analyze_financials(data)` | `{metrics: {name: {period: value}}, anomalies: [{title, detail, numbers, score}]}` |
| validation.py | Person 3 | `run_validation(data, official_rows, source_register, peer_data, llm)` | insight schema (see bottom of validation.py) |
| app.py / report.py | Person 4 | UI + `build_pdf(data, insight)` | PDF bytes |

Financial records also carry `period_months` (3, 12, or e.g. 15 when a company changed its year-end).
Validation leaves non-standard periods out of every comparison and says so in the data-quality flags.

`analysis.py` is still a placeholder. Replace the logic, keep the function names and return shapes, and the app keeps working.

Official source-linked figures go in `data/official/<TICKER>_evidence.csv` and `<TICKER>_register.csv`; the app loads them automatically.
Set `GEMINI_API_KEY` to turn on LLM wording.

## How the agent decides (for judges)
- **Ticker:** alias table, then screener's own search, so a company we never listed still resolves.
- **Is it Track C?** NSE's classification from screener: broad sector "Fast Moving Consumer Goods", or sector
  "Consumer Durables" / "Consumer Services". Anything else (banks, autos, IT) returns `insufficient_evidence`
  with the reason, instead of a forced finding.
- **Angle is investigated only if** it is unusual vs the company's own history (robust z >= 1.5) AND the move is
  material (gap >= 5 pp, days >= 5, margin >= 1.5 pp YoY, cash ratio >= 0.2). Thresholds are at the top of validation.py.
- **History used for scoring excludes detected one-off quarters** (e.g. a demerger gain), and the result says which.
- **Peer control (P2):** peers come from `TRACK_C_PEERS` in research.py (3-5 per company); a company not in the map
  gets screener's own same-industry list. Peers are fetched numbers-only, in parallel. Result is
  company-specific / partly shared / sector-wide, with each peer's own reading shown.
- **Filing quotes (P3):** research.py reads the latest investor presentation, earnings-call transcript, results
  press releases and results filings from NSE's announcements feed (PDF text via PyMuPDF), and keeps sentences that
  state a fact, with page numbers. The quarter is set by the filing date; the document's own words only confirm it.
  A filing quote that states the same number we computed (same quarter) counts as an independent supporting source.
- **Price vs volume (P4):** UVG / USG are read from the filings in each company's own wording. `test_price_vs_volume`
  splits sales growth into volume and price/mix. Red-team checks specific to it: segment wording in the volume quote,
  USG far from reported revenue growth (= probably a segment figure, ignored), low volume base, like-for-like basis,
  and whether there was enough history to score it (else a fixed rule is used and the result says so).
