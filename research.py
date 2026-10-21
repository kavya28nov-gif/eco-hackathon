# research.py — Person 1: Public-Data Pipeline
import json
import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional
from ddgs import DDGS

# 1. Map common Track C (FMCG / Consumer) names to official NSE Symbols
TRACK_C_TICKERS = {
    "hul": "HINDUNILVR",
    "hindustan unilever": "HINDUNILVR",
    "hindustan unilever limited": "HINDUNILVR",
    "itc": "ITC",
    "itc limited": "ITC",
    "nestle": "NESTLEIND",
    "nestle india": "NESTLEIND",
    "britannia": "BRITANNIA",
    "britannia industries": "BRITANNIA",
    "tata consumer": "TATACONSUM",
    "tata consumer products": "TATACONSUM",
    "asian paints": "ASIANPAINT",
    "titan": "TITAN",
    "titan company": "TITAN",
    "trent": "TRENT",
    "dabur": "DABUR",
    "dabur india": "DABUR",
    "marico": "MARICO",
    "godrej consumer": "GODREJCP",
    "colgate": "COLPAL",
    "varun beverages": "VBL",
    "avenue supermarts": "DMART",
    "dmart": "DMART"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "data", "cache")

# How much history to keep. 12 quarters gives 8 year-on-year points, which is the
# minimum for the robust z-score in validation.py to mean something (8 quarters
# only gave 4 points, too thin -- one odd quarter could swing the whole score).
QUARTERS_TO_KEEP = 12
YEARS_TO_KEEP = 7

# NSE's own industry classification, as shown on screener's peers section.
# Track C = these broad sectors / sectors. We are not keeping a hand-made list of
# companies here, so a company the judges pick live is also classified correctly.
TRACK_C_BROAD_SECTORS = ("Fast Moving Consumer Goods",)
TRACK_C_SECTORS = ("Consumer Durables", "Consumer Services")

# 2. Peer map for the peer-control check (ONE place only -- app.py and tests read it from here).
# Rule for picking: same kind of product and same customer, listed in India, 3 to 5 names.
# If a company is not here, we fall back to screener's own same-industry peer list,
# so a company the judges pick live still gets peers.
TRACK_C_PEERS = {
    "HINDUNILVR": ["ITC", "NESTLEIND", "DABUR", "GODREJCP", "COLPAL"],
    "ITC":        ["HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR"],
    "NESTLEIND":  ["BRITANNIA", "HINDUNILVR", "TATACONSUM", "ITC"],
    "BRITANNIA":  ["NESTLEIND", "ITC", "TATACONSUM", "HINDUNILVR"],
    "TATACONSUM": ["HINDUNILVR", "NESTLEIND", "BRITANNIA", "MARICO"],
    "ASIANPAINT": ["BERGEPAINT", "KANSAINER", "INDIGOPNTS", "JSWDULUX"],
    "TITAN":      ["KALYANKJIL", "SENCO", "THANGAMAYL"],
    "TRENT":      ["DMART", "VMART", "SHOPERSTOP"],
    "DMART":      ["TRENT", "VMART", "SHOPERSTOP"],
    "DABUR":      ["MARICO", "GODREJCP", "COLPAL", "EMAMILTD", "HINDUNILVR"],
    "MARICO":     ["DABUR", "GODREJCP", "EMAMILTD", "COLPAL", "HINDUNILVR"],
    "GODREJCP":   ["DABUR", "MARICO", "HINDUNILVR", "EMAMILTD"],
    "COLPAL":     ["HINDUNILVR", "DABUR", "GODREJCP"],
}
MAX_PEERS = 5


def _search_screener(company_name: str) -> Optional[str]:
    """Asks screener's own search box for the ticker. Returns None if nothing found
    or the site is not reachable -- then the caller falls back to the raw input."""
    try:
        resp = requests.get("https://www.screener.in/api/company/search/",
                            params={"q": company_name, "v": 3, "fts": 1},
                            headers=HEADERS, timeout=15)
        for hit in resp.json():
            # url looks like "/company/JUBLFOOD/consolidated/"; the last item is
            # always "Search everywhere: ..." which has no company id, so skip that
            match = re.match(r"/company/([^/]+)/", hit.get("url") or "")
            if hit.get("id") and match:
                return match.group(1).upper()
    except Exception as e:
        print(f"[Warning] screener search failed for '{company_name}': {e}")
    return None


def resolve_ticker(company_name: str, allow_online: bool = True) -> str:
    """Converts a user-entered company name into an NSE ticker symbol.
    Order: exact alias -> alias as whole words inside the input -> screener search
    -> assume the user typed the ticker itself. allow_online=False is for offline
    mode, where we must not touch the network at all."""
    clean = company_name.lower().strip()
    if clean in TRACK_C_TICKERS:
        return TRACK_C_TICKERS[clean]
    # already a ticker we know (e.g. a peer like "KANSAINER") -- no need to search
    known = set(TRACK_C_TICKERS.values()) | set(TRACK_C_PEERS) | {p for ps in TRACK_C_PEERS.values() for p in ps}
    if company_name.strip().upper() in known:
        return company_name.strip().upper()
    # "Hindustan Unilever Ltd" should still hit "hindustan unilever". Whole-word match
    # only, because plain substring match was sending "in" to Hindustan Unilever.
    for key, ticker in sorted(TRACK_C_TICKERS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(key)}\b", clean):
            return ticker
    if allow_online:
        found = _search_screener(company_name)
        if found:
            return found
    # Otherwise assume the user typed the ticker directly (e.g. "EMAMILTD").
    # Tickers never have spaces, so "HDFC BANK" becomes "HDFCBANK".
    return re.sub(r"\s+", "", company_name.upper())


def _clean_number(val_str: str) -> Optional[float]:
    """Converts strings like '61,896' or '24%' into float, or None if missing."""
    if not val_str:
        return None
    cleaned = re.sub(r"[,%\s]", "", str(val_str))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _period_months(col_name: str, is_quarterly: bool) -> int:
    """Length of the period in months. Screener writes '15m' after the year when a
    company changes its year-end (e.g. 'Mar 2024 15m'). Such a year is NOT comparable
    with a normal 12 month year, so we record its real length and let validation skip it."""
    match = re.search(r"\d{4}\s*(\d{1,2})m\b", col_name)
    if match:
        return int(match.group(1))
    return 3 if is_quarterly else 12


def _parse_period(col_name: str, is_quarterly: bool) -> str:
    """
    Normalizes column headers like 'Mar 2024' into 'FY2024' (for annual)
    or 'Mar 2024' into 'Q4_FY2024' (for quarterly).
    An annual column that does not end in March (e.g. 'Dec 2023') is NOT an Indian
    financial year, so it stays as 'Dec_2023' instead of being passed off as FY2023.
    """
    col_name = col_name.strip()
    match = re.search(r"([A-Za-z]{3})\s+(\d{4})", col_name)
    if not match:
        return col_name.replace(" ", "_")

    month, year = match.group(1).title(), int(match.group(2))
    if not is_quarterly:
        if month != "Mar":
            return f"{month}_{year}"
        return f"FY{year}"
    
    # Indian FY mapping (Apr-Jun = Q1, Jul-Sep = Q2, Oct-Dec = Q3, Jan-Mar = Q4)
    if month == "Jun":
        return f"Q1_FY{year + 1}"
    elif month == "Sep":
        return f"Q2_FY{year + 1}"
    elif month == "Dec":
        return f"Q3_FY{year + 1}"
    elif month == "Mar":
        return f"Q4_FY{year}"
    return f"{month}_{year}"


def _latest_year(soup) -> int:
    """Latest year in the annual P&L header, 0 if the table is empty or missing.
    Only the section is not enough: a company with no subsidiaries has an empty
    consolidated page, and some pages have only the label column and no years."""
    pl = soup.find("section", id="profit-loss")
    table = pl.find("table", class_="data-table") if pl else None
    if not table or not table.find("thead"):
        return 0
    years = [int(y) for th in table.find("thead").find_all("th")
             for y in re.findall(r"\b(20\d{2}|19\d{2})", th.get_text())]
    return max(years, default=0)


def _get_screener_page(ticker: str):
    """Downloads the screener page once. Returns (soup, url) or (None, url).
    Consolidated is preferred, but only if it is current. Colgate's consolidated page,
    for example, stops in 2010 (old subsidiaries) while standalone is up to date --
    so when consolidated is behind, we also try standalone and keep the newer one."""
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    best, best_url, best_year = None, urls[0], 0
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except Exception as e:
            print(f"[Warning] screener not reachable for {ticker}: {e}")
            continue
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        year = _latest_year(soup)
        if year > best_year:
            best, best_url, best_year = soup, url, year
        # a consolidated page with last year's (or this year's) numbers is good enough
        if best_year >= int(time.strftime("%Y")) - 1:
            break
    return best, best_url


def scrape_industry_info(soup) -> Dict[str, str]:
    """Reads NSE's classification from screener's peers section, e.g.
    {'broad_sector': 'Fast Moving Consumer Goods', ..., 'industry': 'Packaged Foods'}.
    Empty dict if the section is not there -- validation then treats it as unknown."""
    info: Dict[str, str] = {}
    peers = soup.find("section", id="peers") if soup else None
    if not peers:
        return info
    for a in peers.find_all("a"):
        title = (a.get("title") or "").strip()
        if title in ("Broad Sector", "Sector", "Broad Industry", "Industry"):
            info[title.lower().replace(" ", "_")] = a.get_text(strip=True)
    return info


def _screener_peers(soup, ticker: str) -> List[str]:
    """Screener's own same-industry peers (ranked by market cap). Used only when the
    company is not in TRACK_C_PEERS. Empty list if anything fails."""
    tag = soup.find(attrs={"data-warehouse-id": True}) if soup else None
    if not tag:
        return []
    try:
        resp = requests.get(f"https://www.screener.in/api/company/{tag['data-warehouse-id']}/peers/",
                            headers=HEADERS, timeout=15)
        found = re.findall(r'href="/company/([^/]+)/', resp.text)
    except Exception as e:
        print(f"[Warning] screener peers failed for {ticker}: {e}")
        return []
    # keep order, drop the company itself and duplicates
    return [t for t in dict.fromkeys(found) if t != ticker][:MAX_PEERS]


def peers_for(ticker: str) -> List[str]:
    """Peer tickers for a company: our hand-checked map first, else what screener
    gave last time we fetched it (stored in the cache). Works offline."""
    if ticker in TRACK_C_PEERS:
        return TRACK_C_PEERS[ticker][:MAX_PEERS]
    if os.path.exists(cache_path(ticker)):
        with open(cache_path(ticker), encoding="utf-8") as f:
            return json.load(f).get("peer_tickers", [])
    return []


def is_track_c(industry_info: Dict[str, str]) -> Optional[bool]:
    """True / False from NSE's classification, None if we do not know (no info)."""
    if not industry_info:
        return None
    return (industry_info.get("broad_sector") in TRACK_C_BROAD_SECTORS
            or industry_info.get("sector") in TRACK_C_SECTORS)


def scrape_structured_financials(ticker: str, soup=None, used_url: str = "") -> List[Dict[str, Any]]:
    """
    Scrapes multi-year Annual and Quarterly P&L, Balance Sheet, and Cash Flow
    in INR Crore for any Indian listed company.
    Pass soup/used_url if the page is already downloaded, else it downloads itself.
    """
    records: List[Dict[str, Any]] = []
    if soup is None:
        soup, used_url = _get_screener_page(ticker)

    if not soup:
        return records

    # Mapping of raw row names to standardized metric names for Person 2
    row_metric_map = {
        "Sales": ("revenue_from_operations", "INR crore"),
        "Revenue": ("revenue_from_operations", "INR crore"),
        "Operating Profit": ("operating_profit", "INR crore"),
        "OPM %": ("operating_margin_pct", "percentage"),
        "Net Profit": ("net_profit", "INR crore"),
        "EPS in Rs": ("eps", "INR"),
        "Borrowings": ("total_debt", "INR crore"),
        "Other Liabilities": ("other_liabilities", "INR crore"),
        "Cash from Operating Activity": ("operating_cash_flow", "INR crore"),
        "Debtor Days": ("debtor_days", "days"),
        "Inventory Days": ("inventory_days", "days"),
        "Working Capital Days": ("working_capital_days", "days"),
        "ROCE %": ("roce_pct", "percentage"),
    }

    sections = [
        ("quarters", True, "Quarterly Financial Results Filing"),
        ("profit-loss", False, "Annual Profit & Loss Statement"),
        ("balance-sheet", False, "Annual Balance Sheet Filing"),
        ("cash-flow", False, "Annual Cash Flow Statement"),
        ("ratios", False, "Working Capital & Operational Ratios"),
    ]

    for section_id, is_quarterly, doc_label in sections:
        sec = soup.find("section", id=section_id)
        if not sec:
            continue
        table = sec.find("table", class_="data-table")
        if not table:
            continue

        # Extract period headers
        headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
        periods = headers[1:]  # First column is the metric name

        for tr in table.find("tbody").find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not cols or len(cols) < 2:
                continue

            raw_label = re.sub(r"\+$", "", cols[0]).strip()
            if raw_label not in row_metric_map:
                continue

            metric_name, unit = row_metric_map[raw_label]
            values = cols[1:]

            # Screener gives a TTM column also at the end. Drop it BEFORE slicing,
            # else it eats one of our slots and we get one quarter/year less.
            pairs = [(p, v) for p, v in zip(periods, values) if "TTM" not in p]
            slice_len = QUARTERS_TO_KEEP if is_quarterly else YEARS_TO_KEEP
            recent_pairs = pairs[-slice_len:]

            for raw_period, raw_val in recent_pairs:
                norm_period = _parse_period(raw_period, is_quarterly)
                val = _clean_number(raw_val)

                records.append({
                    "company": ticker,
                    "metric": metric_name,
                    "value": val,  # None if missing (never guess!)
                    "unit": unit,
                    "period": norm_period,
                    "period_type": "quarterly" if is_quarterly else "annual",
                    "period_months": _period_months(raw_period, is_quarterly),
                    "doc_title": f"{ticker} {doc_label} ({norm_period})",
                    "source_url": used_url
                })

    return records


def search_public_filings_and_excerpts(company_name: str, ticker: str) -> List[Dict[str, Any]]:
    """
    Uses DuckDuckGo Search to pull real links & excerpts from Investor Presentations,
    Earnings Call Transcripts, and Exchange Announcements for Track C angles
    (Volume vs Price growth, Quick Commerce/Rural distribution shifts, Working Capital).
    """
    excerpts: List[Dict[str, Any]] = []
    queries = [
        f"{company_name} {ticker} investor presentation underlying volume growth price quarterly results",
        f"{company_name} {ticker} earnings call transcript quick commerce modern trade rural demand",
        f"{company_name} {ticker} annual report trade receivables channel inventory filing"
    ]

    try:
        with DDGS() as ddgs:
            for q in queries:
                results = list(ddgs.text(q, max_results=3))
                for r in results:
                    excerpts.append({
                        "company": ticker,
                        "topic": q.split(ticker)[-1].strip(),
                        "doc_title": r.get("title", "Public Filing / Financial Report"),
                        "excerpt": r.get("body", ""),
                        "source_url": r.get("href", "")
                    })
                time.sleep(0.5)
    except Exception as e:
        print(f"[Warning] Live search encountered an issue: {e}")

    return excerpts


def fetch_company_data(company_name: str, save_to_cache: bool = True,
                       include_text: bool = True) -> Dict[str, Any]:
    """
    MAIN FUNCTION CALLED BY PERSON 2 AND PERSON 4.
    Accepts any company name (e.g., 'HUL', 'ITC', 'Britannia') and returns
    structured financial records + source-linked qualitative excerpts.
    A successful fetch is also saved to data/cache/<TICKER>.json, so the offline
    demo mode has a backup for every company we have ever run.
    include_text=False skips searches and PDFs -- used for peers, where we only
    need the numbers, so 5 peers take seconds and not minutes.
    """
    ticker = resolve_ticker(company_name)
    flags: List[str] = []

    # 1. Fetch structured multi-year & multi-quarter numbers (+ NSE industry, from the same page)
    soup, used_url = _get_screener_page(ticker)
    financial_records = scrape_structured_financials(ticker, soup, used_url)
    industry_info = scrape_industry_info(soup)
    peer_tickers = TRACK_C_PEERS.get(ticker) or _screener_peers(soup, ticker)
    if not financial_records:
        flags.append(f"No financial tables found on screener for ticker '{ticker}' "
                     f"(wrong name, or site not reachable).")
    elif soup is not None and _latest_year(soup) < int(time.strftime("%Y")) - 1:
        flags.append(f"Latest annual data for {ticker} is from {_latest_year(soup)} -- "
                     f"stale, findings may not describe the company today.")
    odd = sorted({r["period"] for r in financial_records if r["period_months"] not in (3, 12)
                  or (r["period_type"] == "annual" and not r["period"].startswith("FY"))})
    if odd:
        flags.append(f"Non-standard periods (year-end change / not 12 months): {', '.join(odd)}. "
                     f"Kept as-is, but not comparable with normal years.")

    # 2. Fetch qualitative excerpts & filing URLs for triangulation
    text_excerpts = search_public_filings_and_excerpts(company_name, ticker) if include_text else []

    data = {
        "company": company_name,
        "ticker": ticker,
        "sector": "Track C - Consumer, FMCG & Retail",
        "record_count": len(financial_records),
        "financial_records": financial_records,
        "text_excerpts": text_excerpts,
        # added fields (old fields above are frozen, do not rename them)
        "industry_info": industry_info,
        "is_track_c": is_track_c(industry_info),
        "peer_tickers": peer_tickers,
        "data_quality_flags": flags,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    if save_to_cache and financial_records:
        if not include_text and os.path.exists(cache_path(ticker)):
            # HUL is also ITC's peer. A light peer fetch must not wipe out the quotes
            # we saved when HUL itself was run, so we carry the old text across.
            with open(cache_path(ticker), encoding="utf-8") as f:
                old = json.load(f)
            data["text_excerpts"] = old.get("text_excerpts", [])
            data["company"] = old.get("company", data["company"])   # keep "HUL", not "HINDUNILVR"
        save_cache(data)
    return data


def fetch_peers(tickers: List[str], progress=None) -> Dict[str, Dict[str, Any]]:
    """Numbers-only fetch for each peer, all at the same time (threads), because
    one by one 5 peers was taking ~50 seconds. A peer that fails is just skipped."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(fetch_company_data, t, True, False): t for t in tickers}
        for job in as_completed(jobs):
            t = jobs[job]
            try:
                d = job.result()
            except Exception as e:
                d = None
                print(f"[Warning] peer {t} failed: {e}")
            if d and d.get("financial_records"):
                out[t] = d
            if progress:
                progress(f"Peer {t}: {'ok' if t in out else 'no data, skipped'}")
    return out


# ---------------------------------------------------------------------------
# Offline cache -- data/cache/<TICKER>.json, one file per company
# ---------------------------------------------------------------------------

def cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.json")


def save_cache(data: Dict[str, Any]) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(data["ticker"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    return path


def cached_tickers() -> List[str]:
    if not os.path.isdir(CACHE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(CACHE_DIR) if f.endswith(".json"))


def load_cache(company_name: str) -> Optional[Dict[str, Any]]:
    """Offline mode: finds the cached file for whatever the user typed, without any
    network call. Tries the alias table, then the typed text as a ticker, then the
    company name stored inside each cached file. None if we have never cached it."""
    candidates = [resolve_ticker(company_name, allow_online=False)]
    typed = company_name.lower().strip()
    for t in cached_tickers():
        try:
            with open(cache_path(t), encoding="utf-8") as f:
                stored_name = json.load(f).get("company", "").lower().strip()
        except (OSError, json.JSONDecodeError):
            continue
        if stored_name == typed:
            candidates.append(t)
    for t in candidates:
        if os.path.exists(cache_path(t)):
            with open(cache_path(t), encoding="utf-8") as f:
                return json.load(f)
    return None


# Quick test when you run `python research.py` in your terminal
if __name__ == "__main__":
    print("Testing fetch_company_data('HUL')...")
    data = fetch_company_data("HUL")
    print(f"Ticker resolved: {data['ticker']}")
    print(f"Total structured financial records extracted: {data['record_count']}")
    
    # Show first 5 records
    df = pd.DataFrame(data["financial_records"])
    print("\nSample Financial Records:")
    print(df.head(10).to_string(index=False))
    
    print(f"\nTotal qualitative excerpts extracted: {len(data['text_excerpts'])}")
    if data["text_excerpts"]:
        print("Sample Excerpt:", data["text_excerpts"][0])
