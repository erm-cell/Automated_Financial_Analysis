import os
import re
import math
import hashlib
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf

# ============================================================
# APP SETTINGS AND DISPLAY STYLE
# ============================================================
st.set_page_config(
    page_title="Global Stock Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.25 !important;
        min-height: 1.5rem;
        font-size: 0.88rem !important;
    }
    [data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        font-size: clamp(1.15rem, 2.2vw, 1.85rem) !important;
    }
    [data-testid="stMetricDelta"] {
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stDataFrame"] {
        width: 100%;
    }
    .small-note {font-size: 0.86rem; opacity: 0.82;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Global Stock Research")
st.caption(
    "One dashboard for price history, company health, valuation, news, "
    "country risk, world events, and one year scenarios."
)

# ============================================================
# GENERAL HELPERS
# ============================================================

def get_secret(name: str) -> str:
    """Read a key from Streamlit secrets or environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fmt_price(value, currency=""):
    if pd.isna(value):
        return "Not available"
    return f"{currency} {float(value):,.2f}".strip()


def fmt_pct(value, digits=1):
    if pd.isna(value):
        return "Not available"
    return f"{float(value):+.{digits}f}%"


def fmt_money(value, currency=""):
    if pd.isna(value):
        return "Not reported"
    value = float(value)
    absolute = abs(value)
    if absolute >= 1_000_000_000_000:
        return f"{currency} {value / 1_000_000_000_000:,.2f}T".strip()
    if absolute >= 1_000_000_000:
        return f"{currency} {value / 1_000_000_000:,.2f}B".strip()
    if absolute >= 1_000_000:
        return f"{currency} {value / 1_000_000:,.2f}M".strip()
    return f"{currency} {value:,.2f}".strip()


def normalise_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def flatten_dict(data, parent="", sep=" / "):
    output = {}
    if not isinstance(data, dict):
        return output
    for key, value in data.items():
        name = f"{parent}{sep}{key}" if parent else str(key)
        if isinstance(value, dict):
            output.update(flatten_dict(value, name, sep))
        else:
            output[name] = value
    return output


def recursive_value(data, aliases):
    wanted = {normalise_key(x) for x in aliases}
    if isinstance(data, dict):
        for key, value in data.items():
            if normalise_key(key) in wanted:
                number = safe_float(value)
                if pd.notna(number):
                    return number
        for value in data.values():
            found = recursive_value(value, aliases)
            if pd.notna(found):
                return found
    elif isinstance(data, list):
        for item in data:
            found = recursive_value(item, aliases)
            if pd.notna(found):
                return found
    return np.nan


def dataframe_value(frame, aliases):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return np.nan
    lookup = {normalise_key(index): index for index in frame.index}
    for alias in aliases:
        match = lookup.get(normalise_key(alias))
        if match is not None:
            row = frame.loc[match]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            values = pd.to_numeric(row, errors="coerce").dropna()
            if len(values):
                return safe_float(values.iloc[0])
    return np.nan


def latest_record(rows):
    if not isinstance(rows, list) or not rows:
        return {}
    valid = [x for x in rows if isinstance(x, dict)]
    if not valid:
        return {}
    valid.sort(
        key=lambda x: str(x.get("fiscal_date") or x.get("date") or ""),
        reverse=True,
    )
    return valid[0]


def statement_summary(rows, kind, limit=5):
    """Show a small set of familiar statement lines with short column names."""
    fields = {
        "income": [
            ("Revenue", ["total_revenue", "revenue", "sales", "total revenue"]),
            ("Gross profit", ["gross_profit", "gross_profit_value"]),
            ("Operating profit", ["operating_income", "operating_income_value"]),
            ("Net income", ["net_income", "net_income_common_stockholders", "net_income_to_common", "net_income_continuous_operations"]),
            ("EPS", ["diluted_eps", "diluted_eps_ttm", "basic_eps"]),
        ],
        "balance": [
            ("Total assets", ["total_assets"]),
            ("Cash", ["cash_and_cash_equivalents", "cash_cash_equivalents_and_short_term_investments", "total_cash"]),
            ("Total debt", ["total_debt", "long_term_debt"]),
            ("Equity", ["total_stockholders_equity", "total_stockholder_equity", "stockholders_equity", "shareholders_equity"]),
            ("Current ratio", ["current_ratio"]),
        ],
        "cashflow": [
            ("Operating cash", ["operating_cash_flow", "cash_flow_from_continuing_operating_activities"]),
            ("Capital spend", ["capital_expenditures", "capital_expenditure"]),
            ("Free cash flow", ["free_cash_flow"]),
            ("Dividends paid", ["dividend_paid", "dividends_paid"]),
        ],
    }
    selected = fields[kind]
    records = []

    if isinstance(rows, pd.DataFrame) and not rows.empty:
        index_lookup = {normalise_key(index): index for index in rows.index}
        columns = list(rows.columns)
        try:
            columns = sorted(columns, reverse=True)
        except Exception:
            pass
        for column in columns[:limit]:
            record = {"Report date": str(column)[:10]}
            for label, aliases in selected:
                value = np.nan
                for alias in aliases:
                    index = index_lookup.get(normalise_key(alias))
                    if index is not None:
                        value = safe_float(rows.loc[index, column])
                        break
                record[label] = value
            records.append(record)

    elif isinstance(rows, list):
        valid = [row for row in rows if isinstance(row, dict)]
        valid.sort(key=lambda row: str(row.get("fiscal_date") or row.get("date") or ""), reverse=True)
        for row in valid[:limit]:
            record = {"Report date": str(row.get("fiscal_date") or row.get("date") or "Not reported")[:10]}
            for label, aliases in selected:
                record[label] = recursive_value(row, aliases)
            records.append(record)

    return pd.DataFrame(records)

# ============================================================
# GLOBAL PRICE DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_yahoo_data(symbol, period):
    ticker = yf.Ticker(symbol)
    history = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=False,
        timeout=25,
    )
    if history is None or history.empty or "Close" not in history:
        raise ValueError(
            "No price history was returned. Check the Yahoo ticker format "
            "and whether Yahoo covers that exchange."
        )
    history = history.copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=["Close"]).sort_index()
    info = {}
    try:
        info = ticker.get_info() or {}
    except Exception:
        try:
            info = ticker.info or {}
        except Exception:
            info = {}
    statements = {}
    for key, attr in [
        ("income", "financials"),
        ("balance", "balance_sheet"),
        ("cashflow", "cashflow"),
    ]:
        try:
            value = getattr(ticker, attr)
            statements[key] = value if isinstance(value, pd.DataFrame) else pd.DataFrame()
        except Exception:
            statements[key] = pd.DataFrame()
    meta = {
        "currency": info.get("currency") or "",
        "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
        "country": info.get("country") or "",
        "company": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "provider": "Yahoo Finance",
        "profile": info,
        "statements": statements,
    }
    return history, meta


def twelve_request(endpoint, params, api_key):
    request_params = dict(params)
    request_params["apikey"] = api_key
    response = requests.get(
        f"https://api.twelvedata.com/{endpoint}",
        params=request_params,
        timeout=30,
        headers={"User-Agent": "GlobalStockResearch/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Twelve Data returned an error."))
    if isinstance(payload, dict) and payload.get("code") and payload.get("message"):
        raise RuntimeError(payload["message"])
    return payload


@st.cache_data(ttl=900, show_spinner=False)
def load_twelve_data(symbol, exchange, api_key, period):
    period_days = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650}.get(period, 1825)
    start_date = (datetime.now(timezone.utc) - timedelta(days=period_days)).date().isoformat()
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start_date,
        "outputsize": 5000,
        "order": "asc",
        "adjust": "all",
    }
    if exchange.strip():
        params["exchange"] = exchange.strip()
    payload = twelve_request("time_series", params, api_key)
    values = payload.get("values", []) if isinstance(payload, dict) else []
    if not values:
        raise ValueError(
            "Twelve Data returned no prices. Check the symbol, exchange, "
            "API plan, and API key."
        )
    history = pd.DataFrame(values)
    if "datetime" not in history.columns:
        raise ValueError("The price response did not include dates.")
    history["Date"] = pd.to_datetime(history["datetime"], errors="coerce")
    history = history.set_index("Date").sort_index()
    history.index.name = "Date"
    for column in ["open", "high", "low", "close", "volume"]:
        if column in history.columns:
            history[column.title()] = pd.to_numeric(history[column], errors="coerce")
    if "Close" not in history.columns:
        raise ValueError("The price response did not include closing prices.")
    history = history.dropna(subset=["Close"])
    history = history[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in history.columns]]

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    profile = {}
    statements = {"income": [], "balance": [], "cashflow": []}
    statement_errors = []

    # These endpoints may require a higher subscription tier.
    for endpoint, key in [
        ("profile", "profile"),
        ("income_statement", "income"),
        ("balance_sheet", "balance"),
        ("cash_flow", "cashflow"),
    ]:
        try:
            statement_payload = twelve_request(
                endpoint,
                {"symbol": symbol, "period": "annual", "outputsize": 5, **({"exchange": exchange.strip()} if exchange.strip() else {})},
                api_key,
            )
            if key == "profile":
                profile = statement_payload if isinstance(statement_payload, dict) else {}
            else:
                statements[key] = statement_payload.get(key, []) if isinstance(statement_payload, dict) else []
        except Exception as exc:
            statement_errors.append(f"{endpoint}: {exc}")

    company = profile.get("name") or meta.get("name") or symbol
    combined_meta = {
        "currency": meta.get("currency") or profile.get("currency") or "",
        "exchange": meta.get("exchange") or profile.get("exchange") or exchange,
        "country": profile.get("country") or "",
        "company": company,
        "sector": profile.get("industry") or profile.get("sector") or "",
        "industry": profile.get("industry") or "",
        "provider": "Twelve Data",
        "profile": profile,
        "statements": statements,
        "statement_errors": statement_errors,
    }
    return history, combined_meta

# ============================================================
# TECHNICAL AND FUNDAMENTAL FACTORS
# ============================================================

def rsi_value(prices, window=14):
    change = prices.diff()
    gain = change.clip(lower=0).rolling(window).mean()
    loss = (-change.clip(upper=0)).rolling(window).mean()
    relative = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative))
    return safe_float(rsi.iloc[-1]) if len(rsi.dropna()) else np.nan


def price_metrics(history):
    close = history["Close"].dropna().astype(float)
    daily_returns = close.pct_change().dropna()
    if len(close) < 2:
        raise ValueError("At least two closing prices are needed.")
    latest = float(close.iloc[-1])
    annual_vol = float(daily_returns.std(ddof=1) * np.sqrt(252) * 100) if len(daily_returns) > 1 else np.nan
    sma50 = safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else np.nan
    sma200 = safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    high_water = close.cummax()
    drawdown = (close / high_water - 1.0) * 100
    max_drawdown = float(drawdown.min())

    def past_return(days):
        if len(close) <= days:
            return np.nan
        return float((latest / close.iloc[-days - 1] - 1) * 100)

    return {
        "close": close,
        "returns": daily_returns,
        "latest": latest,
        "median": float(close.median()),
        "average": float(close.mean()),
        "high_52w": float(close.tail(252).max()),
        "low_52w": float(close.tail(252).min()),
        "return_1m": past_return(21),
        "return_3m": past_return(63),
        "return_6m": past_return(126),
        "return_1y": past_return(252),
        "volatility": annual_vol,
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": rsi_value(close),
        "max_drawdown": max_drawdown,
        "last_date": close.index[-1],
    }


def get_fundamental_metrics(meta):
    info = meta.get("profile", {}) or {}
    statements = meta.get("statements", {}) or {}
    income = statements.get("income", pd.DataFrame())
    balance = statements.get("balance", pd.DataFrame())
    cashflow = statements.get("cashflow", pd.DataFrame())

    if isinstance(income, pd.DataFrame):
        revenue = dataframe_value(income, ["Total Revenue", "Revenue", "Sales"])
        gross_profit = dataframe_value(income, ["Gross Profit"])
        operating_income = dataframe_value(income, ["Operating Income", "Operating Income Loss"])
        net_income = dataframe_value(income, ["Net Income", "Net Income Common Stockholders"])
        revenue_series = pd.Series(dtype=float)
        try:
            for alias in ["Total Revenue", "Revenue", "Sales"]:
                rows = {normalise_key(i): i for i in income.index}
                match = rows.get(normalise_key(alias))
                if match is not None:
                    revenue_series = pd.to_numeric(income.loc[match], errors="coerce").dropna()
                    break
        except Exception:
            pass
        revenue_growth = np.nan
        if len(revenue_series) > 1 and revenue_series.iloc[1] != 0:
            revenue_growth = (float(revenue_series.iloc[0]) / float(revenue_series.iloc[1]) - 1) * 100
    else:
        income_rows = income if isinstance(income, list) else []
        latest_income = latest_record(income_rows)
        revenue = recursive_value(latest_income, ["total_revenue", "revenue", "sales"])
        gross_profit = recursive_value(latest_income, ["gross_profit", "gross_profit_value"])
        operating_income = recursive_value(latest_income, ["operating_income", "operating_income_value"])
        net_income = recursive_value(latest_income, ["net_income", "net_income_common_stockholders", "net_income_to_common"])
        ordered = sorted(income_rows, key=lambda x: str(x.get("fiscal_date", "")), reverse=True)
        revenue_growth = np.nan
        if len(ordered) > 1:
            newer = recursive_value(ordered[0], ["total_revenue", "revenue", "sales"])
            older = recursive_value(ordered[1], ["total_revenue", "revenue", "sales"])
            if pd.notna(newer) and pd.notna(older) and older != 0:
                revenue_growth = (newer / older - 1) * 100

    if isinstance(balance, pd.DataFrame):
        total_debt = dataframe_value(balance, ["Total Debt", "Long Term Debt"])
        cash = dataframe_value(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments"])
        equity = dataframe_value(balance, ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"])
        current_assets = dataframe_value(balance, ["Current Assets"])
        current_liabilities = dataframe_value(balance, ["Current Liabilities"])
    else:
        balance_rows = balance if isinstance(balance, list) else []
        latest_balance = latest_record(balance_rows)
        total_debt = recursive_value(latest_balance, ["total_debt", "long_term_debt"])
        cash = recursive_value(latest_balance, ["cash_and_cash_equivalents", "cash_cash_equivalents_and_short_term_investments", "total_cash"])
        equity = recursive_value(latest_balance, ["total_stockholders_equity", "total_stockholder_equity", "shareholders_equity"])
        current_assets = recursive_value(latest_balance, ["total_current_assets"])
        current_liabilities = recursive_value(latest_balance, ["total_current_liabilities"])

    if isinstance(cashflow, pd.DataFrame):
        operating_cash_flow = dataframe_value(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        capex = dataframe_value(cashflow, ["Capital Expenditure", "Capital Expenditures"])
        free_cash_flow = dataframe_value(cashflow, ["Free Cash Flow"])
    else:
        cash_rows = cashflow if isinstance(cashflow, list) else []
        latest_cash = latest_record(cash_rows)
        operating_cash_flow = recursive_value(latest_cash, ["operating_cash_flow", "cash_flow_from_continuing_operating_activities"])
        capex = recursive_value(latest_cash, ["capital_expenditures", "capital_expenditure"])
        free_cash_flow = recursive_value(latest_cash, ["free_cash_flow"])

    if pd.isna(free_cash_flow) and pd.notna(operating_cash_flow) and pd.notna(capex):
        free_cash_flow = operating_cash_flow + capex if capex < 0 else operating_cash_flow - capex

    # Some providers report debt/equity as a percentage. Convert to a ratio.
    debt_equity = safe_float(info.get("debtToEquity"))
    if pd.isna(debt_equity) and pd.notna(total_debt) and pd.notna(equity) and equity != 0:
        debt_equity = total_debt / equity
    elif pd.notna(debt_equity) and debt_equity > 10:
        debt_equity = debt_equity / 100.0

    current_ratio = (
        current_assets / current_liabilities
        if pd.notna(current_assets) and pd.notna(current_liabilities) and current_liabilities != 0
        else np.nan
    )

    gross_margin = gross_profit / revenue * 100 if pd.notna(gross_profit) and pd.notna(revenue) and revenue else np.nan
    operating_margin = operating_income / revenue * 100 if pd.notna(operating_income) and pd.notna(revenue) and revenue else np.nan
    net_margin = net_income / revenue * 100 if pd.notna(net_income) and pd.notna(revenue) and revenue else np.nan

    return {
        "revenue": revenue,
        "revenue_growth": revenue_growth,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "operating_income": operating_income,
        "operating_margin": operating_margin,
        "net_income": net_income,
        "net_margin": net_margin,
        "operating_cash_flow": operating_cash_flow,
        "capex": capex,
        "free_cash_flow": free_cash_flow,
        "cash": cash,
        "debt": total_debt,
        "equity": equity,
        "debt_equity": debt_equity,
        "current_ratio": current_ratio,
        "trailing_pe": safe_float(info.get("trailingPE")),
        "forward_pe": safe_float(info.get("forwardPE")),
        "price_to_book": safe_float(info.get("priceToBook")),
        "ev_ebitda": safe_float(info.get("enterpriseToEbitda")),
        "dividend_yield": safe_float(info.get("dividendYield")),
        "market_cap": safe_float(info.get("marketCap")),
        "industry_pe": safe_float(info.get("industryPE")),
        "sector_pe": safe_float(info.get("sectorPE")),
        "target_mean": safe_float(info.get("targetMeanPrice")),
        "target_high": safe_float(info.get("targetHighPrice")),
        "target_low": safe_float(info.get("targetLowPrice")),
        "analyst_count": safe_float(info.get("numberOfAnalystOpinions")),
    }


def financial_quality(metrics):
    checks = []
    score = 50.0
    measured = 0

    def add(name, value, good_text, bad_text, points):
        nonlocal score, measured
        if pd.isna(value):
            checks.append((name, "Not reported", 0.0))
        elif value > 0:
            measured += 1
            score += points
            checks.append((name, good_text, points))
        else:
            measured += 1
            score -= points
            checks.append((name, bad_text, -points))

    add("Revenue growth", metrics["revenue_growth"], "Growing", "Falling", 15)
    add("Operating profit", metrics["operating_margin"], "Positive margin", "Loss or no margin", 12)
    add("Net income", metrics["net_income"], "Profit", "Loss", 8)
    add("Free cash flow", metrics["free_cash_flow"], "Cash positive", "Cash negative", 10)

    debt_ratio = metrics["debt_equity"]
    if pd.isna(debt_ratio):
        checks.append(("Debt compared with equity", "Not reported", 0.0))
    elif debt_ratio <= 1:
        measured += 1
        score += 5
        checks.append(("Debt compared with equity", "Lower debt", 5.0))
    elif debt_ratio >= 2:
        measured += 1
        score -= 5
        checks.append(("Debt compared with equity", "Higher debt", -5.0))
    else:
        measured += 1
        checks.append(("Debt compared with equity", "Middle range", 0.0))

    if measured == 0:
        return None, checks
    score = float(np.clip(score, 0, 100))
    return score, checks

# ============================================================
# RECENT NEWS AND WORLD EVENTS
# ============================================================

POSITIVE_WORDS = [
    "earnings beat", "revenue growth", "profit growth", "record revenue",
    "strong demand", "margin expansion", "new contract", "product launch",
    "approval granted", "ceasefire", "peace deal", "sanctions lifted",
    "trade deal", "tariff reduction", "inflation cools", "rate cut",
    "debt reduced", "cash flow improves", "guidance raised",
]
NEGATIVE_WORDS = [
    "earnings miss", "revenue decline", "profit decline", "weak demand",
    "margin falls", "lawsuit", "antitrust", "investigation", "recall",
    "fine imposed", "guidance cut", "supply disruption", "factory shutdown",
    "war", "invasion", "airstrike", "missile", "armed conflict",
    "sanctions imposed", "sanctions expanded", "export ban", "export controls",
    "blockade", "tariff increase", "trade restrictions", "shipping disruption",
    "recession", "inflation rises", "rate hike", "debt crisis",
]

EVENT_TAGS = {
    "War and conflict": ["war", "invasion", "airstrike", "missile", "armed conflict", "military operation", "ceasefire"],
    "Sanctions and trade": ["sanction", "export ban", "export control", "tariff", "trade restriction", "embargo"],
    "Supply chain": ["supply chain", "factory shutdown", "shipping disruption", "port closure", "shortage", "production delay"],
    "Company results": ["earnings", "revenue", "profit", "guidance", "quarterly results", "annual results"],
    "Law and rules": ["lawsuit", "antitrust", "investigation", "regulation", "fine", "court"],
    "Economy": ["inflation", "interest rate", "central bank", "GDP", "unemployment", "recession", "bond yield"],
}


def classify_headline(title):
    text = (title or "").lower()
    positive = [term for term in POSITIVE_WORDS if term in text]
    negative = [term for term in NEGATIVE_WORDS if term in text]
    score = float(np.clip(0.5 * len(positive) - 0.55 * len(negative), -1, 1))
    tags = [label for label, words in EVENT_TAGS.items() if any(word in text for word in words)]
    if not tags:
        tags = ["General news"]
    if positive and negative:
        signal = "Mixed headline"
    elif positive:
        signal = "Positive terms"
    elif negative:
        signal = "Risk terms"
    else:
        signal = "No clear signal"
    reason = ", ".join((positive + negative)[:4]) if positive or negative else "No matched terms"
    return score, signal, ", ".join(tags), reason


@st.cache_data(ttl=900, show_spinner=False)
def get_recent_news(query, days, max_records=75):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(days))
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(int(max_records), 250),
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": now.strftime("%Y%m%d%H%M%S"),
    }
    response = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params=params,
        timeout=30,
        headers={"User-Agent": "GlobalStockResearch/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("articles", []) if isinstance(payload, dict) else []
    articles = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        url = row.get("url") or row.get("url_mobile")
        raw_seen = str(row.get("seendate") or "").strip()
        indexed = pd.to_datetime(raw_seen, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
        if pd.isna(indexed):
            indexed = pd.to_datetime(raw_seen, utc=True, errors="coerce")
        if not title or not url or pd.isna(indexed):
            continue
        if indexed < pd.Timestamp(start):
            continue
        score, signal, tag, reason = classify_headline(title)
        age_days = max(0.0, (pd.Timestamp(now) - indexed).total_seconds() / 86400)
        articles.append({
            "title": title,
            "url": url,
            "domain": row.get("domain") or "Source not named",
            "indexed": indexed,
            "source_country": row.get("sourcecountry") or "",
            "language": row.get("language") or "",
            "score": score,
            "signal": signal,
            "tag": tag,
            "reason": reason,
            "weight": math.exp(-age_days / 5.0),
        })
    return articles


def weighted_news_score(articles):
    if not articles:
        return np.nan
    weights = np.array([x["weight"] for x in articles], dtype=float)
    scores = np.array([x["score"] for x in articles], dtype=float)
    if weights.sum() == 0:
        return float(scores.mean())
    return float(np.average(scores, weights=weights))


def build_news_queries(company, symbol, home_country, regions):
    company_name = (company or symbol).strip()
    company_query = f'"{company_name}" OR "{symbol}"'
    country = (home_country or "").strip()
    region_terms = [x.strip() for x in regions.split(",") if x.strip()]
    watched_places = ([country] if country else []) + [
        x for x in region_terms if x.lower() != country.lower()
    ]
    place_query = " OR ".join(f'"{x}"' for x in watched_places)
    macro_terms = (
        'inflation OR "interest rates" OR "central bank" OR GDP OR '
        'unemployment OR recession OR tariffs OR trade OR budget OR currency '
        'OR export OR imports OR energy'
    )
    country_query = (
        f'({place_query}) AND ({macro_terms})'
        if place_query else f'({macro_terms})'
    )
    # This global query is not restricted to the watchlist, so conflicts outside
    # the selected regions can still appear in the news results.
    world_query = (
        '(war OR conflict OR sanctions OR invasion OR airstrike OR missile '
        'OR blockade OR ceasefire OR "export controls" OR tariffs OR embargo '
        'OR "shipping disruption" OR "military operation" OR "trade restrictions" '
        'OR "supply chain disruption")'
    )
    return {
        "Company": company_query,
        "Country and regions": country_query,
        "World risk": world_query,
    }

# ============================================================
# PRICE FORECAST AND RISK
# ============================================================

def moving_block_bootstrap(log_returns, current_price, annual_adjustment_pct, seed):
    """100,000 one-year outcomes using 5-day historical return blocks."""
    values = np.asarray(log_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 60:
        raise ValueError("At least 60 daily returns are needed for the forecast.")

    simulations = 100_000
    horizon = 252
    block_length = 5
    blocks_needed = math.ceil(horizon / block_length)
    rng = np.random.default_rng(seed)
    base_terminal = np.empty(simulations, dtype=float)
    chunk_size = 1_000
    adjustment_daily = (annual_adjustment_pct / 100.0) / horizon

    # Use circular blocks to retain short runs of market movement.
    offsets = np.arange(block_length)
    for start in range(0, simulations, chunk_size):
        stop = min(start + chunk_size, simulations)
        n = stop - start
        block_starts = rng.integers(0, len(values), size=(n, blocks_needed))
        indices = (block_starts[:, :, None] + offsets[None, None, :]) % len(values)
        sampled = values[indices].reshape(n, -1)[:, :horizon]
        total_log_return = sampled.sum(axis=1)
        base_terminal[start:stop] = current_price * np.exp(total_log_return)

    adjusted_terminal = base_terminal * np.exp(annual_adjustment_pct / 100.0)
    batches = adjusted_terminal.reshape(500, 200)
    batch_loss = (batches < current_price).mean(axis=1) * 100

    p05, median, p95 = np.percentile(adjusted_terminal, [5, 50, 95])
    lower_tail = adjusted_terminal[adjusted_terminal <= p05]
    var95 = max(0.0, (current_price - p05) / current_price * 100)
    cvar95 = max(0.0, (current_price - float(lower_tail.mean())) / current_price * 100) if len(lower_tail) else np.nan

    preview_count = 24
    starts = rng.integers(0, len(values), size=(preview_count, blocks_needed))
    idx = (starts[:, :, None] + offsets[None, None, :]) % len(values)
    sampled = values[idx].reshape(preview_count, -1)[:, :horizon]
    preview = current_price * np.exp(
        np.cumsum(sampled + adjustment_daily, axis=1)
    )

    return {
        "base_terminal": base_terminal,
        "terminal": adjusted_terminal,
        "median": float(median),
        "p05": float(p05),
        "p95": float(p95),
        "base_median": float(np.median(base_terminal)),
        "chance_up": float((adjusted_terminal > current_price).mean() * 100),
        "chance_down": float((adjusted_terminal < current_price).mean() * 100),
        "batch_median": float(np.median(batch_loss)),
        "batch_low": float(np.percentile(batch_loss, 5)),
        "batch_high": float(np.percentile(batch_loss, 95)),
        "var95": float(var95),
        "cvar95": float(cvar95),
        "preview": preview,
        "simulations": simulations,
        "batches": 500,
        "items_per_batch": 200,
    }


def factor_adjustments(price_data, fundamentals, quality_score, info, news_scores):
    factors = []

    # Recent price momentum is a small scenario input, not a guarantee it continues.
    momentum = price_data["return_6m"]
    if pd.notna(momentum):
        impact = float(np.clip(momentum * 0.04, -1.5, 1.5))
        factors.append(("Price momentum", impact, f"Six month return: {momentum:+.1f}%. Small capped trend scenario."))
    else:
        factors.append(("Price momentum", 0.0, "Not enough price history. No adjustment."))

    # Company health score is a simple rule score based on reported financial data.
    if quality_score is not None:
        impact = float(np.clip((quality_score - 50) / 50 * 1.5, -1.5, 1.5))
        factors.append(("Company health", impact, f"Financial health score: {quality_score:.0f}/100. Rule based, not industry adjusted."))
    else:
        factors.append(("Company health", 0.0, "Financial data not available. No adjustment."))

    # Relative valuation is used only when the provider supplies an industry PE.
    pe = fundamentals.get("trailing_pe", np.nan)
    industry_pe = fundamentals.get("industry_pe", np.nan)
    sector_pe = fundamentals.get("sector_pe", np.nan)
    peer_pe = industry_pe if pd.notna(industry_pe) and industry_pe > 0 else sector_pe
    if pd.notna(pe) and pe > 0 and pd.notna(peer_pe) and peer_pe > 0:
        premium = pe / peer_pe - 1
        impact = float(np.clip(-premium * 1.0, -1.0, 1.0))
        factors.append(("Valuation", impact, f"P/E {pe:.1f}x compared with peer reference {peer_pe:.1f}x. Limited valuation scenario."))
    else:
        factors.append(("Valuation", 0.0, "No usable peer P/E reference. Displayed, but not used in forecast."))

    news_caps = {"Company news": 1.0, "Country news": 0.5, "World risk news": 1.0}
    for key, cap in news_caps.items():
        score = news_scores.get(key, np.nan)
        if pd.notna(score):
            impact = float(np.clip(score, -1, 1) * cap)
            factors.append((key, impact, f"Recent headline score: {score:+.2f}. Assumed impact capped at ±{cap:.1f} points."))
        else:
            factors.append((key, 0.0, "No recent headlines. No adjustment."))

    total = sum(row[1] for row in factors)
    total = float(np.clip(total, -5.0, 5.0))
    return factors, total

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Stock settings")
provider = st.sidebar.selectbox(
    "Price source",
    ["Yahoo Finance", "Twelve Data"],
    help="Yahoo uses exchange suffixes. Twelve Data uses a ticker and optional exchange name or MIC code.",
)
ticker = st.sidebar.text_input("Stock ticker", value="AAPL").strip().upper()
exchange = ""
api_key = ""
if provider == "Yahoo Finance":
    st.sidebar.caption("Examples: AAPL, 7203.T, 0700.HK, RELIANCE.NS, VOD.L, SAP.DE")
else:
    exchange = st.sidebar.text_input("Exchange (optional)", value="", help="Examples: NASDAQ, XNAS, Tokyo Stock Exchange")
    api_key = st.sidebar.text_input(
        "Twelve Data API key",
        value=get_secret("TWELVE_DATA_API_KEY"),
        type="password",
        help="You can also save this as TWELVE_DATA_API_KEY in Streamlit Secrets.",
    ).strip()
    st.sidebar.caption("Twelve Data access to each exchange depends on your plan.")

history_period = st.sidebar.selectbox("Price history", ["2y", "5y", "10y"], index=1)
news_days = st.sidebar.selectbox("News window", [3, 7, 14, 30], index=1)
home_country = st.sidebar.text_input("Company country override", value="", help="Leave blank to use the company profile country.")
region_watch = st.sidebar.text_input(
    "Supply and risk regions",
    value="China, Taiwan, Japan, South Korea, India, Europe, Russia, Ukraine, Middle East",
    help="Edit this list for the regions that matter to this company.",
)
run_button = st.sidebar.button("Run full research", type="primary", use_container_width=True)
st.sidebar.caption("Forecast period: 1 year. Simulations: 100,000. Risk groups: 500.")

# ============================================================
# RUN RESEARCH
# ============================================================
if run_button:
    if not ticker:
        st.error("Enter a stock ticker.")
        st.stop()
    if provider == "Twelve Data" and not api_key:
        st.error("Enter your Twelve Data API key, or choose Yahoo Finance.")
        st.stop()

    try:
        with st.spinner("Loading market data and company information..."):
            if provider == "Yahoo Finance":
                history, meta = load_yahoo_data(ticker, history_period)
            else:
                history, meta = load_twelve_data(ticker, exchange, api_key, history_period)
        price = price_metrics(history)
        fundamentals = get_fundamental_metrics(meta)
        quality_score, quality_checks = financial_quality(fundamentals)
    except Exception as exc:
        st.error(f"Could not load this stock: {exc}")
        st.info("Check the ticker format, exchange, provider access, and data plan.")
        st.stop()

    company = meta.get("company") or ticker
    currency = meta.get("currency") or "Local units"
    detected_country = home_country.strip() or meta.get("country") or ""
    country = detected_country or "Not reported"
    profile = meta.get("profile", {}) or {}

    queries = build_news_queries(company, ticker, detected_country, region_watch)
    news_groups = {}
    news_errors = {}
    with st.spinner(f"Finding news from the last {news_days} days..."):
        with ThreadPoolExecutor(max_workers=3) as pool:
            future_map = {
                pool.submit(get_recent_news, query, news_days): category
                for category, query in queries.items()
            }
            for future, category in future_map.items():
                try:
                    news_groups[category] = future.result()
                except Exception as exc:
                    news_groups[category] = []
                    news_errors[category] = str(exc)

    # Remove duplicate URLs across the three searches.
    all_news = []
    seen = set()
    for category, items in news_groups.items():
        for item in items:
            if item["url"] in seen:
                continue
            seen.add(item["url"])
            item["category"] = category
            all_news.append(item)
    all_news.sort(key=lambda x: x["indexed"], reverse=True)

    news_scores = {
        "Company news": weighted_news_score(news_groups.get("Company", [])),
        "Country news": weighted_news_score(news_groups.get("Country and regions", [])),
        "World risk news": weighted_news_score(news_groups.get("World risk", [])),
    }

    factors, total_adjustment = factor_adjustments(
        price,
        fundamentals,
        quality_score,
        fundamentals,
        news_scores,
    )

    seed_text = f"{ticker}|{price['last_date']}|GlobalStockResearch"
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    try:
        with st.spinner("Running 100,000 one year simulations and 500 risk groups..."):
            forecast = moving_block_bootstrap(
                np.log(price["close"] / price["close"].shift(1)).dropna().to_numpy(),
                price["latest"],
                total_adjustment,
                seed,
            )
    except Exception as exc:
        st.error(f"Forecast could not be calculated: {exc}")
        st.stop()

    # ========================================================
    # TOP OF DASHBOARD
    # ========================================================
    st.subheader(f"{company}  |  {ticker}")
    top_line = f"{meta.get('exchange') or 'Exchange not reported'}  ·  {country}  ·  {currency}  ·  Data date: {pd.Timestamp(price['last_date']).strftime('%Y-%m-%d')}"
    st.caption(top_line)

    # Short labels and wrapped values prevent the clipped labels in the earlier version.
    a, b, c, d = st.columns(4)
    a.metric("Latest price", fmt_price(price["latest"], currency))
    b.metric("1 year median", fmt_price(forecast["median"], currency), fmt_pct((forecast["median"] / price["latest"] - 1) * 100))
    c.metric("Price range low", fmt_price(forecast["p05"], currency))
    d.metric("Price range high", fmt_price(forecast["p95"], currency))

    st.caption(
        f"Research scenario adjustment: {total_adjustment:+.2f} percentage points per year. "
        "This is a transparent set of assumptions, not a proven prediction formula."
    )

    tabs = st.tabs([
        "Main view",
        "Price and risk",
        "Company health",
        "News and world events",
        "Forecast reasons",
        "Sources and settings",
    ])

    # ========================================================
    # TAB 1: MAIN VIEW
    # ========================================================
    with tabs[0]:
        st.markdown("### Stock at a glance")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("1 month", fmt_pct(price["return_1m"]))
        p2.metric("3 months", fmt_pct(price["return_3m"]))
        p3.metric("6 months", fmt_pct(price["return_6m"]))
        p4.metric("1 year", fmt_pct(price["return_1y"]))

        p5, p6, p7, p8 = st.columns(4)
        p5.metric("Price median", fmt_price(price["median"], currency))
        p6.metric("Price average", fmt_price(price["average"], currency))
        p7.metric("52 week high", fmt_price(price["high_52w"], currency))
        p8.metric("52 week low", fmt_price(price["low_52w"], currency))

        chart = go.Figure()
        chart.add_trace(go.Scatter(
            x=price["close"].index,
            y=price["close"].values,
            mode="lines",
            name="Adjusted close",
        ))
        if pd.notna(price["sma50"]):
            sma50_series = price["close"].rolling(50).mean()
            chart.add_trace(go.Scatter(x=sma50_series.index, y=sma50_series, mode="lines", name="50 day average"))
        if pd.notna(price["sma200"]):
            sma200_series = price["close"].rolling(200).mean()
            chart.add_trace(go.Scatter(x=sma200_series.index, y=sma200_series, mode="lines", name="200 day average"))
        chart.update_layout(
            title="Price history",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency})",
            hovermode="x unified",
            template="plotly_white",
            legend_title="Line",
        )
        st.plotly_chart(chart, use_container_width=True)

        st.markdown("### Trend and risk signs")
        t1, t2, t3, t4 = st.columns(4)
        trend = "Above both averages" if pd.notna(price["sma50"]) and pd.notna(price["sma200"]) and price["latest"] > price["sma50"] > price["sma200"] else "Below both averages" if pd.notna(price["sma50"]) and pd.notna(price["sma200"]) and price["latest"] < price["sma50"] < price["sma200"] else "Mixed trend"
        t1.metric("Trend", trend)
        t2.metric("RSI 14", f"{price['rsi14']:.1f}" if pd.notna(price["rsi14"]) else "Not available")
        t3.metric("Price swings", f"{price['volatility']:.1f}%" if pd.notna(price["volatility"]) else "Not available")
        t4.metric("Past worst fall", f"{price['max_drawdown']:.1f}%")
        st.caption("RSI is a price speed measure. It is not a buy or sell signal by itself.")

    # ========================================================
    # TAB 2: PRICE AND RISK
    # ========================================================
    with tabs[1]:
        st.markdown("### One year price range")
        r1, r2, r3 = st.columns(3)
        r1.metric("Low case", fmt_price(forecast["p05"], currency))
        r2.metric("Middle case", fmt_price(forecast["median"], currency))
        r3.metric("High case", fmt_price(forecast["p95"], currency))

        distribution = px.histogram(
            x=forecast["terminal"],
            nbins=70,
            labels={"x": f"One year price ({currency})", "y": "Simulation count"},
            title="100,000 possible one year prices",
        )
        distribution.add_vline(x=price["latest"], line_dash="dash", annotation_text="Today")
        distribution.add_vline(x=forecast["median"], line_dash="dot", annotation_text="Middle case")
        distribution.update_layout(template="plotly_white")
        st.plotly_chart(distribution, use_container_width=True)

        paths = go.Figure()
        for i in range(forecast["preview"].shape[0]):
            paths.add_trace(go.Scatter(
                x=np.arange(1, forecast["preview"].shape[1] + 1),
                y=forecast["preview"][i],
                mode="lines",
                line={"width": 1},
                opacity=0.28,
                showlegend=False,
            ))
        paths.add_hline(y=price["latest"], line_dash="dash", annotation_text="Today")
        paths.update_layout(
            title="24 example paths",
            xaxis_title="Trading days ahead",
            yaxis_title=f"Price ({currency})",
            template="plotly_white",
        )
        st.plotly_chart(paths, use_container_width=True)

        st.markdown("### Risk assessment")
        # Short names and separate values keep every card readable on small screens.
        risk_rows = [
            ("Loss chance", forecast["chance_down"], "%"),
            ("Gain chance", forecast["chance_up"], "%"),
            ("Batch low", forecast["batch_low"], "%"),
            ("Batch middle", forecast["batch_median"], "%"),
            ("Batch high", forecast["batch_high"], "%"),
            ("VaR 95", forecast["var95"], "%"),
            ("CVaR 95", forecast["cvar95"], "%"),
        ]
        for start in range(0, len(risk_rows), 3):
            cols = st.columns(3)
            for col, row in zip(cols, risk_rows[start:start + 3]):
                col.metric(row[0], f"{row[1]:.1f}{row[2]}")

        st.caption(
            "Loss chance is the share of simulations ending below today's price. "
            "The 500 batches each contain 200 simulated outcomes. Batch low, middle, "
            "and high show the 5th percentile, median, and 95th percentile of those "
            "500 loss chance estimates. VaR 95 is the loss at the 5th price percentile. "
            "CVaR 95 is the average loss in the worst 5% of outcomes."
        )

        st.download_button(
            "Download all 100,000 results",
            pd.DataFrame({"one_year_price": forecast["terminal"]}).to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_one_year_simulations.csv",
            mime="text/csv",
        )

    # ========================================================
    # TAB 3: COMPANY HEALTH
    # ========================================================
    with tabs[2]:
        st.markdown("### Business and valuation")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Market value", fmt_money(fundamentals["market_cap"], currency))
        v2.metric("P/E", f"{fundamentals['trailing_pe']:.1f}x" if pd.notna(fundamentals["trailing_pe"]) else "Not reported")
        v3.metric("Forward P/E", f"{fundamentals['forward_pe']:.1f}x" if pd.notna(fundamentals["forward_pe"]) else "Not reported")
        v4.metric("P/B", f"{fundamentals['price_to_book']:.1f}x" if pd.notna(fundamentals["price_to_book"]) else "Not reported")

        v5, v6, v7, v8 = st.columns(4)
        v5.metric("EV / EBITDA", f"{fundamentals['ev_ebitda']:.1f}x" if pd.notna(fundamentals["ev_ebitda"]) else "Not reported")
        v6.metric("Dividend yield", f"{fundamentals['dividend_yield'] * 100:.2f}%" if pd.notna(fundamentals["dividend_yield"]) else "Not reported")
        v7.metric("Debt / equity", f"{fundamentals['debt_equity']:.2f}x" if pd.notna(fundamentals["debt_equity"]) else "Not reported")
        v8.metric("Cash", fmt_money(fundamentals["cash"], currency))

        st.markdown("### Analyst price range")
        a1, a2, a3 = st.columns(3)
        a1.metric("Low target", fmt_price(fundamentals["target_low"], currency))
        a2.metric("Average target", fmt_price(fundamentals["target_mean"], currency))
        a3.metric("High target", fmt_price(fundamentals["target_high"], currency))
        if pd.notna(fundamentals["analyst_count"]):
            st.caption(f"Analyst count: {int(fundamentals['analyst_count'])}. Targets are third party estimates, not our model forecast.")
        else:
            st.caption("Analyst targets were not supplied by this provider.")

        st.markdown("### Company results")
        company_rows = [
            ("Revenue", fmt_money(fundamentals["revenue"], currency)),
            ("Revenue growth", fmt_pct(fundamentals["revenue_growth"])),
            ("Gross profit", fmt_money(fundamentals["gross_profit"], currency)),
            ("Gross margin", fmt_pct(fundamentals["gross_margin"])),
            ("Operating profit", fmt_money(fundamentals["operating_income"], currency)),
            ("Operating margin", fmt_pct(fundamentals["operating_margin"])),
            ("Net income", fmt_money(fundamentals["net_income"], currency)),
            ("Net margin", fmt_pct(fundamentals["net_margin"])),
            ("Cash from operations", fmt_money(fundamentals["operating_cash_flow"], currency)),
            ("Capital spending", fmt_money(fundamentals["capex"], currency)),
            ("Free cash flow", fmt_money(fundamentals["free_cash_flow"], currency)),
            ("Total debt", fmt_money(fundamentals["debt"], currency)),
            ("Shareholder equity", fmt_money(fundamentals["equity"], currency)),
            ("Current ratio", f"{fundamentals['current_ratio']:.2f}x" if pd.notna(fundamentals["current_ratio"]) else "Not reported"),
        ]
        st.dataframe(pd.DataFrame(company_rows, columns=["Measure", "Latest value"]), use_container_width=True, hide_index=True)

        st.markdown("### Financial health score")
        q1, q2 = st.columns([1, 2])
        q1.metric("Health score", f"{quality_score:.0f} / 100" if quality_score is not None else "Not enough data")
        if quality_score is not None:
            q2.progress(int(quality_score))
        else:
            q2.info("The provider did not return enough financial data to score this company.")
        measured_checks = sum(1 for row in quality_checks if row[1] != "Not reported")
        st.caption(f"This score checks growth, profit, cash flow, and debt. Data available for {measured_checks} of 5 checks. It is a simple guide, not a credit rating.")
        st.dataframe(pd.DataFrame(quality_checks, columns=["Check", "Result", "Score change"]), use_container_width=True, hide_index=True)

        statements = meta.get("statements", {}) or {}
        for key, title in [("income", "Income statement"), ("balance", "Balance sheet"), ("cashflow", "Cash flow")]:
            with st.expander(title):
                table = statement_summary(statements.get(key), key)
                if table.empty:
                    st.info("This provider did not return this statement.")
                else:
                    st.dataframe(table, use_container_width=True, hide_index=True)
        if meta.get("statement_errors"):
            st.caption("Some Twelve Data statement fields may need a higher data plan: " + " | ".join(meta["statement_errors"][:3]))

    # ========================================================
    # TAB 4: NEWS AND WORLD EVENTS
    # ========================================================
    with tabs[3]:
        st.markdown(f"### Headlines from the last {news_days} days")
        st.caption(
            "News is limited to the selected recent window. The date shown is when "
            "GDELT indexed the article, not a guarantee of the publisher's original publication time."
        )
        if news_errors:
            for category, error in news_errors.items():
                st.warning(f"{category} search did not load: {error}")

        n1, n2, n3 = st.columns(3)
        n1.metric("Company news", len(news_groups.get("Company", [])))
        n2.metric("Country news", len(news_groups.get("Country and regions", [])))
        n3.metric("World risk news", len(news_groups.get("World risk", [])))

        score_rows = []
        for label, key in [
            ("Company news", "Company news"),
            ("Country news", "Country news"),
            ("World risk", "World risk news"),
        ]:
            value = news_scores[key]
            score_rows.append((label, f"{value:+.2f}" if pd.notna(value) else "No news score"))
        st.dataframe(pd.DataFrame(score_rows, columns=["News group", "Headline score"]), use_container_width=True, hide_index=True)
        st.caption("Scores use words in headlines only. They do not prove that an event caused a price move.")

        if all_news:
            for article in all_news[:30]:
                with st.container(border=True):
                    st.markdown(f"**{article['title']}**")
                    st.caption(
                        f"{article['category']}  ·  {article['tag']}  ·  "
                        f"{article['domain']}  ·  Indexed {article['indexed'].strftime('%Y-%m-%d %H:%M UTC')}  ·  "
                        f"{article['signal']} ({article['score']:+.2f})"
                    )
                    if article["reason"] != "No matched terms":
                        st.caption("Matched words: " + article["reason"])
                    st.markdown(f"[Open original article]({article['url']})")
        else:
            st.info("No matching headlines were returned for this period. The news adjustment is zero.")

        with st.expander("See the exact news searches"):
            for label, query in queries.items():
                st.markdown(f"**{label}**")
                st.code(query, language="text")

    # ========================================================
    # TAB 5: FACTOR EXPLANATION
    # ========================================================
    with tabs[4]:
        st.markdown("### Why the forecast moved")
        st.metric("Total scenario change", f"{total_adjustment:+.2f} points per year")
        st.caption(
            "Each row is a visible model input. The effect sizes are small, capped "
            "scenario assumptions. They are not coefficients proven by a long backtest."
        )
        factor_table = pd.DataFrame(
            [(name, f"{impact:+.2f} points", reason) for name, impact, reason in factors],
            columns=["Factor", "Forecast effect", "Reason"],
        )
        st.dataframe(factor_table, use_container_width=True, hide_index=True)

        st.markdown("### Base case and research case")
        base_change = (forecast["base_median"] / price["latest"] - 1) * 100
        research_change = (forecast["median"] / price["latest"] - 1) * 100
        c1, c2 = st.columns(2)
        c1.metric("Base median", fmt_price(forecast["base_median"], currency), fmt_pct(base_change))
        c2.metric("Research median", fmt_price(forecast["median"], currency), fmt_pct(research_change))
        st.markdown(
            "The base case resamples past daily returns. The research case applies the "
            "visible factor adjustments above. The gap between them shows how much the "
            "assumptions change the result. Neither result is a promise or a reliable "
            "point target on its own."
        )

        if total_adjustment != 0:
            st.info(
                f"The research case shifts the one year log return by {total_adjustment:+.2f} "
                "percentage points compared with the base case. The total adjustment is "
                "capped between minus 5 and plus 5 points."
            )
        else:
            st.info("No extra factor adjustment was applied. The research case equals the base case.")

    # ========================================================
    # TAB 6: SOURCES AND SETTINGS
    # ========================================================
    with tabs[5]:
        st.markdown("### Data sources")
        st.write(f"**Price and company data:** {meta.get('provider')}")
        if provider == "Yahoo Finance":
            st.markdown("[Yahoo Finance market coverage](https://help.yahoo.com/kb/SLN2352.html)")
            st.caption("Yahoo symbols use exchange suffixes. Coverage and delay vary by exchange. yfinance is an independent community tool for research and personal use.")
        else:
            st.markdown("[Twelve Data global market coverage](https://twelvedata.com/stocks/)")
            st.markdown("[Twelve Data API documentation](https://twelvedata.com/docs)")
            st.caption("Access to each country and data type depends on the Twelve Data plan. This app checks the API response and shows missing data.")
        st.markdown("[GDELT news search documentation](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)")
        st.caption("GDELT searches global news in multiple languages, but search coverage is not complete and its index time may differ from the publisher date.")

        st.markdown("### Model settings")
        settings = [
            ("Stock", ticker),
            ("Provider", provider),
            ("Exchange", meta.get("exchange") or "Not reported"),
            ("Currency", currency),
            ("Price history", history_period),
            ("Forecast period", "252 trading days"),
            ("Simulations", "100,000"),
            ("Risk groups", "500 groups of 200"),
            ("News window", f"Last {news_days} days"),
            ("Company country", country),
            ("Risk regions", region_watch),
            ("Model method", "Five day moving block bootstrap"),
            ("Random seed", str(seed)),
            ("Data date", str(price["last_date"])),
        ]
        st.dataframe(pd.DataFrame(settings, columns=["Setting", "Value"]), use_container_width=True, hide_index=True)

        st.markdown("### What this model cannot know")
        st.markdown(
            "The model cannot know future events. It may miss a new war, sudden law change, "
            "fraud, product failure, earnings surprise, exchange closure, or major change in "
            "currency value. Historical returns do not contain every future risk. News scores "
            "read headlines, not full articles. Company accounts may be delayed, incomplete, "
            "or reported under different accounting rules. The scenario effects are not "
            "backtested and should not be treated as proven forecast weights."
        )
        st.markdown(
            "For important facts, open the original article and confirm company results "
            "against the company's investor relations page or the official market regulator "
            "in that country. Forecast prices are in the stock's listing currency; changes "
            "between that currency and your own currency are not included."
        )

    st.divider()
    st.caption(
        f"Finished on {datetime.now().strftime('%Y-%m-%d %H:%M')} local app time. "
        "Prices are in the listing currency. Educational research only, not personal investment advice."
    )
else:
    st.info("Enter a ticker in the sidebar and select Run full research.")
    st.markdown("### Global ticker examples")
    examples = pd.DataFrame(
        [
            ("United States", "AAPL", "Yahoo Finance"),
            ("Japan", "7203.T", "Yahoo Finance"),
            ("Hong Kong", "0700.HK", "Yahoo Finance"),
            ("India", "RELIANCE.NS", "Yahoo Finance"),
            ("United Kingdom", "VOD.L", "Yahoo Finance"),
            ("Germany", "SAP.DE", "Yahoo Finance"),
            ("South Korea", "005930.KS", "Yahoo Finance"),
            ("Indonesia", "BBCA.JK", "Yahoo Finance"),
        ],
        columns=["Market", "Example ticker", "Source"],
    )
    st.dataframe(examples, use_container_width=True, hide_index=True)
    st.caption("Use the exact ticker format shown by your selected data provider. Some exchanges require a paid plan or may not supply every financial field.")
