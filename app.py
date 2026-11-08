
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Quant Financial Research",
    page_icon="📊",
    layout="wide",
)

st.title("Automated Financial Research")
st.caption(
    "Market data, recent news, company fundamentals, geopolitical monitoring, "
    "and a one year scenario model. Research tool only, not investment advice."
)

# ==========================================
# 1. SIDEBAR SETTINGS
# ==========================================

st.sidebar.header("Analysis settings")

ticker_input = st.sidebar.text_input("Stock ticker", value="AAPL")
ticker_symbol = ticker_input.strip().upper()

history_period = st.sidebar.selectbox(
    "Price history",
    ["2y", "5y", "10y"],
    index=1
)

news_days = st.sidebar.selectbox(
    "News freshness window",
    [3, 7, 14, 30],
    index=1
)

news_country = st.sidebar.text_input(
    "Country or main market to monitor",
    value="United States"
)

geo_regions = st.sidebar.text_input(
    "Geopolitical regions to monitor",
    value="China, Taiwan, Russia, Ukraine, Middle East",
    help="Separate regions with commas."
)

news_cap_pct = st.sidebar.slider(
    "Maximum annual news adjustment",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Experimental assumption. Maximum annual drift adjustment in percentage points."
)

run_button = st.sidebar.button(
    "Run full analysis",
    type="primary",
    use_container_width=True
)

st.sidebar.caption(
    "Forecast horizon: 1 year, or 252 trading days. "
    "Price simulations: 100,000. Risk batches: 500."
)


# ==========================================
# 2. MARKET DATA RETRIEVAL
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def get_price_history(symbol, period):

    hist = yf.Ticker(symbol).history(
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=False,
        timeout=25,
    )

    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(
            f"No price history returned for {symbol}. "
            "Check the ticker and Yahoo Finance connection."
        )

    hist = hist.copy()

    hist["Close"] = pd.to_numeric(
        hist["Close"],
        errors="coerce"
    )

    hist = hist.dropna(subset=["Close"])

    if hist.empty:
        raise ValueError("No valid closing prices were returned.")

    return hist


@st.cache_data(ttl=1800, show_spinner=False)
def get_company_data(symbol):

    stock = yf.Ticker(symbol)

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    frames = {}

    for key, attribute in [
        ("income", "financials"),
        ("balance", "balance_sheet"),
        ("cashflow", "cashflow"),
    ]:

        try:
            value = getattr(stock, attribute)

            if isinstance(value, pd.DataFrame):
                frames[key] = value
            else:
                frames[key] = pd.DataFrame()

        except Exception:
            frames[key] = pd.DataFrame()

    return info, frames


# ==========================================
# 3. LATEST NEWS RETRIEVAL
# ==========================================

def parse_gdelt_date(value):

    parsed = pd.to_datetime(
        value,
        utc=True,
        errors="coerce"
    )

    if pd.isna(parsed):
        return None

    return parsed


@st.cache_data(ttl=900, show_spinner=False)
def fetch_gdelt_news(query, days, max_records=50):

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(days))

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": now.strftime("%Y%m%d%H%M%S"),
    }

    response = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params=params,
        headers={
            "User-Agent": "QuantFinancialResearch/1.0 educational dashboard"
        },
        timeout=25,
    )

    response.raise_for_status()

    payload = response.json()

    articles = (
        payload.get("articles", [])
        if isinstance(payload, dict)
        else []
    )

    cutoff = pd.Timestamp(start)

    output = []

    for item in articles:

        if not isinstance(item, dict):
            continue

        seen = parse_gdelt_date(
            item.get("seendate") or item.get("date")
        )

        if seen is None or seen < cutoff:
            continue

        url = item.get("url") or item.get("url_mobile")
        title = (item.get("title") or "").strip()

        if not url or not title:
            continue

        output.append({
            "title": title,
            "url": url,
            "domain": item.get("domain") or "Publisher not identified",
            "seen_at": seen,
            "source_country": item.get("sourcecountry") or "",
            "language": item.get("language") or "",
        })

    return output


# ==========================================
# 4. TRANSPARENT NEWS SCORING
# ==========================================

POSITIVE_TERMS = [
    "earnings beat",
    "revenue growth",
    "profit growth",
    "record revenue",
    "record sales",
    "strong demand",
    "margin expansion",
    "buyback",
    "dividend increase",
    "product approval",
    "partnership",
    "new product",
    "ceasefire",
    "peace deal",
    "peace agreement",
    "de-escalation",
    "sanctions lifted",
    "trade agreement",
    "tariff cut",
    "rate cut",
    "inflation cools",
    "inflation eased",
    "jobs growth",
    "gdp growth",
]

NEGATIVE_TERMS = [
    "earnings miss",
    "revenue decline",
    "profit decline",
    "weak demand",
    "margin compression",
    "lawsuit",
    "antitrust",
    "investigation",
    "recall",
    "fine",
    "downgrade",
    "outage",
    "supply disruption",
    "production delay",
    "war",
    "invasion",
    "missile",
    "airstrike",
    "escalation",
    "new sanctions",
    "sanctions imposed",
    "sanctions expanded",
    "export ban",
    "export controls",
    "blockade",
    "tariff hike",
    "trade restrictions",
    "armed conflict",
    "military operation",
    "shipping disruption",
    "recession",
    "inflation rises",
    "rate hike",
    "unemployment rises",
    "debt crisis",
]


def headline_score(title):

    text = (title or "").lower()

    positive = [
        term for term in POSITIVE_TERMS
        if term in text
    ]

    negative = [
        term for term in NEGATIVE_TERMS
        if term in text
    ]

    raw = (
        0.45 * min(len(positive), 2)
        - 0.55 * min(len(negative), 2)
    )

    score = float(np.clip(raw, -1.0, 1.0))

    reasons = []

    if positive:
        reasons.append(
            "Positive headline terms: "
            + ", ".join(positive[:3])
        )

    if negative:
        reasons.append(
            "Risk headline terms: "
            + ", ".join(negative[:3])
        )

    reason = (
        "; ".join(reasons)
        if reasons
        else "No configured positive or negative terms. Scored neutral."
    )

    return score, reason


def enrich_articles(articles, category):

    enriched = []

    now = pd.Timestamp.now(tz="UTC")

    for article in articles:

        score, reason = headline_score(
            article["title"]
        )

        age_days = max(
            0.0,
            (now - article["seen_at"]).total_seconds() / 86400.0
        )

        recency_weight = float(
            np.exp(-age_days / 5.0)
        )

        enriched.append({
            **article,
            "category": category,
            "score": score,
            "reason": reason,
            "age_days": age_days,
            "recency_weight": recency_weight,
        })

    return enriched


def category_score(articles):

    if not articles:
        return None

    weights = np.array(
        [a["recency_weight"] for a in articles],
        dtype=float
    )

    scores = np.array(
        [a["score"] for a in articles],
        dtype=float
    )

    if weights.sum() == 0:
        return float(scores.mean())

    return float(
        np.average(scores, weights=weights)
    )


def make_queries(symbol, company_name, country, regions_text):

    company_name = (company_name or symbol).strip()

    company_query = (
        f'("{company_name}" OR {symbol} OR "{symbol} stock")'
    )

    country = country.strip() or "United States"

    country_query = (
        f'"{country}" (inflation OR "Federal Reserve" OR '
        '"interest rates" OR GDP OR unemployment OR recession '
        'OR tariffs OR Treasury OR budget)'
    )

    regions = [
        part.strip()
        for part in regions_text.split(",")
        if part.strip()
    ]

    if not regions:
        regions = [country]

    region_group = " OR ".join(
        f'"{region}"' if " " in region else region
        for region in regions
    )

    geo_query = (
        f'({region_group}) (war OR conflict OR sanctions OR tariff '
        'OR "export controls" OR blockade OR ceasefire OR invasion '
        'OR missile OR military OR shipping OR "supply chain")'
    )

    return company_query, country_query, geo_query


# ==========================================
# 5. COMPANY FINANCIAL ANALYSIS
# ==========================================

def normalize_label(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )


def metric_series(frame, aliases):

    if frame is None or frame.empty:
        return pd.Series(dtype=float)

    lookup = {
        normalize_label(index): index
        for index in frame.index
    }

    matched = None

    for alias in aliases:

        key = normalize_label(alias)

        if key in lookup:
            matched = lookup[key]
            break

    if matched is None:
        return pd.Series(dtype=float)

    values = frame.loc[matched]

    if isinstance(values, pd.DataFrame):
        values = values.iloc[0]

    values = pd.to_numeric(
        values,
        errors="coerce"
    ).dropna()

    try:
        values.index = pd.to_datetime(values.index)
        values = values.sort_index(ascending=False)
    except Exception:
        pass

    return values


def latest_value(frame, aliases):

    values = metric_series(frame, aliases)

    if len(values):
        return float(values.iloc[0])

    return np.nan


def format_money(value):

    if pd.isna(value):
        return "Not reported"

    value = float(value)
    absolute = abs(value)

    if absolute >= 1e12:
        return f"${value / 1e12:,.2f}T"

    if absolute >= 1e9:
        return f"${value / 1e9:,.2f}B"

    if absolute >= 1e6:
        return f"${value / 1e6:,.2f}M"

    return f"${value:,.2f}"


def financial_summary(frames):

    income = frames.get("income", pd.DataFrame())
    balance = frames.get("balance", pd.DataFrame())
    cashflow = frames.get("cashflow", pd.DataFrame())

    revenue_series = metric_series(
        income,
        ["Total Revenue", "Revenue"]
    )

    gross_profit = latest_value(
        income,
        ["Gross Profit"]
    )

    operating_income = latest_value(
        income,
        ["Operating Income", "Operating Income Loss"]
    )

    net_income = latest_value(
        income,
        ["Net Income", "Net Income Common Stockholders"]
    )

    cash_from_ops = latest_value(
        cashflow,
        [
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities"
        ]
    )

    capex = latest_value(
        cashflow,
        ["Capital Expenditure", "Capital Expenditures"]
    )

    total_debt = latest_value(
        balance,
        [
            "Total Debt",
            "Long Term Debt And Capital Lease Obligation"
        ]
    )

    cash = latest_value(
        balance,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
        ]
    )

    equity = latest_value(
        balance,
        [
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Total Equity Gross Minority Interest"
        ]
    )

    revenue = (
        float(revenue_series.iloc[0])
        if len(revenue_series)
        else np.nan
    )

    previous_revenue = (
        float(revenue_series.iloc[1])
        if len(revenue_series) > 1
        else np.nan
    )

    if (
        pd.notna(revenue)
        and pd.notna(previous_revenue)
        and previous_revenue != 0
    ):
        revenue_growth = (
            revenue / previous_revenue - 1
        ) * 100
    else:
        revenue_growth = np.nan

    operating_margin = (
        operating_income / revenue * 100
        if pd.notna(operating_income)
        and pd.notna(revenue)
        and revenue
        else np.nan
    )

    net_margin = (
        net_income / revenue * 100
        if pd.notna(net_income)
        and pd.notna(revenue)
        and revenue
        else np.nan
    )

    gross_margin = (
        gross_profit / revenue * 100
        if pd.notna(gross_profit)
        and pd.notna(revenue)
        and revenue
        else np.nan
    )

    if pd.notna(cash_from_ops) and pd.notna(capex):

        if capex < 0:
            free_cash_flow = cash_from_ops + capex
        else:
            free_cash_flow = cash_from_ops - capex

    else:
        free_cash_flow = np.nan

    if (
        pd.notna(total_debt)
        and pd.notna(equity)
        and equity != 0
    ):
        debt_to_equity = total_debt / equity
    else:
        debt_to_equity = np.nan

    rows = [
        ("Revenue", format_money(revenue)),

        (
            "Revenue growth vs previous reported year",
            f"{revenue_growth:.2f}%"
            if pd.notna(revenue_growth)
            else "Not available"
        ),

        ("Gross profit", format_money(gross_profit)),

        (
            "Gross margin",
            f"{gross_margin:.2f}%"
            if pd.notna(gross_margin)
            else "Not available"
        ),

        ("Operating income", format_money(operating_income)),

        (
            "Operating margin",
            f"{operating_margin:.2f}%"
            if pd.notna(operating_margin)
            else "Not available"
        ),

        ("Net income", format_money(net_income)),

        (
            "Net margin",
            f"{net_margin:.2f}%"
            if pd.notna(net_margin)
            else "Not available"
        ),

        ("Operating cash flow", format_money(cash_from_ops)),
        ("Capital expenditure", format_money(capex)),
        ("Estimated free cash flow", format_money(free_cash_flow)),
        ("Cash and short term investments", format_money(cash)),
        ("Total debt", format_money(total_debt)),

        (
            "Debt to equity",
            f"{debt_to_equity:.2f}x"
            if pd.notna(debt_to_equity)
            else "Not available"
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["Metric", "Latest available value"]
    )


def statement_display(frame, row_limit=18):

    if frame is None or frame.empty:
        return pd.DataFrame()

    result = frame.head(row_limit).copy().T

    result.index = [
        str(value)[:10]
        for value in result.index
    ]

    result.index.name = "Reporting period"

    return result


# ==========================================
# 6. MONTE CARLO SIMULATION
# ==========================================

def run_monte_carlo(
    close_prices,
    news_score,
    max_news_adjust_pct
):

    log_returns = np.log(
        close_prices / close_prices.shift(1)
    ).dropna().to_numpy(dtype=float)

    log_returns = log_returns[
        np.isfinite(log_returns)
    ]

    if len(log_returns) < 60:
        raise ValueError(
            "At least 60 valid daily returns are needed."
        )

    simulations = 100_000
    trading_days = 252

    current_price = float(close_prices.iloc[-1])

    annual_news_adjustment = (
        float(news_score)
        * float(max_news_adjust_pct)
        / 100.0
    )

    daily_news_adjustment = (
        annual_news_adjustment / trading_days
    )

    rng = np.random.default_rng(20261108)

    terminal_prices = np.empty(
        simulations,
        dtype=float
    )

    chunk_size = 5_000

    for start in range(0, simulations, chunk_size):

        stop = min(
            start + chunk_size,
            simulations
        )

        sampled = rng.choice(
            log_returns,
            size=(stop - start, trading_days),
            replace=True
        )

        terminal_prices[start:stop] = (
            current_price
            * np.exp(
                sampled.sum(axis=1)
                + daily_news_adjustment * trading_days
            )
        )

    # 500 batches, each containing 200 outcomes.

    batches = terminal_prices.reshape(
        500,
        simulations // 500
    )

    batch_loss_probabilities = (
        batches < current_price
    ).mean(axis=1) * 100.0

    p05, p50, p95 = np.percentile(
        terminal_prices,
        [5, 50, 95]
    )

    lower_tail = terminal_prices[
        terminal_prices <= p05
    ]

    var_95_pct = max(
        0.0,
        (current_price - p05) / current_price * 100
    )

    if len(lower_tail):

        cvar_95_pct = max(
            0.0,
            (current_price - float(lower_tail.mean()))
            / current_price * 100
        )

    else:
        cvar_95_pct = np.nan

    # Only 30 sample paths are generated for visualization.

    preview_returns = rng.choice(
        log_returns,
        size=(30, trading_days),
        replace=True
    )

    preview_paths = current_price * np.exp(
        np.cumsum(
            preview_returns + daily_news_adjustment,
            axis=1
        )
    )

    return {
        "current_price": current_price,
        "terminal_prices": terminal_prices,
        "p05": float(p05),
        "median": float(p50),
        "p95": float(p95),

        "probability_loss": float(
            (terminal_prices < current_price).mean() * 100
        ),

        "median_batch_loss": float(
            np.median(batch_loss_probabilities)
        ),

        "batch_loss_p05": float(
            np.percentile(batch_loss_probabilities, 5)
        ),

        "batch_loss_p95": float(
            np.percentile(batch_loss_probabilities, 95)
        ),

        "var_95_pct": float(var_95_pct),
        "cvar_95_pct": float(cvar_95_pct),
        "preview_paths": preview_paths,

        "annual_news_adjustment_pct": (
            annual_news_adjustment * 100
        ),

        "history_daily_volatility_pct": float(
            np.std(log_returns, ddof=1)
            * np.sqrt(252) * 100
        ),

        "history_annualized_mean_log_return_pct": float(
            np.mean(log_returns) * 252 * 100
        ),
    }


# ==========================================
# 7. MAIN APPLICATION
# ==========================================

if run_button:

    if not ticker_symbol:
        st.error("Enter a stock ticker first.")
        st.stop()

    try:

        with st.spinner(
            f"Retrieving {ticker_symbol} market data and company profile..."
        ):

            history = get_price_history(
                ticker_symbol,
                history_period
            )

            info, frames = get_company_data(
                ticker_symbol
            )

    except Exception as exc:

        st.error(
            f"Market data retrieval failed: {exc}"
        )

        st.stop()

    company_name = (
        info.get("longName")
        or info.get("shortName")
        or ticker_symbol
    )

    company_country = (
        info.get("country")
        or news_country
    )

    sector = info.get("sector") or "Not available"
    industry = info.get("industry") or "Not available"

    close = history["Close"].dropna()

    daily_returns = close.pct_change().dropna()

    historical_median = float(close.median())
    historical_mean = float(close.mean())

    annual_vol = (
        float(daily_returns.std() * np.sqrt(252) * 100)
        if len(daily_returns) > 1
        else np.nan
    )

    company_query, country_query, geo_query = make_queries(
        ticker_symbol,
        company_name,
        news_country,
        geo_regions
    )

    st.subheader(
        f"{company_name} ({ticker_symbol})"
    )

    st.caption(
        f"Sector: {sector} | "
        f"Industry: {industry} | "
        f"Company country in provider profile: {company_country}"
    )

    # ======================================
    # 8. GET THREE NEWS STREAMS
    # ======================================

    news_errors = {}
    news_results = {}

    news_jobs = {
        "Company": company_query,
        "Country and macro": country_query,
        "Geopolitical": geo_query,
    }

    with st.spinner(
        f"Searching current news from the last {news_days} days..."
    ):

        with ThreadPoolExecutor(max_workers=3) as executor:

            future_map = {
                executor.submit(
                    fetch_gdelt_news,
                    query,
                    int(news_days),
                    50
                ): category

                for category, query in news_jobs.items()
            }

            for future, category in future_map.items():

                try:
                    news_results[category] = future.result()

                except Exception as exc:

                    news_results[category] = []
                    news_errors[category] = str(exc)

    articles = []

    for category, items in news_results.items():

        articles.extend(
            enrich_articles(items, category)
        )

    # Remove duplicate article URLs.

    seen_urls = set()
    unique_articles = []

    for article in sorted(
        articles,
        key=lambda x: x["seen_at"],
        reverse=True
    ):

        if article["url"] not in seen_urls:

            seen_urls.add(article["url"])
            unique_articles.append(article)

    articles = unique_articles

    company_score = category_score([
        a for a in articles
        if a["category"] == "Company"
    ])

    country_score = category_score([
        a for a in articles
        if a["category"] == "Country and macro"
    ])

    geo_score = category_score([
        a for a in articles
        if a["category"] == "Geopolitical"
    ])

    # Category weights are explicit and can be adjusted later.

    group_weights = {
        "Company": 0.45,
        "Country and macro": 0.20,
        "Geopolitical": 0.35,
    }

    available_scores = []

    for category, score in [
        ("Company", company_score),
        ("Country and macro", country_score),
        ("Geopolitical", geo_score),
    ]:

        if score is not None:

            available_scores.append(
                (group_weights[category], score)
            )

    if available_scores:

        total_weight = sum(
            weight for weight, score in available_scores
        )

        combined_news_score = sum(
            weight * score
            for weight, score in available_scores
        ) / total_weight

        combined_news_score = float(combined_news_score)

    else:

        combined_news_score = 0.0

    # ======================================
    # 9. RUN 100,000 SIMULATIONS
    # ======================================

    with st.spinner(
        "Running 100,000 one year price simulations..."
    ):

        try:

            forecast = run_monte_carlo(
                close,
                combined_news_score,
                news_cap_pct
            )

        except Exception as exc:

            st.error(f"Simulation failed: {exc}")
            st.stop()

    # ======================================
    # 10. MARKET OVERVIEW
    # ======================================

    st.divider()

    st.subheader("1. Market overview")

    top1, top2, top3, top4 = st.columns(4)

    top1.metric(
        "Latest adjusted close",
        f"${close.iloc[-1]:,.2f}"
    )

    top2.metric(
        "Historical median",
        f"${historical_median:,.2f}"
    )

    top3.metric(
        "Historical average",
        f"${historical_mean:,.2f}"
    )

    top4.metric(
        "Annualized historical volatility",
        f"{annual_vol:.2f}%"
        if pd.notna(annual_vol)
        else "Not available"
    )

    overview_tab, news_tab, fundamentals_tab, forecast_tab, methods_tab = st.tabs(
        [
            "Overview",
            "News and geopolitics",
            "Company financials",
            "Forecast and risk",
            "Evidence and methodology"
        ]
    )

    # ======================================
    # 11. OVERVIEW TAB
    # ======================================

    with overview_tab:

        st.markdown(
            "#### Historical adjusted closing price"
        )

        price_figure = go.Figure()

        price_figure.add_trace(
            go.Scatter(
                x=close.index,
                y=close.values,
                mode="lines",
                name="Adjusted close"
            )
        )

        price_figure.add_hline(
            y=historical_median,
            line_dash="dash",
            annotation_text="Historical median"
        )

        price_figure.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (listing currency)",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(
            price_figure,
            use_container_width=True
        )

        st.markdown("#### Recent price data")

        st.dataframe(
            history.tail(15),
            use_container_width=True
        )

        st.download_button(
            "Download historical prices CSV",
            history.to_csv().encode("utf-8"),
            file_name=f"{ticker_symbol}_historical_prices.csv",
            mime="text/csv"
        )

    # ======================================
    # 12. NEWS AND GEOPOLITICS TAB
    # ======================================

    with news_tab:

        st.markdown(
            f"#### News search window: last {news_days} days"
        )

        st.caption(
            "Only GDELT articles inside the selected date window "
            "are included. The headline score is a transparent "
            "keyword rule, not a human verified interpretation."
        )

        if news_errors:

            for category, error in news_errors.items():

                st.warning(
                    f"{category} news could not be retrieved: {error}"
                )

        n1, n2, n3, n4 = st.columns(4)

        n1.metric(
            "Company headlines",
            len(news_results.get("Company", []))
        )

        n2.metric(
            "Country headlines",
            len(news_results.get("Country and macro", []))
        )

        n3.metric(
            "Geopolitical headlines",
            len(news_results.get("Geopolitical", []))
        )

        n4.metric(
            "Combined news score",
            f"{combined_news_score:+.2f}"
        )

        score_cols = st.columns(3)

        for col, title, value in [
            (score_cols[0], "Company score", company_score),
            (score_cols[1], "Country and macro score", country_score),
            (score_cols[2], "Geopolitical score", geo_score),
        ]:

            col.metric(
                title,
                f"{value:+.2f}"
                if value is not None
                else "No recent articles"
            )

        st.info(
            f"News adjustment applied to forecast: "
            f"{forecast['annual_news_adjustment_pct']:+.2f} "
            "percentage points of annual log drift. "
            "This is an experimental sensitivity adjustment, "
            "not a measured causal effect."
        )

        if articles:

            news_table = pd.DataFrame([
                {
                    "GDELT timestamp (UTC)": a["seen_at"].strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "Category": a["category"],
                    "Publisher domain": a["domain"],
                    "Headline": a["title"],
                    "Headline score": round(a["score"], 2),
                    "Scoring reason": a["reason"],
                    "Original article": a["url"],
                }
                for a in articles
            ])

            st.dataframe(
                news_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Original article": st.column_config.LinkColumn(
                        "Original article",
                        display_text="Open source"
                    )
                }
            )

        else:

            st.warning(
                "No matching recent headlines were returned. "
                "The forecast therefore uses no news adjustment."
            )

        st.markdown("#### Search coverage")

        st.write(f"Company query: `{company_query}`")
        st.write(f"Country query: `{country_query}`")
        st.write(f"Geopolitical query: `{geo_query}`")

    # ======================================
    # 13. FUNDAMENTAL ANALYSIS TAB
    # ======================================

    with fundamentals_tab:

        st.markdown("#### Company financial health")

        fundamental_table = financial_summary(frames)

        st.dataframe(
            fundamental_table,
            use_container_width=True,
            hide_index=True
        )

        pe_col, pb_col, cap_col, div_col = st.columns(4)

        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        market_cap = info.get("marketCap")
        dividend_yield = info.get("dividendYield")

        pe_col.metric(
            "Trailing P/E",
            f"{pe:.2f}x"
            if isinstance(pe, (int, float))
            else "Not available"
        )

        pb_col.metric(
            "Price to book",
            f"{pb:.2f}x"
            if isinstance(pb, (int, float))
            else "Not available"
        )

        cap_col.metric(
            "Market capitalization",
            format_money(market_cap)
            if isinstance(market_cap, (int, float))
            else "Not available"
        )

        div_col.metric(
            "Dividend yield",
            f"{dividend_yield * 100:.2f}%"
            if isinstance(dividend_yield, (int, float))
            else "Not available"
        )

        with st.expander("Income statement data"):

            table = statement_display(
                frames.get("income")
            )

            if table.empty:
                st.info("Income statement data was not returned.")
            else:
                st.dataframe(
                    table,
                    use_container_width=True
                )

        with st.expander("Balance sheet data"):

            table = statement_display(
                frames.get("balance")
            )

            if table.empty:
                st.info("Balance sheet data was not returned.")
            else:
                st.dataframe(
                    table,
                    use_container_width=True
                )

        with st.expander("Cash flow statement data"):

            table = statement_display(
                frames.get("cashflow")
            )

            if table.empty:
                st.info("Cash flow data was not returned.")
            else:
                st.dataframe(
                    table,
                    use_container_width=True
                )

        st.caption(
            "Financial statements and valuation fields come from "
            "Yahoo Finance through yfinance. Confirm important "
            "figures against official company filings."
        )

    # ======================================
    # 14. FORECAST AND RISK TAB
    # ======================================

    with forecast_tab:

        st.markdown(
            "#### One year bootstrap simulation"
        )

        st.caption(
            "100,000 simulated terminal prices. Historical daily "
            "log returns are sampled with replacement over 252 "
            "trading days."
        )

        f1, f2, f3, f4 = st.columns(4)

        f1.metric(
            "Median one year price",
            f"${forecast['median']:,.2f}",
            f"{(
                forecast['median'] / forecast['current_price'] - 1
            ) * 100:+.2f}%"
        )

        f2.metric(
            "5th percentile",
            f"${forecast['p05']:,.2f}"
        )

        f3.metric(
            "95th percentile",
            f"${forecast['p95']:,.2f}"
        )

        f4.metric(
            "Probability of ending below today",
            f"{forecast['probability_loss']:.2f}%"
        )

        dist_figure = px.histogram(
            x=forecast["terminal_prices"],
            nbins=80,
            labels={
                "x": "Simulated one year price",
                "y": "Number of simulations"
            },
            title="Distribution of 100,000 simulated one year prices"
        )

        dist_figure.add_vline(
            x=forecast["current_price"],
            line_dash="dash",
            annotation_text="Current price"
        )

        dist_figure.add_vline(
            x=forecast["median"],
            line_dash="dot",
            annotation_text="Median forecast"
        )

        dist_figure.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            dist_figure,
            use_container_width=True
        )

        # Sample simulation paths.

        preview = forecast["preview_paths"]

        path_figure = go.Figure()

        for i in range(preview.shape[0]):

            path_figure.add_trace(
                go.Scatter(
                    x=np.arange(
                        1,
                        preview.shape[1] + 1
                    ),
                    y=preview[i],
                    mode="lines",
                    line={"width": 1},
                    opacity=0.25,
                    showlegend=False
                )
            )

        path_figure.add_hline(
            y=forecast["current_price"],
            line_dash="dash",
            annotation_text="Starting price"
        )

        path_figure.update_layout(
            title="30 sample paths from the simulation",
            xaxis_title="Trading days ahead",
            yaxis_title="Simulated price",
            template="plotly_white"
        )

        st.plotly_chart(
            path_figure,
            use_container_width=True
        )

        # Risk analysis.

        st.markdown("#### Risk assessment")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Median loss probability across 500 batches",
            f"{forecast['median_batch_loss']:.2f}%"
        )

        r2.metric(
            "5th to 95th percentile batch risk",
            f"{forecast['batch_loss_p05']:.2f}% "
            f"to {forecast['batch_loss_p95']:.2f}%"
        )

        r3.metric(
            "95% Value at Risk",
            f"{forecast['var_95_pct']:.2f}%"
        )

        r4.metric(
            "95% Conditional Value at Risk",
            f"{forecast['cvar_95_pct']:.2f}%"
        )

        st.write(
            "The 500 batch estimates split the 100,000 simulated "
            "outcomes into 500 groups of 200. The median is the "
            "median of those 500 estimated loss probabilities. "
            "VaR is the loss at the 5th percentile. Conditional "
            "VaR is the average loss among outcomes at or below "
            "that percentile."
        )

        st.download_button(
            "Download 100,000 simulated terminal prices CSV",
            pd.DataFrame({
                "simulated_terminal_price":
                    forecast["terminal_prices"]
            }).to_csv(index=False).encode("utf-8"),
            file_name=(
                f"{ticker_symbol}_100000_simulated_terminal_prices.csv"
            ),
            mime="text/csv"
        )

    # ======================================
    # 15. EVIDENCE AND METHODOLOGY TAB
    # ======================================

    with methods_tab:

        st.markdown(
            "#### Evidence and data provenance"
        )

        st.write(
            "Market prices, company profile, valuation ratios, "
            "and company financial statements: Yahoo Finance "
            "through yfinance."
        )

        st.write(
            "Recent company, country, and geopolitical headlines: "
            "GDELT DOC article search. Each returned article is "
            "displayed with its publisher domain, timestamp, "
            "and original URL."
        )

        st.write(
            "GDELT is a news discovery index. A headline and URL "
            "show that a publisher page was indexed. They do not "
            "prove that every claim in the article is correct. "
            "Open and verify important claims at the publisher "
            "or primary source."
        )

        st.markdown(
            "#### Exact model settings"
        )

        settings = pd.DataFrame(
            [
                ("Asset", ticker_symbol),
                ("Historical price window", history_period),
                ("Forecast horizon", "252 trading days"),
                ("Price simulations", "100,000"),
                ("Risk batches", "500 batches × 200 outcomes"),
                ("News lookback", f"{news_days} days"),
                ("Country monitored", news_country),
                ("Geopolitical regions", geo_regions),
                (
                    "Maximum news drift adjustment",
                    f"±{news_cap_pct:.1f} percentage points per year"
                ),
                (
                    "News scoring",
                    "Keyword score applied to headlines only"
                ),
                (
                    "Simulation method",
                    "Historical daily log return bootstrap"
                ),
                ("Random seed", "20261108"),
            ],
            columns=["Setting", "Value"]
        )

        st.dataframe(
            settings,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "#### Important limitations"
        )

        st.write(
            "1. The news score is a simple keyword heuristic. "
            "It does not understand context, verify causation, "
            "or read every full article. "

            "2. The annual news adjustment cap is a user selected "
            "sensitivity assumption, not a learned coefficient. "

            "3. The bootstrap assumes historical daily returns "
            "are a useful guide to possible future outcomes. "
            "It may understate structural breaks, new wars, "
            "policy shocks, liquidity events, and unprecedented "
            "conditions. "

            "4. The model does not know the future and cannot "
            "produce guaranteed buy or sell instructions. "

            "5. Yahoo Finance coverage can be incomplete or "
            "delayed. Confirm company financials with official filings."
        )

    st.divider()

    st.caption(
        f"Analysis completed using price data ending "
        f"{close.index[-1]}. News lookback: {news_days} days. "
        "Verify important articles and financial filings."
    )

else:

    st.info(
        "Enter a ticker and click Run full analysis. "
        "AAPL and a one year forecast are ready as defaults."
    )

    st.markdown(
        "This dashboard includes current company headlines, "
        "country economic and policy news, geopolitical monitoring, "
        "company financial statements, 100,000 historical bootstrap "
        "simulations, and a 500 batch loss probability estimate."
    )


st.divider()

st.caption(
    "Educational financial research software. "
    "Not a recommendation to buy, sell, or hold any security."
)
