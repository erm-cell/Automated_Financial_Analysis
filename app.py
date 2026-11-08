import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Automated Financial Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Automated Financial Analysis")
st.caption("Step 1: Stock data retrieval and historical price analysis")

st.sidebar.header("Analysis settings")
with st.sidebar.form("analysis_form"):
    ticker_input = st.text_input("Stock ticker", value="AAPL").strip().upper()
    prediction_horizon = st.selectbox(
        "Future prediction horizon",
        ["1 year", "2 years", "3 years", "5 years"],
        index=0,
    )
    history_period = st.selectbox(
        "Historical data window",
        ["1 year", "2 years", "5 years", "10 years"],
        index=2,
    )
    submitted = st.form_submit_button("Analyze stock", type="primary")

st.info(
    "This first version retrieves historical market data and calculates descriptive statistics. "
    "The selected prediction horizon is saved for the forecasting stage, but no future price is predicted yet."
)


@st.cache_data(ttl=900, show_spinner=False)
def get_stock_history(symbol: str, period: str) -> pd.DataFrame:
    """Retrieve daily, auto-adjusted price history from Yahoo Finance via yfinance."""
    stock = yf.Ticker(symbol)
    history = stock.history(
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=False,
        timeout=20,
    )
    if history is None or history.empty:
        raise ValueError(
            f"No price history was returned for '{symbol}'. Check the ticker symbol "
            "and try again. Some exchanges or symbols may not be supported."
        )
    if "Close" not in history.columns:
        raise ValueError("The data provider did not return a daily closing price column.")

    history = history.copy()
    history["Close"] = pd.to_numeric(history["Close"], errors="coerce")
    history = history.dropna(subset=["Close"])
    if history.empty:
        raise ValueError("The returned price history did not contain usable closing prices.")
    return history


def format_number(value: float) -> str:
    return f"{value:,.2f}"


if submitted:
    if not ticker_input:
        st.error("Enter a stock ticker, such as AAPL.")
        st.stop()

    try:
        with st.spinner(f"Retrieving {ticker_input} price history..."):
            history = get_stock_history(ticker_input, history_period)

        prices = history["Close"].dropna()
        daily_returns = prices.pct_change().dropna()
        latest_price = float(prices.iloc[-1])
        historical_median = float(prices.median())
        historical_mean = float(prices.mean())
        one_year_prices = prices.tail(252)
        one_year_high = float(one_year_prices.max())
        one_year_low = float(one_year_prices.min())
        annualized_volatility = (
            float(daily_returns.std(ddof=1) * math.sqrt(252) * 100)
            if len(daily_returns) > 1
            else float("nan")
        )
        median_daily_return = (
            float(daily_returns.median() * 100)
            if not daily_returns.empty
            else float("nan")
        )

        latest_date = pd.Timestamp(prices.index[-1]).strftime("%Y-%m-%d")
        retrieval_time = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S WIB")

        st.subheader(f"{ticker_input} historical analysis")
        st.caption(
            f"Latest available daily close: {latest_date} | "
            f"Retrieved: {retrieval_time} | "
            f"Historical window: {history_period} | "
            f"Prediction horizon selected: {prediction_horizon}"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest available close", format_number(latest_price))
        col2.metric("Historical median close", format_number(historical_median))
        col3.metric("Historical average close", format_number(historical_mean))

        col4, col5, col6 = st.columns(3)
        col4.metric("52-week high", format_number(one_year_high))
        col5.metric("52-week low", format_number(one_year_low))
        col6.metric(
            "Annualized historical volatility",
            f"{annualized_volatility:.2f}%" if pd.notna(annualized_volatility) else "Unavailable",
        )

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Historical closing prices")
            chart_data = prices.rename("Adjusted close")
            st.line_chart(chart_data, use_container_width=True)
        with right:
            st.subheader("Additional statistics")
            st.write(f"**Median daily return:** {median_daily_return:.4f}%" if pd.notna(median_daily_return) else "**Median daily return:** Unavailable")
            st.write(f"**Trading days retrieved:** {len(prices):,}")
            st.write(f"**First date:** {pd.Timestamp(prices.index[0]).strftime('%Y-%m-%d')}")
            st.write(f"**Last date:** {latest_date}")

        st.download_button(
            "Download retrieved price history (CSV)",
            data=history.to_csv().encode("utf-8"),
            file_name=f"{ticker_input}_{history_period}_price_history.csv",
            mime="text/csv",
        )

        st.caption(
            "Method note: prices are auto-adjusted by the data library for events such as splits and dividends. "
            "The 52-week high and low are calculated from the latest 252 available trading observations in the retrieved data. "
            "Annualized volatility is the standard deviation of daily close-to-close returns multiplied by √252."
        )

    except Exception as exc:
        st.error(f"Could not complete the analysis: {exc}")
        st.caption(
            "Possible causes include an unsupported ticker, a temporary data-provider issue, "
            "or a request limit. Try again later or verify the ticker symbol."
        )
else:
    st.subheader("Ready to analyze")
    st.write("The default ticker is **AAPL**. Select your settings in the sidebar, then click **Analyze stock**.")
    st.write("This stage establishes the market-data connection. Forecasting and buy or sell research will be added only after the input data is working reliably.")

st.divider()
st.caption(
    "Data source: Yahoo Finance accessed through the yfinance open-source Python library. "
    "This is a research and educational prototype, not investment advice or a guaranteed live quote."
)
