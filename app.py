
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Quant Financial Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("Automated Financial Analysis")
st.caption("Stock research and historical price analysis")

st.sidebar.header("Analysis Settings")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="AAPL"
).strip().upper()

period = st.sidebar.selectbox(
    "Historical data period",
    ["1y", "2y", "5y", "10y", "max"],
    index=2
)

st.sidebar.info("Future prediction horizon: 1 year")

@st.cache_data(ttl=300)
def get_stock_data(symbol, selected_period):
    stock = yf.Ticker(symbol)

    data = stock.history(
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        timeout=30
    )

    if data.empty:
        raise ValueError(
            "Yahoo Finance returned no price data."
        )

    data = data.dropna(subset=["Close"])

    if data.empty:
        raise ValueError("No valid closing prices found.")

    return data

if st.button("Analyze Stock", type="primary"):

    try:
        with st.spinner("Retrieving stock data..."):

            data = get_stock_data(ticker, period)

            prices = data["Close"]

            current_price = float(prices.iloc[-1])
            median_price = float(prices.median())
            mean_price = float(prices.mean())

            returns = prices.pct_change().dropna()

            volatility = (
                float(returns.std() * np.sqrt(252) * 100)
                if len(returns) > 1 else 0
            )

            highest = float(prices.max())
            lowest = float(prices.min())

        st.success(f"Successfully retrieved {ticker} data!")

        st.subheader("Historical Price Analysis")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest closing price",
            f"${current_price:,.2f}"
        )

        col2.metric(
            "Historical median",
            f"${median_price:,.2f}"
        )

        col3.metric(
            "Historical average",
            f"${mean_price:,.2f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Highest price",
            f"${highest:,.2f}"
        )

        col5.metric(
            "Lowest price",
            f"${lowest:,.2f}"
        )

        col6.metric(
            "Annualized volatility",
            f"{volatility:.2f}%"
        )

        st.subheader("Historical Price Chart")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                mode="lines",
                name="Closing price"
            )
        )

        fig.add_hline(
            y=median_price,
            line_dash="dash",
            annotation_text="Historical median"
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Historical Data")

        st.dataframe(
            data.tail(10),
            use_container_width=True
        )

        csv = data.to_csv().encode("utf-8")

        st.download_button(
            "Download Historical Data CSV",
            csv,
            file_name=f"{ticker}_historical_data.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error("Stock analysis failed.")

        st.code(str(e))

        st.info(
            "Check the ticker, data provider connection, "
            "and Streamlit logs."
        )

st.divider()

st.caption(
    "Data provided through Yahoo Finance via yfinance. "
    "Historical analysis only. Not investment advice."
)
