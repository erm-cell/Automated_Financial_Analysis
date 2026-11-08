# Automated Financial Analysis

First implementation stage: a Streamlit stock dashboard that retrieves daily historical prices and calculates descriptive price statistics.

## Included in this version

- Stock ticker input (defaults to AAPL)
- Configurable historical data window
- Selected future prediction horizon (stored for a later forecasting stage)
- Latest available daily close
- Historical median and average closing price
- 52-week high and low based on the latest 252 available trading observations
- Annualized historical volatility
- Median daily return
- Historical price chart
- CSV export of retrieved prices

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Data note

This educational prototype retrieves data through the open-source `yfinance` library, which accesses Yahoo Finance's publicly available APIs. It is not affiliated with or endorsed by Yahoo. Review Yahoo's terms for permitted data use. Data can be delayed, incomplete, unavailable, or rate-limited.

## Current limitation

This version does not predict future prices and does not generate buy or sell recommendations. Those features should be implemented only after data retrieval and validation are working reliably.
