# app.py
# -------------------------------------------------------
# Step 1: Multi-stock Streamlit dashboard
# Run with: uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta


# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")


# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input(
    "Stock Tickers (comma-separated)",
    value="AAPL,MSFT,NVDA"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    min_value=date(1970, 1, 1)
)
end_date = st.sidebar.date_input(
    "End Date",
    value=date.today(),
    min_value=date(1970, 1, 1)
)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()


# -- Data download ----------------------------------------
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Download daily stock data from Yahoo Finance."""
    return yf.download(tickers, start=start, end=end, progress=False)


# -- Main logic -------------------------------------------
if tickers:
    try:
        df = load_data(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

    if df.empty:
        st.error("No data found. Check the ticker symbols and try again.")
        st.stop()

    # Pull out close prices in a way that works for one or many tickers
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            st.error("Close price data not found.")
            st.stop()
        close_prices = df["Close"].copy()
    else:
        close_prices = pd.DataFrame(df["Close"])
        close_prices.columns = [tickers[0]]

    # Remove invalid all-empty columns
    close_prices = close_prices.dropna(axis=1, how="all")

    if close_prices.empty:
        st.error("No valid closing price data was returned.")
        st.stop()

    st.subheader("Stock Price Comparison")

    fig_prices = go.Figure()

    for ticker in close_prices.columns:
        fig_prices.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices[ticker],
                mode="lines",
                name=ticker
            )
        )

    fig_prices.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig_prices, width="stretch")

    with st.expander("View Closing Prices"):
        st.dataframe(close_prices.tail(60), width="stretch")

else:
    st.info("Enter stock tickers in the sidebar to get started.")