# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
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
def load_data(all_tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Download daily stock data from Yahoo Finance."""
    return yf.download(all_tickers, start=start, end=end, progress=False)


# -- Main logic -------------------------------------------
if tickers:
    benchmark = "^GSPC"
    all_tickers = tickers + [benchmark]

    try:
        df = load_data(all_tickers, start_date, end_date)
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
        close_prices.columns = [all_tickers[0]]

    # Remove all-empty columns
    close_prices = close_prices.dropna(axis=1, how="all")

    if close_prices.empty:
        st.error("No valid closing price data was returned.")
        st.stop()

    # Rename benchmark for cleaner display
    if "^GSPC" in close_prices.columns:
        close_prices = close_prices.rename(columns={"^GSPC": "S&P 500"})

    st.subheader("Stock Price Comparison with S&P 500 Benchmark")

    fig_prices = go.Figure()

    for col in close_prices.columns:
        fig_prices.add_trace(
            go.Scatter(
                x=close_prices.index,
                y=close_prices[col],
                mode="lines",
                name=col
            )
        )

    fig_prices.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig_prices, width="stretch")

    # Normalized comparison chart
    st.subheader("Normalized Performance Comparison")

    normalized_prices = close_prices / close_prices.iloc[0] * 100

    fig_norm = go.Figure()

    for col in normalized_prices.columns:
        fig_norm.add_trace(
            go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[col],
                mode="lines",
                name=col
            )
        )

    fig_norm.update_layout(
        xaxis_title="Date",
        yaxis_title="Indexed Value (Start = 100)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig_norm, width="stretch")
    # -- Compute returns -------------------------------------
    returns = close_prices.pct_change().dropna()

    # -- Summary statistics ----------------------------------
    st.subheader("Summary Statistics")

    summary_stats = pd.DataFrame({
        "Mean Daily Return": returns.mean(),
        "Volatility (Daily)": returns.std(),
    })

    summary_stats["Annual Return"] = summary_stats["Mean Daily Return"] * 252
    summary_stats["Annual Volatility"] = summary_stats["Volatility (Daily)"] * (252 ** 0.5)

    risk_free_rate = 0.045

    summary_stats["Sharpe Ratio"] = (
        (summary_stats["Annual Return"] - risk_free_rate)
        / summary_stats["Annual Volatility"]
    )

    st.dataframe(
        summary_stats.style.format({
            "Mean Daily Return": "{:.4%}",
            "Volatility (Daily)": "{:.4%}",
            "Annual Return": "{:.2%}",
            "Annual Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
        }),
        width="stretch"
    )
    # -- Equal-weight portfolio ---------------------------
    st.subheader("Equal-Weight Portfolio Performance")

    # Remove benchmark from portfolio
    asset_columns = [col for col in close_prices.columns if col != "S&P 500"]

    if len(asset_columns) < 1:
        st.warning("No valid assets for portfolio calculation.")
    else:
        # Portfolio returns (equal weight)
        portfolio_returns = returns[asset_columns].mean(axis=1)

        # Cumulative return
        portfolio_cum = (1 + portfolio_returns).cumprod()

        fig_port = go.Figure()

        fig_port.add_trace(
            go.Scatter(
                x=portfolio_cum.index,
                y=portfolio_cum,
                mode="lines",
                name="Equal-Weight Portfolio",
                line=dict(color="black", width=3)
            )
        )

        # Add S&P 500 for comparison (if available)
        if "S&P 500" in returns.columns:
            sp500_cum = (1 + returns["S&P 500"]).cumprod()

            fig_port.add_trace(
                go.Scatter(
                    x=sp500_cum.index,
                    y=sp500_cum,
                    mode="lines",
                    name="S&P 500",
                    line=dict(dash="dash")
                )
            )

        fig_port.update_layout(
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_port, width="stretch")

    with st.expander("View Closing Prices"):
        st.dataframe(close_prices.tail(60), width="stretch")

    with st.expander("View Normalized Prices"):
        st.dataframe(normalized_prices.tail(60), width="stretch")

else:
    st.info("Enter stock tickers in the sidebar to get started.")