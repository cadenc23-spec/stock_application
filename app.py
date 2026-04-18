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
import plotly.express as px
from scipy import stats


# -- Page configuration ----------------------------------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")
st.markdown("""
Analyze and compare multiple stocks against the S&P 500 benchmark.  
This dashboard provides insights into performance, risk, correlations, and portfolio behavior using an equal-weight strategy.
""")

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
corr_window = st.sidebar.slider(
    "Rolling Correlation Window (days)",
    min_value=10,
    max_value=120,
    value=30,
    step=5
)

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.5, step=0.1
) / 100

corr_pair_input = st.sidebar.text_input(
    "Rolling Correlation Pair",
    value="AAPL,MSFT"
)
corr_pair = [t.strip().upper() for t in corr_pair_input.split(",") if t.strip()]

qq_choice = st.sidebar.text_input(
    "Q-Q Plot Asset",
    value="PORTFOLIO"
).upper().strip()

vol_window = st.sidebar.slider(
    "Rolling Volatility Window (days)",
    min_value=10,
    max_value=120,
    value=30,
    step=5
)

two_asset_input = st.sidebar.text_input(
    "Two-Asset Explorer Pair",
    value="AAPL,MSFT"
)
two_asset_pair = [t.strip().upper() for t in two_asset_input.split(",") if t.strip()]

asset1_weight = st.sidebar.slider(
    "Weight in First Asset (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=5
) / 100

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
    # Assets only (exclude benchmark from portfolio construction)
    asset_columns = [col for col in close_prices.columns if col != "S&P 500"]
    asset_returns = returns.drop(columns=["S&P 500"], errors="ignore")

    # Equal-weight portfolio objects used in multiple sections
    portfolio_returns = None
    portfolio_cum = None
    portfolio_weights = None

    if len(asset_columns) >= 1:
        portfolio_weights = pd.Series(
            1 / len(asset_columns),
            index=asset_columns,
            name="Weight"
        )
        portfolio_returns = asset_returns[asset_columns].mean(axis=1)
        portfolio_cum = (1 + portfolio_returns).cumprod()

    # Two-asset portfolio objects
    two_asset_portfolio_returns = None
    two_asset_portfolio_cum = None
    two_asset_volatility = None
    pair1 = None
    pair2 = None
    w1 = asset1_weight
    w2 = 1 - w1

    if len(two_asset_pair) == 2:
        pair1, pair2 = two_asset_pair[0], two_asset_pair[1]

        if pair1 in asset_returns.columns and pair2 in asset_returns.columns:
            two_asset_portfolio_returns = (
                w1 * asset_returns[pair1] + w2 * asset_returns[pair2]
            )
            two_asset_portfolio_cum = (1 + two_asset_portfolio_returns).cumprod()
            two_asset_volatility = (
                two_asset_portfolio_returns.rolling(window=vol_window).std() * (252 ** 0.5)
            )
    
    # -- Summary statistics ----------------------------------
    st.subheader("Summary Statistics")

    summary_stats = pd.DataFrame({
        "Mean Daily Return": returns.mean(),
        "Volatility (Daily)": returns.std(),
    })

    summary_stats["Annual Return"] = summary_stats["Mean Daily Return"] * 252
    summary_stats["Annual Volatility"] = summary_stats["Volatility (Daily)"] * (252 ** 0.5)
    summary_stats["Sharpe Ratio"] = (
        (summary_stats["Annual Return"] - risk_free_rate)
        / summary_stats["Annual Volatility"]
    )

    if portfolio_returns is not None:
        portfolio_row = pd.DataFrame({
            "Mean Daily Return": [portfolio_returns.mean()],
            "Volatility (Daily)": [portfolio_returns.std()],
            "Annual Return": [portfolio_returns.mean() * 252],
            "Annual Volatility": [portfolio_returns.std() * (252 ** 0.5)],
            "Sharpe Ratio": [((portfolio_returns.mean() * 252) - risk_free_rate) / (portfolio_returns.std() * (252 ** 0.5))]
        }, index=["Equal-Weight Portfolio"])

        summary_stats = pd.concat([summary_stats, portfolio_row])

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

        # -- Best / Worst performer ---------------------------
    st.subheader("Top Performers")

    cumulative_returns = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1

    best_stock = cumulative_returns.idxmax()
    worst_stock = cumulative_returns.idxmin()

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Performer",
        best_stock,
        f"{cumulative_returns[best_stock]:.2%}"
    )

    col2.metric(
        "Worst Performer",
        worst_stock,
        f"{cumulative_returns[worst_stock]:.2%}"
    )
    # -- Equal-weight portfolio ---------------------------
    st.subheader("Equal-Weight Portfolio Performance")

    if portfolio_returns is None or portfolio_cum is None:
        st.warning("No valid assets for portfolio calculation.")
    else:
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

    # -- Two-Asset Portfolio Explorer ---------------------
    st.subheader("Two-Asset Portfolio Explorer")

    if len(two_asset_pair) != 2:
        st.warning("Enter exactly two tickers for the two-asset explorer, like AAPL,MSFT.")
    elif two_asset_portfolio_returns is None or two_asset_portfolio_cum is None:
        st.warning("Both two-asset explorer tickers must be in your selected stock list.")
    else:
        st.markdown(
            f"Portfolio weights: **{pair1} = {w1:.0%}**, **{pair2} = {w2:.0%}**"
        )

        fig_two_asset = go.Figure()

        fig_two_asset.add_trace(
            go.Scatter(
                x=two_asset_portfolio_cum.index,
                y=two_asset_portfolio_cum,
                mode="lines",
                name="Two-Asset Portfolio",
                line=dict(width=3)
            )
        )

        fig_two_asset.add_trace(
            go.Scatter(
                x=(1 + asset_returns[pair1]).cumprod().index,
                y=(1 + asset_returns[pair1]).cumprod(),
                mode="lines",
                name=pair1
            )
        )

        fig_two_asset.add_trace(
            go.Scatter(
                x=(1 + asset_returns[pair2]).cumprod().index,
                y=(1 + asset_returns[pair2]).cumprod(),
                mode="lines",
                name=pair2
            )
        )

        fig_two_asset.update_layout(
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_two_asset, width="stretch")

        st.subheader("Two-Asset Portfolio Rolling Volatility")

        fig_two_vol = go.Figure()

        fig_two_vol.add_trace(
            go.Scatter(
                x=two_asset_volatility.index,
                y=two_asset_volatility,
                mode="lines",
                name="Portfolio Rolling Volatility",
                line=dict(width=2)
            )
        )

        fig_two_vol.update_layout(
            xaxis_title="Date",
            yaxis_title="Annualized Volatility",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig_two_vol, width="stretch")

        # -- Max Drawdown -------------------------------------
        running_max = portfolio_cum.cummax()
        drawdown = (portfolio_cum - running_max) / running_max
        max_drawdown = drawdown.min()

        st.metric("Portfolio Max Drawdown", f"{max_drawdown:.2%}")
        
    # -- Correlation matrix -------------------------------
    st.subheader("Correlation Matrix")

    asset_returns = returns.drop(columns=["S&P 500"], errors="ignore")
    corr_matrix = asset_returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Stock Return Correlation"
    )

    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, width="stretch")
    # -- Rolling correlation ------------------------------
    st.subheader("Rolling Correlation")

    asset_returns = returns.drop(columns=["S&P 500"], errors="ignore")

    if len(corr_pair) != 2:
        st.warning("Enter exactly two tickers for rolling correlation, like AAPL,MSFT.")
    elif corr_pair[0] not in asset_returns.columns or corr_pair[1] not in asset_returns.columns:
        st.warning("One or both rolling-correlation tickers are not in your selected stock list.")
    else:
        rolling_corr = (
            asset_returns[corr_pair[0]]
            .rolling(window=corr_window)
            .corr(asset_returns[corr_pair[1]])
        )

        fig_roll_corr = go.Figure()

        fig_roll_corr.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                mode="lines",
                name=f"{corr_pair[0]} vs {corr_pair[1]}",
                line=dict(width=2)
            )
        )

        fig_roll_corr.update_layout(
            xaxis_title="Date",
            yaxis_title="Rolling Correlation",
            template="plotly_white",
            height=450
        )

        st.plotly_chart(fig_roll_corr, width="stretch")
    
        # -- Q-Q plot -----------------------------------------
    st.subheader("Q-Q Plot")

    asset_returns = returns.drop(columns=["S&P 500"], errors="ignore")

    if len(asset_columns) < 1:
        st.warning("No valid assets available for Q-Q plot.")
    else:
        if qq_choice == "PORTFOLIO":
            qq_series = portfolio_returns.dropna()
            qq_label = "Equal-Weight Portfolio"
        elif qq_choice in asset_returns.columns:
            qq_series = asset_returns[qq_choice].dropna()
            qq_label = qq_choice
        else:
            st.warning("Enter PORTFOLIO or one of your selected stock tickers for the Q-Q plot.")
            qq_series = None

        if qq_series is not None and len(qq_series) > 0:
            theoretical, sample = stats.probplot(qq_series, dist="norm", fit=False)

            fig_qq = go.Figure()

            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical,
                    y=sample,
                    mode="markers",
                    name=qq_label
                )
            )

            # Reference line
            slope, intercept, r = stats.probplot(qq_series, dist="norm")[1]
            x_line = [min(theoretical), max(theoretical)]
            y_line = [slope * x + intercept for x in x_line]

            fig_qq.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Reference Line"
                )
            )

            fig_qq.update_layout(
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                template="plotly_white",
                height=450,
                title=f"Q-Q Plot: {qq_label}"
            )

            st.plotly_chart(fig_qq, width="stretch")
    # -- Portfolio explorer -------------------------------
    st.subheader("Portfolio Explorer")

    if portfolio_returns is None or portfolio_cum is None or portfolio_weights is None:
        st.warning("Portfolio explorer is unavailable because no valid portfolio assets were found.")
    else:
        # Portfolio summary table
        portfolio_summary = pd.DataFrame({
            "Metric": [
                "Number of Assets",
                "Average Portfolio Weight",
                "Mean Daily Return",
                "Daily Volatility",
                "Annual Return",
                "Annual Volatility",
                "Sharpe Ratio",
                "Cumulative Return"
            ],
            "Value": [
                len(asset_columns),
                portfolio_weights.mean(),
                portfolio_returns.mean(),
                portfolio_returns.std(),
                portfolio_returns.mean() * 252,
                portfolio_returns.std() * (252 ** 0.5),
                ((portfolio_returns.mean() * 252) - risk_free_rate) / (portfolio_returns.std() * (252 ** 0.5)),
                portfolio_cum.iloc[-1] - 1
            ]
        })

        st.markdown("**Portfolio Summary**")
        st.dataframe(
            portfolio_summary.style.format({
                "Value": lambda x: (
                    f"{x:.2%}" if isinstance(x, (int, float)) and abs(x) < 10 and x != len(asset_columns)
                    else f"{x:.4f}" if isinstance(x, (int, float))
                    else x
                )
            }),
            width="stretch"
        )

        st.markdown("**Portfolio Weights**")
        weights_df = portfolio_weights.reset_index()
        weights_df.columns = ["Asset", "Weight"]
        st.dataframe(
            weights_df.style.format({"Weight": "{:.2%}"}),
            width="stretch"
        )

        st.markdown("**Portfolio Return Series**")
        portfolio_df = pd.DataFrame({
            "Portfolio Return": portfolio_returns,
            "Portfolio Growth": portfolio_cum
        })

        if "S&P 500" in returns.columns:
            portfolio_df["S&P 500 Return"] = returns["S&P 500"]
            portfolio_df["S&P 500 Growth"] = (1 + returns["S&P 500"]).cumprod()

        st.dataframe(
            portfolio_df.tail(60).style.format({
                "Portfolio Return": "{:.4%}",
                "Portfolio Growth": "{:.4f}",
                "S&P 500 Return": "{:.4%}",
                "S&P 500 Growth": "{:.4f}",
            }),
            width="stretch"
        )

        st.markdown("**Asset Return Explorer**")
        st.dataframe(
            asset_returns.tail(60).style.format("{:.4%}"),
            width="stretch"
        )
    csv = close_prices.to_csv().encode("utf-8")

    st.download_button(
        label="Download Price Data",
        data=csv,
        file_name="stock_data.csv",
        mime="text/csv",
    )
    with st.expander("View Closing Prices"):
        st.dataframe(close_prices.tail(60), width="stretch")

    with st.expander("View Normalized Prices"):
        st.dataframe(normalized_prices.tail(60), width="stretch")

else:
    st.info("Enter stock tickers in the sidebar to get started.")