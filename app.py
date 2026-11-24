"""Interactive Streamlit dashboard for the portfolio optimizer.

Run it locally with:
    streamlit run app.py

Pick assets, dates and limits in the sidebar; the app downloads prices (or reads
them from the SQLite database), runs the same optimizer as main.py, and shows the
optimal allocations and risk-return charts.
"""
import math
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from analysis import TRADING_DAYS, annual_cov, annual_returns, asset_summary, daily_returns, portfolio_performance
from data import download_prices, load_prices, save_prices
from optimizer import efficient_frontier, max_sharpe, min_volatility

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

# One fixed color per portfolio, so the same portfolio is the same color in every chart.
COLORS = {"Equal weights": "#2a78d6", "Min volatility": "#eb6834", "Max Sharpe": "#1baf7a"}
MARKERS = {"Equal weights": "circle", "Min volatility": "diamond", "Max Sharpe": "star"}

# Extra choices offered in the sidebar on top of the default tickers in config.py.
POPULAR_TICKERS = ["SPY", "QQQ", "NVDA", "AMZN", "GOOGL", "META", "KO", "WMT", "BND", "VNQ"]


# ---------------------------------------------------------------- data loading
def tickers_to_download(tickers, start, end):
    """Tickers the database doesn't have (or doesn't cover the chosen dates for)."""
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            coverage = pd.read_sql_query(
                "SELECT ticker, MIN(date) AS first, MAX(date) AS last FROM prices GROUP BY ticker",
                conn,
            ).set_index("ticker")
    except (sqlite3.Error, pd.errors.DatabaseError):
        return list(tickers)  # no database yet (e.g. a fresh deploy)

    slack = timedelta(days=7)  # weekends and holidays mean the exact dates may have no price
    missing = []
    for t in tickers:
        if t not in coverage.index:
            missing.append(t)
            continue
        first = date.fromisoformat(coverage.loc[t, "first"])
        last = date.fromisoformat(coverage.loc[t, "last"])
        if first > start + slack or last < min(end, date.today()) - slack:
            missing.append(t)
    return missing


@st.cache_data(ttl=60 * 60 * 24, show_spinner="Loading prices...")
def get_prices(tickers, start, end):
    """Make sure the database has the prices we need, then read them back with SQL."""
    missing = tickers_to_download(tickers, start, end)
    if missing:
        downloaded = download_prices(missing, start.isoformat(), end.isoformat())
        if not downloaded.empty:
            save_prices(downloaded)

    # Only keep tickers that actually have data (a typo like "APPL" returns nothing).
    with sqlite3.connect(config.DB_PATH) as conn:
        found = pd.read_sql_query(
            f"SELECT DISTINCT ticker FROM prices WHERE ticker IN ({','.join('?' * len(tickers))})",
            conn,
            params=list(tickers),
        )["ticker"].tolist()
    available = [t for t in tickers if t in found]
    if not available:
        return pd.DataFrame()
    return load_prices(available, start.isoformat(), end.isoformat())


@st.cache_data(show_spinner="Optimizing portfolios...")
def optimize(returns, rf, max_weight):
    """Run the optimizer and return weights for each strategy plus the efficient frontier."""
    mu, cov = annual_returns(returns), annual_cov(returns)
    n = len(mu)
    weights = {
        "Equal weights": np.full(n, 1 / n),
        "Min volatility": min_volatility(mu, cov, max_weight),
        "Max Sharpe": max_sharpe(mu, cov, rf, max_weight),
    }
    frontier = efficient_frontier(mu, cov, max_weight, n_points=30)
    return weights, frontier


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")

picked = st.sidebar.multiselect(
    "Assets",
    options=config.TICKERS + [t for t in POPULAR_TICKERS if t not in config.TICKERS],
    default=config.TICKERS,
    help="Yahoo Finance ticker symbols.",
)
extra = st.sidebar.text_input("Add other tickers", placeholder="e.g. TSLA, COST")
extra_tickers = [t.strip().upper() for t in extra.split(",") if t.strip()]
tickers = list(dict.fromkeys(picked + extra_tickers))  # remove duplicates, keep order

start = st.sidebar.date_input("Start date", value=date.fromisoformat(config.START_DATE),
                              min_value=date(2000, 1, 1), max_value=date.today())
end = st.sidebar.date_input("End date", value=date.fromisoformat(config.END_DATE),
                            min_value=date(2000, 1, 1), max_value=date.today())

rf = st.sidebar.slider(
    "Risk-free rate (%)", 0.0, 8.0, config.RISK_FREE_RATE * 100, 0.25,
    help="Annual return of a 'safe' investment like T-bills. Used in the Sharpe ratio.",
) / 100

# With n assets, the cap must be at least 1/n, otherwise the weights can't add up to 100%.
min_cap = math.ceil(100 / max(len(tickers), 2))
max_weight = st.sidebar.slider(
    "Max weight per asset (%)", min_cap, 100, max(min_cap, int(config.MAX_WEIGHT * 100)), 5,
    help="Stops the optimizer from putting everything into one asset.",
) / 100

st.sidebar.caption("Educational project, not investment advice.")

# ---------------------------------------------------------------- checks
st.title("📈 Portfolio Optimizer")
st.write(
    "Finds the best way to split money across the assets you choose, using "
    "**mean-variance (Markowitz) optimization** on historical daily prices."
)

if len(tickers) < 2:
    st.info("Pick at least two assets in the sidebar.")
    st.stop()
if start >= end - timedelta(days=90):
    st.warning("Choose a date range of at least a few months.")
    st.stop()

try:
    prices = get_prices(tuple(tickers), start, end)
except Exception as err:  # network problems, Yahoo rate limits, ...
    st.error(f"Couldn't load prices: {err}")
    st.stop()

dropped = [t for t in tickers if t not in prices.columns]
if dropped:
    st.warning(f"No price data found for: {', '.join(dropped)}. They were left out.")
if prices.shape[1] < 2 or len(prices) < 60:
    st.error("Not enough data to optimize. Try other assets or a longer date range.")
    st.stop()

tickers = list(prices.columns)
max_weight = max(max_weight, 1 / len(tickers))  # in case some tickers were dropped
returns = daily_returns(prices)
mu, cov = annual_returns(returns), annual_cov(returns)

try:
    weights, frontier = optimize(returns, rf, max_weight)
except RuntimeError as err:
    st.error(f"The optimizer couldn't find a solution: {err}")
    st.stop()

st.caption(
    f"Using {len(prices):,} trading days from {prices.index[0]:%b %d, %Y} "
    f"to {prices.index[-1]:%b %d, %Y} for {len(tickers)} assets."
)

# ---------------------------------------------------------------- headline numbers
perf = {}
for name, w in weights.items():
    ret, vol = portfolio_performance(w, mu, cov)
    perf[name] = {"Expected return": ret, "Volatility": vol, "Sharpe": (ret - rf) / vol}

cols = st.columns(3)
for col, (name, p) in zip(cols, perf.items()):
    with col.container(border=True):
        st.markdown(f"**{name}**")
        st.markdown(
            "<div style='display:flex; justify-content:space-between; gap:0.5rem'>"
            + "".join(
                f"<div><div style='font-size:0.85rem; opacity:0.7'>{label}</div>"
                f"<div style='font-size:1.6rem; font-weight:600'>{value}</div></div>"
                for label, value in [
                    ("Return", f"{p['Expected return']:.1%}"),
                    ("Risk", f"{p['Volatility']:.1%}"),
                    ("Sharpe", f"{p['Sharpe']:.2f}"),
                ]
            )
            + "</div>",
            unsafe_allow_html=True,
        )

tab_alloc, tab_frontier, tab_growth, tab_assets, tab_help = st.tabs(
    ["Allocations", "Risk vs. return", "Historical growth", "Asset stats", "What do these mean?"]
)

# ---------------------------------------------------------------- allocations
with tab_alloc:
    alloc = pd.DataFrame(weights, index=tickers)
    fig = go.Figure()
    for name in alloc.columns:
        fig.add_bar(
            name=name, x=alloc.index, y=alloc[name], marker_color=COLORS[name],
            hovertemplate="%{x}: %{y:.1%}<extra>" + name + "</extra>",
        )
    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.08, height=420,
        yaxis=dict(title="Share of portfolio", tickformat=".0%", rangemode="tozero"),
        legend=dict(orientation="h", y=1.1), margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    table = alloc.copy()
    table = table[(table > 0.0005).any(axis=1)]
    st.dataframe(table.style.format("{:.1%}"), width="stretch")
    st.download_button(
        "Download weights (CSV)", alloc.round(4).to_csv().encode(), "weights.csv", "text/csv"
    )

# ---------------------------------------------------------------- risk vs return
with tab_frontier:
    summary = asset_summary(returns, rf)
    fig = go.Figure()
    if len(frontier):
        fig.add_scatter(
            x=frontier[:, 0], y=frontier[:, 1], mode="lines", name="Efficient frontier",
            line=dict(color="#7a7974", width=2),
            hovertemplate="Risk %{x:.1%}<br>Return %{y:.1%}<extra>Frontier</extra>",
        )
    fig.add_scatter(
        x=summary["Volatility"], y=summary["Annual return"], mode="markers+text",
        name="Individual assets", text=summary.index, textposition="top right",
        marker=dict(size=9, color="#a3a29b"),
        hovertemplate="%{text}<br>Risk %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    )
    for name, p in perf.items():
        fig.add_scatter(
            x=[p["Volatility"]], y=[p["Expected return"]], mode="markers", name=name,
            marker=dict(size=16, color=COLORS[name], symbol=MARKERS[name],
                        line=dict(width=2, color="white")),
            hovertemplate=f"{name}<br>Risk %{{x:.1%}}<br>Return %{{y:.1%}}<extra></extra>",
        )
    fig.update_layout(
        height=520, margin=dict(t=40, b=10), legend=dict(orientation="h", y=1.08),
        xaxis=dict(title="Volatility (annual risk)", tickformat=".0%"),
        yaxis=dict(title="Expected annual return", tickformat=".0%"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each gray dot is one asset. The line is the efficient frontier: the highest "
        "return you can get for each level of risk. Points below it are inefficient."
    )

# ---------------------------------------------------------------- growth of $1
with tab_growth:
    fig = go.Figure()
    for name, w in weights.items():
        growth = (1 + returns @ w).cumprod()
        fig.add_scatter(
            x=growth.index, y=growth, mode="lines", name=name,
            line=dict(color=COLORS[name], width=2),
            hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra>" + name + "</extra>",
        )
    fig.update_layout(
        height=450, margin=dict(t=40, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        yaxis=dict(title="Value of $1 invested", tickprefix="$"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "⚠️ In-sample: these weights were chosen using this same period, so the result "
        "looks better than it would in real life. See backtest.py for an out-of-sample test."
    )

# ---------------------------------------------------------------- asset stats
with tab_assets:
    stats = asset_summary(returns, rf)
    stats["Weight in Max Sharpe"] = pd.Series(weights["Max Sharpe"], index=tickers)
    st.dataframe(
        stats.style.format({
            "Annual return": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}",
            "Weight in Max Sharpe": "{:.1%}",
        }),
        width="stretch",
    )
    st.markdown("**Correlation** (1 = always move together, 0 = unrelated, −1 = opposite)")
    st.dataframe(
        returns.corr().style.format("{:.2f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1),
        width="stretch",
    )

# ---------------------------------------------------------------- explainer
with tab_help:
    st.markdown(
        f"""
- **Expected return**: the average yearly gain, based on past daily returns × {TRADING_DAYS} trading days.
- **Volatility (risk)**: how much returns bounce around each year (standard deviation). Higher = bumpier ride.
- **Sharpe ratio**: (return − risk-free rate) ÷ volatility. Extra return earned per unit of risk; higher is better.
- **Equal weights**: the simple baseline, same amount in every asset.
- **Min volatility**: the least bumpy mix possible.
- **Max Sharpe**: the mix with the best return for the risk taken.
- **Efficient frontier**: the best possible return at every risk level. Diversifying across assets that
  don't move together lets a portfolio be less risky than any single asset in it.

Constraints used: weights add up to 100%, no short selling, and at most {max_weight:.0%} in one asset.
        """
    )
