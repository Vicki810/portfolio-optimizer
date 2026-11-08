"""Run the full pipeline: download -> store in SQL -> analyze -> optimize -> chart."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from analysis import annual_cov, annual_returns, asset_summary, daily_returns, portfolio_performance
from data import download_prices, load_prices, save_prices
from optimizer import efficient_frontier, max_sharpe, min_volatility


def describe(name, weights, mean_returns, cov, rf):
    ret, vol = portfolio_performance(weights, mean_returns, cov)
    print(f"\n=== {name} ===")
    print(f"Expected annual return: {ret:6.1%}")
    print(f"Volatility (risk):      {vol:6.1%}")
    print(f"Sharpe ratio:           {(ret - rf) / vol:6.2f}")
    alloc = pd.Series(weights, index=mean_returns.index)
    print("Weights:")
    for ticker, w in alloc[alloc > 0.001].sort_values(ascending=False).items():
        print(f"  {ticker:5s} {w:6.1%}")
    return ret, vol


def plot_frontier(summary, frontier, special_points, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier[:, 0], frontier[:, 1], "-", lw=2, label="Efficient frontier")
    ax.scatter(summary["Volatility"], summary["Annual return"], s=40, color="gray")
    for ticker, row in summary.iterrows():
        ax.annotate(ticker, (row["Volatility"], row["Annual return"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=9)
    for label, (ret, vol), marker in special_points:
        ax.scatter(vol, ret, s=180, marker=marker, zorder=5, label=label)
    ax.set_xlabel("Volatility (annual)")
    ax.set_ylabel("Expected return (annual)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Risk vs. return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Downloading prices...")
    prices = download_prices(config.TICKERS, config.START_DATE, config.END_DATE)
    rows = save_prices(prices)
    print(f"Saved {rows} rows to {config.DB_PATH}")

    prices = load_prices(config.TICKERS, config.START_DATE, config.END_DATE)
    print(f"Loaded {len(prices)} trading days for {len(prices.columns)} assets from SQL")

    returns = daily_returns(prices)
    mean_returns = annual_returns(returns)
    cov = annual_cov(returns)
    rf = config.RISK_FREE_RATE

    summary = asset_summary(returns, rf)
    print("\n=== Individual assets ===")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

    equal = np.full(len(mean_returns), 1 / len(mean_returns))
    w_min = min_volatility(mean_returns, cov, config.MAX_WEIGHT)
    w_sharpe = max_sharpe(mean_returns, cov, rf, config.MAX_WEIGHT)

    p_equal = describe("Equal weights (baseline)", equal, mean_returns, cov, rf)
    p_min = describe("Minimum volatility", w_min, mean_returns, cov, rf)
    p_sharpe = describe("Maximum Sharpe ratio", w_sharpe, mean_returns, cov, rf)

    pd.DataFrame(
        {"Equal": equal, "MinVol": w_min, "MaxSharpe": w_sharpe}, index=mean_returns.index
    ).round(4).to_csv(os.path.join(config.OUTPUT_DIR, "weights.csv"))

    frontier = efficient_frontier(mean_returns, cov, config.MAX_WEIGHT)
    chart = os.path.join(config.OUTPUT_DIR, "efficient_frontier.png")
    plot_frontier(
        summary,
        frontier,
        [("Equal weights", p_equal, "o"), ("Min volatility", p_min, "D"), ("Max Sharpe", p_sharpe, "*")],
        chart,
    )
    print(f"\nSaved chart to {chart} and weights to {config.OUTPUT_DIR}/weights.csv")


if __name__ == "__main__":
    main()
