"""Backtest: choose weights using older data, then test them on newer data the optimizer never saw.

Run main.py at least once first so portfolio.db has prices in it.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from analysis import TRADING_DAYS, annual_cov, annual_returns, daily_returns
from data import load_prices
from optimizer import max_sharpe, min_volatility

# Training period: everything before this date. Test period: this date onward.
SPLIT_DATE = "2024-01-01"


def realized_stats(port_returns, rf):
    """What actually happened to a portfolio over the test period."""
    ann_ret = port_returns.mean() * TRADING_DAYS
    ann_vol = port_returns.std() * np.sqrt(TRADING_DAYS)
    return {
        "Annual return": ann_ret,
        "Volatility": ann_vol,
        "Sharpe": (ann_ret - rf) / ann_vol,
        "Total growth": (1 + port_returns).prod() - 1,
    }


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    rf = config.RISK_FREE_RATE

    prices = load_prices(config.TICKERS, config.START_DATE, config.END_DATE)
    returns = daily_returns(prices)
    train = returns[returns.index < SPLIT_DATE]
    test = returns[returns.index >= SPLIT_DATE]
    print(f"Training: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"Testing:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")

    # 1. Choose weights using ONLY the training data.
    mu, cov = annual_returns(train), annual_cov(train)
    n = len(config.TICKERS)
    portfolios = {
        "Equal weights": np.full(n, 1 / n),
        "Min volatility": min_volatility(mu, cov, config.MAX_WEIGHT),
        "Max Sharpe": max_sharpe(mu, cov, rf, config.MAX_WEIGHT),
    }

    # 2. Freeze those weights and apply them to the test period's daily returns.
    test_returns = {name: test @ w for name, w in portfolios.items()}

    # 3. Measure what actually happened.
    results = pd.DataFrame({name: realized_stats(r, rf) for name, r in test_returns.items()}).T
    print("\n=== Test-period results (data the optimizer never saw) ===")
    print(results.to_string(float_format=lambda x: f"{x:.3f}"))

    eq_vol = results.loc["Equal weights", "Volatility"]
    print("\n=== Volatility vs. equal-weight baseline ===")
    for name in ["Min volatility", "Max Sharpe"]:
        change = (eq_vol - results.loc[name, "Volatility"]) / eq_vol
        direction = "lower" if change > 0 else "higher"
        print(f"{name}: {abs(change):.0%} {direction} volatility than equal weights")

    # Save weights and a chart of how $1 would have grown.
    pd.DataFrame(portfolios, index=config.TICKERS).round(4).to_csv(
        os.path.join(config.OUTPUT_DIR, "backtest_weights.csv")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, r in test_returns.items():
        ax.plot((1 + r).cumprod(), label=name)
    ax.set_title(f"Growth of $1 in the test period (weights chosen before {SPLIT_DATE})")
    ax.set_ylabel("Value of $1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    chart = os.path.join(config.OUTPUT_DIR, "backtest_growth.png")
    fig.savefig(chart, dpi=150)
    plt.close(fig)
    print(f"\nSaved chart to {chart}")


if __name__ == "__main__":
    main()
