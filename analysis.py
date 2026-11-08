"""Return and risk calculations."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252  # stock markets are open roughly 252 days a year


def daily_returns(prices):
    """Percent change from one day's close to the next."""
    return prices.pct_change().dropna()


def annual_returns(returns):
    """Average daily return scaled up to a year."""
    return returns.mean() * TRADING_DAYS


def annual_cov(returns):
    """Covariance matrix scaled to a year: how assets move together."""
    return returns.cov() * TRADING_DAYS


def portfolio_performance(weights, mean_returns, cov):
    """Expected annual return and volatility of a portfolio with the given weights."""
    weights = np.asarray(weights)
    ret = weights @ mean_returns
    vol = np.sqrt(weights @ cov @ weights)
    return ret, vol


def asset_summary(returns, risk_free_rate):
    """One row per asset: annual return, volatility, Sharpe ratio."""
    ret = annual_returns(returns)
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    return pd.DataFrame(
        {
            "Annual return": ret,
            "Volatility": vol,
            "Sharpe": (ret - risk_free_rate) / vol,
        }
    ).sort_values("Sharpe", ascending=False)
