"""Mean-variance portfolio optimization (Markowitz) using scipy."""
import numpy as np
from scipy.optimize import minimize

from analysis import portfolio_performance


def _setup(n_assets, max_weight):
    start = np.full(n_assets, 1 / n_assets)  # initial guess: equal weights
    bounds = [(0.0, max_weight)] * n_assets  # no short selling, cap per asset
    weights_sum_to_one = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    return start, bounds, [weights_sum_to_one]


def _solve(objective, n_assets, max_weight, extra_constraints=()):
    start, bounds, constraints = _setup(n_assets, max_weight)
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints + list(extra_constraints),
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    weights = np.clip(result.x, 0, None)
    return weights / weights.sum()  # tidy tiny rounding errors


def min_volatility(mean_returns, cov, max_weight):
    """Lowest-risk portfolio."""
    return _solve(
        lambda w: portfolio_performance(w, mean_returns, cov)[1],
        len(mean_returns),
        max_weight,
    )


def max_sharpe(mean_returns, cov, risk_free_rate, max_weight):
    """Portfolio with the best return per unit of risk."""

    def negative_sharpe(w):
        ret, vol = portfolio_performance(w, mean_returns, cov)
        return -(ret - risk_free_rate) / vol

    return _solve(negative_sharpe, len(mean_returns), max_weight)


def max_return(mean_returns, cov, max_weight):
    """Highest-return portfolio allowed by the weight cap."""
    return _solve(
        lambda w: -portfolio_performance(w, mean_returns, cov)[0],
        len(mean_returns),
        max_weight,
    )


def efficient_frontier(mean_returns, cov, max_weight, n_points=40):
    """For a range of target returns, find the lowest-risk portfolio hitting each one."""
    low_w = min_volatility(mean_returns, cov, max_weight)
    high_w = max_return(mean_returns, cov, max_weight)
    low_ret = portfolio_performance(low_w, mean_returns, cov)[0]
    high_ret = portfolio_performance(high_w, mean_returns, cov)[0]

    points = []
    for target in np.linspace(low_ret, high_ret, n_points):
        hit_target = {
            "type": "eq",
            "fun": lambda w, t=target: portfolio_performance(w, mean_returns, cov)[0] - t,
        }
        try:
            w = _solve(
                lambda w: portfolio_performance(w, mean_returns, cov)[1],
                len(mean_returns),
                max_weight,
                extra_constraints=[hit_target],
            )
        except RuntimeError:
            continue
        ret, vol = portfolio_performance(w, mean_returns, cov)
        points.append((vol, ret))
    return np.array(points)
