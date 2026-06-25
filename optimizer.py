import numpy as np
from scipy.optimize import minimize


RF_RATE = 0.045  # risk-free rate (approx current T-bill yield, annualized)


def portfolio_stats(weights, mean_returns, cov_matrix):
    """
    Given a weight vector, compute annualized return, annualized volatility,
    and Sharpe ratio for the portfolio.

    - Return:     dot product of weights and mean returns, × 12 to annualize
    - Volatility: sqrt(w^T · Σ · w) — the covariance matrix Σ captures
                  how correlated each pair of assets is.  If two assets always
                  move together, you get no diversification benefit.
    - Sharpe:     (return − risk-free rate) / volatility — return per unit of
                  risk you're actually taking.  Higher is better.
    """
    w   = np.array(weights)
    ret = float(np.dot(w, mean_returns)) * 12
    vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix * 12, w))))
    sharpe = (ret - RF_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _base_constraints(n):
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

def _bounds(n):
    return tuple((0.0, 1.0) for _ in range(n))   # no short selling


def max_sharpe_portfolio(mean_returns, cov_matrix):
    """
    Maximize Sharpe ratio — the portfolio with the best return per unit of risk.
    This is the classic 'tangency portfolio' on the efficient frontier.
    scipy.minimize can only minimize, so we minimize negative Sharpe.
    """
    n    = len(mean_returns)
    x0   = np.full(n, 1 / n)

    result = minimize(
        lambda w: -portfolio_stats(w, mean_returns, cov_matrix)[2],
        x0,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_base_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result


def min_variance_portfolio(mean_returns, cov_matrix):
    """
    Minimize total portfolio volatility with no return constraint.
    This gives you the lowest-risk portfolio that exists anywhere
    in the investable universe of these tickers.
    """
    n  = len(mean_returns)
    x0 = np.full(n, 1 / n)

    result = minimize(
        lambda w: portfolio_stats(w, mean_returns, cov_matrix)[1],
        x0,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_base_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result


def efficient_frontier(mean_returns, cov_matrix, n_points=150):
    """
    Sweep from the minimum possible return to the maximum, and at each
    target return find the minimum-variance portfolio that hits it.
    The collection of those portfolios traces out the efficient frontier.
    """
    n       = len(mean_returns)
    min_ret = float(mean_returns.min()) * 12
    max_ret = float(mean_returns.max()) * 12
    targets = np.linspace(min_ret, max_ret, n_points)

    vols, rets, sharpes = [], [], []

    for target in targets:
        constraints = _base_constraints(n) + [
            {"type": "eq", "fun": lambda w, t=target:
             portfolio_stats(w, mean_returns, cov_matrix)[0] - t}
        ]
        res = minimize(
            lambda w: portfolio_stats(w, mean_returns, cov_matrix)[1],
            np.full(n, 1 / n),
            method="SLSQP",
            bounds=_bounds(n),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success:
            r, v, s = portfolio_stats(res.x, mean_returns, cov_matrix)
            vols.append(v); rets.append(r); sharpes.append(s)

    return np.array(vols), np.array(rets), np.array(sharpes)
