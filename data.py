import yfinance as yf
import numpy as np
import pandas as pd


def fetch_returns(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """
    Download monthly adjusted close prices for each ticker and return
    a DataFrame of log returns.  Log return = ln(price_today / price_yesterday).
    We use log returns instead of simple % returns because they're additive
    over time — a +10% month followed by a -10% month is not flat in simple
    returns, but log returns handle this correctly.
    """
    raw = yf.download(tickers, period=period, interval="1mo",
                      auto_adjust=True, progress=False)["Close"]

    # yfinance returns a Series (not DataFrame) when only one ticker is passed
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Drop any column that's entirely NaN (bad ticker)
    raw = raw.dropna(axis=1, how="all")

    # Log returns: ln(P_t / P_{t-1}), drop the first NaN row
    returns = np.log(raw / raw.shift(1)).dropna()
    return returns


def mean_and_cov(returns: pd.DataFrame):
    """
    Returns the mean monthly log return and covariance matrix for each ticker.
    The covariance matrix captures how much two assets move together —
    assets that move in opposite directions reduce portfolio volatility.
    """
    mean_returns = returns.mean()        # average monthly return per ticker
    cov_matrix   = returns.cov()         # N×N matrix of pairwise covariances
    return mean_returns, cov_matrix
