# Markowitz Portfolio Optimizer

**Live:** https://portfolio-optimizer-7octrbjvzbzqbs7dwvj7fv.streamlit.app/

Find the efficient frontier for any set of stocks — Modern Portfolio Theory
implemented from scratch with NumPy and SciPy, visualized in Streamlit.

## What it does

- Enter any list of tickers → downloads 5 years of monthly prices (yfinance)
- Computes the **max-Sharpe portfolio** (best return per unit of risk) and the
  **minimum-variance portfolio** (lowest possible volatility)
- Traces the full **efficient frontier** — the curve of optimal portfolios
- Interactive Plotly chart: every frontier point, both optimal portfolios, and
  each individual asset plotted in risk/return space
- Shows optimal weight allocations for both portfolios

## The math

All optimization is implemented by hand (SciPy `minimize` with SLSQP), not a
portfolio library:

| Piece | How |
|---|---|
| Returns | Monthly log returns: `ln(P_t / P_{t-1})` — additive over time, unlike simple returns |
| Expected return | Mean monthly log return × 12 (annualized) |
| Risk | `sqrt(wᵀ Σ w)` — the covariance matrix Σ is what captures diversification |
| Sharpe ratio | `(return − risk-free) / volatility`, risk-free ≈ 4.5% (T-bill) |
| Max Sharpe | Minimize negative Sharpe subject to weights ≥ 0 and Σw = 1 |
| Frontier | For each target return, minimize variance subject to hitting that return |

The core insight MPT formalizes: a portfolio's risk is not the average of its
assets' risks — correlation is everything. Two volatile assets that move
opposite each other can combine into a calm portfolio.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python · Streamlit · NumPy · SciPy · pandas · yfinance · Plotly

Built by Julian Quevedo ([julianq.dev](https://cashredo.github.io/julianq.dev))
