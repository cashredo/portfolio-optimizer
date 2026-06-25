import streamlit as st
import plotly.graph_objects as go
import numpy as np

from data import fetch_returns, mean_and_cov
from optimizer import (
    portfolio_stats,
    max_sharpe_portfolio,
    min_variance_portfolio,
    efficient_frontier,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("Markowitz Portfolio Optimizer")
st.caption("Find the efficient frontier for any set of stocks — built from scratch using Modern Portfolio Theory.")
st.markdown("*Built by [Julian Quevedo](https://cashredo.github.io/julianq.dev)*", unsafe_allow_html=False)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    ticker_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, NVDA, MSFT, SPY, BND",
        help="Enter 2–10 stock or ETF symbols",
    )
    period = st.selectbox("Historical period", ["1y", "3y", "5y", "10y"], index=2)
    rf_override = st.slider("Risk-free rate (%)", 0.0, 8.0, 4.5, 0.1) / 100

    run = st.button("Optimize →", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
    **How to read this chart**

    Every dot is a possible portfolio.
    The **efficient frontier** is the curve
    along the top-left edge — those are the
    portfolios that give the best return for
    each level of risk.

    🔵 **Max Sharpe** — highest return per
    unit of risk (tangency portfolio)

    🔴 **Min Variance** — lowest total risk
    """)

# ── Main logic ────────────────────────────────────────────────────────────────
if not run:
    st.info("Set your tickers in the sidebar and click **Optimize →** to run.")
    st.stop()

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if len(tickers) < 2:
    st.error("Enter at least 2 tickers."); st.stop()

with st.spinner("Fetching price data…"):
    try:
        returns = fetch_returns(tickers, period)
    except Exception as e:
        st.error(f"Data fetch failed: {e}"); st.stop()

valid_tickers = list(returns.columns)
if len(valid_tickers) < 2:
    st.error("Couldn't get data for enough tickers. Check your symbols."); st.stop()

if set(valid_tickers) != set(tickers):
    st.warning(f"Couldn't load: {set(tickers) - set(valid_tickers)}. Continuing with: {valid_tickers}")

mean_returns, cov_matrix = mean_and_cov(returns)

with st.spinner("Computing efficient frontier…"):
    f_vols, f_rets, f_sharpes = efficient_frontier(mean_returns, cov_matrix, n_points=150)

    ms_res  = max_sharpe_portfolio(mean_returns, cov_matrix)
    mv_res  = min_variance_portfolio(mean_returns, cov_matrix)

ms_ret, ms_vol, ms_sharpe = portfolio_stats(ms_res.x, mean_returns, cov_matrix)
mv_ret, mv_vol, mv_sharpe = portfolio_stats(mv_res.x, mean_returns, cov_matrix)

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Max Sharpe Return",    f"{ms_ret*100:.1f}%")
col2.metric("Max Sharpe Volatility",f"{ms_vol*100:.1f}%")
col3.metric("Max Sharpe Ratio",     f"{ms_sharpe:.2f}")
col4.metric("Min Variance Vol",     f"{mv_vol*100:.1f}%")

# ── Efficient frontier chart ──────────────────────────────────────────────────
fig = go.Figure()

# Frontier curve, colored by Sharpe ratio
fig.add_trace(go.Scatter(
    x=f_vols * 100, y=f_rets * 100,
    mode="markers",
    marker=dict(
        size=5,
        color=f_sharpes,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Sharpe"),
    ),
    name="Efficient Frontier",
    hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
))

# Max Sharpe point
fig.add_trace(go.Scatter(
    x=[ms_vol * 100], y=[ms_ret * 100],
    mode="markers+text",
    marker=dict(size=14, color="#1d4ed8", symbol="star"),
    text=["Max Sharpe"], textposition="top right",
    name="Max Sharpe",
    hovertemplate=f"Return: {ms_ret*100:.1f}%<br>Vol: {ms_vol*100:.1f}%<br>Sharpe: {ms_sharpe:.2f}<extra>Max Sharpe</extra>",
))

# Min Variance point
fig.add_trace(go.Scatter(
    x=[mv_vol * 100], y=[mv_ret * 100],
    mode="markers+text",
    marker=dict(size=14, color="#dc2626", symbol="diamond"),
    text=["Min Vol"], textposition="top right",
    name="Min Variance",
    hovertemplate=f"Return: {mv_ret*100:.1f}%<br>Vol: {mv_vol*100:.1f}%<br>Sharpe: {mv_sharpe:.2f}<extra>Min Variance</extra>",
))

fig.update_layout(
    title="Efficient Frontier",
    xaxis_title="Annualized Volatility (%)",
    yaxis_title="Annualized Return (%)",
    height=520,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ── Weight breakdowns ─────────────────────────────────────────────────────────
col_ms, col_mv = st.columns(2)

def weight_chart(weights, tickers, title, color):
    df_w = {t: round(w * 100, 1) for t, w in zip(tickers, weights) if w > 0.001}
    pie = go.Figure(go.Pie(
        labels=list(df_w.keys()),
        values=list(df_w.values()),
        hole=0.4,
        marker_colors=[color] * len(df_w),
    ))
    pie.update_layout(title=title, height=320, showlegend=True,
                      margin=dict(t=40, b=0, l=0, r=0))
    return pie

with col_ms:
    st.subheader("Max Sharpe weights")
    st.plotly_chart(weight_chart(ms_res.x, valid_tickers, "Max Sharpe", "#1d4ed8"),
                    use_container_width=True)
    for t, w in zip(valid_tickers, ms_res.x):
        if w > 0.001:
            st.write(f"**{t}** — {w*100:.1f}%")

with col_mv:
    st.subheader("Min Variance weights")
    st.plotly_chart(weight_chart(mv_res.x, valid_tickers, "Min Variance", "#dc2626"),
                    use_container_width=True)
    for t, w in zip(valid_tickers, mv_res.x):
        if w > 0.001:
            st.write(f"**{t}** — {w*100:.1f}%")

# ── Correlation heatmap ───────────────────────────────────────────────────────
st.subheader("Correlation matrix")
st.caption("Shows how much each pair of assets moves together. Lower correlation = more diversification benefit.")

corr = returns.corr().round(2)
heat = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale="RdBu", zmid=0,
    text=corr.values, texttemplate="%{text}",
))
heat.update_layout(height=400, margin=dict(t=10, b=10))
st.plotly_chart(heat, use_container_width=True)
