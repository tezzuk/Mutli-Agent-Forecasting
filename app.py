"""
Streamlit dashboard — Multi-Agent Financial Forecasting
3 tabs: Forecast vs Actual | Agent Weights | Performance Breakdown
"""
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Forecasting",
    page_icon="📈",
    layout="wide",
)

RESULTS_PATH = pathlib.Path(__file__).parent / "results" / "backtest_results.csv"
METRICS_PATH = pathlib.Path(__file__).parent / "results" / "metrics_report.csv"

AGENT_NAMES  = ["TrendAgent", "MomentumAgent", "VolatilityAgent", "SequenceAgent"]
AGENT_COLORS = {"TrendAgent": "#4361ee", "MomentumAgent": "#f77f00",
                "VolatilityAgent": "#06d6a0", "SequenceAgent": "#7209b7",
                "Hedge Ensemble": "#ef233c", "Equal Weight": "#8b8fa8",
                "Buy & Hold": "#adb5bd"}

REGIME_BANDS = [
    {"x0": "2020-02-20", "x1": "2020-04-07",
     "label": "COVID Crash", "color": "rgba(239,35,60,0.12)"},
    {"x0": "2022-01-03", "x1": "2022-12-30",
     "label": "Rate Hike Cycle", "color": "rgba(114,9,183,0.10)"},
]


# ── Data loading (cached) ─────────────────────────────────────────────────────
def _mtime(path: pathlib.Path) -> float:
    """File modification time — used as a cache key so regenerated CSVs auto-reload."""
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def load_results(_mtime_key: float) -> pd.DataFrame | None:
    if not RESULTS_PATH.exists():
        return None
    df = pd.read_csv(RESULTS_PATH, index_col=0, parse_dates=True)
    return df


@st.cache_data
def load_metrics(_mtime_key: float) -> pd.DataFrame | None:
    if not METRICS_PATH.exists():
        return None
    return pd.read_csv(METRICS_PATH)


def add_regime_bands(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    for r in REGIME_BANDS:
        if r["x0"] >= str(df.index[0]) and r["x1"] <= str(df.index[-1]):
            fig.add_vrect(x0=r["x0"], x1=r["x1"],
                          fillcolor=r["color"], line_width=0,
                          annotation_text=r["label"],
                          annotation_position="top left")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

results = load_results(_mtime(RESULTS_PATH))
metrics = load_metrics(_mtime(METRICS_PATH))

if results is None:
    st.error(
        "**No backtest results found.**\n\n"
        "Run the backtest first:\n```\npython -m pipeline.backtest\n```"
    )
    st.stop()

# Date range
min_date = results.index.min().date()
max_date = results.index.max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if len(date_range) == 2:
    start_date, end_date = date_range
    results = results.loc[str(start_date): str(end_date)]

# Agent toggles
st.sidebar.markdown("**Agents to display**")
active_agents = [a for a in AGENT_NAMES
                 if st.sidebar.checkbox(a, value=True, key=f"chk_{a}")]

# Eta info
st.sidebar.markdown("---")
st.sidebar.markdown("**Hedge config (info)**")
st.sidebar.info("Hedge weights agents by predictive accuracy (mean-normalized MSE), "
                "η=0.2. Fixed-Share α=0.05 mixes a uniform component back each step so "
                "no agent dies and the ensemble stays regime-adaptive. (A P&L-aligned "
                "directional loss was tested but chased daily noise — see CORRECTIONS.md.)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Multi-Agent Financial Forecasting")
st.caption("Hedge algorithm · Walk-forward backtest · SPY 2010–2024 · Stamatics IIT Kanpur")

# KPI row
col1, col2, col3, col4 = st.columns(4)
n_days  = len(results)
n_folds = int(results["fold"].max()) + 1 if "fold" in results.columns else "—"

strategy_ret = np.sign(results["ensemble_pred"].values) * results["actual"].values
bh_ret       = results["actual"].values

with col1:
    st.metric("Trading Days", f"{n_days:,}")
with col2:
    st.metric("Backtest Folds", n_folds)
with col3:
    sr = float(np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252)) if np.std(strategy_ret) else 0
    st.metric("Ensemble Sharpe", f"{sr:.2f}")
with col4:
    da = float(np.mean(np.sign(results["actual"].values) == np.sign(results["ensemble_pred"].values)))
    st.metric("Ensemble Dir. Acc.", f"{da:.1%}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📊 Forecast vs Actual", "⚖️ Agent Weights Over Time", "🏆 Performance Breakdown"]
)


# ── Tab 1: Forecast vs Actual ─────────────────────────────────────────────────
with tab1:
    st.subheader("Predicted vs Actual Log Returns")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results["actual"],
        name="Actual", line=dict(color="#1a1a2e", width=1), opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results["ensemble_pred"],
        name="Hedge Ensemble", line=dict(color=AGENT_COLORS["Hedge Ensemble"], width=1.5)
    ))
    for agent in active_agents:
        col = f"{agent}_pred"
        if col in results.columns:
            fig.add_trace(go.Scatter(
                x=results.index, y=results[col],
                name=agent, line=dict(color=AGENT_COLORS[agent], width=1),
                opacity=0.5, visible="legendonly"
            ))

    fig = add_regime_bands(fig, results)
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.15),
                      xaxis_title="Date", yaxis_title="Log Return",
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    if metrics is not None:
        st.subheader("Metrics Table")
        st.dataframe(
            metrics.style.background_gradient(subset=["Sharpe Ratio"], cmap="Blues")
                         .format({"Sharpe Ratio": "{:.3f}", "Max Drawdown": "{:.3f}",
                                  "Directional Accuracy": "{:.3f}", "MAE": "{:.6f}"}),
            use_container_width=True,
        )


# ── Tab 2: Agent Weights Over Time ────────────────────────────────────────────
with tab2:
    st.subheader("Hedge Weight Evolution")
    st.caption("Weights sum to 1 at every step. Heavier agents predicted better recently.")

    weight_cols = [f"{a}_weight" for a in active_agents if f"{a}_weight" in results.columns]
    wdf = results[weight_cols].copy()
    wdf.columns = [c.replace("_weight", "") for c in wdf.columns]

    # Stacked area chart
    fig2 = go.Figure()
    for agent in wdf.columns:
        fig2.add_trace(go.Scatter(
            x=wdf.index, y=wdf[agent],
            name=agent,
            fill="tonexty" if agent != wdf.columns[0] else "tozeroy",
            line=dict(color=AGENT_COLORS.get(agent, "#999"), width=0.5),
            stackgroup="one",
        ))

    fig2 = add_regime_bands(fig2, results)
    fig2.update_layout(height=400, yaxis_title="Weight", xaxis_title="Date",
                       yaxis=dict(tickformat=".0%", range=[0, 1]),
                       legend=dict(orientation="h", y=-0.15),
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

    # Final weights bar
    if len(weight_cols) > 0:
        final_weights = {a: float(results[f"{a}_weight"].iloc[-1])
                         for a in active_agents if f"{a}_weight" in results.columns}
        st.subheader("Current (Final) Agent Weights")
        fw_df = pd.DataFrame(list(final_weights.items()), columns=["Agent", "Weight"])
        fig_fw = px.bar(fw_df, x="Agent", y="Weight",
                        color="Agent",
                        color_discrete_map=AGENT_COLORS,
                        text_auto=".1%")
        fig_fw.update_layout(height=280, showlegend=False,
                              plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_fw, use_container_width=True)


# ── Tab 3: Performance Breakdown ──────────────────────────────────────────────
with tab3:
    if metrics is not None:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Sharpe Ratio by Model")
            fig3 = px.bar(
                metrics.sort_values("Sharpe Ratio", ascending=True),
                x="Sharpe Ratio", y="Label", orientation="h",
                color="Label", color_discrete_map=AGENT_COLORS,
                text_auto=".3f",
            )
            fig3.update_layout(height=320, showlegend=False,
                                plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.subheader("Directional Accuracy by Model")
            fig4 = px.bar(
                metrics.sort_values("Directional Accuracy", ascending=True),
                x="Directional Accuracy", y="Label", orientation="h",
                color="Label", color_discrete_map=AGENT_COLORS,
                text_auto=".1%",
            )
            fig4.add_vline(x=0.5, line_dash="dash", line_color="red",
                           annotation_text="Random (50%)")
            fig4.update_layout(height=320, showlegend=False,
                                plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig4, use_container_width=True)

        # Equity curves
        st.subheader("$10,000 Equity Curves")
        eq_fig = go.Figure()
        bh_ret_arr = results["actual"].values

        def equity(preds, actuals):
            return 10_000 * np.exp(np.cumsum(np.sign(preds) * actuals))

        eq_fig.add_trace(go.Scatter(
            x=results.index, y=equity(results["ensemble_pred"].values, bh_ret_arr),
            name="Hedge Ensemble", line=dict(color=AGENT_COLORS["Hedge Ensemble"], width=2)
        ))
        eq_fig.add_trace(go.Scatter(
            x=results.index, y=10_000 * np.exp(np.cumsum(bh_ret_arr)),
            name="Buy & Hold", line=dict(color=AGENT_COLORS["Buy & Hold"], width=1.5, dash="dash")
        ))
        for agent in active_agents:
            col = f"{agent}_pred"
            if col in results.columns:
                eq_fig.add_trace(go.Scatter(
                    x=results.index, y=equity(results[col].values, bh_ret_arr),
                    name=agent, line=dict(color=AGENT_COLORS[agent], width=1),
                    visible="legendonly",
                ))

        eq_fig = add_regime_bands(eq_fig, results)
        eq_fig.update_layout(height=400, yaxis_title="Portfolio Value ($)",
                              xaxis_title="Date",
                              legend=dict(orientation="h", y=-0.15),
                              plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(eq_fig, use_container_width=True)

    else:
        st.info("Run the backtest and `full_metrics_report` to populate this tab.")

st.markdown("---")
st.caption("Stamatics IIT Kanpur · Mentor: Aayushman Tripathi · Hedge algorithm: Freund & Schapire (1997)")
