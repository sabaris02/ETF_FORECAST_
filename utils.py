"""
utils.py
--------
Shared utilities: plotting helpers, formatting, and download tools.
All charts are built with Plotly for interactivity.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "gold": "#F5A623",
    "silver": "#A8A9AD",
    "primary": "#1E90FF",
    "success": "#00C48C",
    "danger": "#FF4C61",
    "neutral": "#8892A4",
    "bg": "#0E1117",
    "card": "#1A1D23",
    "line_1": "#00C48C",
    "line_2": "#FF8C42",
}


def format_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def format_num(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


# ─── Price Chart ──────────────────────────────────────────────────────────────
def plot_price_chart(
    df: pd.DataFrame,
    etf_label: str,
    show_indicators: bool = False,
) -> go.Figure:
    """
    Interactive candlestick chart with optional EMA and Bollinger Band overlays.
    """
    is_gold = "Gold" in etf_label
    accent = COLORS["gold"] if is_gold else COLORS["silver"]

    rows = 2 if show_indicators else 1
    row_heights = [0.7, 0.3] if show_indicators else [1.0]
    specs = [[{"type": "xy"}]] * rows
    subplot_titles = [f"{etf_label} Price", "RSI (14)"] if show_indicators else [f"{etf_label} Price"]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ── Candlestick ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
        ),
        row=1, col=1,
    )

    # ── EMA overlays ─────────────────────────────────────────────────────
    if show_indicators and "ema_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["ema_20"],
                name="EMA 20", line=dict(color="#FF8C42", width=1.2),
            ),
            row=1, col=1,
        )
    if show_indicators and "ema_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["ema_50"],
                name="EMA 50", line=dict(color="#7B68EE", width=1.2),
            ),
            row=1, col=1,
        )

    # ── Bollinger Bands ───────────────────────────────────────────────────
    if show_indicators and "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["bb_upper"],
                name="BB Upper", line=dict(color="#4FC3F7", width=0.8, dash="dot"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["bb_lower"],
                name="BB Lower", line=dict(color="#4FC3F7", width=0.8, dash="dot"),
                fill="tonexty", fillcolor="rgba(79,195,247,0.05)",
            ),
            row=1, col=1,
        )

    # ── RSI ───────────────────────────────────────────────────────────────
    if show_indicators and "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["rsi_14"],
                name="RSI", line=dict(color=accent, width=1.5),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"], row=2, col=1, opacity=0.6)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["success"], row=2, col=1, opacity=0.6)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        height=500 if show_indicators else 420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        font=dict(family="Courier New, monospace", size=11),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def plot_equity_curves(backtest_result: dict) -> go.Figure:
    """Plot strategy vs Buy-and-Hold cumulative return curves."""
    strat = backtest_result["strategy_cum"]
    bah = backtest_result["bah_cum"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strat.index, y=(strat - 1) * 100,
        name="ML Strategy", line=dict(color=COLORS["success"], width=2),
        fill="tozeroy", fillcolor="rgba(0,196,140,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=bah.index, y=(bah - 1) * 100,
        name="Buy & Hold", line=dict(color=COLORS["line_2"], width=2, dash="dot"),
    ))
    fig.update_layout(
        title="Equity Curves — Cumulative Return (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Cumulative Return (%)",
        height=380,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Courier New, monospace"),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    top = fi_df.head(top_n).sort_values("Importance")
    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker=dict(
            color=top["Importance"],
            colorscale="Teal",
            showscale=False,
        ),
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(300, top_n * 20),
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(family="Courier New, monospace"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Annotated heatmap for confusion matrix."""
    labels = ["Down (0)", "Up (1)"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(family="Courier New, monospace"),
    )
    return fig


def plot_feature_heatmap(fi_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """Single-row heatmap for feature importances."""
    top = fi_df.head(top_n)
    fig = go.Figure(go.Heatmap(
        z=[top["Importance"].values],
        x=top["Feature"].values,
        colorscale="Plasma",
        showscale=True,
    ))
    fig.update_layout(
        title=f"Feature Importance Heatmap (Top {top_n})",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=10, r=10, t=45, b=80),
        font=dict(family="Courier New, monospace"),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
    )
    return fig


def predictions_to_csv(df: pd.DataFrame, oof_preds: np.ndarray, task: str) -> bytes:
    """Create a downloadable CSV of dates + predictions + actuals."""
    valid_mask = ~np.isnan(oof_preds)
    out = pd.DataFrame({
        "Date": df.index[valid_mask],
        "Actual": df[f"{task}_target"].values[valid_mask],
        "Predicted": oof_preds[valid_mask],
        "Close": df["Close"].values[valid_mask],
    })
    return out.to_csv(index=False).encode("utf-8")


def metric_card(label: str, value: str, delta: str | None = None, color: str = "#00C48C"):
    """Render a styled metric using Streamlit markdown."""
    delta_html = f'<span style="color:{color};font-size:0.8em;">{delta}</span>' if delta else ""
    st.markdown(
        f"""
        <div style="
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:10px;
            padding:14px 18px;
            margin-bottom:8px;
        ">
            <div style="color:#8892A4;font-size:0.78em;font-family:'Courier New',monospace;letter-spacing:1px;text-transform:uppercase">{label}</div>
            <div style="color:#FFFFFF;font-size:1.5em;font-weight:700;font-family:'Courier New',monospace;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
