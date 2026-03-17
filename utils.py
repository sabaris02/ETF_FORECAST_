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
    Interactive price chart.
    - Without indicators : single candlestick pane.
    - With indicators    : 3 panes — Price / RSI / MACD — with colour zones
                           and annotations so even a newcomer can read them.
    """
    is_gold = "Gold" in etf_label
    accent  = COLORS["gold"] if is_gold else COLORS["silver"]

    if show_indicators:
        # ── 3-pane layout: Price (55%) | RSI (22%) | MACD (23%) ───────────
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.55, 0.22, 0.23],
            subplot_titles=[
                f"{etf_label} — Price with EMA & Bollinger Bands",
                "RSI (14)  ·  Momentum Oscillator",
                "MACD  ·  Trend Momentum",
            ],
        )
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{etf_label} Price"])

    # ── Pane 1: Candlestick ───────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            name="Price",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
            increasing_fillcolor=COLORS["success"],
            decreasing_fillcolor=COLORS["danger"],
        ),
        row=1, col=1,
    )

    if show_indicators:
        # ── EMA overlays ──────────────────────────────────────────────────
        if "ema_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["ema_20"],
                    name="EMA 20",
                    line=dict(color="#FF8C42", width=1.6),
                    hovertemplate="EMA 20: %{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
        if "ema_50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["ema_50"],
                    name="EMA 50",
                    line=dict(color="#7B68EE", width=1.6),
                    hovertemplate="EMA 50: %{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

        # ── Bollinger Band shaded region ──────────────────────────────────
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["bb_upper"],
                    name="BB Upper",
                    line=dict(color="rgba(79,195,247,0.5)", width=1, dash="dot"),
                    hovertemplate="BB Upper: %{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["bb_lower"],
                    name="BB Lower",
                    line=dict(color="rgba(79,195,247,0.5)", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(79,195,247,0.07)",
                    hovertemplate="BB Lower: %{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

        # ── Pane 2: RSI with colour zones ─────────────────────────────────
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"]

            # Red overbought zone (70–100)
            fig.add_hrect(y0=70, y1=100, row=2, col=1,
                          fillcolor="rgba(255,76,97,0.12)", line_width=0)
            # Green oversold zone (0–30)
            fig.add_hrect(y0=0, y1=30, row=2, col=1,
                          fillcolor="rgba(0,196,140,0.12)", line_width=0)

            fig.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"],
                          row=2, col=1, opacity=0.7,
                          annotation_text="Overbought (70)",
                          annotation_position="top left",
                          annotation_font_color=COLORS["danger"],
                          annotation_font_size=10)
            fig.add_hline(y=30, line_dash="dash", line_color=COLORS["success"],
                          row=2, col=1, opacity=0.7,
                          annotation_text="Oversold (30)",
                          annotation_position="bottom left",
                          annotation_font_color=COLORS["success"],
                          annotation_font_size=10)
            fig.add_hline(y=50, line_dash="dot", line_color="#8892A4",
                          row=2, col=1, opacity=0.4)

            # Colour the RSI line based on zone
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=rsi,
                    name="RSI (14)",
                    line=dict(color=accent, width=1.8),
                    hovertemplate="RSI: %{y:.1f}<extra></extra>",
                ),
                row=2, col=1,
            )

        # ── Pane 3: MACD ──────────────────────────────────────────────────
        if all(c in df.columns for c in ["macd", "macd_signal", "macd_hist"]):
            hist = df["macd_hist"]
            bar_colors = [
                COLORS["success"] if v >= 0 else COLORS["danger"]
                for v in hist
            ]
            fig.add_trace(
                go.Bar(
                    x=df.index, y=hist,
                    name="MACD Histogram",
                    marker_color=bar_colors,
                    opacity=0.65,
                    hovertemplate="Histogram: %{y:.5f}<extra></extra>",
                ),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["macd"],
                    name="MACD Line",
                    line=dict(color="#1E90FF", width=1.6),
                    hovertemplate="MACD: %{y:.5f}<extra></extra>",
                ),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["macd_signal"],
                    name="Signal Line",
                    line=dict(color="#FF8C42", width=1.6, dash="dash"),
                    hovertemplate="Signal: %{y:.5f}<extra></extra>",
                ),
                row=3, col=1,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#8892A4",
                          row=3, col=1, opacity=0.5)

    # ── Common layout ─────────────────────────────────────────────────────
    chart_height = 760 if show_indicators else 420
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        xaxis_rangeslider_visible=False,
        height=chart_height,
        margin=dict(l=10, r=150, t=50, b=10),
        legend=dict(
            orientation="v",
            x=1.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=10),
        ),
        font=dict(family="Courier New, monospace", size=11),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

    # RSI y-axis fixed range
    if show_indicators:
        fig.update_yaxes(range=[0, 100], row=2, col=1, fixedrange=True)

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
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
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
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
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
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
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
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
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
