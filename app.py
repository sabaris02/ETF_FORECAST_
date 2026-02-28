"""
app.py
------
Main Streamlit application entry point.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import time

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="India ETF ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ─────────────────────────────────────────────────────────────
from data_loader import get_full_dataset, ETF_TICKERS
from feature_engineering import build_features, get_X_y
from models import (
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    train_with_timeseries_split,
    aggregate_fold_metrics,
    get_feature_importance,
    save_model,
    walk_forward_train,
)
from backtest import run_backtest
from utils import (
    plot_price_chart,
    plot_equity_curves,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_feature_heatmap,
    predictions_to_csv,
    metric_card,
    format_pct,
    format_num,
    COLORS,
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    .main { background: #0A0C10; }
    .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0D1117 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    /* Section header */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7em;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #4A90D9;
        border-bottom: 1px solid rgba(74,144,217,0.3);
        padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }

    /* Status badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.72em;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .badge-gold { background: rgba(245,166,35,0.15); color: #F5A623; border: 1px solid rgba(245,166,35,0.3); }
    .badge-silver { background: rgba(168,169,173,0.15); color: #C0C0C0; border: 1px solid rgba(168,169,173,0.3); }
    .badge-reg { background: rgba(0,196,140,0.12); color: #00C48C; border: 1px solid rgba(0,196,140,0.25); }
    .badge-cls { background: rgba(30,144,255,0.12); color: #1E90FF; border: 1px solid rgba(30,144,255,0.25); }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Session State Init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "trained": False,
        "result": None,
        "backtest": None,
        "df": None,
        "feature_cols": None,
        "fi_df": None,
        "wf_preds": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:12px 0 18px">
            <div style="font-size:2em;">📈</div>
            <div style="font-size:1.1em;font-weight:700;color:#FFFFFF;letter-spacing:1px;">ETF ML DASHBOARD</div>
            <div style="font-size:0.68em;color:#4A90D9;letter-spacing:2px;margin-top:2px;">INDIA • GOLD • SILVER</div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.07);margin:0 0 16px">
        """,
        unsafe_allow_html=True,
    )

    # ── ETF Selection ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Asset</div>', unsafe_allow_html=True)
    etf_label = st.selectbox(
        "Select ETF",
        list(ETF_TICKERS.keys()),
        label_visibility="collapsed",
    )
    is_gold = "Gold" in etf_label
    badge_class = "badge-gold" if is_gold else "badge-silver"
    badge_text = "GOLD" if is_gold else "SILVER"
    ticker_str = ETF_TICKERS[etf_label]
    st.markdown(
        f'<span class="badge {badge_class}">{badge_text}</span> '
        f'<span style="color:#8892A4;font-size:0.78em;">{ticker_str}</span>',
        unsafe_allow_html=True,
    )

    # ── Task & Model ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
    task = st.radio(
        "Task", ["Regression", "Classification"], horizontal=True, label_visibility="collapsed"
    ).lower()

    model_choices = list(REGRESSION_MODELS.keys()) if task == "regression" else list(CLASSIFICATION_MODELS.keys())
    model_name = st.selectbox("Algorithm", model_choices, label_visibility="collapsed")

    n_splits = st.slider("CV Folds (TimeSeriesSplit)", 3, 10, 5)

    # ── Options ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Options</div>', unsafe_allow_html=True)
    show_indicators = st.toggle("Show Technical Indicators", value=True)
    run_walkforward = st.toggle("Walk-Forward Backtest", value=False)
    save_to_disk = st.toggle("Save Trained Model", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    train_btn = st.button("🚀  TRAIN MODEL", width='stretch', type="primary")


# ─── Header ───────────────────────────────────────────────────────────────────
col_title, col_badges = st.columns([3, 1])
with col_title:
    st.markdown(
        f"""
        <h1 style="margin:0;font-size:1.6em;font-weight:700;letter-spacing:-0.5px;">
            {etf_label}
        </h1>
        <div style="color:#8892A4;font-size:0.82em;margin-top:4px;">
            ML-powered return prediction · 5-year daily data · TimeSeriesSplit validation
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_badges:
    task_badge = "badge-reg" if task == "regression" else "badge-cls"
    task_label = "REGRESSION" if task == "regression" else "CLASSIFICATION"
    st.markdown(
        f"""
        <div style="text-align:right;padding-top:10px;">
            <span class="badge {task_badge}">{task_label}</span><br>
            <span style="color:#8892A4;font-size:0.72em;letter-spacing:1px;">{model_name.upper()}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_and_engineer(etf_label: str):
    raw_df = get_full_dataset(etf_label)
    df, feature_cols = build_features(raw_df)
    return df, feature_cols


try:
    with st.spinner("📡 Fetching market data…"):
        df, feature_cols = load_and_engineer(etf_label)
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

st.session_state.df = df
st.session_state.feature_cols = feature_cols


# ─── Section 1: Price Chart ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Market Overview</div>', unsafe_allow_html=True)

info_cols = st.columns(4)
latest_close = df["Close"].iloc[-1]
prev_close = df["Close"].iloc[-2]
daily_ret = (latest_close / prev_close - 1) * 100
ret_color = COLORS["success"] if daily_ret >= 0 else COLORS["danger"]

with info_cols[0]:
    metric_card("Latest Close", f"₹{latest_close:.2f}", f"{daily_ret:+.2f}%", color=ret_color)
with info_cols[1]:
    metric_card("52W High", f"₹{df['Close'].tail(252).max():.2f}")
with info_cols[2]:
    metric_card("52W Low", f"₹{df['Close'].tail(252).min():.2f}")
with info_cols[3]:
    metric_card("Data Points", f"{len(df):,}", f"{feature_cols.__len__()} features")

price_fig = plot_price_chart(df, etf_label, show_indicators=show_indicators)
st.plotly_chart(price_fig, width='stretch', config={"displayModeBar": False})


# ─── Training ─────────────────────────────────────────────────────────────────
if train_btn:
    st.session_state.trained = False
    progress_bar = st.progress(0, text="Preparing data…")

    try:
        X, y = get_X_y(df, feature_cols, task)
        progress_bar.progress(15, text="Running TimeSeriesSplit CV…")

        with st.spinner(f"Training {model_name} with {n_splits}-fold TimeSeriesSplit…"):
            result = train_with_timeseries_split(
                X, y, model_name, task, n_splits=n_splits
            )
        progress_bar.progress(60, text="Running backtest…")

        # Use OOF predictions for backtesting
        if run_walkforward:
            progress_bar.progress(65, text="Walk-forward validation…")
            wf_series = walk_forward_train(
                X, y, df.index, model_name, task
            )
            # Align wf predictions to df
            wf_preds = np.full(len(df), np.nan)
            wf_idx = df.index.get_indexer(wf_series.index, method="nearest")
            for i, pos in enumerate(wf_idx):
                if 0 <= pos < len(wf_preds):
                    wf_preds[pos] = wf_series.iloc[i]
            st.session_state.wf_preds = wf_preds
        else:
            st.session_state.wf_preds = None

        backtest_result = run_backtest(
            df,
            result["oof_preds"],
            task=task,
        )
        progress_bar.progress(90, text="Finalizing…")

        # Feature importances
        fi_df = get_feature_importance(result, feature_cols)

        if save_to_disk:
            saved_path = save_model(result, etf_label)
            st.toast(f"Model saved → {saved_path}", icon="💾")

        st.session_state.result = result
        st.session_state.backtest = backtest_result
        st.session_state.fi_df = fi_df
        st.session_state.trained = True

        progress_bar.progress(100, text="Done!")
        time.sleep(0.5)
        progress_bar.empty()
        st.success("✅ Model trained successfully!")

    except Exception as e:
        progress_bar.empty()
        st.error(f"Training failed: {e}")
        st.code(traceback.format_exc())


# ─── Section 2: Model Metrics ─────────────────────────────────────────────────
if st.session_state.trained:
    result = st.session_state.result
    agg_metrics = aggregate_fold_metrics(result["fold_metrics"])

    st.markdown('<div class="section-header">Model Performance — Cross-Validation</div>', unsafe_allow_html=True)
    st.caption(f"Metrics averaged over {n_splits} TimeSeriesSplit folds (mean ± std)")

    if task == "regression":
        cols = st.columns(4)
        metric_items = [
            ("RMSE", format_num(agg_metrics["RMSE"]["mean"], 6), f"±{agg_metrics['RMSE']['std']:.6f}"),
            ("MAE", format_num(agg_metrics["MAE"]["mean"], 6), f"±{agg_metrics['MAE']['std']:.6f}"),
            ("R²", format_num(agg_metrics["R²"]["mean"], 4), f"±{agg_metrics['R²']['std']:.4f}"),
            ("Dir. Accuracy", format_pct(agg_metrics["Directional Accuracy"]["mean"]),
             f"±{agg_metrics['Directional Accuracy']['std']*100:.2f}%"),
        ]
        for col, (label, val, delta) in zip(cols, metric_items):
            with col:
                metric_card(label, val, delta)

    else:
        # Classification metrics — row 1
        row1 = st.columns(5)
        cls_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        for col, metric_name in zip(row1, cls_metrics):
            with col:
                val = agg_metrics[metric_name]["mean"]
                std = agg_metrics[metric_name]["std"]
                metric_card(metric_name, format_pct(val), f"±{std*100:.2f}%")

        # Confusion matrix from last fold
        last_cm = result["fold_metrics"][-1].get("Confusion Matrix")
        if last_cm is not None:
            cm_col, _ = st.columns([1, 2])
            with cm_col:
                st.plotly_chart(
                    plot_confusion_matrix(last_cm),
                    width='stretch',
                    config={"displayModeBar": False},
                )


# ─── Section 3: Feature Importance ───────────────────────────────────────────
if st.session_state.trained and st.session_state.fi_df is not None:
    fi_df = st.session_state.fi_df
    st.markdown('<div class="section-header">Feature Analysis</div>', unsafe_allow_html=True)

    fi_tab1, fi_tab2 = st.tabs(["📊 Bar Chart", "🗺 Heatmap"])
    with fi_tab1:
        top_n = st.slider("Top N features", 10, min(40, len(fi_df)), 20, key="top_n_bar")
        st.plotly_chart(
            plot_feature_importance(fi_df, top_n=top_n),
            width='stretch',
            config={"displayModeBar": False},
        )
    with fi_tab2:
        top_n_heat = st.slider("Top N features", 10, min(40, len(fi_df)), 25, key="top_n_heat")
        st.plotly_chart(
            plot_feature_heatmap(fi_df, top_n=top_n_heat),
            width='stretch',
            config={"displayModeBar": False},
        )


# ─── Section 4: Backtest ──────────────────────────────────────────────────────
if st.session_state.trained and st.session_state.backtest:
    bt = st.session_state.backtest
    st.markdown('<div class="section-header">Backtest Performance</div>', unsafe_allow_html=True)

    # ── Equity curve ─────────────────────────────────────────────────────
    st.plotly_chart(
        plot_equity_curves(bt),
        width='stretch',
        config={"displayModeBar": False},
    )

    # ── Performance metrics table ─────────────────────────────────────────
    bt_cols = st.columns(2)
    with bt_cols[0]:
        st.markdown(
            '<div style="color:#00C48C;font-size:0.78em;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">ML Strategy</div>',
            unsafe_allow_html=True,
        )
        strat_m = bt["metrics"]["Strategy"]
        metric_card("Cumulative Return", format_pct(strat_m["Cumulative Return"]))
        metric_card("Annualised Return", format_pct(strat_m["Annualised Return"]))
        metric_card("Sharpe Ratio", format_num(strat_m["Sharpe Ratio"], 3))
        metric_card("Max Drawdown", format_pct(strat_m["Max Drawdown"]), color=COLORS["danger"])
        metric_card("Win Rate", format_pct(strat_m["Win Rate"]))

    with bt_cols[1]:
        st.markdown(
            '<div style="color:#FF8C42;font-size:0.78em;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Buy & Hold</div>',
            unsafe_allow_html=True,
        )
        bah_m = bt["metrics"]["Buy & Hold"]
        metric_card("Cumulative Return", format_pct(bah_m["Cumulative Return"]))
        metric_card("Annualised Return", format_pct(bah_m["Annualised Return"]))
        metric_card("Sharpe Ratio", format_num(bah_m["Sharpe Ratio"], 3))
        metric_card("Max Drawdown", format_pct(bah_m["Max Drawdown"]), color=COLORS["danger"])
        metric_card("Win Rate", format_pct(bah_m["Win Rate"]))

    sig_info = bt["metrics"]["Signal Info"]
    st.caption(
        f"📌 Long days: {sig_info['Total Trades (Long Days)']} | "
        f"% time in market: {sig_info['% Days in Market']:.1f}%"
    )

    # ── Walk-forward overlay ──────────────────────────────────────────────
    if st.session_state.wf_preds is not None:
        try:
            wf_bt = run_backtest(df, st.session_state.wf_preds, task=task)
            import plotly.graph_objects as go
            fig_wf = plot_equity_curves(bt)
            fig_wf.add_trace(go.Scatter(
                x=wf_bt["strategy_cum"].index,
                y=(wf_bt["strategy_cum"] - 1) * 100,
                name="Walk-Forward Strategy",
                line=dict(color="#7B68EE", width=2, dash="longdash"),
            ))
            fig_wf.update_layout(title="Equity Curves — Standard vs Walk-Forward")
            st.markdown('<div class="section-header">Walk-Forward vs Standard Backtest</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_wf, width='stretch', config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Walk-forward backtest could not be rendered: {e}")

    # ── Download predictions ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    csv_bytes = predictions_to_csv(df, result["oof_preds"], task)
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label="⬇️  Download Predictions CSV",
            data=csv_bytes,
            file_name=f"{etf_label.split()[0].lower()}_{task}_predictions.csv",
            mime="text/csv",
            width='stretch',
        )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#8892A4;font-size:0.7em;letter-spacing:1px;padding:8px 0;">
        INDIA ETF ML DASHBOARD · DATA: YAHOO FINANCE · MODELS: SCIKIT-LEARN ·
        FOR EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE
    </div>
    """,
    unsafe_allow_html=True,
)
