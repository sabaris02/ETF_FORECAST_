"""
feature_engineering.py
-----------------------
All feature construction. Strictly no lookahead bias:
 - All indicators use only past/current data
 - Targets are created by shifting FORWARD (positive shift)
 - NaN rows dropped AFTER all features are computed
"""

import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)

# ─── Feature Column Registry ──────────────────────────────────────────────────
FEATURE_COLS: list[str] = []  # populated dynamically


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily simple return and log return."""
    df["return_1d"] = df["Close"].pct_change()
    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and volatility at multiple windows."""
    for w in [5, 10, 20, 50]:
        df[f"roll_mean_{w}"] = df["Close"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["Close"].rolling(w).std()
        df[f"roll_mean_ret_{w}"] = df["return_1d"].rolling(w).mean()
    return df


def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Price momentum: close - close[n days ago]."""
    for n in [5, 10, 20]:
        df[f"mom_{n}"] = df["Close"] - df["Close"].shift(n)
        df[f"mom_ret_{n}"] = df["return_1d"].rolling(n).sum()
    return df


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """RSI using ta library."""
    df["rsi_14"] = ta.momentum.RSIIndicator(
        close=df["Close"], window=window
    ).rsi()
    return df


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD line, signal, and histogram."""
    macd_obj = ta.trend.MACD(
        close=df["Close"], window_slow=26, window_fast=12, window_sign=9
    )
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()
    return df


def compute_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Bollinger Band width and %B position."""
    bb = ta.volatility.BollingerBands(
        close=df["Close"], window=window, window_dev=2
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["Close"]
    df["bb_pct"] = bb.bollinger_pband()
    return df


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Exponential Moving Averages (20 and 50) and price distance."""
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_20_dist"] = (df["Close"] - df["ema_20"]) / df["ema_20"]
    df["ema_50_dist"] = (df["Close"] - df["ema_50"]) / df["ema_50"]
    df["ema_cross"] = df["ema_20"] - df["ema_50"]
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range (normalised by Close)."""
    atr_obj = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=window
    )
    df["atr_14"] = atr_obj.average_true_range()
    df["atr_pct"] = df["atr_14"] / df["Close"]
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features."""
    df["vol_change"] = df["Volume"].pct_change()
    df["vol_roll_mean_10"] = df["Volume"].rolling(10).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_roll_mean_10"]
    return df


def compute_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive return-based features from context columns (NIFTY, USDINR, VIX)
    to avoid scale issues and improve model generalisation.
    """
    ctx_map = {
        "NIFTY50": "nifty_ret",
        "USDINR": "usdinr_ret",
        "INDIAVIX": "vix_change",
    }
    for col, feat_name in ctx_map.items():
        if col in df.columns:
            df[feat_name] = df[col].pct_change()
            df[f"{feat_name}_5d"] = df[col].pct_change(5)

    # Lag features to avoid lookahead
    if "INDIAVIX" in df.columns:
        df["vix_level"] = df["INDIAVIX"]
        df["vix_roll_mean_5"] = df["INDIAVIX"].rolling(5).mean()
    return df


def compute_lag_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Lag the daily return and log return for AR-style features."""
    for lag in range(1, lags + 1):
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)
        df[f"log_ret_lag_{lag}"] = df["log_return_1d"].shift(lag)
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create regression and classification targets.
    Uses FORWARD shift — no lookahead.

    regression_target  : next-day return
    classification_target: 1 if next-day return > 0, else 0
    """
    df["regression_target"] = df["return_1d"].shift(-1)
    df["classification_target"] = (df["regression_target"] > 0).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline: apply all feature engineering steps in correct order.
    Returns a clean DataFrame with feature columns and target columns.
    All rows with NaN in feature or target columns are dropped.
    """
    df = df.copy()

    df = compute_returns(df)
    df = compute_rolling_stats(df)
    df = compute_momentum(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_ema(df)
    df = compute_atr(df)
    df = compute_volume_features(df)
    df = compute_context_features(df)
    df = compute_lag_features(df)
    df = create_targets(df)

    # ── Identify all model input features ──────────────────────────────────
    exclude = {
        "Open", "High", "Low", "Close", "Volume",
        "NIFTY50", "USDINR", "INDIAVIX",
        "regression_target", "classification_target",
    }
    feature_cols = [c for c in df.columns if c not in exclude]

    # Drop rows with any NaN in features or targets
    required_cols = feature_cols + ["regression_target", "classification_target"]
    df.dropna(subset=required_cols, inplace=True)

    logger.info(
        f"Feature engineering complete: {len(df)} rows, {len(feature_cols)} features"
    )
    return df, feature_cols


def get_X_y(df: pd.DataFrame, feature_cols: list[str], task: str):
    """
    Split DataFrame into X (features) and y (target).

    task: 'regression' or 'classification'
    """
    target_col = f"{task}_target"
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y
