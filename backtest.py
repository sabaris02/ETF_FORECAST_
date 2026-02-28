"""
backtest.py
-----------
Simple trading strategy backtesting:
  - If prediction > 0 (or class == 1) → Buy (hold)
  - Otherwise → Cash (no position)
Computes cumulative returns, Sharpe ratio, max drawdown.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.065  # approx Indian risk-free (10Y G-Sec yield), annualised


def _validate_inputs(actual_returns: pd.Series, predictions: np.ndarray):
    if len(actual_returns) != len(predictions):
        raise ValueError(
            f"Length mismatch: actual_returns={len(actual_returns)}, "
            f"predictions={len(predictions)}"
        )


def build_strategy_returns(
    actual_returns: pd.Series,
    predictions: np.ndarray,
    task: str = "regression",
) -> pd.Series:
    """
    Generate daily strategy returns.

    For regression : signal = 1 if pred > 0 else 0
    For classification: signal = 1 if pred == 1 else 0
    """
    _validate_inputs(actual_returns, predictions)

    if task == "regression":
        signal = (predictions > 0).astype(float)
    else:
        signal = (predictions == 1).astype(float)

    # Strategy return = signal * actual_return (long-only, no leverage)
    strategy_ret = pd.Series(
        signal * actual_returns.values,
        index=actual_returns.index,
        name="strategy_return",
    )
    return strategy_ret


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert daily returns to cumulative wealth series starting at 1."""
    return (1 + returns).cumprod()


def sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - RISK_FREE_RATE / periods
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / returns.std())


def max_drawdown(cum_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a fraction)."""
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    return float(drawdown.min())


def annualised_return(returns: pd.Series, periods: int = 252) -> float:
    """Compound annualised return."""
    total = (1 + returns).prod()
    n_years = len(returns) / periods
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def run_backtest(
    df: pd.DataFrame,
    predictions: np.ndarray,
    task: str = "regression",
    oof_indices: np.ndarray | None = None,
) -> dict:
    """
    Full backtest pipeline.

    Parameters
    ----------
    df            : Full feature-engineered DataFrame (has actual returns)
    predictions   : OOF predictions array (aligned to df index or oof_indices)
    task          : 'regression' or 'classification'
    oof_indices   : If provided, subset df to these row positions

    Returns
    -------
    dict with equity curves, metrics, and signal series
    """
    actual_ret = df["return_1d"].copy()

    if oof_indices is not None:
        # Use only rows where we have OOF predictions
        valid_mask = ~np.isnan(predictions)
        valid_positions = np.where(valid_mask)[0]
        if len(valid_positions) == 0:
            raise ValueError("No valid OOF predictions found.")
        actual_ret = actual_ret.iloc[valid_positions]
        preds = predictions[valid_mask]
    else:
        preds = predictions
        # Drop NaN predictions
        valid_mask = ~np.isnan(preds)
        actual_ret = actual_ret[valid_mask]
        preds = preds[valid_mask]

    # But we use next-day actual return for evaluation
    # actual_ret here corresponds to the day AFTER the prediction date
    # Since targets are shifted, actual_ret already lines up correctly
    strategy_ret = build_strategy_returns(actual_ret, preds, task)
    bah_ret = actual_ret.rename("bah_return")

    strat_cum = cumulative_returns(strategy_ret).rename("Strategy")
    bah_cum = cumulative_returns(bah_ret).rename("Buy & Hold")

    # ── Signal statistics ─────────────────────────────────────────────────
    if task == "regression":
        signal = (preds > 0).astype(float)
    else:
        signal = (preds == 1).astype(float)

    n_trades = int(signal.sum())
    signal_pct = float(signal.mean() * 100)

    metrics = {
        "Strategy": {
            "Cumulative Return": float(strat_cum.iloc[-1] - 1),
            "Annualised Return": annualised_return(strategy_ret),
            "Sharpe Ratio": sharpe_ratio(strategy_ret),
            "Max Drawdown": max_drawdown(strat_cum),
            "Win Rate": float((strategy_ret[strategy_ret != 0] > 0).mean()),
        },
        "Buy & Hold": {
            "Cumulative Return": float(bah_cum.iloc[-1] - 1),
            "Annualised Return": annualised_return(bah_ret),
            "Sharpe Ratio": sharpe_ratio(bah_ret),
            "Max Drawdown": max_drawdown(bah_cum),
            "Win Rate": float((bah_ret > 0).mean()),
        },
        "Signal Info": {
            "Total Trades (Long Days)": n_trades,
            "% Days in Market": signal_pct,
        },
    }

    return {
        "strategy_returns": strategy_ret,
        "bah_returns": bah_ret,
        "strategy_cum": strat_cum,
        "bah_cum": bah_cum,
        "metrics": metrics,
        "signal": pd.Series(signal, index=actual_ret.index),
    }
