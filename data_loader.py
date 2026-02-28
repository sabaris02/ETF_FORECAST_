"""
data_loader.py
--------------
Handles all data downloading and preprocessing from yfinance.
No forward-looking leakage. Adjusted close prices only.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ─── Ticker Definitions ───────────────────────────────────────────────────────
ETF_TICKERS = {
    "Gold ETF (GOLDBEES)": "GOLDBEES.NS",
    "Silver ETF (SILVERBEES)": "SILVERBEES.NS",
}

CONTEXT_TICKERS = {
    "NIFTY50": "^NSEI",
    "USDINR": "INR=X",
    "INDIAVIX": "^INDIAVIX",
}

LOOKBACK_YEARS = 5


@st.cache_data(ttl=3600, show_spinner=False)
def download_etf_data(ticker: str, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """
    Download adjusted close OHLCV data for the given ticker.
    Returns a clean DataFrame with no NaN rows in OHLCV columns.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365 + 30)  # extra buffer

    try:
        raw = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,   # gives adjusted OHLCV directly
            progress=False,
            threads=False,
        )
    except Exception as e:
        logger.error(f"Failed to download {ticker}: {e}")
        raise RuntimeError(f"Could not download data for {ticker}. Check connectivity.")

    if raw.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Drop rows where Close is NaN
    df.dropna(subset=["Close"], inplace=True)
    # Forward-fill remaining minor gaps (weekends already excluded by yfinance)
    df.ffill(inplace=True)

    logger.info(f"Downloaded {len(df)} rows for {ticker}")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def download_context_data(years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """
    Download market-context features: NIFTY50, USDINR, India VIX.
    Returns a DataFrame aligned to trading days with forward-fill.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365 + 60)

    frames = {}
    for name, ticker in CONTEXT_TICKERS.items():
        try:
            raw = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if not raw.empty:
                s = raw["Close"].copy()
                s.index = pd.to_datetime(s.index)
                frames[name] = s
        except Exception as e:
            logger.warning(f"Could not download context ticker {ticker}: {e}")

    if not frames:
        return pd.DataFrame()

    ctx = pd.DataFrame(frames)
    ctx.sort_index(inplace=True)
    ctx.ffill(inplace=True)
    ctx.dropna(how="all", inplace=True)
    return ctx


def merge_with_context(etf_df: pd.DataFrame, ctx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join ETF data with context data on the date index.
    Context columns are forward-filled to handle non-overlapping trading days.
    """
    if ctx_df.empty:
        return etf_df.copy()

    merged = etf_df.join(ctx_df, how="left")
    merged.ffill(inplace=True)
    # Drop rows where all context columns are still NaN (very beginning of series)
    ctx_cols = [c for c in ctx_df.columns if c in merged.columns]
    merged.dropna(subset=ctx_cols, inplace=True)
    return merged


def get_full_dataset(etf_label: str) -> pd.DataFrame:
    """
    Convenience wrapper: download ETF + context, merge, return clean DataFrame.
    """
    ticker = ETF_TICKERS[etf_label]
    etf_df = download_etf_data(ticker)
    ctx_df = download_context_data()
    full_df = merge_with_context(etf_df, ctx_df)
    return full_df
