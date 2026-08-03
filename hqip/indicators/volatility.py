"""
HQIP v3 — Volatility Indicators
=================================
ATR, Bollinger Bands, squeeze detection, Keltner Channels, regime classification.

All functions accept a pandas DataFrame with at least ``close``
(and ``high``/``low`` where needed).  NaN-safe and vectorized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Raw True Range (not averaged)."""
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)


# ── public API ────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing via EMA).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    pd.Series
        ATR values.
    """
    tr = _true_range(df)
    return _ema(tr, period)


def bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std: float = 2.0,
) -> dict[str, pd.Series]:
    """Bollinger Bands and derived metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close``.
    period : int
        SMA look-back (default 20).
    std : float
        Standard-deviation multiplier (default 2.0).

    Returns
    -------
    dict
        ``upper``  — upper band
        ``middle`` — SMA basis
        ``lower``  — lower band
        ``width``  — (upper − lower) / middle  — normalised bandwidth
        ``pct_b``  — (close − lower) / (upper − lower)  — %B position
    """
    close = df["close"]
    middle = _sma(close, period)
    rolling_std = close.rolling(period, min_periods=1).std()

    upper = middle + std * rolling_std
    lower = middle - std * rolling_std

    mid_safe = middle.replace(0, 1e-10)
    band_range = (upper - lower).replace(0, 1e-10)

    width = (upper - lower) / mid_safe
    pct_b = (close - lower) / band_range

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "width": width,
        "pct_b": pct_b,
    }


def bollinger_squeeze(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> bool:
    """Detect a Bollinger-Keltner squeeze on the latest bar.

    A squeeze occurs when both Bollinger bands are **inside** the
    Keltner Channels, indicating low-volatility compression that
    often precedes an explosive move.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    bb_period, bb_std : int, float
        Bollinger parameters.
    kc_period : int
        Keltner EMA period.
    kc_mult : float
        Keltner ATR multiplier.

    Returns
    -------
    bool
        ``True`` if squeeze is active on the last bar.
    """
    bb = bollinger(df, bb_period, bb_std)
    kc = keltner(df, kc_period, kc_mult)

    bb_upper = bb["upper"].iloc[-1]
    bb_lower = bb["lower"].iloc[-1]
    kc_upper = kc["upper"].iloc[-1]
    kc_lower = kc["lower"].iloc[-1]

    if any(pd.isna(v) for v in [bb_upper, bb_lower, kc_upper, kc_lower]):
        return False

    return bool(bb_upper < kc_upper and bb_lower > kc_lower)


def keltner(
    df: pd.DataFrame,
    period: int = 20,
    mult: float = 1.5,
) -> dict[str, pd.Series]:
    """Keltner Channels — EMA-based with ATR width.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int
        EMA and ATR look-back (default 20).
    mult : float
        ATR multiplier for channel width (default 1.5).

    Returns
    -------
    dict
        ``upper``, ``middle``, ``lower`` — pd.Series.
    """
    close = df["close"]
    middle = _ema(close, period)
    atr_val = atr(df, period)

    upper = middle + mult * atr_val
    lower = middle - mult * atr_val

    return {"upper": upper, "middle": middle, "lower": lower}


def volatility_regime(df: pd.DataFrame, period: int = 100) -> str:
    """Classify the current volatility regime.

    Compares the most recent ATR% (ATR/close) against its own
    rolling distribution over ``period`` bars.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    period : int
        Distribution look-back (default 100).

    Returns
    -------
    str
        One of ``'low'``, ``'normal'``, ``'high'``, ``'extreme'``.
    """
    if df.empty or len(df) < 20:
        return "normal"

    atr_val = atr(df, 14)
    atr_pct = atr_val / df["close"].replace(0, 1e-10) * 100.0

    current = atr_pct.iloc[-1]
    if pd.isna(current):
        return "normal"

    recent = atr_pct.iloc[-period:]
    mean_val = recent.mean()
    std_val = recent.std()

    if pd.isna(std_val) or std_val == 0:
        return "normal"

    z = (current - mean_val) / std_val

    if z > 2.5:
        return "extreme"
    if z > 1.0:
        return "high"
    if z < -1.0:
        return "low"
    return "normal"
