"""
Stop-loss calculation module.

Provides ATR-based, swing-based, structure-based, trailing, and
dynamic stop-loss strategies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def atr_stop(
    entry: float,
    atr: float,
    direction: str,
    mult: float = 1.5,
) -> float:
    """
    ATR-based stop-loss.

    Parameters
    ----------
    entry : float
        Entry price.
    atr : float
        Current ATR value.
    direction : str
        "BUY" or "SELL".
    mult : float
        ATR multiplier (default 1.5).

    Returns
    -------
    float
        Stop-loss price.
    """
    d = direction.upper()
    if d == "BUY":
        return round(entry - (atr * mult), 6)
    elif d == "SELL":
        return round(entry + (atr * mult), 6)
    raise ValueError(f"direction must be 'BUY' or 'SELL', got '{direction}'")


def swing_stop(df: pd.DataFrame, direction: str) -> float:
    """
    Stop-loss placed at the most recent swing low (BUY) or swing high (SELL).

    Expects a DataFrame with columns ``high`` and ``low``.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.  Must contain 'high' and 'low' columns.
    direction : str
        "BUY" or "SELL".

    Returns
    -------
    float
        Swing-based stop price.
    """
    d = direction.upper()
    if d == "BUY":
        # Swing low = minimum of recent lows (last 20 bars)
        swing = float(df["low"].tail(20).min())
    elif d == "SELL":
        # Swing high = maximum of recent highs (last 20 bars)
        swing = float(df["high"].tail(20).max())
    else:
        raise ValueError(f"direction must be 'BUY' or 'SELL', got '{direction}'")
    return round(swing, 6)


def structure_stop(df: pd.DataFrame, direction: str) -> float:
    """
    Structure-based stop — uses the lowest low / highest high of the
    current structure (impulse wave).

    Identifies structure by looking for the most recent "structure
    break" point: the lowest low before the latest higher-high (BUY)
    or highest high before the latest lower-low (SELL).

    Expects columns: ``high``, ``low``.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'high' and 'low' columns.
    direction : str
        "BUY" or "SELL".

    Returns
    -------
    float
        Structure stop price.
    """
    d = direction.upper()
    if len(df) < 5:
        return swing_stop(df, direction)

    highs = df["high"].values
    lows = df["low"].values

    if d == "BUY":
        # Find last swing low before the most recent higher high
        recent_high = highs[-1]
        # Walk backwards to find the last higher high
        hh_idx = len(highs) - 1
        for i in range(len(highs) - 2, -1, -1):
            if highs[i] < recent_high:
                # This is a higher high; the structure low is between here and end
                break
            hh_idx = i
        # The structure stop is the lowest low in this swing
        structure_low = float(lows[hh_idx:].min())
        return round(structure_low, 6)
    elif d == "SELL":
        recent_low = lows[-1]
        ll_idx = len(lows) - 1
        for i in range(len(lows) - 2, -1, -1):
            if lows[i] > recent_low:
                break
            ll_idx = i
        structure_high = float(highs[ll_idx:].max())
        return round(structure_high, 6)
    raise ValueError(f"direction must be 'BUY' or 'SELL', got '{direction}'")


def trailing_stop(
    entry: float,
    current_price: float,
    atr: float,
    direction: str,
) -> float:
    """
    Trailing stop that follows price by an ATR multiple.

    For a BUY position the stop trails below the current price;
    for a SELL it trails above.

    Parameters
    ----------
    entry : float
        Original entry price (used as initial floor).
    current_price : float
        Latest market price.
    atr : float
        Current ATR value.
    direction : str
        "BUY" or "SELL".

    Returns
    -------
    float
        Trailing stop price.
    """
    d = direction.upper()
    trail_distance = atr * 1.5

    if d == "BUY":
        trailing = current_price - trail_distance
        # Never move stop below entry (or previous stop if you store state)
        return round(max(trailing, entry - trail_distance), 6)
    elif d == "SELL":
        trailing = current_price + trail_distance
        return round(min(trailing, entry + trail_distance), 6)
    raise ValueError(f"direction must be 'BUY' or 'SELL', got '{direction}'")


def dynamic_stop(
    entry: float,
    df: pd.DataFrame,
    direction: str,
    method: str = "auto",
) -> float:
    """
    Dynamic stop-loss that selects the best method automatically.

    Parameters
    ----------
    entry : float
        Entry price.
    df : pd.DataFrame
        OHLCV data with 'high', 'low', and optionally 'atr' columns.
    direction : str
        "BUY" or "SELL".
    method : str
        One of 'auto', 'atr', 'swing', 'structure'.

        - 'auto'     : tries structure first, falls back to ATR.
        - 'atr'      : ATR-based stop.
        - 'swing'    : Swing-based stop.
        - 'structure': Structure-based stop.

    Returns
    -------
    float
        Stop-loss price.
    """
    m = method.lower()

    if m == "atr" or (m == "auto" and "atr" in df.columns):
        atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns else _calc_atr(df)
        return atr_stop(entry, atr_val, direction, mult=1.5)

    if m == "swing":
        return swing_stop(df, direction)

    if m == "structure":
        return structure_stop(df, direction)

    # Auto: prefer structure, fall back to swing
    if m == "auto":
        try:
            return structure_stop(df, direction)
        except Exception:
            return swing_stop(df, direction)

    raise ValueError(f"Unknown method '{method}'")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR from OHLCV DataFrame when not pre-computed."""
    if len(df) < period + 1:
        return float(df["high"].max() - df["low"].min())

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values if "close" in df.columns else df["open"].values

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    atr = float(np.mean(tr[-period:]))
    return atr
