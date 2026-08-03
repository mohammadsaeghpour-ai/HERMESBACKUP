"""
Market Structure Analysis — ICT methodology.

Market Structure tracks the sequence of swing highs and swing lows to determine:
- Trend direction (higher highs / higher lows = bullish, vice versa)
- Break of Structure (BOS) — continuation: price breaks the last swing in the
  trend direction
- Change of Character (CHoCH) — reversal: price breaks the last swing against
  the established trend

This is the foundational layer of ICT analysis.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has required OHLCV columns."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")
    if len(df) < 5:
        raise ValueError("DataFrame must have at least 5 rows")


def detect_swings(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
    """
    Detect swing highs and swing lows.

    A swing high is a candle whose high is the highest within `lookback`
    candles on each side. A swing low is analogous using the low.

    Args:
        df: OHLCV DataFrame.
        lookback: Number of candles on each side to confirm a swing point.

    Returns:
        Tuple of (swing_highs, swing_lows) where each is a list of
        (price, index) tuples sorted by index ascending.
    """
    _validate_ohlcv(df)

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(lookback, n - lookback):
        # Swing High: high[i] is max in the window
        window_highs = highs[i - lookback : i + lookback + 1]
        if highs[i] == np.max(window_highs) and not np.any(np.isnan(window_highs)):
            swing_highs.append((float(highs[i]), i))

        # Swing Low: low[i] is min in the window
        window_lows = lows[i - lookback : i + lookback + 1]
        if lows[i] == np.min(window_lows) and not np.any(np.isnan(window_lows)):
            swing_lows.append((float(lows[i]), i))

    return swing_highs, swing_lows


def detect_bos(df: pd.DataFrame, lookback: int = 5) -> str:
    """
    Detect Break of Structure (BOS) — trend continuation.

    Bullish BOS: Price breaks above the most recent swing high while
    in an uptrend (higher highs, higher lows).
    Bearish BOS: Price breaks below the most recent swing low while
    in a downtrend (lower highs, lower lows).

    Args:
        df: OHLCV DataFrame.
        lookback: Swing detection lookback.

    Returns:
        'bullish_bos', 'bearish_bos', or 'none'.
    """
    _validate_ohlcv(df)

    swing_highs, swing_lows = detect_swings(df, lookback)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "none"

    closes = df["close"].values
    n = len(df)
    last_close = closes[-1]

    # Determine current trend from last two swings
    # Bullish trend: last swing high > previous swing high AND last swing low > previous
    last_sh = swing_highs[-1]
    prev_sh = swing_highs[-2]
    last_sl = swing_lows[-1]
    prev_sl = swing_lows[-2]

    bullish_trend = last_sh[0] > prev_sh[0] and last_sl[0] > prev_sl[0]
    bearish_trend = last_sh[0] < prev_sh[0] and last_sl[0] < prev_sl[0]

    if bullish_trend and last_close > last_sh[0]:
        # Confirmed bullish BOS — price broke above the last swing high
        return "bullish_bos"
    elif bearish_trend and last_close < last_sl[0]:
        # Confirmed bearish BOS — price broke below the last swing low
        return "bearish_bos"

    return "none"


def detect_choch(df: pd.DataFrame, lookback: int = 5) -> str:
    """
    Detect Change of Character (CHoCH) — trend reversal.

    Bullish CHoCH: Price was making lower lows (downtrend) but then breaks
    above the most recent lower high.
    Bearish CHoCH: Price was making higher highs (uptrend) but then breaks
    below the most recent higher low.

    Args:
        df: OHLCV DataFrame.
        lookback: Swing detection lookback.

    Returns:
        'bullish_choch', 'bearish_choch', or 'none'.
    """
    _validate_ohlcv(df)

    swing_highs, swing_lows = detect_swings(df, lookback)
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "none"

    closes = df["close"].values
    last_close = closes[-1]

    # Check for bearish CHoCH: was uptrend, now breaks below last higher low
    # Uptrend = last two swing highs ascending, last two swing lows ascending
    sh1, sh2, sh3 = swing_highs[-3][0], swing_highs[-2][0], swing_highs[-1][0]
    sl1, sl2, sl3 = swing_lows[-3][0], swing_lows[-2][0], swing_lows[-1][0]

    was_uptrend = sh3 > sh2 and sl3 > sl2
    was_downtrend = sh3 < sh2 and sl3 < sl2

    if was_uptrend and last_close < sl3:
        return "bearish_choch"
    elif was_downtrend and last_close > sh3:
        return "bullish_choch"

    return "none"


def analyze_structure(df: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Comprehensive market structure analysis.

    Combines swing detection, BOS, and CHoCH to give a full picture of
    current market structure.

    Args:
        df: OHLCV DataFrame.
        lookback: Swing detection lookback.

    Returns:
        Dict with keys:
        - trend: 'bullish', 'bearish', or 'ranging'
        - bos: result from detect_bos()
        - choch: result from detect_choch()
        - swing_highs: list of (price, index)
        - swing_lows: list of (price, index)
        - last_high: most recent swing high price (or None)
        - last_low: most recent swing low price (or None)
        - structure: 'HH_HL', 'LH_LL', 'HH_LL', 'LH_HL', or 'unknown'
    """
    _validate_ohlcv(df)

    swing_highs, swing_lows = detect_swings(df, lookback)
    bos = detect_bos(df, lookback)
    choch = detect_choch(df, lookback)

    trend = "ranging"
    structure = "unknown"
    last_high = None
    last_low = None

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_sh = swing_highs[-1]
        prev_sh = swing_highs[-2]
        last_sl = swing_lows[-1]
        prev_sl = swing_lows[-2]
        last_high = last_sh[0]
        last_low = last_sl[0]

        higher_high = last_sh[0] > prev_sh[0]
        higher_low = last_sl[0] > prev_sl[0]
        lower_high = last_sh[0] < prev_sh[0]
        lower_low = last_sl[0] < prev_sl[0]

        if higher_high and higher_low:
            trend = "bullish"
            structure = "HH_HL"
        elif lower_high and lower_low:
            trend = "bearish"
            structure = "LH_LL"
        elif higher_high and lower_low:
            structure = "HH_LL"
            trend = "ranging"  # Expansion / volatile
        elif lower_high and higher_low:
            structure = "LH_HL"
            trend = "ranging"  # Consolidation / squeeze

    return {
        "trend": trend,
        "bos": bos,
        "choch": choch,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "last_high": last_high,
        "last_low": last_low,
        "structure": structure,
    }
