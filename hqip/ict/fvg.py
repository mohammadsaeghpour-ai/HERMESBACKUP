"""
Fair Value Gap (FVG) Detection — ICT methodology.

A Fair Value Gap is a three-candle pattern where a price gap exists between
the first and third candle. These gaps represent price inefficiencies that
markets tend to revisit (fill).

- Bull FVG: gap between candle[i-2].high and candle[i].low (gap UP)
- Bear FVG: gap between candle[i].low and candle[i-2].high (gap DOWN)

FVGs are significant when the gap exceeds 0.05% of the current price.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has required OHLCV columns."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")
    if len(df) < 3:
        raise ValueError("DataFrame must have at least 3 rows")


def detect_bull_fvg(df: pd.DataFrame, min_gap_pct: float = 0.0005) -> List[Tuple[float, float, int, bool]]:
    """
    Detect bullish Fair Value Gaps.

    A bull FVG exists when candle[i].low > candle[i-2].high, meaning there is
    an unmitigated gap between the first and third candle in the sequence.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
        min_gap_pct: Minimum gap size as percentage of price (default 0.05%).

    Returns:
        List of (gap_low, gap_high, index, filled) tuples.
        - gap_low: Bottom of the FVG (candle[i-2].high).
        - gap_high: Top of the FVG (candle[i].low).
        - index: Position of the middle candle (i-1) that created the gap.
        - filled: Whether price has returned to fill the gap.
    """
    _validate_ohlcv(df)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    results = []

    for i in range(2, n):
        gap_low = highs[i - 2]   # top of first candle
        gap_high = lows[i]        # bottom of third candle

        if gap_high <= gap_low:
            continue  # No gap — prices overlap

        gap_size = gap_high - gap_low
        price_level = (gap_high + gap_low) / 2.0
        if price_level <= 0:
            continue

        gap_pct = gap_size / price_level
        if gap_pct < min_gap_pct:
            continue  # Gap too small to be significant

        # Check if the gap has been filled by subsequent price action
        filled = False
        for j in range(i + 1, n):
            if lows[j] <= gap_low:
                filled = True
                break

        results.append((gap_low, gap_high, i - 1, filled))

    return results


def detect_bear_fvg(df: pd.DataFrame, min_gap_pct: float = 0.0005) -> List[Tuple[float, float, int, bool]]:
    """
    Detect bearish Fair Value Gaps.

    A bear FVG exists when candle[i].high < candle[i-2].low, meaning there is
    an unmitigated gap between the first and third candle in the sequence.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
        min_gap_pct: Minimum gap size as percentage of price (default 0.05%).

    Returns:
        List of (gap_low, gap_high, index, filled) tuples.
        - gap_low: Bottom of the FVG (candle[i].high).
        - gap_high: Top of the FVG (candle[i-2].low).
        - index: Position of the middle candle (i-1) that created the gap.
        - filled: Whether price has returned to fill the gap.
    """
    _validate_ohlcv(df)

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    results = []

    for i in range(2, n):
        gap_low = highs[i]        # top of third candle
        gap_high = lows[i - 2]    # bottom of first candle

        if gap_high <= gap_low:
            continue  # No gap — prices overlap

        gap_size = gap_high - gap_low
        price_level = (gap_high + gap_low) / 2.0
        if price_level <= 0:
            continue

        gap_pct = gap_size / price_level
        if gap_pct < min_gap_pct:
            continue  # Gap too small to be significant

        # Check if the gap has been filled by subsequent price action
        filled = False
        for j in range(i + 1, n):
            if highs[j] >= gap_high:
                filled = True
                break

        results.append((gap_low, gap_high, i - 1, filled))

    return results


def get_fvg_fill_status(
    df: pd.DataFrame, gap_low: float, gap_high: float, start_index: int
) -> Tuple[bool, float]:
    """
    Check if an FVG has been filled and how much.

    Args:
        df: OHLCV DataFrame.
        gap_low: Lower boundary of the FVG.
        gap_high: Upper boundary of the FVG.
        start_index: Index to start checking from.

    Returns:
        Tuple of (is_filled, fill_pct).
        fill_pct is 0.0 (no fill) to 1.0 (fully filled).
    """
    _validate_ohlcv(df)
    highs = df["high"].values
    lows = df["low"].values
    gap_mid = (gap_high + gap_low) / 2.0
    gap_range = gap_high - gap_low

    if gap_range <= 0:
        return True, 1.0

    max_penetration = 0.0
    n = len(df)

    for j in range(max(0, start_index + 1), n):
        if lows[j] <= gap_mid:
            # Price penetrated into the FVG
            penetration = min((gap_mid - lows[j]) / gap_range, 1.0)
            max_penetration = max(max_penetration, penetration)
        if highs[j] >= gap_high and lows[j] <= gap_low:
            return True, 1.0

    return max_penetration >= 1.0, max_penetration
