"""
Premium/Discount Arrays — ICT methodology.

ICT uses the range between the highest high and lowest low over a lookback
period to define three zones:

- Premium zone: Upper half (above equilibrium) — sellers dominate, good for shorts
- Discount zone: Lower half (below equilibrium) — buyers dominate, good for longs
- Equilibrium: The midpoint — fair value

The optimal entry for longs is in the discount zone, and for shorts in the premium zone.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has required OHLCV columns."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")


def equilibrium_price(df: pd.DataFrame, lookback: int = 50) -> float:
    """
    Calculate the equilibrium (midpoint) price over the lookback period.

    Equilibrium = (highest_high + lowest_low) / 2

    Args:
        df: OHLCV DataFrame.
        lookback: Number of candles to look back.

    Returns:
        The equilibrium price as a float.
    """
    _validate_ohlcv(df)

    window = df.iloc[-lookback:] if len(df) >= lookback else df
    high = window["high"].max()
    low = window["low"].min()

    return float((high + low) / 2.0)


def premium_discount(df: pd.DataFrame, lookback: int = 50) -> str:
    """
    Determine if current price is in premium, discount, or equilibrium zone.

    Uses the range of highs/lows over the lookback period:
    - Premium: current price > equilibrium
    - Discount: current price < equilibrium
    - Equilibrium: price within ±0.1% of the midpoint

    Args:
        df: OHLCV DataFrame.
        lookback: Number of candles to look back.

    Returns:
        'premium', 'discount', or 'equilibrium'.
    """
    _validate_ohlcv(df)

    window = df.iloc[-lookback:] if len(df) >= lookback else df
    high = window["high"].max()
    low = window["low"].min()
    eq = (high + low) / 2.0
    current = df["close"].iloc[-1]

    if eq <= 0:
        return "equilibrium"

    deviation_pct = abs(current - eq) / eq
    if deviation_pct < 0.001:  # Within 0.1%
        return "equilibrium"
    elif current > eq:
        return "premium"
    else:
        return "discount"


def get_pd_levels(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Get complete Premium/Discount level information.

    Divides the lookback range into zones:
    - Premium zone: 50%–100% of the range (above equilibrium)
    - Discount zone: 0%–50% of the range (below equilibrium)
    - Sub-zones at 25%, 50%, 75% for precision

    Args:
        df: OHLCV DataFrame.
        lookback: Number of candles to look back.

    Returns:
        Dict with keys:
        - range_high: Highest high in the lookback period
        - range_low: Lowest low in the lookback period
        - equilibrium: Midpoint price
        - premium_zone: Dict with 'start' and 'end' (equilibrium to range_high)
        - discount_zone: Dict with 'start' and 'end' (range_low to equilibrium)
        - quarter_levels: List of [25%, 50%, 75%] of the range
        - current_zone: 'premium', 'discount', or 'equilibrium'
        - current_position: 0.0 (at range low) to 1.0 (at range high)
    """
    _validate_ohlcv(df)

    window = df.iloc[-lookback:] if len(df) >= lookback else df
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    eq = (range_high + range_low) / 2.0
    rng = range_high - range_low
    current = float(df["close"].iloc[-1])

    if rng <= 0:
        return {
            "range_high": range_high,
            "range_low": range_low,
            "equilibrium": eq,
            "premium_zone": {"start": eq, "end": range_high},
            "discount_zone": {"start": range_low, "end": eq},
            "quarter_levels": [eq, eq, eq],
            "current_zone": "equilibrium",
            "current_position": 0.5,
        }

    position = (current - range_low) / rng
    position = float(np.clip(position, 0.0, 1.0))

    q25 = range_low + rng * 0.25
    q50 = eq
    q75 = range_low + rng * 0.75

    return {
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": eq,
        "premium_zone": {"start": eq, "end": range_high},
        "discount_zone": {"start": range_low, "end": eq},
        "quarter_levels": [q25, q50, q75],
        "current_zone": premium_discount(df, lookback),
        "current_position": position,
    }


def is_in_premium(df: pd.DataFrame, lookback: int = 50, threshold: float = 0.5) -> bool:
    """
    Check if current price is above the equilibrium + threshold of range.

    Useful for filtering shorts (only short in premium zone).

    Args:
        df: OHLCV DataFrame.
        lookback: Lookback period.
        threshold: Minimum position in range (0.0–1.0) to consider premium.

    Returns:
        True if price is in the upper premium area.
    """
    levels = get_pd_levels(df, lookback)
    return levels["current_position"] >= threshold


def is_in_discount(df: pd.DataFrame, lookback: int = 50, threshold: float = 0.5) -> bool:
    """
    Check if current price is below the equilibrium - threshold of range.

    Useful for filtering longs (only long in discount zone).

    Args:
        df: OHLCV DataFrame.
        lookback: Lookback period.
        threshold: Maximum position in range (0.0–1.0) to consider discount.

    Returns:
        True if price is in the lower discount area.
    """
    levels = get_pd_levels(df, lookback)
    return levels["current_position"] <= (1.0 - threshold)
