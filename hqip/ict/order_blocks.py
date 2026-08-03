"""
Order Block Detection — ICT methodology.

An Order Block (OB) is the last opposing candle before a significant price move.
It represents a zone where institutional orders were likely placed.

- Bullish OB: Last bearish (close < open) candle before a strong bullish move.
- Bearish OB: Last bullish (close > open) candle before a strong bearish move.

Strength is scored based on move magnitude, volume, and subsequent candle count.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has required OHLCV columns."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")
    if len(df) < 3:
        raise ValueError("DataFrame must have at least 3 rows")


def _strength_score(
    move_magnitude: float,
    move_pct: float,
    volume_ratio: float,
    candle_count: int,
) -> float:
    """
    Compute OB strength from 0.0 to 1.0 based on multiple factors.

    Args:
        move_magnitude: Absolute price move size.
        move_pct: Percentage move relative to price.
        volume_ratio: Volume of move relative to average volume.
        candle_count: Number of candles in the move.
    Returns:
        Strength score in [0.0, 1.0].
    """
    # Weighted combination of factors (each normalised roughly 0-1)
    magnitude_score = np.clip(move_pct * 100, 0, 1)  # 1% move → 1.0
    volume_score = np.clip(volume_ratio / 3.0, 0, 1)  # 3x avg vol → 1.0
    candle_score = np.clip(candle_count / 5.0, 0, 1)  # 5+ candles → 1.0
    raw = 0.5 * magnitude_score + 0.3 * volume_score + 0.2 * candle_score
    return float(np.clip(raw, 0.0, 1.0))


def _detect_opposing_candle(
    df: pd.DataFrame,
    lookback: int,
    is_bullish: bool,
) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Detect opposing candles that precede strong moves.

    Args:
        df: OHLCV DataFrame.
        lookback: Number of candles to check for the subsequent move.
        is_bullish: True for bullish OB (bearish candle before up move).

    Returns:
        List of (ob_price, ob_high, ob_low, strength, move_magnitude, index).
    """
    _validate_ohlcv(df)
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    vols = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    mid = (highs + lows) / 2.0

    avg_vol = pd.Series(vols, index=df.index).rolling(20, min_periods=1).mean().values

    results = []
    n = len(df)

    # Need at least 1 candle + lookback candles ahead
    for i in range(1, n - lookback):
        # Check if candle i is the opposing candle
        candle_bullish = closes[i] > opens[i]
        is_opposing = (not candle_bullish) if is_bullish else candle_bullish
        if not is_opposing:
            continue

        # Check for a strong move in the `lookback` candles after i
        future_slice = slice(i + 1, min(i + 1 + lookback, n))
        if future_slice.start >= n:
            continue

        future_highs = highs[future_slice]
        future_lows = lows[future_slice]
        future_closes = closes[future_slice]

        if is_bullish:
            # Strong bullish move: future high should significantly exceed OB high
            move_high = np.max(future_highs)
            move_magnitude = move_high - highs[i]
        else:
            # Strong bearish move: future low should significantly drop below OB low
            move_low = np.min(future_lows)
            move_magnitude = lows[i] - move_low

        # Require minimum move of 0.3% of price
        price_level = mid[i]
        if price_level <= 0:
            continue
        move_pct = move_magnitude / price_level
        if move_pct < 0.003:
            continue

        # Volume confirmation: move volume should be above average
        move_vols = vols[future_slice]
        vol_ratio = np.mean(move_vols) / max(np.mean(avg_vol[future_slice]), 1e-10)

        # Count how many candles continue the move direction
        if is_bullish:
            continue_count = int(np.sum(future_closes > closes[i]))
        else:
            continue_count = int(np.sum(future_closes < closes[i]))

        strength = _strength_score(move_magnitude, move_pct, vol_ratio, continue_count)
        ob_price = closes[i]

        results.append((ob_price, highs[i], lows[i], strength, move_magnitude, i))

    return results


def detect_bull_ob(df: pd.DataFrame, lookback: int = 5) -> List[Tuple[float, float, int]]:
    """
    Detect bullish Order Blocks.

    A bullish OB is the last bearish candle before a strong bullish move.
    The OB zone is defined by the high and low of that candle.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
        lookback: Number of candles to evaluate for the subsequent move.

    Returns:
        List of (ob_price, strength, index) tuples.
        ob_price is the midpoint of the OB zone (high+low)/2.
        strength is 0.0–1.0 (higher = stronger institutional footprint).
        index is the position in the original DataFrame.
    """
    _validate_ohlcv(df)
    raw = _detect_opposing_candle(df, lookback, is_bullish=True)
    return [(r[0], r[3], r[5]) for r in raw]


def detect_bear_ob(df: pd.DataFrame, lookback: int = 5) -> List[Tuple[float, float, int]]:
    """
    Detect bearish Order Blocks.

    A bearish OB is the last bullish candle before a strong bearish move.
    The OB zone is defined by the high and low of that candle.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
        lookback: Number of candles to evaluate for the subsequent move.

    Returns:
        List of (ob_price, strength, index) tuples.
        ob_price is the midpoint of the OB zone (high+low)/2.
        strength is 0.0–1.0 (higher = stronger institutional footprint).
        index is the position in the original DataFrame.
    """
    _validate_ohlcv(df)
    raw = _detect_opposing_candle(df, lookback, is_bullish=False)
    return [(r[0], r[3], r[5]) for r in raw]


def get_ob_zone(df: pd.DataFrame, index: int) -> Tuple[float, float]:
    """
    Get the OB zone (high, low) for a specific candle index.

    Args:
        df: OHLCV DataFrame.
        index: Candle index.

    Returns:
        Tuple of (zone_high, zone_low).
    """
    _validate_ohlcv(df)
    return float(df["high"].iloc[index]), float(df["low"].iloc[index])
