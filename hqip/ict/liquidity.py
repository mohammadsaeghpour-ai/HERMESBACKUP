"""
Liquidity Analysis — ICT methodology.

Liquidity refers to pools of stop-loss orders that accumulate at obvious
technical levels (equal highs, equal lows, swing points). Institutions
engineer price moves to "sweep" this liquidity before reversing.

- Buy-Side Liquidity (BSL): Stops above equal highs — swept by a fake breakout up
- Sell-Side Liquidity (SSL): Stops below equal lows — swept by a fake breakdown down
- Liquidity Sweep: Price takes out the pool then reverses aggressively
- Stop Clusters: Zones where retail stops are likely concentrated
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


def find_bsl(df: pd.DataFrame, lookback: int = 20, tolerance_pct: float = 0.001) -> List[Tuple[float, int, int]]:
    """
    Find Buy-Side Liquidity (BSL) pools — clusters of equal highs.

    BSL forms when price creates multiple swing highs at similar levels,
    attracting stop-loss orders above those highs.

    Args:
        df: OHLCV DataFrame.
        lookback: Number of candles to scan for swing points.
        tolerance_pct: Max deviation between highs to consider them "equal"
                       as a percentage of price.

    Returns:
        List of (price_level, start_index, end_index) tuples.
        - price_level: The approximate level of the liquidity pool.
        - start_index: First candle index in the pool.
        - end_index: Last candle index in the pool.
    """
    _validate_ohlcv(df)
    from .structure import detect_swings

    swing_highs, _ = detect_swings(df, lookback=max(3, lookback // 4))
    if len(swing_highs) < 2:
        return []

    highs_arr = df["high"].values
    pools = []
    used = set()

    for i in range(len(swing_highs)):
        if i in used:
            continue
        level_i = swing_highs[i][0]
        idx_i = swing_highs[i][1]
        cluster = [i]
        for j in range(i + 1, len(swing_highs)):
            if j in used:
                continue
            level_j = swing_highs[j][0]
            if abs(level_i - level_j) / max(level_i, 1e-10) <= tolerance_pct:
                cluster.append(j)
                used.add(j)
        used.add(i)

        if len(cluster) >= 2:
            indices = [swing_highs[c][1] for c in cluster]
            avg_level = np.mean([swing_highs[c][0] for c in cluster])
            pools.append((float(avg_level), min(indices), max(indices)))

    return pools


def find_ssl(df: pd.DataFrame, lookback: int = 20, tolerance_pct: float = 0.001) -> List[Tuple[float, int, int]]:
    """
    Find Sell-Side Liquidity (SSL) pools — clusters of equal lows.

    SSL forms when price creates multiple swing lows at similar levels,
    attracting stop-loss orders below those lows.

    Args:
        df: OHLCV DataFrame.
        lookback: Number of candles to scan for swing points.
        tolerance_pct: Max deviation between lows to consider them "equal"
                       as a percentage of price.

    Returns:
        List of (price_level, start_index, end_index) tuples.
        - price_level: The approximate level of the liquidity pool.
        - start_index: First candle index in the pool.
        - end_index: Last candle index in the pool.
    """
    _validate_ohlcv(df)
    from .structure import detect_swings

    _, swing_lows = detect_swings(df, lookback=max(3, lookback // 4))
    if len(swing_lows) < 2:
        return []

    pools = []
    used = set()

    for i in range(len(swing_lows)):
        if i in used:
            continue
        level_i = swing_lows[i][0]
        cluster = [i]
        for j in range(i + 1, len(swing_lows)):
            if j in used:
                continue
            level_j = swing_lows[j][0]
            if abs(level_i - level_j) / max(level_i, 1e-10) <= tolerance_pct:
                cluster.append(j)
                used.add(j)
        used.add(i)

        if len(cluster) >= 2:
            indices = [swing_lows[c][1] for c in cluster]
            avg_level = np.mean([swing_lows[c][0] for c in cluster])
            pools.append((float(avg_level), min(indices), max(indices)))

    return pools


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect liquidity sweeps.

    A liquidity sweep occurs when price:
    1. Takes out an equal high/low level (pierces above/below the pool)
    2. Then reverses aggressively, closing back inside the range

    Args:
        df: OHLCV DataFrame.
        lookback: Scanning window for liquidity pools.

    Returns:
        Dict with keys:
        - swept: bool — whether a sweep is detected at the latest bar
        - direction: 'up' or 'down' — direction of the sweep wick
        - level: price level that was swept
        - confidence: 0.0–1.0 — strength of the sweep signal
    """
    _validate_ohlcv(df)

    bsl = find_bsl(df, lookback)
    ssl = find_ssl(df, lookback)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    n = len(df)

    result = {"swept": False, "direction": None, "level": None, "confidence": 0.0}

    # Check last candle for sweep of BSL
    last_h = highs[-1]
    last_l = lows[-1]
    last_c = closes[-1]
    last_o = opens[-1]

    for level, _, end_idx in bsl:
        if end_idx >= n - 1:
            continue
        # Sweep up: high pierces above pool, but close is below → bearish sweep
        if last_h > level and last_c < level:
            overshoot = (last_h - level) / max(level, 1e-10)
            confidence = min(overshoot / 0.005, 1.0)  # 0.5% overshoot = max confidence
            if confidence > result["confidence"]:
                result = {
                    "swept": True,
                    "direction": "up",
                    "level": float(level),
                    "confidence": confidence,
                }

    # Check last candle for sweep of SSL
    for level, _, end_idx in ssl:
        if end_idx >= n - 1:
            continue
        # Sweep down: low pierces below pool, but close is above → bullish sweep
        if last_l < level and last_c > level:
            overshoot = (level - last_l) / max(level, 1e-10)
            confidence = min(overshoot / 0.005, 1.0)
            if confidence > result["confidence"]:
                result = {
                    "swept": True,
                    "direction": "down",
                    "level": float(level),
                    "confidence": confidence,
                }

    return result


def estimate_stop_clusters(df: pd.DataFrame, lookback: int = 20) -> List[Tuple[float, float, str]]:
    """
    Estimate zones where stop-loss orders are likely clustered.

    Stops cluster above swing highs (buy stops) and below swing lows (sell stops),
    as well as around obvious round numbers and liquidity pools.

    Args:
        df: OHLCV DataFrame.
        lookback: Scanning window.

    Returns:
        List of (zone_center, zone_width, zone_type) tuples.
        - zone_center: Midpoint of the stop cluster zone.
        - zone_width: Width of the zone (± from center).
        - zone_type: 'buy_stop' or 'sell_stop'.
    """
    _validate_ohlcv(df)
    from .structure import detect_swings

    swing_highs, swing_lows = detect_swings(df, lookback=max(3, lookback // 4))
    closes = df["close"].values
    current_price = closes[-1]

    clusters = []

    # Buy stop clusters above swing highs
    for price, idx in swing_highs:
        # Stops are placed slightly above the high
        zone_center = price * 1.001  # 0.1% above
        zone_width = price * 0.002   # ±0.2% band
        clusters.append((zone_center, zone_width, "buy_stop"))

    # Sell stop clusters below swing lows
    for price, idx in swing_lows:
        zone_center = price * 0.999  # 0.1% below
        zone_width = price * 0.002
        clusters.append((zone_center, zone_width, "sell_stop"))

    # Round number clusters (psychological levels)
    if current_price > 0:
        magnitude = 10 ** int(np.log10(max(current_price, 1)))
        round_step = magnitude / 10  # nearest 10% of order magnitude
        if round_step > 0:
            nearest_round = round(current_price / round_step) * round_step
            for offset in [-round_step, 0, round_step]:
                level = nearest_round + offset
                if level > 0:
                    zone_width = round_step * 0.05
                    zone_type = "buy_stop" if level > current_price else "sell_stop"
                    clusters.append((level, zone_width, zone_type))

    # Sort by distance from current price
    clusters.sort(key=lambda x: abs(x[0] - current_price))
    return clusters
