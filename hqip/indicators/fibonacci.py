"""
HQIP v3 — Fibonacci Indicators
================================
Retracement levels, extensions, multi-timeframe clusters, Optimal Trade Entry
(OTE) zones, and nearest-level lookup.

All functions use only pandas/numpy, are NaN-safe, and return plain
dicts / lists for easy consumption by other modules.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


# ── public API ────────────────────────────────────────────────────────

def fibonacci_levels(high: float, low: float) -> dict[float, float]:
    """Standard Fibonacci retracement levels.

    Parameters
    ----------
    high : float
        Swing high price.
    low : float
        Swing low price.

    Returns
    -------
    dict
        Keys are the Fibonacci ratios (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0).
        Values are the corresponding price levels (descending from ``high``).
    """
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    rng = high - low
    return {r: high - r * rng for r in ratios}


def fibonacci_extensions(
    high: float,
    low: float,
    retrace_low: float,
) -> dict[float, float]:
    """Fibonacci extension levels from a retracement.

    Parameters
    ----------
    high : float
        Original swing high.
    low : float
        Original swing low.
    retrace_low : float
        Lowest point of the retracement (where price turned back up).

    Returns
    -------
    dict
        Keys: 1.0, 1.272, 1.618, 2.0.
        Values: projected target prices.
    """
    imp = high - low  # impulse leg
    ratios = [1.0, 1.272, 1.618, 2.0]
    return {r: retrace_low + imp * r for r in ratios}


def fibonacci_cluster(
    df: pd.DataFrame,
    lookback: int = 100,
) -> list[tuple[float, float]]:
    """Multi-swing Fibonacci cluster analysis.

    Identifies several recent swing highs/lows, computes retracement
    levels for each pair, and finds prices where multiple levels
    converge (a "cluster").

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``.
    lookback : int
        Number of recent bars to search for swing points (default 100).

    Returns
    -------
    list of (price, strength)
        Sorted by strength descending.  ``strength`` is the count of
        Fibonacci levels within ±0.3 % of the price.
    """
    if df.empty or lookback < 20:
        return []

    recent = df.iloc[-lookback:]
    highs = recent["high"].values
    lows = recent["low"].values
    close = recent["close"].values
    n = len(recent)

    # Detect swing highs/lows with a ±2 bar window
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    window = 2
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append((i, lows[i]))

    if not swing_highs or not swing_lows:
        return []

    # Take the most recent 3 of each to limit combinations
    swing_highs = swing_highs[-3:]
    swing_lows = swing_lows[-3:]

    # Generate all Fibonacci levels from each high/low pair
    all_levels: list[float] = []
    for sh_idx, sh_price in swing_highs:
        for sl_idx, sl_price in swing_lows:
            if sh_price <= sl_price:
                continue
            levels = fibonacci_levels(sh_price, sl_price)
            all_levels.extend(levels.values())

    if not all_levels:
        return []

    all_levels_arr = np.array(all_levels)

    # Current price for proximity weighting
    current_price = float(close[-1])

    # Cluster detection: find prices with most nearby fib levels
    clusters: dict[float, float] = {}

    for lvl in all_levels_arr:
        # Snap to the nearest 0.1% grid point to merge close levels
        grid = round(lvl / (current_price * 0.001)) * (current_price * 0.001)
        grid = round(grid, 2)
        clusters[grid] = clusters.get(grid, 0) + 1

    # Also count nearby levels within tolerance
    result: list[tuple[float, float]] = []
    tol = current_price * 0.003  # 0.3 %

    for price, count in clusters.items():
        # Bonus: count how many raw levels are within tolerance
        nearby = int(np.sum(np.abs(all_levels_arr - price) < tol))
        strength = float(max(count, nearby))
        if strength >= 2:
            result.append((price, strength))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def ote_zone(
    high: float,
    low: float,
) -> dict[str, float]:
    """Optimal Trade Entry (OTE) zone — 0.618–0.786 Fibonacci retracement.

    Parameters
    ----------
    high : float
        Swing high.
    low : float
        Swing low.

    Returns
    -------
    dict
        ``mid``     — midpoint of the OTE zone
        ``ote_low`` — 0.786 retracement (lower boundary of OTE)
        ``ote_high`` — 0.618 retracement (upper boundary of OTE)
    """
    rng = high - low
    ote_high = high - 0.618 * rng  # upper price (shallower fib)
    ote_low = high - 0.786 * rng   # lower price (deeper fib)
    mid = (ote_high + ote_low) / 2.0

    return {"mid": mid, "ote_low": ote_low, "ote_high": ote_high}


def nearest_fib(
    price: float,
    levels: dict[float, float],
) -> dict[str, Union[float, str]]:
    """Find the nearest Fibonacci level to a given price.

    Parameters
    ----------
    price : float
        Current price.
    levels : dict
        Fibonacci levels dict (ratio → price), e.g. from ``fibonacci_levels()``.

    Returns
    -------
    dict
        ``level``    — the Fibonacci ratio (e.g. 0.618)
        ``distance`` — absolute price distance to the level
        ``pct``      — percentage distance (positive = above, negative = below)
    """
    if not levels:
        return {"level": 0.0, "distance": 0.0, "pct": 0.0}

    best_ratio = 0.0
    best_dist = float("inf")

    for ratio, lvl_price in levels.items():
        dist = abs(price - lvl_price)
        if dist < best_dist:
            best_dist = dist
            best_ratio = ratio

    lvl_price = levels[best_ratio]
    pct = ((price - lvl_price) / lvl_price * 100.0) if lvl_price != 0 else 0.0

    return {
        "level": best_ratio,
        "distance": best_dist,
        "pct": pct,
    }
