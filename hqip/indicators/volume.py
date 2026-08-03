"""
HQIP v3 — Volume Indicators
============================
OBV, VWAP, VWAP bands, MFI, volume profile, absorption detection, composite score.

All functions use only pandas/numpy, are NaN-safe, and handle edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


# ── public API ────────────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — vectorised implementation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close`` and ``volume``.

    Returns
    -------
    pd.Series
        OBV cumulative series.
    """
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def obv_divergence(df: pd.DataFrame) -> dict[str, bool]:
    """Detect bullish / bearish OBV divergence vs price over recent bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close`` and ``volume``.

    Returns
    -------
    dict
        ``bullish`` — price lower low + OBV higher low.
        ``bearian`` — price higher high + OBV lower high.
    """
    close = df["close"]
    obv_vals = obv(df)
    lookback = min(60, len(close) - 1)

    if lookback < 10:
        return {"bullish": False, "bearian": False}

    c = close.iloc[-lookback:]
    o = obv_vals.iloc[-lookback:]

    # Swing lows
    lows = [i for i in range(2, len(c) - 2)
            if c.iloc[i] <= c.iloc[i - 1] and c.iloc[i] <= c.iloc[i + 1]]

    # Swing highs
    highs = [i for i in range(2, len(c) - 2)
             if c.iloc[i] >= c.iloc[i - 1] and c.iloc[i] >= c.iloc[i + 1]]

    bullish = bearian = False

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if c.iloc[i2] < c.iloc[i1] and o.iloc[i2] > o.iloc[i1]:
            bullish = True

    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if c.iloc[i2] > c.iloc[i1] and o.iloc[i2] < o.iloc[i1]:
            bearian = True

    return {"bullish": bullish, "bearian": bearian}


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (session-cumulative).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``, ``volume``.

    Returns
    -------
    pd.Series
        VWAP values.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, 1e-10)


def vwap_bands(
    df: pd.DataFrame,
    std: float = 1.5,
) -> dict[str, pd.Series]:
    """VWAP ±N standard-deviation bands.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``, ``volume``.
    std : float
        Number of stdevs for the bands (default 1.5).

    Returns
    -------
    dict
        ``upper`` — VWAP + std bands
        ``lower`` — VWAP − std bands
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap_val = vwap(df)

    # Cumulative variance of TP weighted by volume
    cum_vol = df["volume"].cumsum().replace(0, 1e-10)
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_tp2_vol = ((tp ** 2) * df["volume"]).cumsum()

    variance = (cum_tp2_vol / cum_vol) - (cum_tp_vol / cum_vol) ** 2
    variance = variance.clip(lower=0)  # numerical guard
    band_std = np.sqrt(variance)

    return {
        "upper": vwap_val + std * band_std,
        "lower": vwap_val - std * band_std,
    }


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``, ``volume``.
    period : int
        Look-back (default 14).

    Returns
    -------
    pd.Series
        MFI values (0–100).
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_mf = tp * df["volume"]

    tp_diff = tp.diff()
    pos_mf = raw_mf.where(tp_diff > 0, 0.0)
    neg_mf = raw_mf.where(tp_diff < 0, 0.0)

    pos_sum = pos_mf.rolling(period, min_periods=1).sum()
    neg_sum = neg_mf.rolling(period, min_periods=1).sum()

    mfr = pos_sum / neg_sum.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + mfr))


def volume_profile(
    df: pd.DataFrame,
    bins: int = 24,
) -> dict[str, object]:
    """Volume-at-price profile.

    Distributes each bar's volume across price bins between the
    session high and low.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``, ``volume``.
    bins : int
        Number of price buckets (default 24).

    Returns
    -------
    dict
        ``prices``  — array of bin centre prices
        ``volumes`` — array of total volume per bin
        ``poc``     — price at the Point of Control (highest volume bin)
    """
    price_min = df["low"].min()
    price_max = df["high"].max()

    if price_min == price_max:
        mid = float(price_min)
        return {
            "prices": np.array([mid]),
            "volumes": np.array([df["volume"].sum()]),
            "poc": mid,
        }

    edges = np.linspace(price_min, price_max, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    vol_at_price = np.zeros(bins, dtype=float)

    low_vals = df["low"].values
    high_vals = df["high"].values
    vol_vals = df["volume"].values

    for i in range(len(df)):
        lo, hi, v = low_vals[i], high_vals[i], vol_vals[i]
        mask = (centres >= lo) & (centres <= hi)
        n_bins = mask.sum()
        if n_bins > 0:
            vol_at_price[mask] += v / n_bins

    poc_idx = int(np.argmax(vol_at_price))

    return {
        "prices": centres,
        "volumes": vol_at_price,
        "poc": float(centres[poc_idx]),
    }


def volume_absorption(df: pd.DataFrame, lookback: int = 20) -> dict[str, object]:
    """Detect volume absorption patterns.

    Absorption occurs when large volume accompanies small price changes,
    indicating aggressive orders being absorbed by the opposing side.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``, ``volume``.
    lookback : int
        Number of recent bars to evaluate (default 20).

    Returns
    -------
    dict
        ``detected``  — bool, True if absorption detected
        ``direction`` — ``'bullish'``, ``'bearish'``, or ``'none'``
    """
    if df.empty or len(df) < lookback:
        return {"detected": False, "direction": "none"}

    recent = df.iloc[-lookback:]
    vol = recent["volume"]
    body = (recent["close"] - recent["open"]).abs()
    atr_val = recent["high"] - recent["low"]  # simple range as proxy

    vol_mean = vol.mean()
    vol_std = vol.std()

    if pd.isna(vol_mean) or pd.isna(vol_std) or vol_std == 0:
        return {"detected": False, "direction": "none"}

    # High volume bars with small body relative to range
    high_vol = vol > vol_mean + 1.5 * vol_std
    small_body = body < 0.3 * atr_val.replace(0, 1e-10)
    absorption_mask = high_vol & small_body

    count = absorption_mask.sum()
    if count < 2:
        return {"detected": False, "direction": "none"}

    # Determine direction: net price change on absorption bars
    price_change = recent.loc[absorption_mask, "close"].diff()
    net = price_change.sum()

    if net > 0:
        direction = "bullish"
    elif net < 0:
        direction = "bearish"
    else:
        direction = "none"

    return {"detected": True, "direction": direction}


def volume_score(df: pd.DataFrame) -> float:
    """Composite volume score in **−100 … +100**.

    Sub-scores and weights:
    • OBV trend vs price trend  (weight 25)
    • MFI position              (weight 20)
    • Volume ratio vs average   (weight 20)
    • OBV divergence            (weight 15)
    • Volume absorption bias    (weight 20)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.

    Returns
    -------
    float
        Score from −100 (bearish volume) to +100 (bullish volume).
    """
    if df.empty or len(df) < 20:
        return 0.0

    scores: list[float] = []
    weights: list[float] = []

    # 1. OBV trend: OBV slope vs price slope (last 20 bars)
    lookback = min(20, len(df) - 1)
    obv_vals = obv(df)
    close_vals = df["close"]

    obv_slope = obv_vals.iloc[-1] - obv_vals.iloc[-lookback]
    price_slope = close_vals.iloc[-1] - close_vals.iloc[-lookback]
    obv_safe = abs(obv_slope) + 1e-10

    if obv_slope > 0 and price_slope > 0:
        scores.append(80.0)
    elif obv_slope > 0 and price_slope <= 0:
        scores.append(50.0)  # bullish divergence
    elif obv_slope < 0 and price_slope > 0:
        scores.append(-50.0)  # bearish divergence
    else:
        scores.append(-80.0)
    weights.append(25)

    # 2. MFI
    mfi_val = mfi(df).iloc[-1]
    if pd.notna(mfi_val):
        scores.append((mfi_val - 50) * 2)
        weights.append(20)

    # 3. Volume ratio
    vol = df["volume"]
    vol_avg = vol.rolling(20, min_periods=1).mean()
    vol_ratio = vol.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1.0
    # High volume amplifies the existing direction
    last_return = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100 if len(df) > 1 else 0
    vol_amplified = np.clip(last_return * min(vol_ratio, 3.0) * 10, -100, 100)
    scores.append(float(vol_amplified))
    weights.append(20)

    # 4. OBV divergence
    div = obv_divergence(df)
    div_score = 0.0
    if div["bullish"]:
        div_score = 70.0
    elif div["bearian"]:
        div_score = -70.0
    scores.append(div_score)
    weights.append(15)

    # 5. Volume absorption
    abs_result = volume_absorption(df)
    abs_score = 0.0
    if abs_result["detected"]:
        if abs_result["direction"] == "bullish":
            abs_score = 60.0
        elif abs_result["direction"] == "bearish":
            abs_score = -60.0
    scores.append(abs_score)
    weights.append(20)

    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return float(np.clip(
        sum(s * w for s, w in zip(scores, weights)) / total_w, -100, 100
    ))
