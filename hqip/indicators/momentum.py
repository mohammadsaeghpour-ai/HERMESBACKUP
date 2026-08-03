"""
HQIP v3 — Momentum Indicators
==============================
RSI, MACD, Stochastic, CCI, Williams %R, divergence detectors, composite score.

All functions use only pandas/numpy, are NaN-safe, and return Series or dicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Internal EMA — identical formula to ``trend.ema``."""
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range (not averaged)."""
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)


# ── public API ────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing via EMA).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close``.
    period : int
        Look-back (default 14).

    Returns
    -------
    pd.Series
        RSI in 0–100 range.
    """
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_divergence(df: pd.DataFrame, period: int = 14) -> dict[str, bool]:
    """Detect bullish / bearish RSI divergence vs price over recent bars.

    Compares the last two swing lows (bullish) and swing highs (bearish)
    using a simple peak/trough finder over the last 60 bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close``.
    period : int
        RSI look-back (default 14).

    Returns
    -------
    dict
        ``bullish`` — True if price makes lower low but RSI makes higher low.
        ``bearian`` — True if price makes higher high but RSI makes lower high.
    """
    close = df["close"]
    rsi_vals = rsi(df, period)
    lookback = min(60, len(close) - 1)

    if lookback < 10:
        return {"bullish": False, "bearian": False}

    c = close.iloc[-lookback:]
    r = rsi_vals.iloc[-lookback:]

    # Find local minima (swing lows) — index i is a trough if both neighbours are higher
    lows_idx = []
    for i in range(2, len(c) - 2):
        if c.iloc[i] <= c.iloc[i - 1] and c.iloc[i] <= c.iloc[i + 1]:
            lows_idx.append(i)

    highs_idx = []
    for i in range(2, len(c) - 2):
        if c.iloc[i] >= c.iloc[i - 1] and c.iloc[i] >= c.iloc[i + 1]:
            highs_idx.append(i)

    bullish = False
    bearian = False

    # Bullish divergence: price lower low, RSI higher low
    if len(lows_idx) >= 2:
        i1, i2 = lows_idx[-2], lows_idx[-1]
        if c.iloc[i2] < c.iloc[i1] and r.iloc[i2] > r.iloc[i1]:
            bullish = True

    # Bearish divergence: price higher high, RSI lower high
    if len(highs_idx) >= 2:
        i1, i2 = highs_idx[-2], highs_idx[-1]
        if c.iloc[i2] > c.iloc[i1] and r.iloc[i2] < r.iloc[i1]:
            bearian = True

    return {"bullish": bullish, "bearian": bearian}


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """MACD line, signal line, histogram.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close``.
    fast, slow, signal : int
        Standard MACD parameters.

    Returns
    -------
    dict
        ``macd``      — MACD line (pd.Series)
        ``signal``    — signal line (pd.Series)
        ``histogram`` — MACD − signal (pd.Series)
    """
    close = df["close"]
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def macd_divergence(df: pd.DataFrame) -> dict[str, bool]:
    """Detect bullish / bearish MACD histogram divergence vs price.

    Uses last 60 bars and simple peak/trough detection.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close``.

    Returns
    -------
    dict
        ``bullish`` / ``bearian`` — booleans.
    """
    close = df["close"]
    hist = macd(df)["histogram"]
    lookback = min(60, len(close) - 1)

    if lookback < 10:
        return {"bullish": False, "bearian": False}

    c = close.iloc[-lookback:]
    h = hist.iloc[-lookback:]

    # Troughs in price
    lows_idx = []
    for i in range(2, len(c) - 2):
        if c.iloc[i] <= c.iloc[i - 1] and c.iloc[i] <= c.iloc[i + 1]:
            lows_idx.append(i)

    # Peaks in price
    highs_idx = []
    for i in range(2, len(c) - 2):
        if c.iloc[i] >= c.iloc[i - 1] and c.iloc[i] >= c.iloc[i + 1]:
            highs_idx.append(i)

    bullish = False
    bearian = False

    if len(lows_idx) >= 2:
        i1, i2 = lows_idx[-2], lows_idx[-1]
        if c.iloc[i2] < c.iloc[i1] and h.iloc[i2] > h.iloc[i1]:
            bullish = True

    if len(highs_idx) >= 2:
        i1, i2 = highs_idx[-2], highs_idx[-1]
        if c.iloc[i2] > c.iloc[i1] and h.iloc[i2] < h.iloc[i1]:
            bearian = True

    return {"bullish": bullish, "bearian": bearian}


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> dict[str, pd.Series]:
    """Stochastic Oscillator (%K, %D).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    k : int
        %K look-back (default 14).
    d : int
        %D smoothing period (default 3).

    Returns
    -------
    dict
        ``k`` — %K line (pd.Series, 0–100)
        ``d`` — %D line (pd.Series, SMA of %K)
    """
    low_min = df["low"].rolling(k, min_periods=1).min()
    high_max = df["high"].rolling(k, min_periods=1).max()
    denom = (high_max - low_min).replace(0, 1e-10)
    k_line = 100.0 * (df["close"] - low_min) / denom
    d_line = _sma(k_line, d)
    return {"k": k_line, "d": d_line}


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int
        Look-back (default 20).

    Returns
    -------
    pd.Series
        CCI values (unbounded, typically −200..+200 in normal markets).
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(period, min_periods=1).mean()
    mad = tp.rolling(period, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    safe_mad = mad.replace(0, 1e-10)
    return (tp - sma_tp) / (0.015 * safe_mad)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — momentum oscillator (−100 to 0).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int
        Look-back (default 14).

    Returns
    -------
    pd.Series
        Williams %R values.
    """
    hh = df["high"].rolling(period, min_periods=1).max()
    ll = df["low"].rolling(period, min_periods=1).min()
    denom = (hh - ll).replace(0, 1e-10)
    return -100.0 * (hh - df["close"]) / denom


def momentum_score(df: pd.DataFrame) -> float:
    """Composite momentum score in **−100 … +100**.

    Sub-scores and weights:
    • RSI position          (weight 25)
    • MACD histogram sign   (weight 25)
    • Stochastic %K vs %D   (weight 15)
    • CCI position          (weight 15)
    • Williams %R           (weight 10)
    • RSI divergence bonus  (weight 10)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.

    Returns
    -------
    float
        Score from −100 (extremely bearish momentum) to +100.
    """
    if df.empty or len(df) < 30:
        return 0.0

    scores: list[float] = []
    weights: list[float] = []

    # 1. RSI
    rsi_val = rsi(df).iloc[-1]
    if pd.notna(rsi_val):
        # Map 0..100 → −100..+100
        scores.append((rsi_val - 50) * 2)
        weights.append(25)

    # 2. MACD histogram
    macd_dict = macd(df)
    hist_val = macd_dict["histogram"].iloc[-1]
    hist_prev = macd_dict["histogram"].iloc[-2] if len(df) > 1 else 0.0
    if pd.notna(hist_val) and pd.notna(hist_prev):
        # Normalize: histogram sign and momentum
        score = np.clip(hist_val / (abs(hist_prev) + 1e-10) * 50, -100, 100)
        scores.append(float(score))
        weights.append(25)

    # 3. Stochastic
    stoch = stochastic(df)
    k_val = stoch["k"].iloc[-1]
    d_val = stoch["d"].iloc[-1]
    if pd.notna(k_val) and pd.notna(d_val):
        scores.append((k_val - 50) * 2)
        weights.append(15)

    # 4. CCI
    cci_val = cci(df).iloc[-1]
    if pd.notna(cci_val):
        scores.append(np.clip(cci_val, -100, 100))
        weights.append(15)

    # 5. Williams %R
    wr_val = williams_r(df).iloc[-1]
    if pd.notna(wr_val):
        # Williams %R is −100..0 → map to −100..+100
        scores.append((wr_val + 50) * 2)
        weights.append(10)

    # 6. RSI divergence bonus
    div = rsi_divergence(df)
    div_score = 0.0
    if div["bullish"]:
        div_score = 60.0
    elif div["bearian"]:
        div_score = -60.0
    scores.append(div_score)
    weights.append(10)

    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return float(np.clip(sum(s * w for s, w in zip(scores, weights)) / total_w, -100, 100))
