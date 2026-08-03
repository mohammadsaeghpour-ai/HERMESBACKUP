"""
HQIP v3 — Trend Indicators
===========================
EMA fan, EMA crosses, SuperTrend, ADX, Ichimoku, composite trend strength.

All functions accept a pandas DataFrame with at least ``close`` (and
``high`` / ``low`` where needed).  They return Series or dicts of Series
so callers can attach results back to the frame or feed them into the
scoring engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Raw EMA with adjust=False (matches TradingView / TA-Lib default)."""
    return series.ewm(span=period, adjust=False).mean()


def _safe_shift(s: pd.Series, n: int = 1) -> pd.Series:
    return s.shift(n)


# ── public API ────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average.

    Parameters
    ----------
    series : pd.Series
        Price or any numeric series.
    period : int
        EMA look-back (must be ≥ 1).

    Returns
    -------
    pd.Series
        EMA values (first ``period - 1`` entries are valid but use
        the recursive formula, not a seed).
    """
    period = max(1, int(period))
    return _ema(series, period)


def ema_fan(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Multi-period EMA fan commonly used for trend framing.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close`` column.

    Returns
    -------
    dict
        Keys: ema_8, ema_13, ema_21, ema_55, ema_89, ema_200.
        Values: pd.Series of EMA values.
    """
    close = df["close"]
    periods = [8, 13, 21, 55, 89, 200]
    return {f"ema_{p}": _ema(close, p) for p in periods}


def ema_cross(df: pd.DataFrame) -> str:
    """Short-term vs long-term EMA directional bias.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close`` column.

    Returns
    -------
    str
        ``'bullish'`` if EMA-20 > EMA-50 on the latest bar,
        ``'bearish'`` if <,
        ``'neutral'`` otherwise.
    """
    close = df["close"]
    ema20 = _ema(close, 20).iloc[-1]
    ema50 = _ema(close, 50).iloc[-1]

    if pd.isna(ema20) or pd.isna(ema50):
        return "neutral"
    if ema20 > ema50:
        return "bullish"
    if ema20 < ema50:
        return "bearish"
    return "neutral"


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    mult: float = 3.0,
) -> dict[str, pd.Series]:
    """SuperTrend indicator (Volatility-adjusted trailing stop).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int
        ATR look-back for the base band (default 10).
    mult : float
        ATR multiplier for upper/lower bands (default 3).

    Returns
    -------
    dict
        ``supertrend`` — trailing stop line (pd.Series)
        ``direction`` — +1 bullish, −1 bearish (pd.Series)
        ``upper`` / ``lower`` — raw bands before ratchet (pd.Series)
    """
    from .volatility import atr as _atr

    atr_val = _atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + mult * atr_val
    lower = hl2 - mult * atr_val

    n = len(df)
    st = pd.Series(np.nan, index=df.index)
    direction = pd.Series(np.nan, index=df.index)

    prev_dir = 1
    prev_lower = lower.iloc[0] if n > 0 else np.nan
    prev_upper = upper.iloc[0] if n > 0 else np.nan

    close_vals = df["close"].values
    upper_vals = upper.values.copy()
    lower_vals = lower.values.copy()
    st_vals = np.full(n, np.nan)
    dir_vals = np.full(n, np.nan)

    for i in range(1, n):
        c = close_vals[i]

        # determine direction
        if c > prev_upper:
            cur_dir = 1
        elif c < prev_lower:
            cur_dir = -1
        else:
            cur_dir = prev_dir

        # ratchet
        if cur_dir == 1:
            lower_vals[i] = max(lower_vals[i], prev_lower) if prev_dir == 1 else lower_vals[i]
            st_vals[i] = lower_vals[i]
        else:
            upper_vals[i] = min(upper_vals[i], prev_upper) if prev_dir == -1 else upper_vals[i]
            st_vals[i] = upper_vals[i]

        dir_vals[i] = cur_dir
        prev_dir = cur_dir
        prev_lower = lower_vals[i]
        prev_upper = upper_vals[i]

    # first bar defaults
    dir_vals[0] = 1
    st_vals[0] = lower_vals[0]

    return {
        "supertrend": pd.Series(st_vals, index=df.index),
        "direction": pd.Series(dir_vals, index=df.index),
        "upper": pd.Series(upper_vals, index=df.index),
        "lower": pd.Series(lower_vals, index=df.index),
    }


def adx(df: pd.DataFrame, period: int = 14) -> dict[str, pd.Series]:
    """Average Directional Index with +DI / −DI.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    period : int
        Smoothing / DX look-back (default 14).

    Returns
    -------
    dict
        ``adx``   — ADX line (pd.Series)
        ``di_plus``  — +DI (pd.Series)
        ``di_minus`` — −DI (pd.Series)
    """
    from .volatility import atr as _atr

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _atr(df, 1) * 1  # True Range (not smoothed)
    # Recompute raw TR for smoothing with Wilder's method (EMA)
    tr_raw = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_s = _ema(tr_raw, period)
    safe_atr = atr_s.replace(0, 1e-10)

    di_plus = 100.0 * _ema(plus_dm, period) / safe_atr
    di_minus = 100.0 * _ema(minus_dm, period) / safe_atr

    di_sum = di_plus + di_minus
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum.replace(0, 1e-10)
    adx_val = _ema(dx, period)

    return {
        "adx": adx_val,
        "di_plus": di_plus,
        "di_minus": di_minus,
    }


def ichimoku(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Ichimoku Kinko Hyo — 5-line cloud system.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.

    Returns
    -------
    dict
        Keys: tenkan, kijun, senkou_a, senkou_b, chikou.
        ``senkou_a`` and ``senkou_b`` are shifted forward 26 periods
        (leading spans).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    def _mid(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
        return (h.rolling(n).max() + l.rolling(n).min()) / 2.0

    tenkan = _mid(high, low, 9)
    kijun = _mid(high, low, 26)

    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = _mid(high, low, 52).shift(26)
    chikou = close.shift(-26)

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }


def trend_strength(df: pd.DataFrame) -> float:
    """Composite trend-strength score in **−100 … +100**.

    Blends:
    • EMA-20 vs EMA-50 position         (weight 20)
    • EMA-200 price relative             (weight 25)
    • SuperTrend direction               (weight 20)
    • ADX directional bias               (weight 20)
    • Ichimoku cloud position            (weight 15)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (needs ≥ 200 bars for best accuracy).

    Returns
    -------
    float
        Score from −100 (extremely bearish) to +100 (extremely bullish).
    """
    if df.empty or len(df) < 30:
        return 0.0

    scores: list[float] = []
    weights: list[float] = []

    close = df["close"]
    last_close = close.iloc[-1]

    # 1. EMA-20 vs EMA-50
    ema20 = _ema(close, 20).iloc[-1]
    ema50 = _ema(close, 50).iloc[-1]
    if pd.notna(ema20) and pd.notna(ema50) and ema50 != 0:
        diff_pct = (ema20 - ema50) / abs(ema50) * 100.0
        scores.append(np.clip(diff_pct * 10, -100, 100))
        weights.append(20)

    # 2. Price vs EMA-200
    ema200 = _ema(close, 200).iloc[-1]
    if pd.notna(ema200) and ema200 != 0:
        diff_pct = (last_close - ema200) / abs(ema200) * 100.0
        scores.append(np.clip(diff_pct * 10, -100, 100))
        weights.append(25)

    # 3. SuperTrend
    st = supertrend(df)
    st_dir = st["direction"].iloc[-1]
    if pd.notna(st_dir):
        scores.append(float(st_dir) * 100)
        weights.append(20)

    # 4. ADX directional bias
    adx_dict = adx(df)
    di_plus_last = adx_dict["di_plus"].iloc[-1]
    di_minus_last = adx_dict["di_minus"].iloc[-1]
    adx_last = adx_dict["adx"].iloc[-1]
    if pd.notna(adx_last) and pd.notna(di_plus_last) and pd.notna(di_minus_last):
        bias = di_plus_last - di_minus_last  # range roughly −200..+200
        # scale to −100..+100, amplified by ADX strength
        strength_factor = min(adx_last / 50.0, 1.0)  # 0..1
        scores.append(np.clip(bias * strength_factor, -100, 100))
        weights.append(20)

    # 5. Ichimoku
    ich = ichimoku(df)
    senkou_a_last = ich["senkou_a"].iloc[-1]
    senkou_b_last = ich["senkou_b"].iloc[-1]
    if pd.notna(senkou_a_last) and pd.notna(senkou_b_last) and senkou_b_last != 0:
        cloud_mid = (senkou_a_last + senkou_b_last) / 2.0
        diff_pct = (last_close - cloud_mid) / abs(cloud_mid) * 100.0
        scores.append(np.clip(diff_pct * 10, -100, 100))
        weights.append(15)

    if not weights:
        return 0.0

    total_w = sum(weights)
    return float(sum(s * w for s, w in zip(scores, weights)) / total_w)
