"""
Position sizing module.

Provides multiple sizing methodologies: fixed-fractional, Kelly-based,
and volatility-adjusted.  Also calculates SL/TP levels from ATR multiples.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def fixed_fractional(
    capital: float,
    risk_pct: float,
    entry: float,
    sl: float,
) -> Dict[str, float]:
    """
    Fixed-fractional position sizing.

    Risk a fixed percentage of capital per trade.

    Parameters
    ----------
    capital : float
        Total account equity.
    risk_pct : float
        Risk per trade as a fraction (e.g. 0.02 for 2 %).
    entry : float
        Planned entry price.
    sl : float
        Stop-loss price.

    Returns
    -------
    dict
        qty          – number of units / contracts to trade.
        margin       – notional exposure (qty × entry).
        risk_usd     – dollar amount at risk.
    """
    risk_distance = abs(entry - sl)
    if risk_distance <= 0:
        return {"qty": 0.0, "margin": 0.0, "risk_usd": 0.0}

    risk_usd = capital * risk_pct
    qty = risk_usd / risk_distance
    margin = qty * entry

    return {
        "qty": round(qty, 6),
        "margin": round(margin, 2),
        "risk_usd": round(risk_usd, 2),
    }


def kelly_sizing(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    entry: float,
    sl: float,
) -> Dict[str, float]:
    """
    Kelly-criterion position sizing.

    Computes the Kelly fraction, then derives position size from it.
    Uses quarter-Kelly as a safety measure.

    Parameters
    ----------
    capital : float
        Total account equity.
    win_rate : float
        Historical win probability (0-1).
    avg_win : float
        Average win in price units (positive).
    avg_loss : float
        Average loss in price units (positive).
    entry : float
        Entry price.
    sl : float
        Stop-loss price.

    Returns
    -------
    dict
        qty, margin, risk_usd, kelly_fraction.
    """
    risk_distance = abs(entry - sl)
    if risk_distance <= 0 or avg_loss <= 0:
        return {"qty": 0.0, "margin": 0.0, "risk_usd": 0.0, "kelly_fraction": 0.0}

    p = float(np.clip(win_rate, 0.0, 1.0))
    q = 1.0 - p
    b = avg_win / avg_loss

    kelly = (p * b - q) / b
    kelly = float(np.clip(kelly, 0.0, 0.25))  # quarter-Kelly cap

    risk_usd = capital * kelly
    qty = risk_usd / risk_distance
    margin = qty * entry

    return {
        "qty": round(qty, 6),
        "margin": round(margin, 2),
        "risk_usd": round(risk_usd, 2),
        "kelly_fraction": round(kelly, 6),
    }


def volatility_sizing(
    capital: float,
    atr: float,
    target_risk_pct: float = 0.02,
) -> Dict[str, float]:
    """
    ATR-based volatility position sizing.

    Sizes the position so that 1 ATR move equals ``target_risk_pct``
    of capital.

    Parameters
    ----------
    capital : float
        Total account equity.
    atr : float
        Current ATR value (price units).
    target_risk_pct : float
        Target risk as fraction of capital per 1 ATR move (default 0.02).

    Returns
    -------
    dict
        qty, margin (estimated from ATR scale), risk_usd.
    """
    if atr <= 0:
        return {"qty": 0.0, "margin": 0.0, "risk_usd": 0.0}

    risk_usd = capital * target_risk_pct
    qty = risk_usd / atr
    # Estimate entry as proportional (caller should supply actual entry)
    margin = qty * atr  # rough proxy

    return {
        "qty": round(qty, 6),
        "margin": round(margin, 2),
        "risk_usd": round(risk_usd, 2),
    }


def calculate_sltp(
    entry: float,
    atr: float,
    direction: str,
    sl_mult: float = 1.5,
    tp1_mult: float = 1.0,
    tp2_mult: float = 2.0,
    tp3_mult: float = 3.0,
) -> Dict[str, float]:
    """
    Calculate stop-loss and take-profit levels from ATR multiples.

    Parameters
    ----------
    entry : float
        Entry price.
    atr : float
        Current ATR value.
    direction : str
        "BUY" or "SELL".
    sl_mult : float
        ATR multiplier for stop-loss (default 1.5).
    tp1_mult : float
        ATR multiplier for first take-profit (default 1.0).
    tp2_mult : float
        ATR multiplier for second take-profit (default 2.0).
    tp3_mult : float
        ATR multiplier for third take-profit (default 3.0).

    Returns
    -------
    dict
        sl, tp1, tp2, tp3.
    """
    d = direction.upper()
    if d == "BUY":
        sl = entry - (atr * sl_mult)
        tp1 = entry + (atr * tp1_mult)
        tp2 = entry + (atr * tp2_mult)
        tp3 = entry + (atr * tp3_mult)
    elif d == "SELL":
        sl = entry + (atr * sl_mult)
        tp1 = entry - (atr * tp1_mult)
        tp2 = entry - (atr * tp2_mult)
        tp3 = entry - (atr * tp3_mult)
    else:
        raise ValueError(f"direction must be 'BUY' or 'SELL', got '{direction}'")

    return {
        "sl": round(sl, 6),
        "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),
        "tp3": round(tp3, 6),
    }
