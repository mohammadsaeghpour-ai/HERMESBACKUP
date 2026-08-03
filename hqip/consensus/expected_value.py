"""
Expected Value calculator for trade decisions.

Computes expected value per unit risk, Kelly criterion position
sizing, and enforces a minimum viable EV threshold.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


# Minimum EV to consider a trade viable (as fraction of risk)
_MIN_VIABLE_EV: float = 0.005  # 0.5%


def minimum_viable_ev() -> float:
    """
    Return the minimum expected value required to take a trade.

    Returns
    -------
    float
        0.005 (i.e. 0.5%).
    """
    return _MIN_VIABLE_EV


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Full Kelly fraction:  f* = (p · b − q) / b

    where p = win probability, q = 1 − p, b = avg_win / avg_loss.

    Returns a fraction of capital to risk.  Negative values mean the
    edge is negative — do not trade.

    Parameters
    ----------
    win_rate : float
        Probability of a winning trade (0-1).
    avg_win : float
        Average win size in risk units (e.g. 1.5 means 1.5R).
    avg_loss : float
        Average loss size in risk units (always positive).

    Returns
    -------
    float
        Kelly fraction (can be negative).  Clamped to [0, 0.25] when
        positive (quarter-Kelly safety cap).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    p = float(np.clip(win_rate, 0.0, 1.0))
    q = 1.0 - p
    b = avg_win / avg_loss

    kelly = (p * b - q) / b

    # Clamp: no negative sizing; cap at 25% (quarter-Kelly)
    return float(np.clip(kelly, 0.0, 0.25))


def calculate_ev(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fees: float = 0.001,
) -> Dict[str, float]:
    """
    Calculate expected value, Kelly fraction, and recommended size.

    EV = (win_rate × avg_win) − (loss_rate × avg_loss) − fees

    Parameters
    ----------
    win_rate : float
        Probability of win (0-1).
    avg_win : float
        Average win in risk units.
    avg_loss : float
        Average loss in risk units (positive).
    fees : float
        Transaction costs as fraction of risk (default 0.001 = 10 bps).

    Returns
    -------
    dict
        Keys: ev, kelly, recommended_size.
        - ev : expected value per unit risk (net of fees).
        - kelly : full Kelly fraction.
        - recommended_size : quarter-Kelly position size (0 if no edge).
    """
    p = float(np.clip(win_rate, 0.0, 1.0))
    q = 1.0 - p

    ev = (p * avg_win) - (q * avg_loss) - fees
    kelly = kelly_criterion(win_rate, avg_win, avg_loss)
    recommended = kelly * 0.25 if kelly > 0 else 0.0  # quarter-Kelly

    return {
        "ev": round(ev, 6),
        "kelly": round(kelly, 6),
        "recommended_size": round(recommended, 6),
    }
