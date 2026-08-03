"""
Portfolio Risk Management Module
=================================
Quantitative risk metrics for trading portfolios.

Provides drawdown analysis, risk-adjusted returns, and statistical
measures for evaluating trading strategy performance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def max_drawdown(pnl_series: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Calculate maximum drawdown from a P&L / equity series.

    Args:
        pnl_series: Cumulative P&L or equity values over time.

    Returns:
        Maximum drawdown as a positive fraction (0.0–1.0).
        Returns 0.0 for an empty or constant series.
    """
    if isinstance(pnl_series, (list, np.ndarray)):
        pnl_series = pd.Series(pnl_series, dtype=float)
    if pnl_series.empty or pnl_series.nunique() <= 1:
        return 0.0

    cumulative = pnl_series.cummax()
    drawdowns = (pnl_series - cumulative) / cumulative.replace(0, np.nan)
    drawdowns = drawdowns.replace([np.inf, -np.inf], np.nan).dropna()

    if drawdowns.empty:
        return 0.0

    return abs(drawdowns.min())


def sharpe_ratio(
    pnl_series: Union[pd.Series, np.ndarray, List[float]],
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio.

    Args:
        pnl_series: Per-period returns (not cumulative).
        rf: Risk-free rate per period (default 0).
        periods_per_year: Annualization factor (252 for daily, 8760 for hourly).

    Returns:
        Annualized Sharpe ratio.
    """
    if isinstance(pnl_series, (list, np.ndarray)):
        pnl_series = pd.Series(pnl_series, dtype=float)
    if len(pnl_series) < 2:
        return 0.0

    excess = pnl_series - rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0

    return float((excess.mean() / std) * math.sqrt(periods_per_year))


def sortino_ratio(
    pnl_series: Union[pd.Series, np.ndarray, List[float]],
    rf: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Annualized Sortino ratio — penalises only downside deviation.

    Args:
        pnl_series: Per-period returns.
        rf: Risk-free rate per period.
        periods_per_year: Annualization factor.
        target_return: Minimum acceptable return (MAR).

    Returns:
        Annualized Sortino ratio.
    """
    if isinstance(pnl_series, (list, np.ndarray)):
        pnl_series = pd.Series(pnl_series, dtype=float)
    if len(pnl_series) < 2:
        return 0.0

    excess = pnl_series - rf
    downside = excess[excess < target_return]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else 0.0

    downside_std = math.sqrt((downside ** 2).mean())
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float((excess.mean() / downside_std) * math.sqrt(periods_per_year))


def win_rate(pnl_series: Union[pd.Series, np.ndarray, List[float]]) -> float:
    """
    Fraction of positive-return periods.

    Args:
        pnl_series: Per-period P&L values.

    Returns:
        Win rate as a fraction between 0.0 and 1.0.
    """
    if isinstance(pnl_series, (list, np.ndarray)):
        pnl_series = pd.Series(pnl_series, dtype=float)
    if pnl_series.empty:
        return 0.0

    return float((pnl_series > 0).sum() / len(pnl_series))


def profit_factor(
    pnl_series: Union[pd.Series, np.ndarray, List[float]],
) -> float:
    """
    Profit factor — gross profits divided by gross losses.

    Args:
        pnl_series: Per-period P&L values.

    Returns:
        Profit factor (>1 is profitable). Returns inf for zero losses.
    """
    if isinstance(pnl_series, (list, np.ndarray)):
        pnl_series = pd.Series(pnl_series, dtype=float)

    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = abs(pnl_series[pnl_series < 0].sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def risk_of_ruin(
    capital: float,
    risk_per_trade: float,
    win_rate: float,
    max_trades: int,
    ruin_threshold: float = 0.0,
) -> float:
    """
    Probability of hitting the ruin threshold before max_trades.

    Uses a recursive Markov-chain approach:
    P(ruin | current_capital) is computed bottom-up from 0 to capital.

    Args:
        capital: Starting capital.
        risk_per_trade: Amount risked per trade (must be > 0).
        win_rate: Probability of winning a single trade (0–1).
        max_trades: Maximum number of trades allowed.
        ruin_threshold: Capital level that defines ruin (default 0).

    Returns:
        Probability of ruin as a float 0.0–1.0.
    """
    if risk_per_trade <= 0 or capital <= ruin_threshold:
        return 1.0
    if win_rate <= 0.0:
        return 1.0
    if win_rate >= 1.0:
        return 0.0

    # Number of risk units from ruin to current capital
    units = int(capital / risk_per_trade)
    if units <= 0:
        return 1.0

    # Probability of ruin from each level, bottom-up
    q = 1.0 - win_rate  # loss probability
    p = win_rate

    if p == q:  # 50/50
        return min(1.0, ruin_threshold / capital) if ruin_threshold > 0 else 1.0 / (units + 1)

    # Standard risk-of-ruin formula
    q_over_p = q / p
    if q_over_p < 1.0:
        # (q/p)^units is the analytical ruin probability
        return min(1.0, q_over_p ** units)
    else:
        return 1.0


def expectancy(
    trades: Union[pd.Series, np.ndarray, List[float]],
) -> Dict[str, float]:
    """
    Compute expected value, Kelly fraction, and variance from trade results.

    Args:
        trades: Individual trade P&L values.

    Returns:
        Dict with keys:
            ev:       Expected value per trade (mean).
            kelly:    Optimal bet fraction (Kelly criterion).
            variance: Variance of trade results.
    """
    if isinstance(trades, (list, np.ndarray)):
        trades = pd.Series(trades, dtype=float)

    n = len(trades)
    if n == 0:
        return {"ev": 0.0, "kelly": 0.0, "variance": 0.0}

    ev = float(trades.mean())
    variance = float(trades.var(ddof=1)) if n > 1 else 0.0

    # Kelly criterion: f* = (p * b - q) / b
    # where p = win probability, q = loss probability, b = avg_win / avg_loss
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    if wins.empty or losses.empty:
        kelly = 0.0
    else:
        p = len(wins) / n
        q = 1.0 - p
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            kelly = 1.0 if p > 0 else 0.0
        else:
            b = avg_win / avg_loss
            kelly = (p * b - q) / b

    kelly = max(0.0, min(1.0, kelly))  # clamp to [0, 1]

    return {
        "ev": ev,
        "kelly": kelly,
        "variance": variance,
    }


__all__ = [
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate",
    "profit_factor",
    "risk_of_ruin",
    "expectancy",
]
