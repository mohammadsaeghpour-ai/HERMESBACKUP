"""
HQIP v3 — Reports
==================

Automated report generation for trading performance.

Submodules
----------
- ``daily`` : Daily trading summary (P&L, win rate, session breakdown).
"""

from .daily import generate_daily_report

__all__ = ["generate_daily_report"]
