"""
HQIP v3 — Alert System
=======================

Multi-channel alerting for signals, trades, and daily summaries.

Submodules
----------
- ``telegram`` : Telegram Bot API alerts with rich formatting.
- ``webhook``  : Generic webhook-based alerting with ``AlertManager``.
"""

from .telegram import send_signal_alert, send_trade_alert, send_daily_report
from .webhook import AlertManager, send_webhook

__all__ = [
    "send_signal_alert",
    "send_trade_alert",
    "send_daily_report",
    "AlertManager",
    "send_webhook",
]
