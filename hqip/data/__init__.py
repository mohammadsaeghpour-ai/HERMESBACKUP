"""
HQIP v3 — Data Layer
=====================

Market-data acquisition and session management for the trading system.

Modules
-------
- exchange   : Multi-exchange connector via *ccxt* (OKX, Binance, Bybit).
- historical : Bulk historical kline downloads from Binance Vision.
- sessions   : Trading session / kill-zone definitions (ICT methodology).
"""

from hqip.data.exchange import ExchangeConnector
from hqip.data.historical import HistoricalDataManager
from hqip.data.sessions import SessionManager

__all__: list[str] = ["ExchangeConnector", "HistoricalDataManager", "SessionManager"]
