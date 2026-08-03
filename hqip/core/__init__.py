"""
HQIP v3 — Core Data Layer
==========================

The foundational module for the HQIP (High Quality Institutional Patterns) v3
trading system. Provides configuration, logging, and database primitives used
by every other subsystem.

Modules
-------
- config   : System-wide configuration (YAML / env-var overrides).
- logger   : Structured logging with trade, performance, and error sinks.
- database : SQLite persistence for trades, signals, daily PnL, and feature cache.
"""

from hqip.core.config import Config
from hqip.core.logger import HQIPLogger, get_logger
from hqip.core.database import Database

__all__: list[str] = ["Config", "HQIPLogger", "get_logger", "Database"]
