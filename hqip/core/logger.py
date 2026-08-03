"""
HQIP v3 — Structured Logging
=============================

Provides a project-wide logger with:

* **Console** output (human-readable, colour-coded by level).
* **File** output (JSON-lines for machine parsing).
* **Trade** sink — append-only CSV of every trade lifecycle event.
* **Performance** sink — period returns, Sharpe, draw-down.
* **Error** sink — dedicated error log with stack traces.

Usage::

    from hqip.core.logger import get_logger

    log = get_logger("strategy")
    log.info("Signal generated", symbol="BTC/USDT", confidence=0.87)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Custom formatter
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """Emit each log-record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra keyword args attached by the caller.
        for key in ("symbol", "tf", "direction", "confidence", "pnl",
                     "entry", "sl", "tp", "latency_ms", "exchange",
                     "error_type", "stack"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            )
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Human-friendly colour output for the terminal."""

    _COLOURS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[1;31m", # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelname, "")
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return (
            f"{colour}{ts} [{record.levelname:7s}] "
            f"{record.name}: {record.getMessage()}{self._RESET}"
        )


# ---------------------------------------------------------------------------
# Trade / Performance / Error CSV sinks
# ---------------------------------------------------------------------------

class _CSVSink:
    """Append-only CSV writer for structured trade / performance data."""

    def __init__(self, filepath: Path, fieldnames: list[str]) -> None:
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._file = open(filepath, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        # Write header only when the file is empty.
        if filepath.stat().st_size == 0 if filepath.exists() else True:
            self._writer.writeheader()
            self._file.flush()

    def write(self, **kwargs: Any) -> None:
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Public logger class
# ---------------------------------------------------------------------------

_TRADE_FIELDS = [
    "timestamp", "symbol", "tf", "direction", "entry", "sl",
    "tp1", "tp2", "tp3", "pnl", "pnl_pct", "agent_scores", "notes",
]

_PERF_FIELDS = [
    "timestamp", "date", "total_pnl", "daily_pnl", "trades_taken",
    "win_rate", "sharpe", "max_drawdown", "equity",
]

_ERROR_FIELDS = [
    "timestamp", "logger", "level", "message", "exception", "stack",
]


class HQIPLogger:
    """Project-wide logger with trade, performance, and error sinks.

    Parameters
    ----------
    name : str
        Logger namespace (e.g. ``"hqip.core"``).
    log_dir : str | Path
        Directory for log files.
    level : str
        Minimum severity (``DEBUG``, ``INFO``, …).
    """

    def __init__(
        self,
        name: str = "hqip",
        log_dir: str | Path = "logs",
        level: str = "INFO",
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Root HQIP logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False

        # Avoid duplicate handlers on repeated calls.
        if not self._logger.handlers:
            self._setup_handlers()

        # Lazy-init CSV sinks
        self._trade_sink: Optional[_CSVSink] = None
        self._perf_sink: Optional[_CSVSink] = None
        self._error_sink: Optional[_CSVSink] = None

    # ------------------------------------------------------------------
    # Handler wiring
    # ------------------------------------------------------------------
    def _setup_handlers(self) -> None:
        # Console
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(_ConsoleFormatter())
        self._logger.addHandler(console)

        # JSON-lines file
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(
            self._log_dir / f"hqip-{today}.jsonl", encoding="utf-8",
        )
        file_handler.setFormatter(_JSONFormatter())
        self._logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Convenience logging helpers (mirror stdlib API)
    # ------------------------------------------------------------------
    def debug(self, msg: str, **kw: Any) -> None:
        self._logger.debug(msg, extra=kw)

    def info(self, msg: str, **kw: Any) -> None:
        self._logger.info(msg, extra=kw)

    def warning(self, msg: str, **kw: Any) -> None:
        self._logger.warning(msg, extra=kw)

    def error(self, msg: str, *, exc_info: bool = False, **kw: Any) -> None:
        self._logger.error(msg, exc_info=exc_info, extra=kw)
        if exc_info:
            self._log_error(msg, **kw)

    def critical(self, msg: str, **kw: Any) -> None:
        self._logger.critical(msg, exc_info=True, extra=kw)
        self._log_error(msg, **kw)

    # ------------------------------------------------------------------
    # Trade sink
    # ------------------------------------------------------------------
    def _get_trade_sink(self) -> _CSVSink:
        if self._trade_sink is None:
            self._trade_sink = _CSVSink(self._log_dir / "trades.csv", _TRADE_FIELDS)
        return self._trade_sink

    def log_trade(self, **kwargs: Any) -> None:
        """Append a trade record to ``trades.csv``.

        Accepted keyword arguments match :data:`_TRADE_FIELDS`.
        """
        kwargs.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._get_trade_sink().write(**kwargs)

    # ------------------------------------------------------------------
    # Performance sink
    # ------------------------------------------------------------------
    def _get_perf_sink(self) -> _CSVSink:
        if self._perf_sink is None:
            self._perf_sink = _CSVSink(self._log_dir / "performance.csv", _PERF_FIELDS)
        return self._perf_sink

    def log_performance(self, **kwargs: Any) -> None:
        """Append a daily performance record to ``performance.csv``."""
        kwargs.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        kwargs.setdefault("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        self._get_perf_sink().write(**kwargs)

    # ------------------------------------------------------------------
    # Error sink
    # ------------------------------------------------------------------
    def _get_error_sink(self) -> _CSVSink:
        if self._error_sink is None:
            self._error_sink = _CSVSink(self._log_dir / "errors.csv", _ERROR_FIELDS)
        return self._error_sink

    def _log_error(self, msg: str, **kw: Any) -> None:
        import sys as _sys
        exc = _sys.exc_info()
        stack = "".join(traceback.format_exception(*exc)) if exc[1] else ""
        self._get_error_sink().write(
            timestamp=datetime.now(timezone.utc).isoformat(),
            logger=kw.pop("logger_name", self._logger.name),
            level="ERROR",
            message=msg,
            exception=str(exc[1]) if exc[1] else "",
            stack=stack,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close all file sinks."""
        for sink in (self._trade_sink, self._perf_sink, self._error_sink):
            if sink is not None:
                sink.close()
        for handler in self._logger.handlers:
            handler.close()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_LOGGERS: dict[str, HQIPLogger] = {}


def get_logger(
    name: str = "hqip",
    log_dir: str | Path = "logs",
    level: str = "INFO",
) -> HQIPLogger:
    """Return (and cache) a named :class:`HQIPLogger` instance.

    Parameters
    ----------
    name : str
        Logger namespace.  The root logger ``"hqip"`` is used by default;
        subsystems can create child namespaces (e.g. ``"hqip.strategy"``).
    log_dir : str | Path
        Directory for CSV / JSONL log files.
    level : str
        Minimum log level.

    Returns
    -------
    HQIPLogger
    """
    if name not in _LOGGERS:
        _LOGGERS[name] = HQIPLogger(name=name, log_dir=log_dir, level=level)
    return _LOGGERS[name]
