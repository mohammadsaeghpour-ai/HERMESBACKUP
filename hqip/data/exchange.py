"""
HQIP v3 — Multi-Exchange Connector
====================================

Unified interface to OKX, Binance, and Bybit via the `ccxt <https://github.com/ccxt/ccxt>`_
library.  Provides automatic fallback between exchanges, retry with
exponential back-off, and rate-limit awareness.

Usage::

    from hqip.data.exchange import ExchangeConnector

    conn = ExchangeConnector(default="binance")
    ohlcv = await conn.fetch_ohlcv("BTC/USDT", "1h", limit=200)
    ticker = await conn.fetch_ticker("ETH/USDT")
    depth  = await conn.fetch_orderbook("SOL/USDT", limit=10)
    frate  = await conn.fetch_funding_rate("BTC/USDT")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import ccxt  # type: ignore[import-untyped]
import ccxt.async_support as ccxt_async  # type: ignore[import-untyped]

from hqip.core.config import Config, ExchangeCredentials

log = logging.getLogger("hqip.exchange")

# ---------------------------------------------------------------------------
# Exchange factory
# ---------------------------------------------------------------------------

_EXCHANGE_CLASSES: dict[str, type] = {
    "binance": ccxt_async.binance,
    "okx":     ccxt_async.okx,
    "bybit":   ccxt_async.bybit,
}


def _build_exchange(
    name: str, creds: ExchangeCredentials
) -> ccxt_async.Exchange:
    """Instantiate a ccxt async exchange from credentials."""
    cls = _EXCHANGE_CLASSES.get(name.lower())
    if cls is None:
        raise ValueError(f"Unsupported exchange: {name!r}. Choose from {list(_EXCHANGE_CLASSES)}")
    params: dict[str, Any] = {
        "enableRateLimit": True,
        "rateLimit": creds.rate_limit_ms,
    }
    if creds.api_key:
        params["apiKey"] = creds.api_key
        params["secret"] = creds.secret
    if creds.passphrase:
        params["password"] = creds.passphrase
    if creds.sandbox:
        params["sandbox"] = True
    return cls(params)


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

class ExchangeError(Exception):
    """Raised when all exchanges in the fallback chain fail."""


async def _retry(
    fn,
    *args,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    **kwargs,
) -> Any:
    """Execute *fn* with retries and exponential back-off.

    Retries on ccxt network / availability errors only.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except (
            ccxt.NetworkError,
            ccxt.ExchangeNotAvailable,
            ccxt.RequestTimeout,
        ) as exc:
            last_exc = exc
            wait = backoff_base * (2 ** attempt)
            log.warning(
                "Attempt %d/%d failed (%s): %s — retrying in %.1fs",
                attempt + 1, max_retries, type(exc).__name__, exc, wait,
            )
            await asyncio.sleep(wait)
        except ccxt.BaseError as exc:
            # Non-transient errors (bad params, auth) — don't retry.
            raise ExchangeError(str(exc)) from exc
    raise ExchangeError(f"All {max_retries} attempts failed") from last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ExchangeConnector:
    """Async multi-exchange connector with fallback support.

    Parameters
    ----------
    default : str
        Primary exchange name (``"binance"``, ``"okx"``, or ``"bybit"``).
    config : Config, optional
        If *None*, the singleton :class:`Config` is used.

    Example
    -------
    >>> conn = ExchangeConnector("binance")
    >>> candles = await conn.fetch_ohlcv("BTC/USDT", "1h", limit=100)
    """

    def __init__(
        self,
        default: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        self._cfg = config or Config.get()
        self._default = default or self._cfg.exchange.default_exchange
        self._fallback_order = list(self._cfg.exchange.fallback_order)
        # Ensure the default is first in the fallback chain.
        if self._default in self._fallback_order:
            self._fallback_order.remove(self._default)
        self._fallback_order.insert(0, self._default)
        # Lazy-init exchange instances
        self._exchanges: dict[str, ccxt_async.Exchange] = {}

    def _get(self, name: str) -> ccxt_async.Exchange:
        """Return (or create) an exchange instance."""
        if name not in self._exchanges:
            creds = getattr(self._cfg.exchange, name.lower(), None)
            if creds is None:
                raise ValueError(f"No credentials configured for {name}")
            self._exchanges[name] = _build_exchange(name, creds)
        return self._exchanges[name]

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
        since: Optional[int] = None,
    ) -> list[list]:
        """Fetch candlestick data with automatic fallback and retry.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTC/USDT"``.
        timeframe : str
            ccxt timeframe string (``"5m"``, ``"1h"``, ``"1d"``, …).
        limit : int
            Number of candles.
        since : int, optional
            Start timestamp in milliseconds.

        Returns
        -------
        list[list]
            Each entry is ``[timestamp_ms, open, high, low, close, volume]``.
        """
        last_exc: Optional[Exception] = None
        for name in self._fallback_order:
            try:
                ex = self._get(name)
                data = await _retry(ex.fetch_ohlcv, symbol, timeframe, since, limit)
                log.info("fetch_ohlcv %s %s via %s → %d candles", symbol, timeframe, name, len(data))
                return data
            except Exception as exc:
                log.warning("OHLCV fallback %s failed: %s", name, exc)
                last_exc = exc
        raise ExchangeError(f"All exchanges failed for OHLCV {symbol}/{timeframe}") from last_exc

    # ------------------------------------------------------------------
    # Ticker (live price)
    # ------------------------------------------------------------------
    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch the latest ticker (last price, bid/ask, 24 h stats).

        Returns
        -------
        dict
            ccxt-normalised ticker dictionary.
        """
        last_exc: Optional[Exception] = None
        for name in self._fallback_order:
            try:
                ex = self._get(name)
                data = await _retry(ex.fetch_ticker, symbol)
                log.debug("fetch_ticker %s via %s → last=%.2f", symbol, name, data.get("last", 0))
                return data
            except Exception as exc:
                log.warning("Ticker fallback %s failed: %s", name, exc)
                last_exc = exc
        raise ExchangeError(f"All exchanges failed for ticker {symbol}") from last_exc

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------
    async def fetch_orderbook(
        self, symbol: str, limit: int = 20
    ) -> dict[str, Any]:
        """Fetch the order book (bids / asks).

        Parameters
        ----------
        symbol : str
            Trading pair.
        limit : int
            Depth in number of price levels.

        Returns
        -------
        dict
            ``{"bids": [...], "asks": [...], "timestamp": ...}``
        """
        last_exc: Optional[Exception] = None
        for name in self._fallback_order:
            try:
                ex = self._get(name)
                data = await _retry(ex.fetch_order_book, symbol, limit)
                log.debug(
                    "fetch_orderbook %s via %s → %d bids, %d asks",
                    symbol, name, len(data.get("bids", [])), len(data.get("asks", [])),
                )
                return data
            except Exception as exc:
                log.warning("Orderbook fallback %s failed: %s", name, exc)
                last_exc = exc
        raise ExchangeError(f"All exchanges failed for orderbook {symbol}") from last_exc

    # ------------------------------------------------------------------
    # Funding rate (perpetual swaps)
    # ------------------------------------------------------------------
    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Fetch the current funding rate for a perpetual swap pair.

        Returns
        -------
        dict
            ccxt normalised funding-rate dictionary.
        """
        last_exc: Optional[Exception] = None
        for name in self._fallback_order:
            try:
                ex = self._get(name)
                # Some exchanges expose fundingRate via fetchFundingRate;
                # fall back to fetch_ticker['info'] if not available.
                if hasattr(ex, "fetch_funding_rate"):
                    data = await _retry(ex.fetch_funding_rate, symbol)
                else:
                    ticker = await _retry(ex.fetch_ticker, symbol)
                    data = {
                        "symbol": symbol,
                        "fundingRate": ticker.get("info", {}).get("fundingRate"),
                        "fundingTimestamp": ticker.get("info", {}).get("nextFundingTime"),
                    }
                log.debug("fetch_funding_rate %s via %s → %s", symbol, name, data.get("fundingRate"))
                return data
            except Exception as exc:
                log.warning("Funding rate fallback %s failed: %s", name, exc)
                last_exc = exc
        raise ExchangeError(f"All exchanges failed for funding rate {symbol}") from last_exc

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    async def close(self) -> None:
        """Close all underlying exchange connections."""
        for name, ex in self._exchanges.items():
            try:
                await ex.close()
            except Exception as exc:
                log.warning("Error closing %s: %s", name, exc)
        self._exchanges.clear()

    async def __aenter__(self) -> "ExchangeConnector":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"ExchangeConnector(default={self._default!r}, "
            f"fallback={self._fallback_order!r})"
        )
