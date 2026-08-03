"""
HQIP v3 — Historical Data Manager
====================================

Downloads daily kline data from `Binance Vision <https://data.binance.vision/>`_
and stores it locally as CSV files for fast offline analysis and back-testing.

Supported timeframes: 5m, 15m, 30m, 1h, 4h, 1d.

Usage::

    from hqip.data.historical import HistoricalDataManager

    hdm = HistoricalDataManager(data_dir="data/historical")
    # Download last 90 days of BTC/USDT 1h candles
    df = await hdm.download("BTC/USDT", "1h", days=90)
    # Merge with any existing local data
    df = hdm.merge("BTC/USDT", "1h")
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd  # type: ignore[import-untyped]

log = logging.getLogger("hqip.historical")

# Binance Vision public download base URL
_BASE_URL = "https://data.binance.vision"

# Mapping from our short names → Binance Vision interval strings
_INTERVAL_MAP: dict[str, str] = {
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# Columns expected in the Binance kline CSVs.
_KLINE_COLS: list[str] = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]

_DEFAULT_DAYS = 90


class HistoricalDataManager:
    """Download, store, and merge historical kline data from Binance Vision.

    Data is persisted as CSV files under *data_dir*, organised as::

        <data_dir>/<symbol>_<timeframe>.csv

    For example: ``data/historical/BTC_USDT_1h.csv``.

    Parameters
    ----------
    data_dir : str | Path
        Local directory for CSV storage.
    default_days : int
        Number of days to fetch when ``days`` is not specified.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/historical",
        default_days: int = _DEFAULT_DAYS,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._default_days = default_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def download(
        self,
        symbol: str,
        timeframe: str,
        days: Optional[int] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Download daily kline ZIPs from Binance Vision and return as DataFrame.

        Parameters
        ----------
        symbol : str
            Trading pair in Binance format, e.g. ``"BTCUSDT"`` (slash is
            automatically removed).
        timeframe : str
            One of ``"5m"``, ``"15m"``, ``"30m"``, ``"1h"``, ``"4h"``, ``"1d"``.
        days : int, optional
            Look-back window. Defaults to *default_days*.
        end_date : datetime, optional
            End of the download window (UTC). Defaults to *today*.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame for the requested date range.
        """
        import asyncio
        import aiohttp  # type: ignore[import-untyped]

        if timeframe not in _INTERVAL_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}. Use {list(_INTERVAL_MAP)}")

        days = days or self._default_days
        end = (end_date or datetime.now(timezone.utc)).date()
        start = end - timedelta(days=days)
        binance_symbol = symbol.replace("/", "").upper()
        interval = _INTERVAL_MAP[timeframe]

        frames: list[pd.DataFrame] = []
        current = start

        log.info(
            "Downloading %s %s from %s to %s (%d days)",
            symbol, timeframe, start, end, days,
        )

        async with aiohttp.ClientSession() as session:
            while current <= end:
                url = (
                    f"{_BASE_URL}/public/data/klines"
                    f"/{binance_symbol}/{interval}"
                    f"/{binance_symbol}-{interval}-{current.isoformat()}.zip"
                )
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            raw = await resp.read()
                            df = self._parse_zip(raw, timeframe)
                            if df is not None and not df.empty:
                                frames.append(df)
                        else:
                            log.debug("No data for %s (HTTP %d)", current, resp.status)
                except Exception as exc:
                    log.warning("Failed to fetch %s: %s", url, exc)
                current += timedelta(days=1)

        if not frames:
            log.warning("No data downloaded for %s %s", symbol, timeframe)
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values("open_time", inplace=True)
        combined.drop_duplicates(subset=["open_time"], inplace=True)
        combined.reset_index(drop=True, inplace=True)

        # Persist to disk
        csv_path = self._csv_path(symbol, timeframe)
        combined.to_csv(csv_path, index=False)
        log.info(
            "Saved %d rows to %s", len(combined), csv_path,
        )

        return combined

    def merge(
        self,
        symbol: str,
        timeframe: str,
        new_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Merge newly downloaded data with existing local CSV (if any).

        If *new_data* is ``None``, this simply loads the local CSV.

        Parameters
        ----------
        symbol : str
            Trading pair.
        timeframe : str
            Timeframe string.
        new_data : pd.DataFrame, optional
            New data to merge in. Duplicates (by ``open_time``) are dropped.

        Returns
        -------
        pd.DataFrame
            Merged and sorted DataFrame.
        """
        csv_path = self._csv_path(symbol, timeframe)
        existing = self._load_csv(csv_path)

        if new_data is None or new_data.empty:
            return existing

        if existing.empty:
            new_data.sort_values("open_time", inplace=True)
            new_data.drop_duplicates(subset=["open_time"], inplace=True)
            new_data.reset_index(drop=True, inplace=True)
            new_data.to_csv(csv_path, index=False)
            return new_data

        combined = pd.concat([existing, new_data], ignore_index=True)
        combined.sort_values("open_time", inplace=True)
        combined.drop_duplicates(subset=["open_time"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        combined.to_csv(csv_path, index=False)
        log.info("Merged → %d total rows in %s", len(combined), csv_path)
        return combined

    def load(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """Load locally cached CSV (no download).

        Returns an empty DataFrame if the file doesn't exist.
        """
        return self._load_csv(self._csv_path(symbol, timeframe))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _csv_path(self, symbol: str, timeframe: str) -> Path:
        fname = symbol.replace("/", "_").upper() + f"_{timeframe}.csv"
        return self._data_dir / fname

    @staticmethod
    def _parse_zip(raw_bytes: bytes, timeframe: str) -> Optional[pd.DataFrame]:
        """Extract a single CSV from a Binance Vision kline ZIP."""
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f, header=None, names=_KLINE_COLS, dtype=str)
                    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
                    df["close_time"] = pd.to_datetime(df["close_time"].astype(int), unit="ms", utc=True)
                    for col in ("open", "high", "low", "close", "volume",
                                "quote_volume", "taker_buy_base", "taker_buy_quote"):
                        df[col] = df[col].astype(float)
                    df["trades"] = df["trades"].astype(int)
                    df["timeframe"] = timeframe
                    df.drop(columns=["ignore"], inplace=True)
                    return df
        except zipfile.BadZipFile:
            log.warning("Bad ZIP data received")
            return None

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Load a kline CSV, returning an empty DataFrame if absent."""
        if not path.is_file():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            # Restore datetime types if columns exist
            for col in ("open_time", "close_time"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)
            return df
        except Exception as exc:
            log.error("Failed to load %s: %s", path, exc)
            return pd.DataFrame()
