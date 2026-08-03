"""
HQIP v3 — SQLite Database Layer
=================================

Persistent storage for the trading system.  All tables are created
automatically on first use.

Tables
------
- **trades**        : Every order that was sent, including multi-TP levels.
- **signals**       : Agent-generated signals (before execution).
- **daily_pnl**     : Aggregated daily profit & loss.
- **feature_cache** : Cached indicator / feature values to avoid recomputation.

Usage::

    from hqip.core.database import Database

    db = Database("hqip.db")
    db.insert_trade(symbol="BTC/USDT", tf="1h", direction="long", entry=64000.0, ...)
    trades = db.get_trades(symbol="BTC/USDT", since="2026-01-01")
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLES: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT    NOT NULL,
        tf          TEXT    NOT NULL,
        direction   TEXT    NOT NULL CHECK(direction IN ('long', 'short')),
        entry       REAL    NOT NULL,
        sl          REAL,
        tp1         REAL,
        tp2         REAL,
        tp3         REAL,
        pnl         REAL,
        pnl_pct     REAL,
        agent_scores TEXT,                       -- JSON blob
        status      TEXT    DEFAULT 'open',     -- open | closed | partial
        notes       TEXT,
        timestamp   TEXT    NOT NULL,            -- ISO-8601 UTC
        created_at  TEXT    DEFAULT (datetime('now')),
        closed_at   TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol      TEXT    NOT NULL,
        tf          TEXT    NOT NULL,
        direction   TEXT    NOT NULL CHECK(direction IN ('long', 'short')),
        confidence  REAL    NOT NULL,
        agents      TEXT,                        -- JSON: which agents voted + scores
        metadata    TEXT,                        -- JSON: extra context
        timestamp   TEXT    NOT NULL,
        created_at  TEXT    DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_pnl (
        date         TEXT PRIMARY KEY,           -- YYYY-MM-DD
        total_pnl    REAL    DEFAULT 0,
        trades_taken INTEGER DEFAULT 0,
        wins         INTEGER DEFAULT 0,
        losses       INTEGER DEFAULT 0,
        win_rate     REAL    DEFAULT 0,
        max_drawdown REAL    DEFAULT 0,
        equity_eod   REAL,
        created_at   TEXT    DEFAULT (datetime('now')),
        updated_at   TEXT    DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_cache (
        symbol      TEXT    NOT NULL,
        tf          TEXT    NOT NULL,
        feature     TEXT    NOT NULL,
        value       TEXT,                        -- JSON-encoded value
        computed_at TEXT    NOT NULL,
        expires_at  TEXT,
        PRIMARY KEY (symbol, tf, feature)
    );
    """,
]

# Performance indexes
_CREATE_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);",
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_feature_cache_expires ON feature_cache(expires_at);",
]


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """SQLite-backed persistence layer for HQIP v3.

    Parameters
    ----------
    path : str | Path
        File-system path for the SQLite database.
    enable_wal : bool
        Enable WAL journal mode for better read/write concurrency.

    Example
    -------
    >>> db = Database("hqip.db")
    >>> db.insert_trade(
    ...     symbol="BTC/USDT", tf="1h", direction="long",
    ...     entry=64000.0, sl=63200.0, tp1=65600.0, tp2=67200.0,
    ...     tp3=68800.0, agent_scores={"price_action": 0.8},
    ... )
    """

    def __init__(self, path: str | Path = "hqip.db", enable_wal: bool = True) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._enable_wal = enable_wal
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;" if self._enable_wal else "SELECT 1;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._connect() as conn:
            for ddl in _CREATE_TABLES:
                conn.execute(ddl)
            for idx in _CREATE_INDEXES:
                conn.execute(idx)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def insert_trade(
        self,
        *,
        symbol: str,
        tf: str,
        direction: str,
        entry: float,
        sl: Optional[float] = None,
        tp1: Optional[float] = None,
        tp2: Optional[float] = None,
        tp3: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        agent_scores: Optional[dict[str, Any]] = None,
        status: str = "open",
        notes: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Insert a new trade row and return its ``id``.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTC/USDT"``.
        tf : str
            Timeframe that generated the signal, e.g. ``"1h"``.
        direction : str
            ``"long"`` or ``"short"``.
        entry : float
            Entry price.
        sl, tp1, tp2, tp3 : float, optional
            Stop-loss and take-profit levels.
        pnl : float, optional
            Realised profit/loss (set when closing).
        pnl_pct : float, optional
            PnL as a percentage of notional.
        agent_scores : dict, optional
            JSON-serialisable dict of per-agent confidence scores.
        status : str
            ``"open"`` | ``"closed"`` | ``"partial"``.
        notes : str, optional
            Free-text notes.
        timestamp : str, optional
            ISO-8601 UTC string; defaults to *now*.

        Returns
        -------
        int
            Row id of the inserted trade.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        scores_json = json.dumps(agent_scores) if agent_scores else None
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades
                    (symbol, tf, direction, entry, sl, tp1, tp2, tp3,
                     pnl, pnl_pct, agent_scores, status, notes, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, tf, direction, entry, sl, tp1, tp2, tp3,
                 pnl, pnl_pct, scores_json, status, notes, ts),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def close_trade(
        self,
        trade_id: int,
        *,
        pnl: float,
        pnl_pct: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Mark a trade as closed and record final PnL."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET pnl = ?, pnl_pct = ?, notes = COALESCE(?, notes),
                    status = 'closed', closed_at = ?
                WHERE id = ?
                """,
                (pnl, pnl_pct, notes, now, trade_id),
            )

    def get_trades(
        self,
        *,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch trades with optional filters.

        Parameters
        ----------
        symbol : str, optional
            Filter by trading pair.
        status : str, optional
            Filter by trade status.
        since : str, optional
            ISO-8601 lower bound on ``timestamp``.
        limit : int
            Maximum rows returned.

        Returns
        -------
        list[dict]
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if status:
            query += " AND status = ?"
            params.append(status)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def insert_signal(
        self,
        *,
        symbol: str,
        tf: str,
        direction: str,
        confidence: float,
        agents: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Record an agent-generated signal.

        Returns the row id.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        agents_json = json.dumps(agents) if agents else None
        meta_json = json.dumps(metadata) if metadata else None
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals (symbol, tf, direction, confidence, agents, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, tf, direction, confidence, agents_json, meta_json, ts),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_signals(
        self,
        *,
        symbol: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch recent signals."""
        query = "SELECT * FROM signals WHERE 1=1"
        params: list[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Daily PnL
    # ------------------------------------------------------------------
    def upsert_daily_pnl(
        self,
        *,
        date: str,
        total_pnl: float = 0.0,
        trades_taken: int = 0,
        wins: int = 0,
        losses: int = 0,
        max_drawdown: float = 0.0,
        equity_eod: Optional[float] = None,
    ) -> None:
        """Insert or update the daily PnL summary.

        Parameters
        ----------
        date : str
            ``"YYYY-MM-DD"``.
        total_pnl : float
            Net profit/loss for the day.
        trades_taken : int
            Number of round-trip trades.
        wins, losses : int
            Winning / losing trade counts.
        max_drawdown : float
            Maximum intra-day drawdown.
        equity_eod : float, optional
            End-of-day equity.
        """
        win_rate = wins / trades_taken if trades_taken > 0 else 0.0
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_pnl
                    (date, total_pnl, trades_taken, wins, losses, win_rate,
                     max_drawdown, equity_eod, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_pnl    = excluded.total_pnl,
                    trades_taken = excluded.trades_taken,
                    wins         = excluded.wins,
                    losses       = excluded.losses,
                    win_rate     = excluded.win_rate,
                    max_drawdown = excluded.max_drawdown,
                    equity_eod   = excluded.equity_eod,
                    updated_at   = excluded.updated_at
                """,
                (date, total_pnl, trades_taken, wins, losses, win_rate,
                 max_drawdown, equity_eod, now),
            )

    def get_daily_pnl(
        self, *, since: Optional[str] = None, limit: int = 90
    ) -> list[dict[str, Any]]:
        """Fetch daily PnL records."""
        query = "SELECT * FROM daily_pnl WHERE 1=1"
        params: list[Any] = []
        if since:
            query += " AND date >= ?"
            params.append(since)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Feature cache
    # ------------------------------------------------------------------
    def cache_feature(
        self,
        symbol: str,
        tf: str,
        feature: str,
        value: Any,
        ttl_seconds: int = 3600,
    ) -> None:
        """Store a computed feature with an expiry timestamp."""
        now = datetime.now(timezone.utc)
        from datetime import timedelta
        expires = (now + timedelta(seconds=ttl_seconds)).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feature_cache (symbol, tf, feature, value, computed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, tf, feature) DO UPDATE SET
                    value       = excluded.value,
                    computed_at = excluded.computed_at,
                    expires_at  = excluded.expires_at
                """,
                (symbol, tf, feature, json.dumps(value), now.isoformat(), expires),
            )

    def get_cached_feature(
        self, symbol: str, tf: str, feature: str
    ) -> Optional[Any]:
        """Retrieve a cached feature if it hasn't expired.

        Returns ``None`` when the entry is missing or stale.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT value, expires_at FROM feature_cache
                WHERE symbol = ? AND tf = ? AND feature = ?
                """,
                (symbol, tf, feature),
            ).fetchone()
        if row is None:
            return None
        if row["expires_at"] and row["expires_at"] < now_iso:
            return None
        return json.loads(row["value"])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run an arbitrary read query and return rows as dicts."""
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def vacuum(self) -> None:
        """Reclaim disk space (runs ``VACUUM`` outside a transaction)."""
        conn = sqlite3.connect(str(self._path))
        conn.execute("VACUUM;")
        conn.close()

    def __repr__(self) -> str:
        return f"Database(path={self._path!s}, wal={self._enable_wal})"
