"""
HQIP v3 — Configuration
========================

Centralised, hierarchical configuration for the entire trading system.
Every value is overridable via:

1. A YAML file (``config.yaml`` next to this module or path given by
   ``HQIP_CONFIG`` env var).
2. Individual environment variables using the prefix ``HQIP_`` with
   double-underscore separators for nesting (e.g.
   ``HQIP_EXCHANGE__OKX__API_KEY``).

Usage::

    from hqip.core.config import Config

    cfg = Config()                    # load from defaults + YAML + env
    print(cfg.exchange.okx.api_key)
    print(cfg.trading.risk_per_trade)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional YAML support — graceful fallback if PyYAML isn't installed.
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ===== Exchange Config ======================================================

@dataclass(frozen=False)
class ExchangeCredentials:
    """Credentials and rate-limit settings for a single exchange."""

    api_key: str = ""
    secret: str = ""
    passphrase: str = ""  # OKX requires this
    sandbox: bool = False
    rate_limit_ms: int = 1200  # milliseconds between requests


@dataclass(frozen=False)
class ExchangeConfig:
    """Aggregate settings for all supported exchanges."""

    okx: ExchangeCredentials = field(default_factory=ExchangeCredentials)
    binance: ExchangeCredentials = field(default_factory=ExchangeCredentials)
    bybit: ExchangeCredentials = field(default_factory=ExchangeCredentials)
    default_exchange: str = "binance"
    fallback_order: list[str] = field(default_factory=lambda: ["binance", "okx", "bybit"])


# ===== Trading Parameters ====================================================

@dataclass(frozen=False)
class TradingConfig:
    """Capital allocation, leverage, and per-trade risk."""

    capital: float = 10_000.0
    leverage: int = 10
    risk_per_trade: float = 0.01  # 1 % of equity
    max_open_positions: int = 5
    symbols: list[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    ])


# ===== Timeframe Definitions =================================================

# Mapping of human-readable names → ccxt timeframes (milliseconds).
TIMEFRAME_MAP: dict[str, str] = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Ordered from lowest to highest for multi-timeframe analysis.
TIMEFRAME_HIERARCHY: list[str] = ["5m", "15m", "30m", "1h", "4h", "1d"]


# ===== Session Definitions ===================================================

@dataclass(frozen=True)
class SessionDef:
    """Defines a trading session by name, UTC hour range, and volatility."""

    name: str
    start_hour: int
    end_hour: int
    volatility: str  # 'lowest' | 'low' | 'medium' | 'high' | 'highest'


SESSIONS: dict[str, SessionDef] = {
    "asia":     SessionDef("Asia",     0,  8,  "low"),
    "europe":   SessionDef("Europe",   7,  16, "medium"),
    "america":  SessionDef("America",  13, 22, "high"),
    "overlap":  SessionDef("Overlap",  13, 16, "highest"),
    "offhours": SessionDef("Off-Hours", 22, 0, "lowest"),
}

# Volatility multiplier applied to position sizing calculations.
VOLATILITY_MULTIPLIER: dict[str, float] = {
    "lowest":  0.5,
    "low":     0.75,
    "medium":  1.0,
    "high":    1.25,
    "highest": 1.5,
}


# ===== Kill Zones (ICT) =====================================================

@dataclass(frozen=True)
class KillZone:
    """ICT kill-zone — narrow high-probability windows."""

    name: str
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int


KILL_ZONES: dict[str, KillZone] = {
    "london_open":  KillZone("London Open",  7,  0,  10, 0),
    "ny_open":      KillZone("NY Open",      12, 0,  15, 0),
    "london_close": KillZone("London Close", 15, 0,  17, 0),
}


# ===== Indicator Parameters ==================================================

@dataclass(frozen=False)
class IndicatorConfig:
    """Parameters for technical indicators used by the agent pipeline."""

    ema_fast: int = 9
    ema_slow: int = 21
    ema_anchor: int = 200
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_rsi_period: int = 14
    orderbook_depth_levels: int = 20


# ===== Agent Weights =========================================================

@dataclass(frozen=False)
class AgentWeightsConfig:
    """Relative voting weight of each analysis agent."""

    price_action: float = 0.25
    order_flow: float = 0.15
    structure: float = 0.20
    indicator: float = 0.15
    sentiment: float = 0.10
    ml_model: float = 0.15


# ===== Risk Management =======================================================

@dataclass(frozen=False)
class RiskConfig:
    """Risk management parameters."""

    max_daily_drawdown: float = 0.03       # 3 %
    max_weekly_drawdown: float = 0.06      # 6 %
    max_consecutive_losses: int = 5
    cooldown_minutes: int = 30
    trailing_stop_activation_rr: float = 1.5  # R:R ratio to activate trailing
    trailing_stop_distance_pct: float = 0.005  # 0.5 %
    position_sizing: str = "kelly"  # 'fixed' | 'kelly' | 'volatility'


# ===== ML Model Paths ========================================================

@dataclass(frozen=False)
class MLConfig:
    """Paths and settings for machine-learning models."""

    base_dir: Path = field(default_factory=lambda: Path("models"))
    signal_model: str = "signal_model_v3.pkl"
    regime_model: str = "regime_classifier.pkl"
    feature_scaler: str = "feature_scaler.pkl"
    retrain_interval_hours: int = 24


# ===== Database =============================================================

@dataclass(frozen=False)
class DatabaseConfig:
    """SQLite database settings."""

    path: Path = field(default_factory=lambda: Path("hqip.db"))
    enable_wal: bool = True  # write-ahead logging for concurrency


# ===== Root Config ===========================================================

@dataclass(frozen=False)
class Config:
    """Top-level configuration container.

    Loaded once at import; re-instantiation picks up new env-var / YAML values.
    """

    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    agents: AgentWeightsConfig = field(default_factory=AgentWeightsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # ------------------------------------------------------------------
    # Class-level cache so Config() is cheap after the first call.
    # ------------------------------------------------------------------
    _instance: Optional["Config"] = None

    @classmethod
    def load(cls) -> "Config":
        """Build a :class:`Config` from defaults → YAML → environment."""

        cfg = cls()
        cfg._apply_yaml()
        cfg._apply_env()
        return cfg

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------
    def _apply_yaml(self) -> None:
        """Merge values from a YAML file if present."""
        if not _HAS_YAML:
            return
        yaml_path = os.getenv("HQIP_CONFIG")
        if yaml_path is None:
            candidates = [
                Path("config.yaml"),
                Path("hqip.yaml"),
                Path(__file__).resolve().parent.parent / "config.yaml",
            ]
            for p in candidates:
                if p.is_file():
                    yaml_path = str(p)
                    break
        if yaml_path is None or not Path(yaml_path).is_file():
            return
        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        self._merge_dict(raw)

    def _merge_dict(self, raw: dict[str, Any]) -> None:
        """Shallow-merge a flat/nested dict into dataclass fields."""
        for section_name in ("exchange", "trading", "indicators", "agents", "risk", "ml", "database"):
            section_data = raw.get(section_name)
            if section_data is None or not isinstance(section_data, dict):
                continue
            section_obj = getattr(self, section_name, None)
            if section_obj is None:
                continue
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
        # top-level scalars
        for key in ("log_level",):
            if key in raw:
                setattr(self, key, raw[key])

    # ------------------------------------------------------------------
    # Environment variable helpers
    # ------------------------------------------------------------------
    def _apply_env(self) -> None:
        """Override fields from ``HQIP_`` prefixed env vars.

        Convention: ``HQIP_SECTION__FIELD`` → ``self.section.field``.

        Examples::

            HQIP_TRADING__CAPITAL=50000
            HQIP_EXCHANGE__DEFAULT_EXCHANGE=okx
            HQIP_EXCHANGE__OKX__API_KEY=abc
        """
        prefix = "HQIP_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].split("__")
            if len(parts) < 2:
                if parts[0] == "LOG_LEVEL":
                    self.log_level = value
                continue
            section_name, field_name = parts[0].lower(), parts[1]
            section_obj = getattr(self, section_name, None)
            if section_obj is None:
                continue
            # Handle nested exchange credentials: HQIP_EXCHANGE__OKX__API_KEY
            if len(parts) == 3 and section_name == "exchange":
                sub = getattr(section_obj, parts[1].lower(), None)
                if sub is not None and hasattr(sub, parts[2].lower()):
                    setattr(sub, parts[2].lower(), self._coerce(value))
                continue
            if hasattr(section_obj, field_name.lower()):
                setattr(section_obj, field_name.lower(), self._coerce(value))

    @staticmethod
    def _coerce(value: str) -> Any:
        """Best-effort coercion of env-var string to int / float / bool."""
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def get(cls) -> "Config":
        """Return a cached singleton :class:`Config` instance."""
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance
