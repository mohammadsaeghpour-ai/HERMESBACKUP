"""
HQIP Configuration
==================
Central configuration for all symbols, timeframes, agents, and risk parameters.
"""
# ── Symbols ──────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# ── Timeframes (highest → lowest) ────────────────────────
TIMEFRAMES = ["1d", "4h", "1h", "15m"]

# ── Timeframe Weights ────────────────────────────────────
TF_WEIGHTS = {"1d": 6.0, "4h": 4.0, "1h": 3.0, "15m": 2.0}

# ── Consensus Thresholds ─────────────────────────────────
CONSENSUS = {
    "min_weighted_agreement": 0.60,
    "min_confidence": 0.50,
    "min_agents_agree": 3,
}

# ── Risk Defaults ────────────────────────────────────────
RISK = {
    "default_capital": 10000,
    "risk_per_trade_pct": 1.0,
    "max_daily_loss_pct": 3.0,
    "max_trades_per_day": 4,
    "cooldown_minutes": 30,
    "default_leverage": 5,
    "max_leverage": 20,
}

# ── Data ─────────────────────────────────────────────────
DATA = {"candle_limit": 300}

# ── Grade Thresholds ─────────────────────────────────────
GRADES = {
    "A+": {"conf": 0.85, "agree": 0.80},
    "A":  {"conf": 0.75, "agree": 0.75},
    "B+": {"conf": 0.65, "agree": 0.70},
    "B":  {"conf": 0.55, "agree": 0.65},
    "C":  {"conf": 0.45, "agree": 0.55},
}
