"""HermesQuant Global Config"""
OKX_API = "https://www.okx.com/api/v5"
TIMEZONE = "Asia/Tehran"
CAPITAL = 10
LEVERAGE = 20
MAX_RISK_PER_TRADE = 0.03
MAX_DAILY_LOSS = 0.30
DEFAULT_SL_ATR_MULT = 1.5
DEFAULT_TP_ATR_MULT = [2.0, 3.0, 5.0]
MIN_VOTE_RATIO = 0.8  # 4/5
MIN_ADX = 25
MIN_VOLUME_RATIO = 1.0
SESSIONS = {
    "asia": (0, 8),
    "europe": (8, 14),
    "us": (14, 22),
    "late": (22, 24),
}
