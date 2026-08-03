"""
HQIP v3 Indicator Engine
========================
Modular, composable technical indicators built on pandas/numpy.

Modules
-------
- trend      : EMA fan, SuperTrend, ADX, Ichimoku, trend strength
- momentum   : RSI, MACD, Stochastic, CCI, Williams %R, divergences
- volatility : ATR, Bollinger, Keltner, volatility regime
- volume     : OBV, VWAP, MFI, volume profile, absorption detection
- fibonacci  : Retracement levels, extensions, clusters, OTE zones
"""
from .trend import (
    ema,
    ema_fan,
    ema_cross,
    supertrend,
    adx,
    ichimoku,
    trend_strength,
)
from .momentum import (
    rsi,
    rsi_divergence,
    macd,
    macd_divergence,
    stochastic,
    cci,
    williams_r,
    momentum_score,
)
from .volatility import (
    atr,
    bollinger,
    bollinger_squeeze,
    keltner,
    volatility_regime,
)
from .volume import (
    obv,
    obv_divergence,
    vwap,
    vwap_bands,
    mfi,
    volume_profile,
    volume_absorption,
    volume_score,
)
from .fibonacci import (
    fibonacci_levels,
    fibonacci_extensions,
    fibonacci_cluster,
    ote_zone,
    nearest_fib,
)

__all__ = [
    # trend
    "ema", "ema_fan", "ema_cross", "supertrend", "adx", "ichimoku",
    "trend_strength",
    # momentum
    "rsi", "rsi_divergence", "macd", "macd_divergence", "stochastic",
    "cci", "williams_r", "momentum_score",
    # volatility
    "atr", "bollinger", "bollinger_squeeze", "keltner", "volatility_regime",
    # volume
    "obv", "obv_divergence", "vwap", "vwap_bands", "mfi",
    "volume_profile", "volume_absorption", "volume_score",
    # fibonacci
    "fibonacci_levels", "fibonacci_extensions", "fibonacci_cluster",
    "ote_zone", "nearest_fib",
]
