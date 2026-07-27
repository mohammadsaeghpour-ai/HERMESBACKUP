"""
HQIP - High Quality Intelligence Platform
Crypto trading intelligence system with multi-agent technical analysis.
"""

__version__ = "10.0.0"

from hqip.config import SYMBOLS, TIMEFRAMES, TF_WEIGHTS
from hqip.data_platform import DataPlatform
from hqip.indicators import calculate_all_indicators
from hqip.agents.trend_agent import TrendAgent
from hqip.agents.momentum_agent import MomentumAgent
from hqip.agents.volume_agent import VolumeAgent
from hqip.agents.volatility_agent import VolatilityAgent
from hqip.agents.pattern_agent import PatternAgent
from hqip.agents.market_structure_agent import MarketStructureAgent
from hqip.agents.regime_agent import RegimeAgent

__all__ = [
    "DataPlatform",
    "calculate_all_indicators",
    "TrendAgent",
    "MomentumAgent",
    "VolumeAgent",
    "VolatilityAgent",
    "PatternAgent",
    "MarketStructureAgent",
    "RegimeAgent",
    "SYMBOLS",
    "TIMEFRAMES",
    "TF_WEIGHTS",
]
