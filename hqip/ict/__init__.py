"""
ICT (Inner Circle Trader) Module for HQIP v3 Trading System.

Provides institutional-grade analysis concepts:
- Order Blocks (supply/demand zones where institutional orders cluster)
- Fair Value Gaps (price imbalances / inefficiencies)
- Market Structure (BOS, CHoCH, swing points)
- Liquidity (buy-side/sell-side pools, sweeps, stop clusters)
- Premium/Discount Arrays (PD zones for optimal entries)
- Kill Zones (high-probability institutional trading windows)
"""

from .order_blocks import detect_bull_ob, detect_bear_ob
from .fvg import detect_bull_fvg, detect_bear_fvg
from .structure import detect_swings, detect_bos, detect_choch, analyze_structure
from .liquidity import find_bsl, find_ssl, detect_liquidity_sweep, estimate_stop_clusters
from .pd_arrays import premium_discount, equilibrium_price, get_pd_levels
from .killzones import get_killzone, is_killzone_active, killzone_bias

__all__ = [
    # Order Blocks
    "detect_bull_ob",
    "detect_bear_ob",
    # Fair Value Gaps
    "detect_bull_fvg",
    "detect_bear_fvg",
    # Market Structure
    "detect_swings",
    "detect_bos",
    "detect_choch",
    "analyze_structure",
    # Liquidity
    "find_bsl",
    "find_ssl",
    "detect_liquidity_sweep",
    "estimate_stop_clusters",
    # Premium/Discount Arrays
    "premium_discount",
    "equilibrium_price",
    "get_pd_levels",
    # Kill Zones
    "get_killzone",
    "is_killzone_active",
    "killzone_bias",
]
