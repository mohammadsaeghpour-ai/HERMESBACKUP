
"""Debug: Which filter kills the most signals?"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import requests, pandas as pd
from core import indicators as ind
from core.data_types import AgentOutput

from agents.stage0_independent.trend_agent import TrendAgent
from agents.stage0_independent.momentum_agent import MomentumAgent
from agents.stage0_independent.volume_agent import VolumeAgent
from agents.stage0_independent.volatility_agent import VolatilityAgent
from agents.stage0_independent.pattern_agent import PatternAgent
from agents.stage0_independent.dl_forecast_agent import DLForecastAgent
from agents.stage1_meta.regime_agent import RegimeAgent
from agents.stage1_meta.structure_agent import StructureAgent
from agents.stage1_meta.whale_agent import WhaleAgent
from agents.stage2_structure.smc_agent import SMCAgent
from agents.stage2_structure.liquidity_agent import LiquidityAgent
from agents.stage2_structure.wyckoff_agent import WyckoffAgent
from agents.stage2_structure.math_brain_agent import MathBrainAgent
from agents.stage3_decision.game_theory_agent import GameTheoryAgent
from agents.stage3_decision.smart_action_agent import SmartActionAgent
from agents.stage3_decision.ml_agent import MLAgent
from pipeline.probability import bayesian_combine, agent_to_prob, expected_value

def fetch_all(instId, bar, limit=300):
    r = requests.get("https://www.okx.com/api/v5/market/history-candles",
                     params={"instId": instId, "bar": bar, "limit": limit}, timeout=15)
    data = r.json()["data"]
    rows = [{"open": float(c[1]), "high": float(c[2]), "low": float(c[3]),
             "close": float(c[4]), "volume": float(c[5]), "ts": int(c[0])}
            for c in reversed(data)]
    return pd.DataFrame(rows)

agents = [TrendAgent(), MomentumAgent(), VolumeAgent(), VolatilityAgent(),
          PatternAgent(), DLForecastAgent(), RegimeAgent(), StructureAgent(),
          WhaleAgent(), SMCAgent(), LiquidityAgent(), WyckoffAgent(),
          MathBrainAgent(), GameTheoryAgent(), SmartActionAgent(), MLAgent()]

for instId in ["ETH-USDT-SWAP", "BTC-USDT-SWAP"]:
    df = fetch_all(instId, "15m", 280)
    h4_df = fetch_all(instId, "4H", 100)
    _, h4_st = ind.supertrend(h4_df)
    h4_dir = 1 if h4_df.iloc[-1]["close"] > h4_st.iloc[-1] else -1
    
    fail_counts = {"vote": 0, "4H": 0, "ADX": 0, "volume": 0, "EV": 0, "trap": 0, "PASS": 0}
    total_signals = 0
    
    for i in range(50, len(df) - 10):
        w = df.iloc[i-50:i+1].copy().reset_index(drop=True)
        
        results = []
        for a in agents:
            try: results.append(a.analyze(w, instId, "15m"))
            except: results.append(AgentOutput(name=a.name, direction="NEUTRAL"))
        
        buy_w = sum(r.weight for r in results if r.direction == "BUY")
        sell_w = sum(r.weight for r in results if r.direction == "SELL")
        tw = buy_w + sell_w if buy_w + sell_w > 0 else 1
        sig = "BUY" if buy_w > sell_w else ("SELL" if sell_w > buy_w else None)
        
        if sig is None:
            continue
        total_signals += 1
        
        # Gate 1: Vote
        if buy_w/tw < 0.85 and sell_w/tw < 0.85:
            fail_counts["vote"] += 1
            continue
        
        # Gate 2: 4H
        h4_ok = (h4_dir == 1 and sig == "BUY") or (h4_dir == -1 and sig == "SELL")
        if not h4_ok:
            fail_counts["4H"] += 1
            continue
        
        # Gate 3: ADX
        adx_v, _, _ = ind.adx(w)
        adx = adx_v.iloc[-1] if not pd.isna(adx_v.iloc[-1]) else 0
        if adx <= 28:
            fail_counts["ADX"] += 1
            continue
        
        # Gate 4: Volume
        vr = ind.volume_ratio(w).iloc[-1]
        if vr <= 1.0:
            fail_counts["volume"] += 1
            continue
        
        # Gate 6: EV
        priors = [agent_to_prob(r.direction, r.confidence, r.score) for r in results]
        weights = [r.weight for r in results]
        p_up, p_down = bayesian_combine(priors, weights)
        ev = expected_value(p_up if sig=="BUY" else p_down, p_down if sig=="BUY" else p_up, 3.0, 1.0)
        if ev <= 0:
            fail_counts["EV"] += 1
            continue
        
        # Gate 7: Trap
        if buy_w > 0 and sell_w == 0 or sell_w > 0 and buy_w == 0:
            fail_counts["trap"] += 1
            continue
        
        fail_counts["PASS"] += 1
    
    print("\n%s:" % instId)
    print("  Total directional signals: %d" % total_signals)
    for k, v in sorted(fail_counts.items(), key=lambda x: -x[1]):
        pct = v / total_signals * 100 if total_signals > 0 else 0
        print("  %s: %d (%.0f%%)" % (k, v, pct))
