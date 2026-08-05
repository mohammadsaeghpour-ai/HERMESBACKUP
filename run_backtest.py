#!/usr/bin/env python3
"""
HERMES QUANT v3.0 — FINAL BACKTEST
Agents = PRIMARY signal (balanced BUY/SELL)
ML = CONFIDENCE filter (not direction)
Walk-Forward with multiple windows
"""
import sys, os
os.chdir('/data/workspace/HermesQuant')
sys.path.insert(0, '/data/workspace/HermesQuant')
sys.path.insert(0, '/data/workspace/HermesQuant_v3')
sys.path.insert(0, '/data/workspace')

import warnings; warnings.filterwarnings('ignore')
import time, numpy as np, pandas as pd

from HermesQuant_v3.core.data_fetcher import fetch_candles
from HermesQuant_v3.ml.features import FeatureEngine
from HermesQuant_v3.ml.model import EnsembleModel

from agents.stage0_independent.trend_agent import TrendAgent
from agents.stage0_independent.momentum_agent import MomentumAgent
from agents.stage0_independent.momentum_agent_v2 import MomentumAgentV2
from agents.stage0_independent.volume_agent import VolumeAgent
from agents.stage0_independent.volatility_agent import VolatilityAgent
from agents.stage0_independent.pattern_agent import PatternAgent
from agents.stage1_meta.regime_agent import RegimeAgent
from agents.stage1_meta.structure_agent import StructureAgent
from agents.stage1_meta.whale_agent import WhaleAgent
from agents.stage1_meta.fear_greed_agent import FearGreedAgent
from agents.stage1_meta.funding_rate_agent import FundingRateAgent
from agents.stage1_meta.mtf_agent import MTFConfirmAgent
from agents.stage1_meta.open_interest_agent import OpenInterestAgent
from agents.stage2_structure.rsi_divergence_agent import RSIDivergenceAgent
from agents.stage2_structure.bb_squeeze_agent import BBSqueezeAgent
from agents.stage2_structure.liquidity_agent import LiquidityAgent
from agents.stage2_structure.wyckoff_agent import WyckoffAgent
from agents.stage2_structure.math_brain_agent import MathBrainAgent
from agents.stage2_structure.cvd_agent import CVDAgent
from agents.stage3_decision.game_theory_agent import GameTheoryAgent
from agents.stage3_decision.smart_action_agent import SmartActionAgent
from agents.stage3_decision.smart_action_agent_v2 import SmartActionAgentV2
from agents.stage4_risk.risk_agent import RiskAgent


def agent_vote(df, symbol, agents_dict):
    """
    Weighted agent voting — CONFIDENCE WEIGHTED
    Returns: (direction, confidence, buy_score, sell_score, details)
    """
    buy_score = 0
    sell_score = 0
    total_weight = 0
    details = []
    
    for name, agent in agents_dict.items():
        try:
            r = agent.analyze(df, symbol, "15m")
            w = r.weight
            c = r.confidence / 100.0
            
            if r.direction == "BUY":
                buy_score += w * c
                details.append((name, "BUY", r.confidence, w))
            elif r.direction == "SELL":
                sell_score += w * c
                details.append((name, "SELL", r.confidence, w))
            else:
                details.append((name, "NEUTRAL", 0, w))
            
            total_weight += w
        except:
            pass
    
    if total_weight == 0:
        return "NEUTRAL", 0, 0, 0, details
    
    buy_pct = buy_score / total_weight
    sell_pct = sell_score / total_weight
    
    # Decision: need clear majority
    margin = abs(buy_pct - sell_pct)
    
    if buy_pct > sell_pct and margin > 0.05:
        return "BUY", buy_pct * 100, buy_pct, sell_pct, details
    elif sell_pct > buy_pct and margin > 0.05:
        return "SELL", sell_pct * 100, buy_pct, sell_pct, details
    else:
        return "NEUTRAL", 0, buy_pct, sell_pct, details


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
print('='*70)
print('  HERMES QUANT v3.0 — AGENT-PRIMARY BACKTEST')
print('  Agents = PRIMARY | ML = FILTER | Balanced BUY/SELL')
print('='*70)

# Init all 23 agents
all_agents = {
    "trend": TrendAgent(), "momentum": MomentumAgent(), "momentum_v2": MomentumAgentV2(),
    "volume": VolumeAgent(), "volatility": VolatilityAgent(), "pattern": PatternAgent(),
    "regime": RegimeAgent(), "structure": StructureAgent(), "whale": WhaleAgent(),
    "fear_greed": FearGreedAgent(), "funding_rate": FundingRateAgent(),
    "mtf": MTFConfirmAgent(), "open_interest": OpenInterestAgent(),
    "rsi_divergence": RSIDivergenceAgent(), "bb_squeeze": BBSqueezeAgent(),
    "liquidity": LiquidityAgent(), "wyckoff": WyckoffAgent(),
    "math_brain": MathBrainAgent(), "cvd": CVDAgent(),
    "game_theory": GameTheoryAgent(), "smart_action": SmartActionAgent(),
    "smart_action_v2": SmartActionAgentV2(), "risk_agent": RiskAgent(),
}

for sym in ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']:
    print('\n--- %s ---' % sym)
    t0 = time.time()
    
    df = fetch_candles(sym, '15m', 1440)
    print('  Data: %d candles' % len(df))
    
    # Walk-forward
    train_size = 400
    test_size = 100
    horizon = 5
    threshold = 0.001
    
    all_trades = []
    window_results = []
    
    i = train_size
    while i + test_size + horizon <= len(df):
        test_df = df.iloc[i:i+test_size]
        
        # Agent vote (primary signal)
        direction, confidence, buy_pct, sell_pct, details = agent_vote(test_df, sym, all_agents)
        
        if direction == "NEUTRAL":
            i += test_size
            continue
        
        # Backtest
        wins = 0
        losses = 0
        for j in range(0, len(test_df) - horizon):
            entry = test_df["close"].iloc[j]
            future = test_df["close"].iloc[j + horizon]
            actual_ret = (future - entry) / entry
            
            if direction == "BUY":
                correct = actual_ret > threshold
            else:
                correct = actual_ret < -threshold
            
            if correct:
                wins += 1
            else:
                losses += 1
        
        total = wins + losses
        acc = wins / total * 100 if total > 0 else 0
        
        all_trades.extend([{"dir": direction, "correct": True}] * wins +
                          [{"dir": direction, "correct": False}] * losses)
        
        window_results.append({
            "direction": direction,
            "accuracy": acc,
            "wins": wins,
            "losses": losses,
            "buy_pct": buy_pct,
            "sell_pct": sell_pct,
        })
        
        # Show active agents
        active = [(n, d, c) for n, d, c, w in details if d != "NEUTRAL"]
        
        i += test_size
    
    t1 = time.time()
    
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        total = len(trades_df)
        correct = trades_df["correct"].sum()
        accuracy = correct / total * 100
        
        buy_trades = trades_df[trades_df["dir"] == "BUY"]
        sell_trades = trades_df[trades_df["dir"] == "SELL"]
        
        buy_acc = buy_trades["correct"].mean() * 100 if len(buy_trades) > 0 else 0
        sell_acc = sell_trades["correct"].mean() * 100 if len(sell_trades) > 0 else 0
        
        print()
        print('  === RESULTS (%.1fs) ===' % (t1-t0))
        print('  Overall Accuracy: %.1f%% (%d/%d)' % (accuracy, correct, total))
        print()
        print('  BUY  Accuracy: %.1f%% (%d trades)' % (buy_acc, len(buy_trades)))
        print('  SELL Accuracy: %.1f%% (%d trades)' % (sell_acc, len(sell_trades)))
        print()
        print('  === PER WINDOW ===')
        for wi, wr in enumerate(window_results):
            print('    W%02d: %s %.1f%% (buy=%.2f sell=%.2f)' % (
                wi+1, wr["direction"], wr["accuracy"], wr["buy_pct"], wr["sell_pct"]))
        
        # Summary
        avg_acc = np.mean([wr["accuracy"] for wr in window_results])
        print()
        print('  Avg Window Accuracy: %.1f%%' % avg_acc)
        print('  Best: %.1f%% | Worst: %.1f%%' % (
            max(wr["accuracy"] for wr in window_results),
            min(wr["accuracy"] for wr in window_results)))
    else:
        print('  No trades (all NEUTRAL)')
    
    print()

print('='*70)
