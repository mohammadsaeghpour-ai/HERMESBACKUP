"""
Agent Correlation Analysis
Identifies which agents are redundant/correlated
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from core.data_fetcher import fetch_candles

# Import all agents
from agents.stage0_independent.trend_agent import TrendAgent
from agents.stage0_independent.momentum_agent import MomentumAgent
from agents.stage0_independent.volume_agent import VolumeAgent
from agents.stage0_independent.volatility_agent import VolatilityAgent
from agents.stage0_independent.pattern_agent import PatternAgent
from agents.stage1_meta.regime_agent import RegimeAgent
from agents.stage1_meta.structure_agent import StructureAgent
from agents.stage1_meta.whale_agent import WhaleAgent
from agents.stage2_structure.rsi_divergence_agent import RSIDivergenceAgent
from agents.stage2_structure.bb_squeeze_agent import BBSqueezeAgent
from agents.stage2_structure.liquidity_agent import LiquidityAgent
from agents.stage2_structure.wyckoff_agent import WyckoffAgent
from agents.stage2_structure.math_brain_agent import MathBrainAgent
from agents.stage3_decision.game_theory_agent import GameTheoryAgent
from agents.stage3_decision.smart_action_agent import SmartActionAgent


def analyze_correlation(symbol="ETH-USDT-SWAP", timeframe="15m", n_candles=200):
    """Run all agents on historical data and compute correlation matrix"""
    
    df = fetch_candles(symbol, timeframe, n_candles)
    if df is None or len(df) < 100:
        print("ERROR: Not enough data")
        return
    
    agents = [
        TrendAgent(), MomentumAgent(), VolumeAgent(),
        VolatilityAgent(), PatternAgent(),
        RegimeAgent(), StructureAgent(), WhaleAgent(),
        RSIDivergenceAgent(), BBSqueezeAgent(),
        LiquidityAgent(), WyckoffAgent(), MathBrainAgent(),
        GameTheoryAgent(), SmartActionAgent(),
    ]
    
    # Run agents on sliding windows
    LOOKBACK = 50
    results = {a.name: [] for a in agents}
    actual_directions = []
    
    for i in range(LOOKBACK, min(len(df) - 5, LOOKBACK + 150)):
        window = df.iloc[i-LOOKBACK:i+1]
        
        for agent in agents:
            try:
                r = agent.analyze(window, symbol, timeframe)
                # Convert direction to numeric: BUY=1, SELL=-1, NEUTRAL=0
                if r.direction == "BUY":
                    val = r.confidence / 100
                elif r.direction == "SELL":
                    val = -(r.confidence / 100)
                else:
                    val = 0
                results[agent.name].append(val)
            except Exception:
                results[agent.name].append(0)
        
        # Actual direction (next 5 candles)
        if i + 5 < len(df):
            future = (df["close"].iloc[i+5] - df["close"].iloc[i]) / df["close"].iloc[i]
            actual_directions.append(1 if future > 0.001 else (-1 if future < -0.001 else 0))
    
    # Create correlation matrix
    agent_names = list(results.keys())
    corr_data = pd.DataFrame(results)
    
    # Compute correlation with actual direction
    actual_series = pd.Series(actual_directions[:len(corr_data)])
    
    print("="*70)
    print("  AGENT CORRELATION ANALYSIS: %s" % symbol)
    print("="*70)
    
    # Correlation with actual outcome
    print("\n  Correlation with actual direction:")
    print("  " + "-"*40)
    correlations = {}
    for name in agent_names:
        if len(corr_data[name]) == len(actual_series):
            corr = corr_data[name].corr(actual_series)
            correlations[name] = corr
            print("    %20s: %+.3f" % (name, corr))
    
    # Inter-agent correlation
    print("\n  High inter-agent correlations (>0.5):")
    print("  " + "-"*40)
    corr_matrix = corr_data.corr()
    pairs = []
    for i in range(len(agent_names)):
        for j in range(i+1, len(agent_names)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.5:
                pairs.append((agent_names[i], agent_names[j], c))
                print("    %s <-> %s: %.3f" % (agent_names[i], agent_names[j], c))
    
    if not pairs:
        print("    None found")
    
    # Summary
    print("\n  SUMMARY:")
    print("  " + "-"*40)
    
    # Best agents (highest positive correlation with actual)
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print("  Best agents (positive correlation):")
    for name, corr in sorted_corr[:5]:
        print("    %20s: %+.3f" % (name, corr))
    
    print("\n  Worst agents (negative/zero correlation):")
    for name, corr in sorted_corr[-5:]:
        print("    %20s: %+.3f" % (name, corr))
    
    # Redundant agents
    print("\n  Potentially redundant (high inter-correlation):")
    redundant = set()
    for n1, n2, c in pairs:
        if abs(c) > 0.7:
            redundant.add(n2)  # Mark second as redundant
    for name in redundant:
        print("    %s (consider removing)" % name)
    
    return correlations, pairs


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ETH ANALYSIS")
    print("="*70)
    eth_corr, eth_pairs = analyze_correlation("ETH-USDT-SWAP")
    
    print("\n" + "="*70)
    print("  BTC ANALYSIS")
    print("="*70)
    btc_corr, btc_pairs = analyze_correlation("BTC-USDT-SWAP")
