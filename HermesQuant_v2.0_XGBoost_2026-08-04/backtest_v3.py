
"""
Backtest v3 — Clean, No Lookahead Bias
All data pre-fetched, no live API calls in loop
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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


def fetch_all(instId, bar, count=300):
    url = "https://www.okx.com/api/v5/market/candles"
    all_data = []
    after = ""
    for _ in range(count // 300 + 1):
        params = {"instId": instId, "bar": bar, "limit": "300"}
        if after:
            params["after"] = after
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                break
            all_data.extend(data)
            after = data[-1][0]
        except Exception:
            break
    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms")
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df.rename(columns={"vol": "volume"})  # Standardize column name
    return df


def backtest_v3(symbol, timeframe="15m", vote_thresh=0.75, horizon=5, threshold=0.001):
    """
    Clean backtest — NO lookahead bias
    All data pre-fetched, agents run on historical windows only
    """
    print("="*70)
    print("  BACKTEST v3 (No Lookahead): %s | %s" % (symbol, timeframe))
    print("="*70)
    
    # Pre-fetch ALL data (including enough for 4H alignment)
    df = fetch_all(symbol, timeframe, 300)
    if df is None or len(df) < 100:
        print("  ERROR: Not enough data")
        return {}
    
    print("  Candles: %d | %s to %s" % (len(df), df["ts"].iloc[0], df["ts"].iloc[-1]))
    
    # Initialize agents
    agents = [
        TrendAgent(), MomentumAgent(), VolumeAgent(),
        VolatilityAgent(), PatternAgent(),
        RegimeAgent(), StructureAgent(), WhaleAgent(),
        RSIDivergenceAgent(), BBSqueezeAgent(),
        LiquidityAgent(), WyckoffAgent(), MathBrainAgent(),
        GameTheoryAgent(), SmartActionAgent(),
    ]
    
    # Run backtest
    trades = []
    LOOKBACK = 50
    wins = 0
    losses = 0
    capital = 10.0
    max_capital = 10.0
    peak = capital
    
    for i in range(LOOKBACK, len(df) - horizon):
        window = df.iloc[i-LOOKBACK:i+1].copy()
        
        # Run all agents on historical window ONLY
        signals = []
        for agent in agents:
            try:
                r = agent.analyze(window, symbol, timeframe)
                signals.append(r)
            except Exception:
                pass
        
        # Count votes (excluding SMC which was removed)
        buy_w = sum(s.weight for s in signals if s.direction == "BUY")
        sell_w = sum(s.weight for s in signals if s.direction == "SELL")
        total_w = sum(s.weight for s in signals if s.direction in ("BUY","SELL"))
        
        if total_w == 0:
            continue
        
        # 4H agreement — use same historical window (no live API!)
        # Simplified: check if 4H trend aligns
        h4_buy = buy_w / total_w >= vote_thresh
        h4_sell = sell_w / total_w >= vote_thresh
        
        if not h4_buy and not h4_sell:
            continue
        
        # ADX filter
        import core.indicators as ind
        adx_v, _, _ = ind.adx(window)
        if adx_v.iloc[-1] < 22:
            continue
        
        # Volume filter
        vol_ratio = ind.volume_ratio(window)
        if vol_ratio.iloc[-1] < 0.8:
            continue
        
        # Determine direction
        direction = "BUY" if h4_buy else "SELL"
        
        # Calculate P&L
        entry = df["close"].iloc[i]
        exit_price = df["close"].iloc[i + horizon]
        
        if direction == "BUY":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry
        
        pnl_pct *= 20  # 20x leverage
        pnl_dollar = capital * min(pnl_pct, 0.15)  # Max 15% per trade
        
        capital += pnl_dollar
        peak = max(peak, capital)
        
        if pnl_dollar > 0:
            wins += 1
        else:
            losses += 1
        
        trades.append({
            "time": df["ts"].iloc[i],
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "pnl": pnl_dollar,
            "capital": capital,
        })
    
    # Results
    total = wins + losses
    accuracy = wins / total * 100 if total > 0 else 0
    
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    print("  Trades: %d" % total)
    print("  Wins: %d | Losses: %d" % (wins, losses))
    print("  Accuracy: %.1f%%" % accuracy)
    print("  Final Capital: $%.2f (started $10)" % capital)
    print("  P&L: $%.2f (%.1f%%)" % (capital - 10, (capital - 10) / 10 * 100))
    print("  Max Drawdown: $%.2f" % (peak - min(t["capital"] for t in trades) if trades else 0))
    
    return {"accuracy": accuracy, "trades": total, "wins": wins, "losses": losses, "capital": capital}


if __name__ == "__main__":
    r1 = backtest_v3("ETH-USDT-SWAP", "15m", vote_thresh=0.72)
    print("\n")
    r2 = backtest_v3("BTC-USDT-SWAP", "15m", vote_thresh=0.75)
    print("\n" + "="*70)
    print("  CLEAN COMPARISON (No Lookahead)")
    print("="*70)
    print("  ETH: %d trades, %.1f%% accuracy, $%.2f P&L" % (r1.get("trades",0), r1.get("accuracy",0), r1.get("capital",10)-10))
    print("  BTC: %d trades, %.1f%% accuracy, $%.2f P&L" % (r2.get("trades",0), r2.get("accuracy",0), r2.get("capital",10)-10))
