
"""
HermesQuant Backtester v1.0
============================
Run pipeline on historical data for 1 month.
Identify problems, gaps, and improvement areas.
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

tz = timezone(timedelta(hours=3, minutes=30))

# ── Fetch historical data ──
def fetch_all(instId, bar, limit=300):
    r = requests.get("https://www.okx.com/api/v5/market/history-candles",
                     params={"instId": instId, "bar": bar, "limit": limit},
                     timeout=15)
    data = r.json()["data"]
    rows = []
    for c in reversed(data):
        rows.append({
            "open": float(c[1]), "high": float(c[2]),
            "low": float(c[3]), "close": float(c[4]),
            "volume": float(c[5]),
            "ts": int(c[0])
        })
    return pd.DataFrame(rows)

# ── Import system ──
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
from agents.stage4_risk.risk_agent import RiskAgent

from pipeline.probability import bayesian_combine, agent_to_prob, quarter_kelly, expected_value
from pipeline.geometry import agent_to_vector, convergence, resultant, signal_strength

# ── Backtest Engine ──
def backtest(instId="ETH-USDT-SWAP", bar="15m", max_bars=280):
    print("="*70)
    print("  BACKTEST: %s | %s | Last 30 days" % (instId, bar))
    print("="*70)
    
    # Fetch data
    df = fetch_all(instId, bar, max_bars)
    print("  Candles: %d" % len(df))
    print("  Date range: %s to %s" % (
        datetime.fromtimestamp(df["ts"].iloc[0]/1000, tz).strftime("%Y-%m-%d"),
        datetime.fromtimestamp(df["ts"].iloc[-1]/1000, tz).strftime("%Y-%m-%d")))
    
    # Init agents
    agents = [
        TrendAgent(), MomentumAgent(), VolumeAgent(),
        VolatilityAgent(), PatternAgent(), DLForecastAgent(),
        RegimeAgent(), StructureAgent(), WhaleAgent(),
        SMCAgent(), LiquidityAgent(), WyckoffAgent(), MathBrainAgent(),
        GameTheoryAgent(), SmartActionAgent(), MLAgent(),
    ]
    
    # Simulation
    CAPITAL = 10.0
    initial_capital = CAPITAL
    LEVERAGE = 20
    MAX_RISK = 3.0
    
    trades = []
    signals = []
    wins = 0
    losses = 0
    total_trades = 0
    skipped = 0
    errors = 0
    agent_correct = {a.name: {"buy_right": 0, "buy_wrong": 0, "sell_right": 0, "sell_wrong": 0} for a in agents}
    
    # Walk forward: need 50 bars for indicators
    LOOKBACK = 50
    
    for i in range(LOOKBACK, len(df) - 10):
        window = df.iloc[i-LOOKBACK:i+1].copy().reset_index(drop=True)
        current_price = window["close"].iloc[-1]
        future_prices = df["close"].iloc[i+1:i+11].values  # next 10 bars
        
        if len(future_prices) < 5:
            continue
        
        # Run all agents
        agent_results = []
        for agent in agents:
            try:
                r = agent.analyze(window, instId, bar)
                agent_results.append(r)
            except Exception as e:
                errors += 1
                agent_results.append(AgentOutput(name=agent.name, direction="NEUTRAL", confidence=0))
        
        # Vote
        buy_w = sum(r.weight for r in agent_results if r.direction == "BUY")
        sell_w = sum(r.weight for r in agent_results if r.direction == "SELL")
        total_w = buy_w + sell_w if buy_w + sell_w > 0 else 1
        
        # Determine signal
        if buy_w > sell_w and buy_w / total_w >= 0.55:
            signal_dir = "BUY"
        elif sell_w > buy_w and sell_w / total_w >= 0.55:
            signal_dir = "SELL"
        else:
            skipped += 1
            continue
        
        # Check filters
        atr_val = ind.atr(window).iloc[-1]
        adx_v, _, _ = ind.adx(window)
        adx_val = adx_v.iloc[-1] if not pd.isna(adx_v.iloc[-1]) else 0
        vr = ind.volume_ratio(window).iloc[-1]
        
        # Only trade if ADX > 20 (slightly relaxed for backtest)
        if adx_val < 20:
            skipped += 1
            continue
        
        # Calculate outcome
        max_favorable = 0
        max_adverse = 0
        exit_price = current_price
        hit_tp = False
        hit_sl = False
        bars_held = 0
        
        sl_distance = atr_val * 1.5
        tp_distance = atr_val * 3.0
        
        for j, fp in enumerate(future_prices):
            change = fp - current_price if signal_dir == "BUY" else current_price - fp
            adverse = current_price - fp if signal_dir == "BUY" else fp - current_price
            
            if change > max_favorable:
                max_favorable = change
            if adverse > max_adverse:
                max_adverse = adverse
            
            if max_favorable >= tp_distance:
                hit_tp = True
                exit_price = fp
                bars_held = j + 1
                break
            if max_adverse >= sl_distance:
                hit_sl = True
                exit_price = fp
                bars_held = j + 1
                break
        
        if not hit_tp and not hit_sl:
            # Exit at last bar
            exit_price = future_prices[-1]
            bars_held = len(future_prices)
            final_change = exit_price - current_price if signal_dir == "BUY" else current_price - exit_price
            if final_change > 0:
                hit_tp = True
            else:
                hit_sl = True
        
        # Calculate P&L
        if signal_dir == "BUY":
            pnl_pct = (exit_price - current_price) / current_price
        else:
            pnl_pct = (current_price - exit_price) / current_price
        
        pnl_dollar = pnl_pct * LEVERAGE * min(MAX_RISK, CAPITAL * 0.03)
        
        is_win = pnl_dollar > 0
        if is_win:
            wins += 1
        else:
            losses += 1
        total_trades += 1
        CAPITAL += pnl_dollar
        
        # Track agent accuracy
        for r in agent_results:
            if r.direction == "BUY" and is_win:
                agent_correct[r.name]["buy_right"] += 1
            elif r.direction == "BUY" and not is_win:
                agent_correct[r.name]["buy_wrong"] += 1
            elif r.direction == "SELL" and is_win:
                agent_correct[r.name]["sell_right"] += 1
            elif r.direction == "SELL" and not is_win:
                agent_correct[r.name]["sell_wrong"] += 1
        
        signals.append({
            "idx": i,
            "dir": signal_dir,
            "price": current_price,
            "exit": exit_price,
            "pnl": pnl_dollar,
            "win": is_win,
            "bars": bars_held,
            "adx": adx_val,
        })
    
    # ── Results ──
    print("\n" + "="*70)
    print("  BACKTEST RESULTS")
    print("="*70)
    
    if total_trades == 0:
        print("  NO TRADES! System too conservative.")
        print("  Need to relax filters.")
        return {"trades": 0, "accuracy": 0}
    
    accuracy = wins / total_trades * 100 if total_trades > 0 else 0
    
    print("  Total Trades: %d" % total_trades)
    print("  Wins: %d | Losses: %d" % (wins, losses))
    print("  Accuracy: %.1f%%" % accuracy)
    print("  Final Capital: $%.2f (started $%.2f)" % (CAPITAL, initial_capital))
    print("  P&L: $%.2f (%.1f%%)" % (CAPITAL - initial_capital, (CAPITAL/initial_capital - 1)*100))
    
    if signals:
        avg_bars = np.mean([s["bars"] for s in signals])
        avg_win = np.mean([s["pnl"] for s in signals if s["win"]]) if any(s["win"] for s in signals) else 0
        avg_loss = np.mean([s["pnl"] for s in signals if not s["win"]]) if any(not s["win"] for s in signals) else 0
        print("  Avg Bars Held: %.1f" % avg_bars)
        print("  Avg Win: $%.3f | Avg Loss: $%.3f" % (avg_win, avg_loss))
        if avg_loss != 0:
            print("  Profit Factor: %.2f" % (abs(avg_win * wins) / abs(avg_loss * losses) if losses > 0 else float('inf')))
    
    # Agent accuracy
    print("\n  AGENT ACCURACY:")
    for name, counts in sorted(agent_correct.items(), 
                                key=lambda x: sum(x[1].values()), reverse=True):
        total = sum(counts.values())
        if total == 0:
            continue
        right = counts["buy_right"] + counts["sell_right"]
        acc = right / total * 100 if total > 0 else 0
        print("    %15s: %d/%d (%.0f%%) [BUY: %d/%d, SELL: %d/%d]" % (
            name, right, total, acc,
            counts["buy_right"], counts["buy_right"]+counts["buy_wrong"],
            counts["sell_right"], counts["sell_right"]+counts["sell_wrong"]))
    
    # Problems
    print("\n  PROBLEMS IDENTIFIED:")
    if accuracy < 50:
        print("    1. Accuracy below 50% — filters too loose")
    if accuracy > 70:
        print("    1. Accuracy good but too few trades — filters too tight")
    if total_trades < 10:
        print("    2. Too few trades — system too conservative")
    if total_trades > 50:
        print("    2. Too many trades — system too aggressive")
    
    # Win by signal direction
    buy_signals = [s for s in signals if s["dir"] == "BUY"]
    sell_signals = [s for s in signals if s["dir"] == "SELL"]
    if buy_signals:
        buy_acc = sum(1 for s in buy_signals if s["win"]) / len(buy_signals) * 100
        print("    3. BUY accuracy: %.0f%% (%d trades)" % (buy_acc, len(buy_signals)))
    if sell_signals:
        sell_acc = sum(1 for s in sell_signals if s["win"]) / len(sell_signals) * 100
        print("    4. SELL accuracy: %.0f%% (%d trades)" % (sell_acc, len(sell_signals)))
    
    return {
        "trades": total_trades, "accuracy": accuracy,
        "wins": wins, "losses": losses,
        "capital": CAPITAL, "agent_correct": agent_correct,
        "signals": signals
    }


# ── Run backtest for both ──
if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#  LOOP 1: Initial Backtest — Identify Problems")
    print("#"*70)
    
    r_eth = backtest("ETH-USDT-SWAP", "15m", 280)
    print("\n")
    r_btc = backtest("BTC-USDT-SWAP", "15m", 280)
    
    print("\n" + "#"*70)
    print("#  ANALYSIS SUMMARY")
    print("#"*70)
    
    for name, r in [("ETH", r_eth), ("BTC", r_btc)]:
        print("\n  %s: %d trades, %.1f%% accuracy, $%.2f P&L" % (
            name, r.get("trades", 0), r.get("accuracy", 0),
            r.get("capital", 10) - 10))
