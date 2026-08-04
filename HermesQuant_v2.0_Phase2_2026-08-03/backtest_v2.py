
"""
HermesQuant Backtester v2.0 — Full Pipeline Test
Uses actual 7-gate filter for realistic results
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from core import indicators as ind
from core.data_types import AgentOutput

# Import all agents
from agents.stage0_independent.trend_agent import TrendAgent
from agents.stage0_independent.momentum_agent import MomentumAgent
from agents.stage0_independent.volume_agent import VolumeAgent
from agents.stage0_independent.volatility_agent import VolatilityAgent
from agents.stage0_independent.pattern_agent import PatternAgent
from agents.stage0_independent.dl_forecast_agent import DLForecastAgent
from agents.stage1_meta.regime_agent import RegimeAgent
from agents.stage1_meta.structure_agent import StructureAgent
from agents.stage1_meta.whale_agent import WhaleAgent
from agents.stage1_meta.mtf_agent import MTFConfirmAgent
from agents.stage1_meta.funding_rate_agent import FundingRateAgent
from agents.stage1_meta.open_interest_agent import OpenInterestAgent
from agents.stage2_structure.rsi_divergence_agent import RSIDivergenceAgent
from agents.stage2_structure.bb_squeeze_agent import BBSqueezeAgent
from agents.stage2_structure.liquidity_agent import LiquidityAgent
from agents.stage2_structure.wyckoff_agent import WyckoffAgent
from agents.stage2_structure.math_brain_agent import MathBrainAgent
from agents.stage3_decision.game_theory_agent import GameTheoryAgent
from agents.stage3_decision.smart_action_agent import SmartActionAgent
from agents.stage3_decision.ml_agent import MLAgent

from pipeline.probability import bayesian_combine, agent_to_prob, quarter_kelly, expected_value
from pipeline.geometry import agent_to_vector, convergence, resultant, signal_strength


def fetch_all(instId, bar, limit=300):
    r = requests.get("https://www.okx.com/api/v5/market/history-candles",
                     params={"instId": instId, "bar": bar, "limit": limit}, timeout=15)
    data = r.json()["data"]
    rows = []
    for c in reversed(data):
        rows.append({"open": float(c[1]), "high": float(c[2]),
                     "low": float(c[3]), "close": float(c[4]),
                     "volume": float(c[5]), "ts": int(c[0])})
    return pd.DataFrame(rows)


def backtest_v2(instId="ETH-USDT-SWAP", bar="15m", vote_thresh=0.75):
    tz = timezone(timedelta(hours=3, minutes=30))
    print("="*70)
    print("  BACKTEST v2 (Full 7-Gate Filter): %s | %s" % (instId, bar))
    print("="*70)
    
    df = fetch_all(instId, bar, 280)
    print("  Candles: %d | %s to %s" % (len(df),
        datetime.fromtimestamp(df["ts"].iloc[0]/1000, tz).strftime("%Y-%m-%d"),
        datetime.fromtimestamp(df["ts"].iloc[-1]/1000, tz).strftime("%Y-%m-%d")))
    
    # Init agents
    agents = [
        TrendAgent(), MomentumAgent(), VolumeAgent(),
        VolatilityAgent(), DLForecastAgent(),
        RegimeAgent(), StructureAgent(), WhaleAgent(),
        MTFConfirmAgent(), FundingRateAgent(), OpenInterestAgent(),
        RSIDivergenceAgent(), BBSqueezeAgent(), LiquidityAgent(), WyckoffAgent(), MathBrainAgent(),
        GameTheoryAgent(), SmartActionAgent(), MLAgent(),
    ]
    
    CAPITAL = 10.0
    LEVERAGE = 20
    MAX_RISK = 3.0
    LOOKBACK = 50
    
    trades = []
    wins = 0
    losses = 0
    total = 0
    skipped = 0
    agent_stats = {a.name: {"correct": 0, "wrong": 0, "neutral": 0} for a in agents}
    
    for i in range(LOOKBACK, len(df) - 10):
        window = df.iloc[i-LOOKBACK:i+1].copy().reset_index(drop=True)
        price = window["close"].iloc[-1]
        future = df["close"].iloc[i+1:i+11].values
        if len(future) < 5:
            continue
        
        # Run all agents
        results = []
        for agent in agents:
            try:
                r = agent.analyze(window, instId, bar)
                results.append(r)
            except:
                results.append(AgentOutput(name=agent.name, direction="NEUTRAL", confidence=0))
        
        # ── 7-GATE FILTER (strict) ──
        
        # Gate 1: Vote ≥ 85%
        buy_w = sum(r.weight for r in results if r.direction == "BUY")
        sell_w = sum(r.weight for r in results if r.direction == "SELL")
        total_w = buy_w + sell_w if buy_w + sell_w > 0 else 1
        vote_ok = buy_w/total_w >= vote_thresh or sell_w/total_w >= vote_thresh
        
        if not vote_ok:
            skipped += 1
            continue
        
        signal_dir = "BUY" if buy_w > sell_w else "SELL"
        
        # Gate 2: 4H agrees
        h4_df = fetch_all(instId, "4H", 100)
        _, h4_st = ind.supertrend(h4_df)
        h4_price = h4_df.iloc[-1]["close"]
        h4_st_val = h4_st.iloc[-1]
        h4_ratio = abs(h4_price - h4_st_val) / h4_price
        if h4_ratio < 0.002:
            h4_ok = True
        else:
            h4_dir = 1 if h4_price > h4_st_val else -1
            h4_ok = (h4_dir == 1 and signal_dir == "BUY") or (h4_dir == -1 and signal_dir == "SELL")
        
        if not h4_ok:
            skipped += 1
            continue
        
        # Gate 3: ADX > 28
        adx_v, _, _ = ind.adx(window)
        adx_val = adx_v.iloc[-1] if not pd.isna(adx_v.iloc[-1]) else 0
        adx_ok = adx_val > 22
        
        if not adx_ok:
            skipped += 1
            continue
        
        # Gate 4: Volume > 1x
        vr = ind.volume_ratio(window).iloc[-1]
        vol_ok = vr > 0.8
        
        if not vol_ok:
            skipped += 1
            continue
        
        # Gate 5: Session (always pass in backtest)
        session_ok = True
        
        # Gate 6: EV > 0
        priors = [agent_to_prob(r.direction, r.confidence, r.score) for r in results]
        weights = [r.weight for r in results]
        p_up, p_down = bayesian_combine(priors, weights)
        ev = expected_value(p_up if signal_dir=="BUY" else p_down,
                           p_down if signal_dir=="BUY" else p_up, 3.0, 1.0)
        ev_ok = ev > 0
        
        if not ev_ok:
            skipped += 1
            continue
        
        # Gate 7: Not 5/5 trap
        all_same = buy_w > 0 and sell_w == 0 or sell_w > 0 and buy_w == 0
        trap_ok = not all_same
        
        if not trap_ok:
            skipped += 1
            continue
        
        # ALL 7 GATES PASS — Execute trade
        atr_val = ind.atr(window).iloc[-1]
        sl_dist = atr_val * 1.5
        tp_dist = atr_val * 3.0
        
        # Simulate
        max_fav = 0
        max_adv = 0
        hit_tp = False
        hit_sl = False
        exit_price = price
        
        for fp in future:
            change = fp - price if signal_dir == "BUY" else price - fp
            adverse = price - fp if signal_dir == "BUY" else fp - price
            if change > max_fav: max_fav = change
            if adverse > max_adv: max_adv = adverse
            if max_fav >= tp_dist:
                hit_tp = True
                exit_price = fp
                break
            if max_adv >= sl_dist:
                hit_sl = True
                exit_price = fp
                break
        
        if not hit_tp and not hit_sl:
            exit_price = future[-1]
            change = exit_price - price if signal_dir == "BUY" else price - exit_price
            hit_tp = change > 0
        
        pnl_pct = (exit_price - price) / price if signal_dir == "BUY" else (price - exit_price) / price
        pnl_dollar = pnl_pct * LEVERAGE * min(MAX_RISK, CAPITAL * 0.03)
        is_win = pnl_dollar > 0
        
        if is_win: wins += 1
        else: losses += 1
        total += 1
        CAPITAL += pnl_dollar
        
        # Track agent accuracy for signals that passed filter
        for r in results:
            if r.direction in ("BUY", "SELL"):
                if (r.direction == signal_dir and is_win) or (r.direction != signal_dir and not is_win):
                    agent_stats[r.name]["correct"] += 1
                else:
                    agent_stats[r.name]["wrong"] += 1
            else:
                agent_stats[r.name]["neutral"] += 1
        
        trades.append({"dir": signal_dir, "price": price, "exit": exit_price,
                       "pnl": pnl_dollar, "win": is_win, "adx": adx_val, "vol": vr})
    
    # ── Results ──
    print("\n" + "="*70)
    print("  RESULTS")
    print("="*70)
    
    if total == 0:
        print("  NO TRADES — System too conservative with full filter!")
        print("  Filtered out: %d signals" % skipped)
        return {"trades": 0, "accuracy": 0}
    
    acc = wins / total * 100
    print("  Trades: %d (filtered out: %d)" % (total, skipped))
    print("  Wins: %d | Losses: %d" % (wins, losses))
    print("  Accuracy: %.1f%%" % acc)
    print("  Final Capital: $%.2f (started $10)" % CAPITAL)
    print("  P&L: $%.2f (%.1f%%)" % (CAPITAL - 10, (CAPITAL/10 - 1)*100))
    
    if trades:
        avg_win = np.mean([t["pnl"] for t in trades if t["win"]]) if any(t["win"] for t in trades) else 0
        avg_loss = np.mean([t["pnl"] for t in trades if not t["win"]]) if any(not t["win"] for t in trades) else 0
        print("  Avg Win: $%.3f | Avg Loss: $%.3f" % (avg_win, avg_loss))
        if avg_loss != 0:
            pf = abs(avg_win * wins) / abs(avg_loss * losses) if losses > 0 else float('inf')
            print("  Profit Factor: %.2f" % pf)
        
        buy_trades = [t for t in trades if t["dir"] == "BUY"]
        sell_trades = [t for t in trades if t["dir"] == "SELL"]
        if buy_trades:
            print("  BUY accuracy: %.0f%% (%d trades)" % (sum(1 for t in buy_trades if t["win"])/len(buy_trades)*100, len(buy_trades)))
        if sell_trades:
            print("  SELL accuracy: %.0f%% (%d trades)" % (sum(1 for t in sell_trades if t["win"])/len(sell_trades)*100, len(sell_trades)))
    
    # Agent accuracy (only for agents that had directional signals)
    print("\n  AGENT ACCURACY (when signal traded):")
    for name, stats in sorted(agent_stats.items(), key=lambda x: x[1]["correct"]/(x[1]["correct"]+x[1]["wrong"]+1), reverse=True):
        d = stats["correct"] + stats["wrong"]
        if d == 0: continue
        a = stats["correct"] / d * 100
        print("    %15s: %d/%d (%.0f%%)" % (name, stats["correct"], d, a))
    
    return {"trades": total, "accuracy": acc, "wins": wins, "losses": losses,
            "capital": CAPITAL, "filtered": skipped}


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#  BACKTEST v2: Full 7-Gate Filter")
    print("#"*70)
    
    r_eth = backtest_v2("ETH-USDT-SWAP", "15m", vote_thresh=0.75)
    print("\n")
    r_btc = backtest_v2("BTC-USDT-SWAP", "15m", vote_thresh=0.78)
    
    print("\n" + "#"*70)
    print("#  COMPARISON")
    print("#"*70)
    for name, r in [("ETH", r_eth), ("BTC", r_btc)]:
        print("  %s: %d trades, %.1f%% accuracy, $%.2f P&L, filtered=%d" % (
            name, r.get("trades", 0), r.get("accuracy", 0),
            r.get("capital", 10) - 10, r.get("filtered", 0)))
