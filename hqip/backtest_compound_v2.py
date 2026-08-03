#!/usr/bin/env python3
"""
HQIP Compound Backtest v2 — OPTIMIZED
Key fixes:
1. Higher confidence threshold (65%+)
2. Only trade when 3+ agents agree
3. Session filter (avoid Asia low volume)
4. Trailing stop after TP1
5. Max 3 trades/day
"""
import sys, os
sys.path.insert(0, "/data/workspace")
import pandas as pd
import numpy as np
from hqip.absolute_zero import AbsoluteZeroEngine, get_session
from hqip.agents.math_brain_agent import MathBrainAgent
from hqip.agents.game_theory_agent import GameTheoryAgent
from hqip.agents.trend_agent import TrendAgent
from hqip.agents.momentum_agent import MomentumAgent
from hqip.agents.volume_agent import VolumeAgent
from hqip.agents.volatility_agent import VolatilityAgent
from hqip.agents.smc_agent import SMCAgent
from hqip.agents.pattern_agent import PatternAgent
from hqip.indicators import calculate_all_indicators

DATA_DIR = "/data/workspace/hqip/data"
START_CAPITAL = 10.0
LEVERAGE = 20
MAX_RISK_PCT = 0.08  # 8% risk per trade
MAX_TRADES_DAY = 3
MIN_CONFIDENCE = 65
MIN_AGENTS = 3  # Need 3+ agents to agree

print("=" * 55)
print("  HQIP COMPOUND BACKTEST v2 — OPTIMIZED")
print(f"  Start: ${START_CAPITAL} | Leverage: {LEVERAGE}x")
print(f"  Risk: {MAX_RISK_PCT*100:.0f}%/trade | Max: {MAX_TRADES_DAY}/day")
print(f"  Min confidence: {MIN_CONFIDENCE}% | Min agents: {MIN_AGENTS}")
print("=" * 55)

df_15m = pd.read_csv(f"{DATA_DIR}/BTCUSDT_15m_binance.csv")
df_15m["timestamp"] = pd.to_datetime(df_15m["close_time"], unit="us")
df_15m.set_index("timestamp", inplace=True)

print(f"\n  Data: {len(df_15m)} candles | {df_15m.index[0]} to {df_15m.index[-1]}")
days = (df_15m.index[-1] - df_15m.index[0]).days

agents = {
    "MathBrain": MathBrainAgent(),
    "GameTheory": GameTheoryAgent(),
    "Trend": TrendAgent(),
    "Momentum": MomentumAgent(),
    "Volume": VolumeAgent(),
    "Volatility": VolatilityAgent(),
    "SMC": SMCAgent(),
    "Pattern": PatternAgent(),
}

capital = START_CAPITAL
trades = []
daily_count = {}
peak = capital
max_dd = 0
daily_caps = {}
consecutive_losses = 0

for i in range(200, len(df_15m) - 20, 16):
    ts = df_15m.index[i]
    day = str(ts.date())
    
    if daily_count.get(day, 0) >= MAX_TRADES_DAY:
        daily_caps[day] = capital
        continue
    if capital < 2.0:
        break
    
    # Cooldown after 3 consecutive losses
    if consecutive_losses >= 3:
        consecutive_losses = 0
        daily_caps[day] = capital
        continue
    
    df_slice = df_15m.iloc[max(0, i-200):i+1].copy()
    if len(df_slice) < 30:
        continue
    
    df_slice = calculate_all_indicators(df_slice)
    
    hour = ts.hour
    if 13 <= hour < 16: sess = "overlap"
    elif 0 <= hour < 8: sess = "asia"
    elif 7 <= hour < 16: sess = "europe"
    elif 13 <= hour < 22: sess = "america"
    else: sess = "off"
    
    # Skip Asia session (low volume = noise)
    if sess == "asia":
        daily_caps[day] = capital
        continue
    
    results = []
    for name, agent in agents.items():
        try:
            r = agent.analyze(df_slice, symbol="BTCUSDT", timeframe="15m", session_key=sess)
            results.append(r)
        except:
            pass
    
    buys = [r for r in results if r.direction == "BUY"]
    sells = [r for r in results if r.direction == "SELL"]
    
    # Require minimum agents
    if len(buys) >= MIN_AGENTS:
        direction = "BUY"
        avg_conf = np.mean([r.confidence for r in buys])
        avg_score = np.mean([r.score for r in buys])
    elif len(sells) >= MIN_AGENTS:
        direction = "SELL"
        avg_conf = np.mean([r.confidence for r in sells])
        avg_score = np.mean([abs(r.score) for r in sells])
    else:
        daily_caps[day] = capital
        continue
    
    if avg_conf < MIN_CONFIDENCE:
        daily_caps[day] = capital
        continue
    
    entry = float(df_slice.iloc[-1]["close"])
    atr = float(df_slice.iloc[-1]["atr"]) if "atr" in df_slice.columns else entry * 0.01
    sl_pct = max(0.8, min(1.2, atr / entry * 100 * 1.2))
    tp1_pct = sl_pct * 2
    tp2_pct = sl_pct * 3
    tp3_pct = sl_pct * 4
    
    risk = capital * MAX_RISK_PCT
    pos_val = min(risk / (sl_pct / 100), capital * LEVERAGE)
    
    if direction == "BUY":
        sl_p = entry * (1 - sl_pct/100)
        tp1 = entry * (1 + tp1_pct/100)
        tp2 = entry * (1 + tp2_pct/100)
        tp3 = entry * (1 + tp3_pct/100)
    else:
        sl_p = entry * (1 + sl_pct/100)
        tp1 = entry * (1 - tp1_pct/100)
        tp2 = entry * (1 - tp2_pct/100)
        tp3 = entry * (1 - tp3_pct/100)
    
    # Simulate 20 candles (5 hours)
    exit_p = entry
    exit_r = "TIMEOUT"
    tp1_hit = False
    
    for j in range(i+1, min(i+21, len(df_15m))):
        hi = float(df_15m.iloc[j]["high"])
        lo = float(df_15m.iloc[j]["low"])
        
        if direction == "BUY":
            if lo <= sl_p:
                exit_p, exit_r = sl_p, "SL"; break
            if hi >= tp3:
                exit_p, exit_r = tp3, "TP3"; break
            if hi >= tp2:
                exit_p, exit_r = tp2, "TP2"; break
            if hi >= tp1:
                # Trailing stop: move SL to breakeven
                sl_p = entry
                tp1_hit = True
        else:
            if hi >= sl_p:
                exit_p, exit_r = sl_p, "SL"; break
            if lo <= tp3:
                exit_p, exit_r = tp3, "TP3"; break
            if lo <= tp2:
                exit_p, exit_r = tp2, "TP2"; break
            if lo <= tp1:
                sl_p = entry
                tp1_hit = True
    
    if exit_r == "TIMEOUT":
        exit_p = float(df_15m.iloc[min(i+20, len(df_15m)-1)]["close"])
    
    if direction == "BUY":
        pct = (exit_p - entry) / entry
    else:
        pct = (entry - exit_p) / entry
    
    pnl = pos_val * pct
    if exit_r == "SL" and not tp1_hit:
        pnl = -risk
    pnl = max(pnl, -risk)
    pnl = min(pnl, risk * 4)
    
    old = capital
    capital = max(0.01, capital + pnl)
    daily_count[day] = daily_count.get(day, 0) + 1
    daily_caps[day] = capital
    
    if pnl < 0:
        consecutive_losses += 1
    else:
        consecutive_losses = 0
    
    if capital > peak: peak = capital
    dd = (peak - capital) / peak
    if dd > max_dd: max_dd = dd
    
    trades.append({
        "time": str(ts)[:16], "dir": direction, "entry": entry,
        "exit": exit_p, "pnl": pnl, "cap": capital, "reason": exit_r,
        "conf": avg_conf, "n_agents": len(buys) if direction == "BUY" else len(sells),
    })

# Results
wins = sum(1 for t in trades if t["pnl"] > 0)
losses = sum(1 for t in trades if t["pnl"] <= 0)

print(f"\n{'='*55}")
print(f"  RESULTS — {len(trades)} trades over {days} days")
print(f"{'='*55}")
print(f"  Start:    ${START_CAPITAL:.2f}")
print(f"  End:      ${capital:.2f}")
print(f"  Return:   {((capital-START_CAPITAL)/START_CAPITAL*100):+.1f}%")
print(f"  Trades:   {len(trades)} ({len(trades)/max(days,1):.1f}/day)")
print(f"  Wins:     {wins} ({wins/max(len(trades),1)*100:.0f}%)")
print(f"  Losses:   {losses} ({losses/max(len(trades),1)*100:.0f}%)")
print(f"  Max DD:   {max_dd*100:.1f}%")

if wins > 0:
    aw = np.mean([t['pnl'] for t in trades if t['pnl'] > 0])
    print(f"  Avg Win:  ${aw:.2f}")
if losses > 0:
    al = abs(np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]))
    print(f"  Avg Loss: ${al:.2f}")
    print(f"  R:R real: 1:{aw/al:.2f}")

# Growth milestones
print(f"\n  📈 Growth:")
for label, off in [("Week 1",7),("Week 2",14),("Week 4",28),("Week 8",56),("Week 12",84)]:
    target = df_15m.index[0].date() + pd.Timedelta(days=off)
    matching = [v for k,v in daily_caps.items() if k <= str(target)]
    if matching:
        print(f"    {label}: ${matching[-1]:.2f}")

# Top trades
print(f"\n  🏆 Top 10:")
for t in sorted(trades, key=lambda x: x["pnl"], reverse=True)[:10]:
    e = "+" if t["pnl"] > 0 else ""
    print(f"    {t['time'][:16]} {t['dir']:4} {t['reason']:5} {t['n_agents']}ag {t['conf']:.0f}% P&L:{e}{t['pnl']:.2f} -> ${t['cap']:.2f}")

# Last 10
print(f"\n  📊 Last 10:")
for t in trades[-10:]:
    e = "+" if t["pnl"] > 0 else ""
    print(f"    {t['time'][:16]} {t['dir']:4} {t['reason']:5} {t['n_agents']}ag {t['conf']:.0f}% P&L:{e}{t['pnl']:.2f} -> ${t['cap']:.2f}")

print(f"\n⚠️ گذشته ≠ آینده")
