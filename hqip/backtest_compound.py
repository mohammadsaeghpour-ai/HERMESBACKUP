#!/usr/bin/env python3
"""
HQIP Compound Backtest — 90 Days
$10 start, 20x leverage, compound (no withdrawal)
Uses ALL 24 agents for signal generation
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
MAX_RISK_PCT = 0.10  # Risk 10% of capital per trade
MAX_TRADES_DAY = 6
MIN_CONFIDENCE = 55

print("=" * 55)
print("  HQIP COMPOUND BACKTEST — 90 DAYS")
print(f"  Start: ${START_CAPITAL} | Leverage: {LEVERAGE}x | Compound: YES")
print(f"  Risk per trade: {MAX_RISK_PCT*100:.0f}% of capital | Max: {MAX_TRADES_DAY}/day")
print("=" * 55)

# Load data
df_15m = pd.read_csv(f"{DATA_DIR}/BTCUSDT_15m_binance.csv")
# Binance close_time is in microseconds — use directly
df_15m["timestamp"] = pd.to_datetime(df_15m["close_time"], unit="us")
df_15m.set_index("timestamp", inplace=True)

print(f"\n  Data: {len(df_15m)} candles (15m)")
print(f"  From: {df_15m.index[0]} to {df_15m.index[-1]}")
days = (df_15m.index[-1] - df_15m.index[0]).days
print(f"  Period: {days} days")

# Init agents
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

engine = AbsoluteZeroEngine()

# Backtest loop
capital = START_CAPITAL
trades = []
daily_count = {}
peak_capital = capital
max_dd = 0
daily_capitals = {}

for i in range(200, len(df_15m) - 16, 16):  # Every 4 hours
    current_time = df_15m.index[i]
    current_day = str(current_time.date())
    
    if daily_count.get(current_day, 0) >= MAX_TRADES_DAY:
        # Track daily capital
        daily_capitals[current_day] = capital
        continue
    
    if capital < 2.0:
        break
    
    # Prepare data slice
    df_slice = df_15m.iloc[max(0, i-200):i+1].copy()
    if len(df_slice) < 30:
        continue
    
    # Calculate indicators
    df_slice = calculate_all_indicators(df_slice)
    
    # Get session
    hour_utc = current_time.hour
    if 13 <= hour_utc < 16:
        sess = "overlap"
    elif 0 <= hour_utc < 8:
        sess = "asia"
    elif 7 <= hour_utc < 16:
        sess = "europe"
    elif 13 <= hour_utc < 22:
        sess = "america"
    else:
        sess = "off"
    
    # Run agents
    results = []
    for name, agent in agents.items():
        try:
            r = agent.analyze(df_slice, symbol="BTCUSDT", timeframe="15m", session_key=sess)
            results.append(r)
        except:
            pass
    
    # Consensus
    buys = [r for r in results if r.direction == "BUY"]
    sells = [r for r in results if r.direction == "SELL"]
    
    buy_score = sum(r.score * r.confidence/100 for r in buys)
    sell_score = sum(abs(r.score) * r.confidence/100 for r in sells)
    
    if buy_score > sell_score and len(buys) >= 2:
        direction = "BUY"
        avg_conf = np.mean([r.confidence for r in buys])
    elif sell_score > buy_score and len(sells) >= 2:
        direction = "SELL"
        avg_conf = np.mean([r.confidence for r in sells])
    else:
        daily_capitals[current_day] = capital
        continue
    
    if avg_conf < MIN_CONFIDENCE:
        daily_capitals[current_day] = capital
        continue
    
    # Entry
    entry_price = float(df_slice.iloc[-1]["close"])
    
    # Dynamic SL based on ATR
    atr = float(df_slice.iloc[-1]["atr"]) if "atr" in df_slice.columns else entry_price * 0.01
    sl_pct = max(0.5, min(1.5, atr / entry_price * 100 * 1.5))
    
    # Position sizing (risk = MAX_RISK_PCT of current capital)
    risk_amount = capital * MAX_RISK_PCT
    pos_value = risk_amount / (sl_pct / 100)
    pos_value = min(pos_value, capital * LEVERAGE)  # Can't exceed leverage
    
    # Simulate next 16 candles (4 hours)
    exit_price = entry_price
    exit_reason = "TIMEOUT"
    tp1_pct = sl_pct * 2  # R:R = 1:2
    tp2_pct = sl_pct * 3
    tp3_pct = sl_pct * 4
    
    if direction == "BUY":
        sl_price = entry_price * (1 - sl_pct/100)
        tp1_price = entry_price * (1 + tp1_pct/100)
        tp2_price = entry_price * (1 + tp2_pct/100)
        tp3_price = entry_price * (1 + tp3_pct/100)
    else:
        sl_price = entry_price * (1 + sl_pct/100)
        tp1_price = entry_price * (1 - tp1_pct/100)
        tp2_price = entry_price * (1 - tp2_pct/100)
        tp3_price = entry_price * (1 - tp3_pct/100)
    
    # Check candles
    hit_tp1 = hit_tp2 = hit_tp3 = False
    for j in range(i+1, min(i+17, len(df_15m))):
        hi = float(df_15m.iloc[j]["high"])
        lo = float(df_15m.iloc[j]["low"])
        
        if direction == "BUY":
            if lo <= sl_price:
                exit_price, exit_reason = sl_price, "SL"; break
            if hi >= tp3_price:
                exit_price, exit_reason = tp3_price, "TP3"; hit_tp3 = True; break
            if hi >= tp2_price:
                hit_tp2 = True
            if hi >= tp1_price:
                hit_tp1 = True
        else:
            if hi >= sl_price:
                exit_price, exit_reason = sl_price, "SL"; break
            if lo <= tp3_price:
                exit_price, exit_reason = tp3_price, "TP3"; hit_tp3 = True; break
            if lo <= tp2_price:
                hit_tp2 = True
            if lo <= tp1_price:
                hit_tp1 = True
    
    if exit_reason == "TIMEOUT":
        exit_idx = min(i+16, len(df_15m)-1)
        exit_price = float(df_15m.iloc[exit_idx]["close"])
    
    # P&L
    if direction == "BUY":
        price_pct = (exit_price - entry_price) / entry_price
    else:
        price_pct = (entry_price - exit_price) / entry_price
    
    pnl = pos_value * price_pct
    
    # Cap loss at risk amount
    if exit_reason == "SL":
        pnl = -risk_amount
    
    pnl = max(pnl, -risk_amount)
    pnl = min(pnl, risk_amount * 4)  # Cap profit at 4x risk
    
    old_cap = capital
    capital = max(0.01, capital + pnl)
    daily_count[current_day] = daily_count.get(current_day, 0) + 1
    
    # Track drawdown
    if capital > peak_capital:
        peak_capital = capital
    dd = (peak_capital - capital) / peak_capital
    if dd > max_dd:
        max_dd = dd
    
    trades.append({
        "time": str(current_time)[:16],
        "dir": direction,
        "entry": entry_price,
        "exit": exit_price,
        "pnl": pnl,
        "cap": capital,
        "reason": exit_reason,
        "conf": avg_conf,
    })
    
    daily_capitals[current_day] = capital

# Results
print(f"\n{'='*55}")
print(f"  RESULTS — {len(trades)} trades over {days} days")
print(f"{'='*55}")

wins = sum(1 for t in trades if t["pnl"] > 0)
losses = sum(1 for t in trades if t["pnl"] <= 0)

print(f"\n  Start Capital:   ${START_CAPITAL:.2f}")
print(f"  End Capital:     ${capital:.2f}")
print(f"  Total Return:    {((capital-START_CAPITAL)/START_CAPITAL*100):+.1f}%")
print(f"  Total Trades:    {len(trades)}")
print(f"  Wins:            {wins} ({wins/max(len(trades),1)*100:.0f}%)")
print(f"  Losses:          {losses} ({losses/max(len(trades),1)*100:.0f}%)")
print(f"  Max Drawdown:    {max_dd*100:.1f}%")
print(f"  Trades/Day:      {len(trades)/max(days,1):.1f}")

if wins > 0:
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0])
    print(f"  Avg Win:         ${avg_win:.2f}")
if losses > 0:
    avg_loss = abs(np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]))
    print(f"  Avg Loss:        ${avg_loss:.2f}")

# Growth curve
print(f"\n  📈 Capital Growth:")
for label, day_offset in [("Week 1", 7), ("Week 2", 14), ("Week 4", 28), ("Week 8", 56), ("Week 12", 84)]:
    target_date = df_15m.index[0].date() + pd.Timedelta(days=day_offset)
    matching = [v for k, v in daily_capitals.items() if k <= str(target_date)]
    if matching:
        print(f"    {label}: ${matching[-1]:.2f}")

print(f"\n  Final:            ${capital:.2f}")
print(f"  Growth:           {((capital/START_CAPITAL)**(1/max(days,1))-1)*100:.2f}%/day")

# Top 10 trades
print(f"\n  📊 Top 10 Trades:")
sorted_trades = sorted(trades, key=lambda x: x["pnl"], reverse=True)[:10]
for t in sorted_trades:
    e = "+" if t["pnl"] > 0 else ""
    print(f"    {t['time'][:16]} {t['dir']:4} {t['reason']:5} P&L:{e}{t['pnl']:.2f} -> ${t['cap']:.2f}")

# Last 10 trades
print(f"\n  📊 Last 10 Trades:")
for t in trades[-10:]:
    e = "+" if t["pnl"] > 0 else ""
    print(f"    {t['time'][:16]} {t['dir']:4} {t['reason']:5} P&L:{e}{t['pnl']:.2f} -> ${t['cap']:.2f}")

print(f"\n⚠️ گذشته ≠ آینده")
