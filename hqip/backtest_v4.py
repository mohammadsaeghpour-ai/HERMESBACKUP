#!/usr/bin/env python3
"""HQIP Backtest v4 — Multi-Signal with Per-TF Strategies"""
import sys
sys.path.insert(0, "/data/workspace")
import ccxt, pandas as pd, numpy as np
from hqip.strategies import get_all_strategies
from hqip.indicators import calculate_all_indicators

CCXT_SYMBOL = "BTC/USDT"
CAPITAL_START = 10.0
LEVERAGE = 20
SL_PCT_CAPITAL = 0.10
MAX_TRADES_DAY = 3

print("=" * 60)
print("  🧪 HQIP BACKTEST v4 — Multi-Signal Strategies")
print("=" * 60)

# Fetch 15m data
ex = ccxt.okx()
ohlcv = ex.fetch_ohlcv(CCXT_SYMBOL, '15m', limit=300)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = calculate_all_indicators(df)

print(f"  📡 {len(df)} candles: {df.index[0]} → {df.index[-1]}")
days = (df.index[-1] - df.index[0]).total_seconds() / 86400

# Initialize strategies
strategies = get_all_strategies()

capital = CAPITAL_START
trades = []
daily_count = {}

# Scan every 4 hours (16 candles)
for i in range(60, len(df) - 8, 16):
    current_time = df.index[i]
    current_day = str(current_time.date())
    
    if daily_count.get(current_day, 0) >= MAX_TRADES_DAY:
        continue
    if capital < 1.0:
        break
    
    # Get 15m slice
    df_slice = df.iloc[:i+1].copy()
    price = float(df_slice.iloc[-1]["close"])
    
    # Run 15m strategy (primary — we scan every 15m via 4h intervals)
    strat_15m = strategies["15m"]
    result = strat_15m.analyze(df_slice, "BTCUSDT")
    
    direction = result.get("direction", "NO_TRADE")
    confidence = result.get("confidence", 0)
    
    if direction == "NO_TRADE" or confidence < 55:
        continue
    
    entry_price = result.get("entry", price)
    sl_price = result.get("sl", price)
    tp1_price = result.get("tp1", price)
    tp2_price = result.get("tp2", price)
    tp3_price = result.get("tp3", price)
    
    # Simulate trade (next 4-8 candles = 1-2 hours)
    exit_price = entry_price
    exit_reason = "TIMEOUT"
    
    for j in range(i+1, min(i+9, len(df))):
        hi = float(df.iloc[j]["high"])
        lo = float(df.iloc[j]["low"])
        
        if direction == "BUY":
            if lo <= sl_price:
                exit_price, exit_reason = sl_price, "SL"; break
            if hi >= tp1_price:
                exit_price, exit_reason = tp1_price, "TP1"; break
            if hi >= tp2_price:
                exit_price, exit_reason = tp2_price, "TP2"; break
            if hi >= tp3_price:
                exit_price, exit_reason = tp3_price, "TP3"; break
        else:
            if hi >= sl_price:
                exit_price, exit_reason = sl_price, "SL"; break
            if lo <= tp1_price:
                exit_price, exit_reason = tp1_price, "TP1"; break
            if lo <= tp2_price:
                exit_price, exit_reason = tp2_price, "TP2"; break
            if lo <= tp3_price:
                exit_price, exit_reason = tp3_price, "TP3"; break
    
    if exit_reason == "TIMEOUT":
        exit_price = float(df.iloc[min(i+8, len(df)-1)]["close"])
    
    # P&L
    if direction == "BUY":
        pct = (exit_price - entry_price) / entry_price
    else:
        pct = (entry_price - exit_price) / entry_price
    
    risk = capital * SL_PCT_CAPITAL
    sl_dist_pct = abs(sl_price - entry_price) / entry_price if entry_price > 0 else 0.003
    pos_val = min((risk / max(sl_dist_pct, 0.001)) * LEVERAGE, capital * LEVERAGE)
    pnl = pct * pos_val / LEVERAGE
    pnl = np.clip(pnl, -risk, risk * 5)
    if exit_reason == "SL":
        pnl = -risk
    
    old_cap = capital
    capital = max(0.01, capital + pnl)
    daily_count[current_day] = daily_count.get(current_day, 0) + 1
    
    trades.append({
        "time": str(current_time)[:16], "dir": direction,
        "entry": entry_price, "exit": exit_price,
        "pnl": pnl, "cap": capital, "reason": exit_reason,
        "conf": confidence, "sl_pct": result.get("sl_pct", 0),
    })
    
    e = "🟢" if pnl > 0 else "🔴"
    print(f"  {e} {current_time} | {direction:4} | ${entry_price:,.0f} → ${exit_price:,.0f} | "
          f"P&L: {'+'if pnl>=0 else ''}{pnl:.3f} | ${capital:.2f} | {exit_reason}")

# Results
wins = sum(1 for t in trades if t["pnl"] > 0)
losses = sum(1 for t in trades if t["pnl"] <= 0)
total = len(trades)

print("\n" + "=" * 60)
print("  📊 نتایج بک‌تست v4")
print("=" * 60)
print(f"  📅 {days:.1f} روز | 💰 ${CAPITAL_START} → ${capital:.2f}")
print(f"  📈 بازده: {((capital - CAPITAL_START) / CAPITAL_START * 100):+.2f}%")
print(f"  📊 معاملات: {total} | 🟢 {wins} ({wins/max(total,1)*100:.0f}%) | 🔴 {losses} ({losses/max(total,1)*100:.0f}%)")

if wins > 0:
    print(f"  💚 میانگین سود: ${np.mean([t['pnl'] for t in trades if t['pnl']>0]):.4f}")
if losses > 0:
    print(f"  💔 میانگین ضرر: ${np.mean([t['pnl'] for t in trades if t['pnl']<=0]):.4f}")

# Extrapolate
daily_ret = ((capital / CAPITAL_START) ** (1/max(days, 0.1)) - 1)
projected_30 = CAPITAL_START * ((1 + daily_ret) ** 30)
print(f"\n  📈 پیش‌بینی ۳۰ روز:")
print(f"  ├─ معاملات/روز: {total/max(days,1):.1f}")
print(f"  ├─ بازده روزانه: {daily_ret*100:+.2f}%")
print(f"  ├─ سرمایه (۳۰ روز): ${projected_30:.2f}")
print(f"  └─ بازده (۳۰ روز): {((projected_30 - CAPITAL_START) / CAPITAL_START * 100):+.1f}%")

print(f"\n📋 تاریخچه:")
for t in trades:
    e = "🟢" if t["pnl"] > 0 else "🔴"
    print(f"  {t['time']:<18} {e}{t['dir']:<4} ${t['entry']:>9,.0f} → ${t['exit']:>9,.0f} | "
          f"P&L: {'+'if t['pnl']>=0 else ''}{t['pnl']:>6.3f} | ${t['cap']:>7.2f} | {t['reason']}")

print("\n⚠️ گذشته ≠ آینده")
