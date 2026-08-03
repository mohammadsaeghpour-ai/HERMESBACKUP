#!/usr/bin/env python3
"""HQIP Backtest v6 — FIXED R:R Calculation"""
import sys
sys.path.insert(0, "/data/workspace")
import ccxt, pandas as pd, numpy as np
from hqip.strategies_rr import analyze_15m
from hqip.indicators import calculate_all_indicators

CCXT_SYMBOL = "BTC/USDT"
CAPITAL = 10.0
LEVERAGE = 20
SL_CAPITAL_PCT = 0.10  # 10% of capital = $1
MAX_TRADES_DAY = 3

print("=" * 60)
print("  🧪 HQIP BACKTEST v6 — R:R FIXED")
print(f"  💰 ${CAPITAL} | ⚡ {LEVERAGE}x | 🛡️ $1 ضرر | 🎯 $2-4 سود")
print("=" * 60)

ex = ccxt.okx()
ohlcv = ex.fetch_ohlcv(CCXT_SYMBOL, '15m', limit=300)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = calculate_all_indicators(df)

print(f"  📡 {len(df)} کندل: {df.index[0]} → {df.index[-1]}")
days = (df.index[-1] - df.index[0]).total_seconds() / 86400

capital = CAPITAL
trades = []
daily_count = {}

for i in range(60, len(df) - 8, 16):
    current_time = df.index[i]
    current_day = str(current_time.date())
    
    if daily_count.get(current_day, 0) >= MAX_TRADES_DAY:
        continue
    if capital < 1.0:
        break
    
    df_slice = df.iloc[:i+1].copy()
    result = analyze_15m(df_slice, "BTCUSDT")
    
    direction = result.get("direction", "NO_TRADE")
    confidence = result.get("confidence", 0)
    
    if direction == "NO_TRADE" or confidence < 80:
        continue
    
    entry_price = result.get("entry", 0)
    sl_price = result.get("sl", 0)
    tp1_price = result.get("tp1", 0)
    tp2_price = result.get("tp2", 0)
    tp3_price = result.get("tp3", 0)
    
    # Simulate next 4-8 candles
    exit_price = entry_price
    exit_reason = "TIMEOUT"
    
    for j in range(i+1, min(i+9, len(df))):
        hi = float(df.iloc[j]["high"])
        lo = float(df.iloc[j]["low"])
        
        if direction == "BUY":
            if lo <= sl_price:
                exit_price, exit_reason = sl_price, "SL"; break
            if hi >= tp3_price:
                exit_price, exit_reason = tp3_price, "TP3"; break
            if hi >= tp2_price:
                exit_price, exit_reason = tp2_price, "TP2"; break
            if hi >= tp1_price:
                exit_price, exit_reason = tp1_price, "TP1"; break
        else:
            if hi >= sl_price:
                exit_price, exit_reason = sl_price, "SL"; break
            if lo <= tp3_price:
                exit_price, exit_reason = tp3_price, "TP3"; break
            if lo <= tp2_price:
                exit_price, exit_reason = tp2_price, "TP2"; break
            if lo <= tp1_price:
                exit_price, exit_reason = tp1_price, "TP1"; break
    
    if exit_reason == "TIMEOUT":
        exit_price = float(df.iloc[min(i+8, len(df)-1)]["close"])
    
    # ── CORRECT P&L CALCULATION ──
    # Risk = 10% of capital = $1
    risk = capital * SL_CAPITAL_PCT
    
    # Position size (notional): risk / sl_distance_pct
    sl_dist_pct = abs(sl_price - entry_price) / entry_price
    pos_value = risk / max(sl_dist_pct, 0.0001)  # e.g., $1 / 0.005 = $200
    
    # P&L = position_value * price_change_pct (NO division by leverage!)
    if direction == "BUY":
        price_pct = (exit_price - entry_price) / entry_price
    else:
        price_pct = (entry_price - exit_price) / entry_price
    
    pnl = pos_value * price_pct  # This is the CORRECT formula
    
    # Enforce max loss
    if exit_reason == "SL":
        pnl = -risk  # Max loss = $1
    
    # Enforce max profit (cap at 5x risk)
    pnl = np.clip(pnl, -risk, risk * 5)
    
    old_cap = capital
    capital = max(0.01, capital + pnl)
    daily_count[current_day] = daily_count.get(current_day, 0) + 1
    
    trades.append({
        "time": str(current_time)[:16], "dir": direction,
        "entry": entry_price, "exit": exit_price,
        "pnl": pnl, "cap": capital, "reason": exit_reason,
        "conf": confidence,
    })
    
    e = "🟢" if pnl > 0 else "🔴"
    print(f"  {e} {current_time} | {direction:4} | ${entry_price:,.0f} → ${exit_price:,.0f} | "
          f"P&L: {'+'if pnl>=0 else ''}{pnl:.2f} | ${capital:.2f} | {exit_reason}")

# Results
wins = sum(1 for t in trades if t["pnl"] > 0)
losses = sum(1 for t in trades if t["pnl"] <= 0)
total = len(trades)

print("\n" + "=" * 60)
print("  📊 نتایج بک‌تست v6 — R:R FIXED")
print("=" * 60)
print(f"  📅 {days:.1f} روز")
print(f"  💰 ${CAPITAL} → ${capital:.2f}")
print(f"  📈 بازده: {((capital - CAPITAL) / CAPITAL * 100):+.2f}%")
print(f"  📊 معاملات: {total} | 🟢 {wins} ({wins/max(total,1)*100:.0f}%) | 🔴 {losses} ({losses/max(total,1)*100:.0f}%)")

if wins > 0:
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0])
    print(f"  💚 میانگین سود: ${avg_win:.2f}")
if losses > 0:
    avg_loss = abs(np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]))
    print(f"  💔 میانگین ضرر: ${avg_loss:.2f}")

if wins > 0 and losses > 0:
    rr = avg_win / avg_loss
    print(f"  ⚖️ نسبت R:R واقعی: 1:{rr:.1f}")
    
    # Required win rate for profitability
    required_wr = 1 / (1 + rr) * 100
    print(f"  🎯 برای سوددهی به {required_wr:.0f}% win rate نیاز داری")
    print(f"  📊 Win Rate فعلی: {wins/max(total,1)*100:.0f}%")

# Extrapolate
if total > 0:
    daily_ret = ((capital / CAPITAL) ** (1/max(days, 0.1)) - 1)
    projected_30 = CAPITAL * ((1 + daily_ret) ** 30)
    print(f"\n  📈 پیش‌بینی ۳۰ روز:")
    print(f"  ├─ معاملات/روز: {total/max(days,1):.1f}")
    print(f"  ├─ بازده روزانه: {daily_ret*100:+.2f}%")
    print(f"  ├─ سرمایه (۳۰ روز): ${projected_30:.2f}")
    print(f"  └─ بازده (۳۰ روز): {((projected_30 - CAPITAL) / CAPITAL * 100):+.1f}%")

print(f"\n📋 تاریخچه:")
for t in trades:
    e = "🟢" if t["pnl"] > 0 else "🔴"
    print(f"  {t['time']:<18} {e}{t['dir']:<4} ${t['entry']:>9,.0f} → ${t['exit']:>9,.0f} | "
          f"P&L: {'+'if t['pnl']>=0 else ''}{t['pnl']:>6.2f} | ${t['cap']:>7.2f} | {t['reason']}")

print("\n⚠️ گذشته ≠ آینده — این شبیه‌سازیه")
