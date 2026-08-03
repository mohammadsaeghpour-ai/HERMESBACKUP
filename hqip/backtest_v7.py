#!/usr/bin/env python3
"""HQIP Backtest v7 — Wider SL to avoid noise, same $1 risk"""
import sys
sys.path.insert(0, "/data/workspace")
import ccxt, pandas as pd, numpy as np
from hqip.strategies_rr import analyze_15m
from hqip.indicators import calculate_all_indicators

CCXT_SYMBOL = "BTC/USDT"
CAPITAL = 10.0
LEVERAGE = 20
MAX_LOSS_DOLLAR = 1.0  # $1 max loss
MAX_TRADES_DAY = 3

print("=" * 60)
print("  🧪 HQIP BACKTEST v7 — Wider SL (1.0%)")
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
    
    if direction == "NO_TRADE" or confidence < 65:
        continue
    
    entry_price = result.get("entry", 0)
    sl_price = result.get("sl", 0)
    tp1_price = result.get("tp1", 0)
    tp2_price = result.get("tp2", 0)
    tp3_price = result.get("tp3", 0)
    
    # ── WIDER SL/TP ──
    sl_dist_pct = abs(sl_price - entry_price) / entry_price
    
    # Widen SL to 1.0% (from 0.5%) to avoid noise
    new_sl_pct = 0.010  # 1.0%
    new_tp1_pct = 0.020  # 2.0% = $2 profit
    new_tp2_pct = 0.030  # 3.0% = $3 profit
    new_tp3_pct = 0.040  # 4.0% = $4 profit
    
    if direction == "BUY":
        sl_price = entry_price * (1 - new_sl_pct)
        tp1_price = entry_price * (1 + new_tp1_pct)
        tp2_price = entry_price * (1 + new_tp2_pct)
        tp3_price = entry_price * (1 + new_tp3_pct)
    else:
        sl_price = entry_price * (1 + new_sl_pct)
        tp1_price = entry_price * (1 - new_tp1_pct)
        tp2_price = entry_price * (1 - new_tp2_pct)
        tp3_price = entry_price * (1 - new_tp3_pct)
    
    # Simulate next 8-16 candles (2-4 hours for wider targets)
    exit_price = entry_price
    exit_reason = "TIMEOUT"
    
    for j in range(i+1, min(i+17, len(df))):
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
        exit_price = float(df.iloc[min(i+16, len(df)-1)]["close"])
    
    # ── P&L CALCULATION ──
    # Position size: $1 risk / 1.0% SL = $100 notional
    pos_value = MAX_LOSS_DOLLAR / new_sl_pct  # $1 / 0.01 = $100
    
    if direction == "BUY":
        price_pct = (exit_price - entry_price) / entry_price
    else:
        price_pct = (entry_price - exit_price) / entry_price
    
    pnl = pos_value * price_pct
    
    if exit_reason == "SL":
        pnl = -MAX_LOSS_DOLLAR
    
    pnl = np.clip(pnl, -MAX_LOSS_DOLLAR, MAX_LOSS_DOLLAR * 5)
    
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
print("  📊 نتایج بک‌تست v7")
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
    print(f"  ⚖️ R:R واقعی: 1:{rr:.1f}")
    required_wr = 1 / (1 + rr) * 100
    print(f"  🎯 برای سوددهی به {required_wr:.0f}% win rate نیاز داری")
    print(f"  📊 Win Rate فعلی: {wins/max(total,1)*100:.0f}%")

if total > 0:
    daily_ret = ((capital / CAPITAL) ** (1/max(days, 0.1)) - 1)
    projected_30 = CAPITAL * ((1 + daily_ret) ** 30)
    print(f"\n  📈 پیش‌بینی ۳۰ روز:")
    print(f"  ├─ سرمایه (۳۰ روز): ${projected_30:.2f}")
    print(f"  └─ بازده (۳۰ روز): {((projected_30 - CAPITAL) / CAPITAL * 100):+.1f}%")

print(f"\n📋 تاریخچه:")
for t in trades:
    e = "🟢" if t["pnl"] > 0 else "🔴"
    print(f"  {t['time']:<18} {e}{t['dir']:<4} ${t['entry']:>9,.0f} → ${t['exit']:>9,.0f} | "
          f"P&L: {'+'if t['pnl']>=0 else ''}{t['pnl']:>6.2f} | ${t['cap']:>7.2f} | {t['reason']}")

print("\n⚠️ گذشته ≠ آینده")
