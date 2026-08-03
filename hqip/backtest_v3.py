#!/usr/bin/env python3
"""HQIP Backtest v3 — Uses full ManagerAgent with 22 agents"""
import sys
sys.path.insert(0, "/data/workspace")
import ccxt, pandas as pd, numpy as np
from hqip.agents.manager_agent import ManagerAgent

SYMBOL = "BTCUSDT"
CCXT_SYMBOL = "BTC/USDT"
CAPITAL = 10.0
LEVERAGE = 20
SL_PCT = 0.10
MAX_TRADES = 3

print("=" * 60)
print("  🧪 HQIP BACKTEST v3 — Full 22-Agent System")
print("=" * 60)
print(f"  💰 ${CAPITAL} | ⚡ {LEVERAGE}x | 🛡️ SL {SL_PCT*100}% | 📊 {MAX_TRADES}/day")
print("=" * 60)

# Fetch data
print("\n📡 Fetching BTC data...")
ex = ccxt.okx()
ohlcv = ex.fetch_ohlcv(CCXT_SYMBOL, '15m', limit=300)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
print(f"  ✅ {len(df)} candles: {df.index[0]} → {df.index[-1]}")
days = (df.index[-1] - df.index[0]).total_seconds() / 86400

# Run manager agent
print("\n🧠 Running full 22-agent analysis...")
manager = ManagerAgent()

capital = CAPITAL
trades = []
daily_trades = {}

# Scan every 4 hours
for i in range(60, len(df) - 8, 16):
    current_time = df.index[i]
    current_day = str(current_time.date())
    
    if daily_trades.get(current_day, 0) >= MAX_TRADES:
        continue
    if capital < 1.0:
        break
    
    try:
        result = manager.scan(SYMBOL, capital=capital, max_loss=capital * SL_PCT, leverage=LEVERAGE)
    except:
        continue
    
    direction = result.get("direction", "NO_TRADE")
    confidence = result.get("confidence", 0)
    entry_price = result.get("entry")
    sl_price = result.get("sl")
    tp1 = result.get("tp1")
    tp2 = result.get("tp2")
    tp3 = result.get("tp3")
    
    if direction == "NO_TRADE" or confidence < 50 or entry_price is None:
        continue
    if sl_price is None or tp1 is None:
        continue
    
    # Simulate trade
    actual_entry = float(df.iloc[i]["close"])
    
    # Adjust SL/TP relative to actual entry
    sl_dist = abs(sl_price - actual_entry)
    if direction == "BUY":
        sl = actual_entry - sl_dist
        t1 = actual_entry + sl_dist * 1.5
        t2 = actual_entry + sl_dist * 2.5
        t3 = actual_entry + sl_dist * 4.0
    else:
        sl = actual_entry + sl_dist
        t1 = actual_entry - sl_dist * 1.5
        t2 = actual_entry - sl_dist * 2.5
        t3 = actual_entry - sl_dist * 4.0
    
    exit_price = actual_entry
    exit_reason = "TIMEOUT"
    
    for j in range(i+1, min(i+9, len(df))):
        hi = float(df.iloc[j]["high"])
        lo = float(df.iloc[j]["low"])
        
        if direction == "BUY":
            if lo <= sl: exit_price, exit_reason = sl, "SL"; break
            if hi >= t1: exit_price, exit_reason = t1, "TP1"; break
            if hi >= t2: exit_price, exit_reason = t2, "TP2"; break
            if hi >= t3: exit_price, exit_reason = t3, "TP3"; break
        else:
            if hi >= sl: exit_price, exit_reason = sl, "SL"; break
            if lo <= t1: exit_price, exit_reason = t1, "TP1"; break
            if lo <= t2: exit_price, exit_reason = t2, "TP2"; break
            if lo <= t3: exit_price, exit_reason = t3, "TP3"; break
    
    if exit_reason == "TIMEOUT":
        exit_price = float(df.iloc[min(i+8, len(df)-1)]["close"])
    
    # P&L
    if direction == "BUY":
        pct = (exit_price - actual_entry) / actual_entry
    else:
        pct = (actual_entry - exit_price) / actual_entry
    
    risk = capital * SL_PCT
    sl_pct = sl_dist / actual_entry if actual_entry > 0 else 0.005
    pos_val = min((risk / max(sl_pct, 0.001)) * LEVERAGE, capital * LEVERAGE)
    pnl = pct * pos_val / LEVERAGE
    pnl = np.clip(pnl, -risk, risk * 5)
    if exit_reason == "SL":
        pnl = -risk
    
    old_cap = capital
    capital = max(0.01, capital + pnl)
    daily_trades[current_day] = daily_trades.get(current_day, 0) + 1
    
    trades.append({
        "time": str(current_time)[:16], "day": current_day,
        "dir": direction, "entry": actual_entry, "exit": exit_price,
        "pnl": pnl, "cap": capital, "conf": confidence, "reason": exit_reason,
    })
    
    e = "🟢" if pnl > 0 else "🔴"
    print(f"  {e} {current_time} | {direction:4} | ${actual_entry:,.0f} → ${exit_price:,.0f} | "
          f"P&L: {'+'if pnl>=0 else ''}{pnl:.3f} | ${capital:.2f} | {exit_reason}")

# ── Results ────────────────────────────────────────────────
wins = sum(1 for t in trades if t["pnl"] > 0)
losses = sum(1 for t in trades if t["pnl"] <= 0)
total = len(trades)

print("\n" + "=" * 60)
print("  📊 نتایج بک‌تست")
print("=" * 60)
print(f"  📅 دوره: {days:.1f} روز ({df.index[0].date()} → {df.index[-1].date()})")
print(f"  💰 سرمایه اولیه: ${CAPITAL:.2f}")
print(f"  💰 سرمایه نهایی: ${capital:.2f}")
ret = ((capital - CAPITAL) / CAPITAL * 100)
print(f"  📈 بازده کل: {ret:+.2f}%")
print(f"  📊 کل معاملات: {total}")
print(f"  🟢 سودده: {wins} ({wins/max(total,1)*100:.0f}%)")
print(f"  🔴 ضررده: {losses} ({losses/max(total,1)*100:.0f}%)")

if wins > 0:
    print(f"  💚 میانگین سود: ${np.mean([t['pnl'] for t in trades if t['pnl']>0]):.4f}")
if losses > 0:
    print(f"  💔 میانگین ضرر: ${np.mean([t['pnl'] for t in trades if t['pnl']<=0]):.4f}")

# Extrapolate
daily_ret = ((capital / CAPITAL) ** (1/max(days, 0.1)) - 1)
projected_30 = CAPITAL * ((1 + daily_ret) ** 30)

print(f"\n  📈 پیش‌بینی ۳۰ روز:")
print(f"  ├─ معاملات در روز: {total/max(days,1):.1f}")
print(f"  ├─ بازده روزانه: {daily_ret*100:+.2f}%")
print(f"  ├─ سرمایه پیش‌بینی (۳۰ روز): ${projected_30:.2f}")
print(f"  └─ بازده پیش‌بینی (۳۰ روز): {((projected_30 - CAPITAL) / CAPITAL * 100):+.1f}%")

print(f"\n📋 تاریخچه معاملات:")
print("-" * 85)
for t in trades:
    e = "🟢" if t["pnl"] > 0 else "🔴"
    print(f"  {t['time']:<18} {e}{t['dir']:<4} ${t['entry']:>9,.0f} → ${t['exit']:>9,.0f} | "
          f"P&L: {'+'if t['pnl']>=0 else ''}{t['pnl']:>6.3f} | ${t['cap']:>7.2f} | {t['reason']:<6} | {t['conf']:.0f}%")

print("\n⚠️ گذشته ≠ آینده — این شبیه‌سازیه")
print("=" * 60)
