#!/usr/bin/env python3
"""
HQIP Backtest Engine — Compound Trading
- BTC only, 1 month historical
- $10 starting capital, 20x leverage
- 10% stop loss per trade (of current capital)
- 3 trades per day max
- Compound: profits stay in capital
"""
import sys
sys.path.insert(0, "/data/workspace")

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hqip.agents.manager_agent import ManagerAgent
from hqip.indicators import calculate_all_indicators
from hqip.data_platform import DataPlatform

# ── Config ─────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
STARTING_CAPITAL = 10.0
LEVERAGE = 20
STOP_LOSS_PCT = 0.10  # 10% of current capital
MAX_TRADES_PER_DAY = 3
DAYS_BACK = 30
ENTRY_TIMEFRAMES = ["15m", "1h"]

print("=" * 60)
print("  🧪 HQIP BACKTEST ENGINE — Compound Trading")
print("=" * 60)
print(f"  💰 Starting Capital: ${STARTING_CAPITAL:.2f}")
print(f"  ⚡ Leverage: {LEVERAGE}x")
print(f"  🛡️ Stop Loss: {STOP_LOSS_PCT*100:.0f}% of capital per trade")
print(f"  📊 Max Trades/Day: {MAX_TRADES_PER_DAY}")
print(f"  📅 Period: {DAYS_BACK} days back")
print(f"  🔍 Symbol: {SYMBOL}")
print("=" * 60)

# ── Fetch Historical Data ──────────────────────────────────
print("\n📡 Fetching historical data...")
dp = DataPlatform()

# Fetch 15m data for the past month (enough candles)
candles_needed = DAYS_BACK * 24 * 4  # 4 candles per hour on 15m
all_data = {}

for tf in ["15m", "1h", "4h", "1d"]:
    try:
        data = dp.fetch_ohlcv(SYMBOL, tf, limit=min(candles_needed, 1000))
        if data is not None and not data.empty:
            all_data[tf] = data
            print(f"  ✅ {tf}: {len(data)} candles ({data.index[0]} → {data.index[-1]})")
    except Exception as e:
        print(f"  ❌ {tf}: {e}")

if "15m" not in all_data:
    print("❌ No 15m data available. Cannot backtest.")
    sys.exit(1)

# ── Backtest Loop ──────────────────────────────────────────
print("\n🔄 Running backtest simulation...")

capital = STARTING_CAPITAL
trades = []
daily_logs = []
total_trades = 0
winning_trades = 0
losing_trades = 0
max_capital = capital
min_capital = capital

# Get 15m candles for simulation
df_15m = all_data["15m"]
df_1h = all_data.get("1h", df_15m)
df_4h = all_data.get("4h", df_15m)
df_1d = all_data.get("1d", df_15m)

# We'll simulate by walking through 15m candles
# Every 4 candles = 1 hour, we take a signal
# Every 4 signals (4h) we check for 3 trades

# Strategy: scan every 4 hours, take up to 3 trades
# Walk through the data from oldest to newest
scan_interval = 4  # hours between scans
candles_per_scan = scan_interval * 4  # 15m candles per scan

# Start from index 200 (need enough history for indicators)
start_idx = 200
signals_taken_today = 0
current_day = None

print(f"  Starting from candle {start_idx} of {len(df_15m)}")
print(f"  Available candles for backtest: {len(df_15m) - start_idx}")

manager = ManagerAgent()

# Store daily capital snapshots
daily_capital = {}

for scan_idx in range(start_idx, len(df_15m), candles_per_scan):
    # Get current timestamp
    current_time = df_15m.index[scan_idx]
    
    if isinstance(current_time, pd.Timestamp):
        current_day = current_time.date()
    else:
        current_day = str(current_time)[:10]
    
    # Reset trade counter for new day
    if current_day not in daily_capital:
        signals_taken_today = 0
        daily_capital[current_day] = capital
    
    # Check if we've hit daily trade limit
    if signals_taken_today >= MAX_TRADES_PER_DAY:
        continue
    
    # Don't trade if capital is too low
    if capital < 1.0:
        print(f"  💀 Capital below $1. Stopping backtest at {current_time}")
        break
    
    # Prepare data up to current point (no look-ahead!)
    try:
        # Slice data up to current point
        df_slice_15m = df_15m.iloc[:scan_idx+1].copy()
        df_slice_1h = df_1h[df_1h.index <= current_time].copy() if len(df_1h[df_1h.index <= current_time]) >= 50 else None
        df_slice_4h = df_4h[df_4h.index <= current_time].copy() if len(df_4h[df_4h.index <= current_time]) >= 50 else None
        df_slice_1d = df_1d[df_1d.index <= current_time].copy() if len(df_1d[df_1d.index <= current_time]) >= 30 else None
        
        if df_slice_15m is None or len(df_slice_15m) < 50:
            continue
        
        # Calculate indicators
        df_slice_15m = calculate_all_indicators(df_slice_15m)
        if df_slice_1h is not None and len(df_slice_1h) >= 50:
            df_slice_1h = calculate_all_indicators(df_slice_1h)
        if df_slice_4h is not None and len(df_slice_4h) >= 50:
            df_slice_4h = calculate_all_indicators(df_slice_4h)
        if df_slice_1d is not None and len(df_slice_1d) >= 30:
            df_slice_1d = calculate_all_indicators(df_slice_1d)
        
    except Exception as e:
        continue
    
    # Run the 22-agent analysis
    try:
        # Build tf_results manually with the sliced data
        tf_data = {"15m": df_slice_15m}
        if df_slice_1h is not None and len(df_slice_1h) >= 50:
            tf_data["1h"] = df_slice_1h
        if df_slice_4h is not None and len(df_slice_4h) >= 50:
            tf_data["4h"] = df_slice_4h
        if df_slice_1d is not None and len(df_slice_1d) >= 30:
            tf_data["1d"] = df_slice_1d
        
        result = manager.scan(SYMBOL, capital=capital, max_loss=capital * STOP_LOSS_PCT, leverage=LEVERAGE)
        
    except Exception as e:
        continue
    
    direction = result.get("direction", "NO_TRADE")
    confidence = result.get("confidence", 0)
    entry_price = result.get("entry")
    sl_price = result.get("sl")
    tp1_price = result.get("tp1")
    tp2_price = result.get("tp2")
    tp3_price = result.get("tp3")
    
    # Only trade if confidence is high enough
    if direction == "NO_TRADE" or confidence < 60:
        continue
    
    if entry_price is None or sl_price is None or tp1_price is None:
        continue
    
    # ── Simulate the trade ─────────────────────────────────
    # Look at future candles to see what happens
    # Find the candle at entry time
    entry_candle_idx = scan_idx
    entry_candle_price = float(df_15m.iloc[entry_candle_idx]["close"])
    
    # Use actual close as entry (we'd enter at market)
    actual_entry = entry_candle_price
    
    # Calculate position size based on risk
    risk_amount = capital * STOP_LOSS_PCT
    sl_distance_pct = abs(actual_entry - sl_price) / actual_entry
    position_value = (risk_amount / sl_distance_pct) * LEVERAGE if sl_distance_pct > 0 else capital * LEVERAGE
    position_value = min(position_value, capital * LEVERAGE)  # Cap at max leverage
    
    # Simulate next 8 candles (2 hours max hold time)
    max_hold_candles = 8
    trade_closed = False
    exit_price = actual_entry
    exit_reason = "TIMEOUT"
    pnl = 0
    
    for future_idx in range(entry_candle_idx + 1, min(entry_candle_idx + max_hold_candles + 1, len(df_15m))):
        future_high = float(df_15m.iloc[future_idx]["high"])
        future_low = float(df_15m.iloc[future_idx]["low"])
        future_close = float(df_15m.iloc[future_idx]["close"])
        
        if direction == "BUY":
            # Check stop loss (hit low)
            if future_low <= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
                trade_closed = True
                break
            # Check take profit (hit high)
            if future_high >= tp1_price:
                exit_price = tp1_price
                exit_reason = "TP1"
                trade_closed = True
                break
            # Check TP2
            if tp2_price and future_high >= tp2_price:
                exit_price = tp2_price
                exit_reason = "TP2"
                trade_closed = True
                break
            # Check TP3
            if tp3_price and future_high >= tp3_price:
                exit_price = tp3_price
                exit_reason = "TP3"
                trade_closed = True
                break
                
        elif direction == "SELL":
            # Check stop loss (hit high)
            if future_high >= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
                trade_closed = True
                break
            # Check take profit (hit low)
            if future_low <= tp1_price:
                exit_price = tp1_price
                exit_reason = "TP1"
                trade_closed = True
                break
            # Check TP2
            if tp2_price and future_low <= tp2_price:
                exit_price = tp2_price
                exit_reason = "TP2"
                trade_closed = True
                break
            # Check TP3
            if tp3_price and future_low <= tp3_price:
                exit_price = tp3_price
                exit_reason = "TP3"
                trade_closed = True
                break
    
    # If not closed, use last candle close
    if not trade_closed:
        last_idx = min(entry_candle_idx + max_hold_candles, len(df_15m) - 1)
        exit_price = float(df_15m.iloc[last_idx]["close"])
        exit_reason = "TIMEOUT"
    
    # Calculate P&L
    if direction == "BUY":
        price_change_pct = (exit_price - actual_entry) / actual_entry
    else:
        price_change_pct = (actual_entry - exit_price) / actual_entry
    
    pnl = price_change_pct * position_value / LEVERAGE  # P&L on margin
    pnl = np.clip(pnl, -risk_amount, risk_amount * 5)  # Cap losses and gains
    
    # Compound: update capital
    old_capital = capital
    capital += pnl
    capital = max(capital, 0.01)  # Don't go below 0.01
    
    # Track stats
    total_trades += 1
    if pnl > 0:
        winning_trades += 1
    else:
        losing_trades += 1
    
    max_capital = max(max_capital, capital)
    min_capital = min(min_capital, capital)
    signals_taken_today += 1
    
    # Log the trade
    trade_log = {
        "time": str(current_time)[:16],
        "day": str(current_day),
        "direction": direction,
        "entry": actual_entry,
        "exit": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "capital_before": old_capital,
        "capital_after": capital,
        "confidence": confidence,
        "position_value": position_value,
    }
    trades.append(trade_log)
    
    # Update daily capital
    daily_capital[current_day] = capital
    
    # Print progress
    emoji = "🟢" if pnl > 0 else "🔴"
    print(f"  {emoji} {current_time} | {direction} | Entry: ${actual_entry:,.0f} → Exit: ${exit_price:,.0f} | "
          f"P&L: {'+'if pnl>=0 else ''}{pnl:.2f} | Capital: ${capital:.2f} | Reason: {exit_reason}")

# ── Results ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  📊 BACKTEST RESULTS")
print("=" * 60)
print(f"  💰 Starting Capital: ${STARTING_CAPITAL:.2f}")
print(f"  💰 Final Capital: ${capital:.2f}")
print(f"  📈 Total Return: {((capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100):+.1f}%")
print(f"  📊 Total Trades: {total_trades}")
print(f"  🟢 Winning: {winning_trades} ({winning_trades/max(total_trades,1)*100:.0f}%)")
print(f"  🔴 Losing: {losing_trades} ({losing_trades/max(total_trades,1)*100:.0f}%)")
print(f"  📈 Max Capital: ${max_capital:.2f}")
print(f"  📉 Min Capital: ${min_capital:.2f}")

if total_trades > 0:
    win_rate = winning_trades / total_trades * 100
    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] < 0]) if losing_trades > 0 else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
    
    print(f"  🎯 Win Rate: {win_rate:.1f}%")
    print(f"  💚 Avg Win: ${avg_win:.2f}")
    print(f"  💔 Avg Loss: ${avg_loss:.2f}")
    print(f"  ⚖️ Profit Factor: {profit_factor:.2f}")

# ── Daily Capital Curve ───────────────────────────────────
print("\n📅 Daily Capital Curve:")
print("-" * 40)
for day, cap in sorted(daily_capital.items()):
    bar_len = int(cap / max_capital * 30) if max_capital > 0 else 0
    bar = "█" * bar_len
    change = ((cap - STARTING_CAPITAL) / STARTING_CAPITAL * 100)
    print(f"  {day} | ${cap:>7.2f} | {change:>+6.1f}% | {bar}")

# ── Trade Log ─────────────────────────────────────────────
print("\n📋 Trade Log:")
print("-" * 100)
print(f"  {'Time':<18} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'P&L':>8} {'Capital':>8} {'Conf':>5} {'Reason':<8}")
print("-" * 100)
for t in trades:
    emoji = "🟢" if t["pnl"] > 0 else "🔴"
    print(f"  {t['time']:<18} {emoji}{t['direction']:<4} ${t['entry']:>9,.0f} ${t['exit']:>9,.0f} "
          f"{'+'if t['pnl']>=0 else ''}{t['pnl']:>6.2f} ${t['capital_after']:>7.2f} "
          f"{t['confidence']:>4.0f}% {t['exit_reason']:<8}")

print("\n" + "=" * 60)
print("  ⚠️ Past performance ≠ future results")
print("  📊 This is a simulation, not financial advice")
print("=" * 60)
