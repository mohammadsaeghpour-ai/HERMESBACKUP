#!/usr/bin/env python3
"""
HQIP Backtest Engine v2 — Optimized for available data
- BTC only, 3 days available (OKX limit)
- $10 starting, 20x leverage, 10% SL, 3 trades/day
- Compound trading
"""
import sys
sys.path.insert(0, "/data/workspace")
import ccxt, pandas as pd, numpy as np
from hqip.indicators import calculate_all_indicators
from hqip.data_platform import DataPlatform

# ── Config ─────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
CCXT_SYMBOL = "BTC/USDT"
STARTING_CAPITAL = 10.0
LEVERAGE = 20
STOP_LOSS_PCT = 0.10
MAX_TRADES_PER_DAY = 3

print("=" * 60)
print("  🧪 HQIP BACKTEST v2 — Compound Trading")
print("=" * 60)
print(f"  💰 Starting: ${STARTING_CAPITAL:.2f} | ⚡ Leverage: {LEVERAGE}x")
print(f"  🛡️ SL: {STOP_LOSS_PCT*100:.0f}% capital | 📊 Max {MAX_TRADES_PER_DAY} trades/day")
print("=" * 60)

# ── Fetch Data ─────────────────────────────────────────────
print("\n📡 Fetching BTC data from OKX...")
ex = ccxt.okx()
ohlcv = ex.fetch_ohlcv(CCXT_SYMBOL, '15m', limit=300)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

print(f"  ✅ {len(df)} candles: {df.index[0]} → {df.index[-1]}")
days = (df.index[-1] - df.index[0]).total_seconds() / 86400
print(f"  📅 Period: {days:.1f} days")

# Calculate all indicators
df = calculate_all_indicators(df)

# ── Run 22-Agent Analysis at each point ────────────────────
print("\n🧠 Running 22-agent analysis...")

from hqip.agents.trend_agent import TrendAgent
from hqip.agents.momentum_agent import MomentumAgent
from hqip.agents.volume_agent import VolumeAgent
from hqip.agents.volatility_agent import VolatilityAgent
from hqip.agents.pattern_agent import PatternAgent
from hqip.agents.market_structure_agent import MarketStructureAgent
from hqip.agents.regime_agent import RegimeAgent
from hqip.agents.smc_agent import SMCAgent
from hqip.agents.liquidity_agent import LiquidityAgent
from hqip.agents.wyckoff_agent import WyckoffAgent
from hqip.agents.supply_demand_agent import SupplyDemandAgent
from hqip.agents.smart_action_agent import SmartActionAgent

# Initialize all agents
agents = [
    TrendAgent(), MomentumAgent(), VolumeAgent(), VolatilityAgent(),
    PatternAgent(), MarketStructureAgent(), RegimeAgent(),
    SMCAgent(), LiquidityAgent(), WyckoffAgent(), SupplyDemandAgent(),
    SmartActionAgent()
]

# ── Simulation ─────────────────────────────────────────────
capital = STARTING_CAPITAL
trades = []
signals_per_day = {}

# We scan every 4 hours (16 candles on 15m)
SCAN_EVERY = 16
WARMUP = 60  # candles needed for indicators

print(f"  Scanning every {SCAN_EVERY} candles ({SCAN_EVERY*15/60:.0f}h)")
print(f"  Warmup: {WARMUP} candles")

for i in range(WARMUP, len(df) - 8, SCAN_EVERY):  # -8 for future lookback
    current_time = df.index[i]
    current_day = str(current_time.date())
    
    # Reset daily counter
    if current_day not in signals_per_day:
        signals_per_day[current_day] = 0
    
    if signals_per_day[current_day] >= MAX_TRADES_PER_DAY:
        continue
    if capital < 1.0:
        break
    
    # ── Run agents on sliced data ──────────────────────────
    df_slice = df.iloc[:i+1].copy()
    
    buy_score = 0
    sell_score = 0
    total_weight = 0
    smart_money_buy = 0
    smart_money_sell = 0
    smart_money_weight = 0
    
    for agent in agents:
        try:
            result = agent.analyze(df_slice, symbol=SYMBOL, timeframe="15m")
            w = getattr(agent, 'weight', 1.0)
            
            if result.direction == "BUY":
                buy_score += w * (result.confidence / 100)
                if agent.name in ["SMC", "Wyckoff", "Liquidity", "SmartAction", "SupplyDemand"]:
                    smart_money_buy += w * (result.confidence / 100)
                    smart_money_weight += w
            elif result.direction == "SELL":
                sell_score += w * (result.confidence / 100)
                if agent.name in ["SMC", "Wyckoff", "Liquidity", "SmartAction", "SupplyDemand"]:
                    smart_money_sell += w * (result.confidence / 100)
                    smart_money_weight += w
            
            total_weight += w
        except:
            continue
    
    if total_weight == 0:
        continue
    
    net = buy_score - sell_score
    sm_net = smart_money_buy - smart_money_sell
    
    # Direction decision
    if net > 0 and buy_score > sell_score * 1.1:
        direction = "BUY"
        confidence = min(100, (buy_score / total_weight) * 150)
    elif net < 0 and sell_score > buy_score * 1.1:
        direction = "SELL"
        confidence = min(100, (sell_score / total_weight) * 150)
    else:
        continue
    
    # Smart money boost
    if smart_money_weight > 0:
        sm_ratio = abs(sm_net) / smart_money_weight
        if sm_ratio > 0.3 and ((sm_net > 0 and direction == "BUY") or (sm_net < 0 and direction == "SELL")):
            confidence = min(100, confidence * 1.15)
    
    if confidence < 55:
        continue
    
    # ── Calculate Entry/SL/TP ──────────────────────────────
    entry_price = float(df_slice.iloc[-1]["close"])
    
    # ATR-based SL
    if 'atr' in df_slice.columns:
        atr_val = float(df_slice.iloc[-1]["atr"])
        if atr_val > 0:
            sl_distance = atr_val * 1.5
        else:
            sl_distance = entry_price * 0.005
    else:
        sl_distance = entry_price * 0.005
    
    if direction == "BUY":
        sl_price = entry_price - sl_distance
        tp1_price = entry_price + sl_distance * 1.5
        tp2_price = entry_price + sl_distance * 2.5
        tp3_price = entry_price + sl_distance * 4.0
    else:
        sl_price = entry_price + sl_distance
        tp1_price = entry_price - sl_distance * 1.5
        tp2_price = entry_price - sl_distance * 2.5
        tp3_price = entry_price - sl_distance * 4.0
    
    # ── Simulate trade (next 8 candles) ────────────────────
    exit_price = entry_price
    exit_reason = "TIMEOUT"
    
    for j in range(i+1, min(i+9, len(df))):
        hi = float(df.iloc[j]["high"])
        lo = float(df.iloc[j]["low"])
        
        if direction == "BUY":
            if lo <= sl_price:
                exit_price, exit_reason = sl_price, "SL"
                break
            if hi >= tp1_price:
                exit_price, exit_reason = tp1_price, "TP1"
                break
            if hi >= tp2_price:
                exit_price, exit_reason = tp2_price, "TP2"
                break
            if hi >= tp3_price:
                exit_price, exit_reason = tp3_price, "TP3"
                break
        else:  # SELL
            if hi >= sl_price:
                exit_price, exit_reason = sl_price, "SL"
                break
            if lo <= tp1_price:
                exit_price, exit_reason = tp1_price, "TP1"
                break
            if lo <= tp2_price:
                exit_price, exit_reason = tp2_price, "TP2"
                break
            if lo <= tp3_price:
                exit_price, exit_reason = tp3_price, "TP3"
                break
    
    if exit_reason == "TIMEOUT":
        last_j = min(i+8, len(df)-1)
        exit_price = float(df.iloc[last_j]["close"])
    
    # ── Calculate P&L ──────────────────────────────────────
    if direction == "BUY":
        price_pct = (exit_price - entry_price) / entry_price
    else:
        price_pct = (entry_price - exit_price) / entry_price
    
    risk_amount = capital * STOP_LOSS_PCT
    sl_dist_pct = sl_distance / entry_price
    pos_value = min((risk_amount / sl_dist_pct) * LEVERAGE, capital * LEVERAGE)
    
    pnl = price_pct * pos_value / LEVERAGE
    pnl = np.clip(pnl, -risk_amount, risk_amount * 5)
    
    # SL enforcement: max loss is 10% of capital
    if exit_reason == "SL":
        pnl = -risk_amount
    
    old_capital = capital
    capital = max(0.01, capital + pnl)
    
    is_win = pnl > 0
    trades.append({
        "time": str(current_time)[:16],
        "day": current_day,
        "direction": direction,
        "entry": entry_price,
        "exit": exit_price,
        "pnl": pnl,
        "capital": capital,
        "confidence": confidence,
        "reason": exit_reason,
        "smart_money": f"+{smart_money_buy:.1f}/-{smart_money_sell:.1f}"
    })
    
    signals_per_day[current_day] = signals_per_day.get(current_day, 0) + 1
    
    emoji = "🟢" if is_win else "🔴"
    print(f"  {emoji} {current_time} | {direction:4} | ${entry_price:,.0f} → ${exit_price:,.0f} | "
          f"P&L: {'+'if pnl>=0 else ''}{pnl:.3f} | ${capital:.2f} | {exit_reason}")

# ── Results ────────────────────────────────────────────────
wins = sum(1 for t in trades if t["pnl"] > 0)
losses = sum(1 for t in trades if t["pnl"] <= 0)
total = len(trades)

print("\n" + "=" * 60)
print("  📊 BACKTEST RESULTS")
print("=" * 60)
print(f"  📅 Period: {days:.1f} days ({df.index[0].date()} → {df.index[-1].date()})")
print(f"  💰 Starting Capital: ${STARTING_CAPITAL:.2f}")
print(f"  💰 Final Capital: ${capital:.2f}")
print(f"  📈 Total Return: {((capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100):+.2f}%")
print(f"  📊 Total Trades: {total}")
print(f"  🟢 Wins: {wins} ({wins/max(total,1)*100:.0f}%)")
print(f"  🔴 Losses: {losses} ({losses/max(total,1)*100:.0f}%)")

if wins > 0:
    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0])
    print(f"  💚 Avg Win: ${avg_win:.4f}")
if losses > 0:
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0])
    print(f"  💔 Avg Loss: ${avg_loss:.4f}")

# ── Extrapolate to 1 month ────────────────────────────────
trades_per_day_actual = total / max(days, 1)
daily_return_pct = ((capital / STARTING_CAPITAL) ** (1/max(days, 0.1)) - 1) * 100
projected_30d = STARTING_CAPITAL * ((1 + daily_return_pct/100) ** 30)

print(f"\n  📈 === PROJECTED 30-DAY PERFORMANCE ===")
print(f"  📊 Avg trades/day: {trades_per_day_actual:.1f}")
print(f"  📈 Daily return: {daily_return_pct:+.2f}%")
print(f"  💰 Projected capital (30 days): ${projected_30d:.2f}")
print(f"  📈 Projected return (30 days): {((projected_30d - STARTING_CAPITAL) / STARTING_CAPITAL * 100):+.1f}%")

# ── Trade Log ─────────────────────────────────────────────
print(f"\n📋 Trade Log ({total} trades):")
print("-" * 90)
print(f"  {'Time':<18} {'Dir':<5} {'Entry':>10} {'Exit':>10} {'P&L':>8} {'Capital':>8} {'Rsn':<7} {'Conf':>5}")
print("-" * 90)
for t in trades:
    e = "🟢" if t["pnl"] > 0 else "🔴"
    print(f"  {t['time']:<18} {e}{t['direction']:<4} ${t['entry']:>9,.0f} ${t['exit']:>9,.0f} "
          f"{'+'if t['pnl']>=0 else ''}{t['pnl']:>6.3f} ${t['capital']:>7.2f} {t['reason']:<7} {t['confidence']:>4.0f}%")

print("\n" + "=" * 60)
print("  ⚠️ گذشته ≠ آینده — این شبیه‌سازیه، نه مشاوره مالی")
print("=" * 60)
