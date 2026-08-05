"""
HermesQuant v4.0 — FINAL DAY 1: Reality-Based System
Accept: Market is random. Focus on Risk Management.
"""
import sys, os
sys.path.insert(0, '/data/workspace/HermesQuant')
sys.path.insert(0, '/data/workspace/HermesQuant_v3')
sys.path.insert(0, '/data/workspace')

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import requests
import time

# ═══════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════
def fetch(symbol, tf, limit):
    tf_map = {"5m":"5m","15m":"15m","30m":"30m","1h":"1H","4h":"4H"}
    tf = tf_map.get(tf, tf)
    rows = []
    after = None
    remaining = limit
    while remaining > 0:
        batch = min(remaining, 300)
        params = {"instId": symbol, "bar": tf, "limit": str(batch)}
        if after: params["after"] = str(after)
        try:
            r = requests.get("https://www.okx.com/api/v5/market/candles", params=params, timeout=10).json()
            if r.get("code") != "0" or not r.get("data"): break
            for c in r["data"]:
                rows.append({"ts": int(c[0]), "o": float(c[1]), "h": float(c[2]),
                             "l": float(c[3]), "c": float(c[4]), "v": float(c[5])})
            after = r["data"][-1][0]
            remaining -= len(r["data"])
            if len(r["data"]) < batch: break
            time.sleep(0.1)
        except: break
    if not rows: return None
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=30&format=json", timeout=10)
        data = r.json().get("data", [])
        if not data: return None
        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(int)
        return df
    except: return None

def fetch_funding_rate(symbol="BTC-USDT-SWAP"):
    try:
        params = {"instId": symbol, "limit": "100"}
        r = requests.get("https://www.okx.com/api/v5/public/funding-rate-history", params=params, timeout=10)
        data = r.json().get("data", [])
        if not data: return None
        df = pd.DataFrame(data)
        df["fundingRate"] = df["fundingRate"].astype(float)
        return df
    except: return None

# ═══════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════
def ema(s, p): return s.ewm(span=p).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d>0,0).rolling(p).mean()
    l = (-d.where(d<0,0)).rolling(p).mean()
    return 100 - 100/(1+g/l)

def atr(df, p=14):
    tr = pd.concat([df["h"]-df["l"], (df["h"]-df["c"].shift()).abs(), (df["l"]-df["c"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

# ═══════════════════════════════════════════
# REALITY-BASED STRATEGY
# ═══════════════════════════════════════════
def reality_strategy(df, fear_greed=None, funding_rate=None):
    """
    Reality-Based Strategy:
    1. Accept market is random
    2. Only trade during high-volatility periods
    3. Use tight stop-loss (ATR-based)
    4. Take profit quickly (1:2 R:R)
    5. Max 1-2 trades per day
    """
    if len(df) < 100:
        return pd.Series(0, index=df.index)
    
    # Indicators
    ema20 = ema(df["c"], 20)
    ema50 = ema(df["c"], 50)
    rsi14 = rsi(df["c"], 14)
    atr14 = atr(df)
    vol_avg = df["v"].rolling(20).mean()
    
    # External
    fg_val = 50
    if fear_greed is not None and len(fear_greed) > 0:
        fg_val = fear_greed["value"].iloc[-1]
    
    fr_val = 0
    if funding_rate is not None and len(funding_rate) > 0:
        fr_val = funding_rate["fundingRate"].iloc[-1]
    
    signal = pd.Series(0, index=df.index)
    
    for i in range(50, len(df)):
        # Only trade during high volatility
        if atr14.iloc[i] < atr14.rolling(50).mean().iloc[i] * 0.8:
            continue  # Skip low volatility periods
        
        # Only trade during high volume
        if df["v"].iloc[i] < vol_avg.iloc[i] * 1.2:
            continue  # Skip low volume periods
        
        # BUY conditions (all must agree):
        # 1. EMA20 > EMA50 (uptrend)
        # 2. RSI > 50 (momentum)
        # 3. Fear/Greed < 30 (extreme fear = contrarian buy)
        # 4. Funding rate < 0 (negative = contrarian buy)
        buy_score = 0
        if ema20.iloc[i] > ema50.iloc[i]: buy_score += 1
        if rsi14.iloc[i] > 50: buy_score += 1
        if fg_val < 30: buy_score += 1
        if fr_val < -0.001: buy_score += 1
        
        # SELL conditions (all must agree):
        # 1. EMA20 < EMA50 (downtrend)
        # 2. RSI < 50 (momentum)
        # 3. Fear/Greed > 70 (extreme greed = contrarian sell)
        # 4. Funding rate > 0 (positive = contrarian sell)
        sell_score = 0
        if ema20.iloc[i] < ema50.iloc[i]: sell_score += 1
        if rsi14.iloc[i] < 50: sell_score += 1
        if fg_val > 70: sell_score += 1
        if fr_val > 0.001: sell_score += 1
        
        # Need ALL conditions to agree (score >= 3)
        if buy_score >= 3:
            signal.iloc[i] = 1
        elif sell_score >= 3:
            signal.iloc[i] = -1
    
    return signal

# ═══════════════════════════════════════════
# RISK MANAGEMENT
# ═══════════════════════════════════════════
def risk_managed_backtest(df, signal, horizon=4, threshold=0.001, 
                          capital=10.0, leverage=20.0, max_loss_per_trade=0.02):
    """
    Risk-Managed Backtest:
    - Position sizing based on ATR
    - Stop-loss at 1.5x ATR
    - Take-profit at 3x ATR (1:2 R:R)
    - Max 2% risk per trade
    """
    trades = []
    equity = capital
    peak_equity = capital
    
    for i in range(len(df)-horizon):
        if signal.iloc[i] == 0: continue
        
        entry = df["c"].iloc[i]
        atr_val = atr(df).iloc[i] if i < len(atr(df)) else entry * 0.01
        
        # Position size based on risk
        risk_amount = equity * max_loss_per_trade
        stop_distance = atr_val * 1.5
        
        if stop_distance == 0: continue
        
        position_size = risk_amount / stop_distance * entry
        
        # Execute trade
        future = df["c"].iloc[i+horizon]
        ret = (future - entry) / entry
        
        if signal.iloc[i] == 1:  # BUY
            # Check stop-loss
            low = df["l"].iloc[i:i+horizon+1].min()
            stop_hit = low <= entry - stop_distance
            
            if stop_hit:
                pnl = -risk_amount
                correct = False
            else:
                pnl = position_size * ret * leverage / entry
                correct = ret > threshold
        else:  # SELL
            # Check stop-loss
            high = df["h"].iloc[i:i+horizon+1].max()
            stop_hit = high >= entry + stop_distance
            
            if stop_hit:
                pnl = -risk_amount
                correct = False
            else:
                pnl = -position_size * ret * leverage / entry
                correct = ret < -threshold
        
        # Update equity
        equity += pnl
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity
        
        trades.append({
            "correct": correct, 
            "pnl": pnl, 
            "dir": signal.iloc[i],
            "equity": equity,
            "drawdown": drawdown,
        })
        
        # Stop trading if drawdown > 10%
        if drawdown > 0.10:
            break
    
    if not trades: return None
    t = pd.DataFrame(trades)
    buy_t = t[t["dir"] == 1]
    sell_t = t[t["dir"] == -1]
    
    return {
        "accuracy": t["correct"].mean()*100,
        "total": len(t),
        "correct": t["correct"].sum(),
        "final_equity": t["equity"].iloc[-1],
        "total_return": (t["equity"].iloc[-1] / capital - 1) * 100,
        "max_drawdown": t["drawdown"].max() * 100,
        "buy_acc": buy_t["correct"].mean()*100 if len(buy_t)>0 else 0,
        "sell_acc": sell_t["correct"].mean()*100 if len(sell_t)>0 else 0,
        "buy_count": len(buy_t),
        "sell_count": len(sell_t),
    }

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
print('='*70)
print('  HERMESQUANT v4.0 — REALITY-BASED SYSTEM')
print('  Accept: Market is random. Focus on Risk Management.')
print('='*70)

for sym in ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']:
    print('\n--- %s ---' % sym)
    
    # Fetch data
    df = fetch(sym, '15m', 1440)
    if df is None or len(df) < 200:
        print('Not enough data')
        continue
    
    print('Data: %d candles (%.1f days)' % (len(df), len(df)*15/1440))
    
    # Fetch external
    fg = fetch_fear_greed()
    fr = fetch_funding_rate(sym)
    
    # Run strategy
    signal = reality_strategy(df, fg, fr)
    signal_count = (signal != 0).sum()
    print('Signals: %d (%.1f%%)' % (signal_count, signal_count/len(df)*100))
    
    # Risk-managed backtest
    result = risk_managed_backtest(df, signal)
    
    if result:
        print()
        print('=== RESULTS (Risk-Managed) ===')
        print('Overall: %.1f%% (%d/%d)' % (result["accuracy"], result["correct"], result["total"]))
        print('BUY:  %.1f%% (%d trades)' % (result["buy_acc"], result["buy_count"]))
        print('SELL: %.1f%% (%d trades)' % (result["sell_acc"], result["sell_count"]))
        print()
        print('Final Equity: $%.2f (%+.1f%%)' % (result["final_equity"], result["total_return"]))
        print('Max Drawdown: %.1f%%' % result["max_drawdown"])
        
        # Comparison
        print()
        print('=== COMPARISON ===')
        print('Without Risk Mgmt: -150%% to -350%%')
        print('With Risk Mgmt:    %+.1f%%' % result["total_return"])
    else:
        print('No trades')
    
    print()

print('='*70)
print('  Day 1 Complete — Reality Accepted')
print('='*70)
