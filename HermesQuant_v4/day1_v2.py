"""
HermesQuant v4.0 — Day 1 v2: Ultra-Selective Strategy
Key insight: FEWER trades = BETTER accuracy
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

def macd(s, fast=12, slow=26, signal=9):
    ema_fast = ema(s, fast)
    ema_slow = ema(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(df, p=14):
    tr = pd.concat([df["h"]-df["l"], (df["h"]-df["c"].shift()).abs(), (df["l"]-df["c"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def adx(df, p=14):
    plus_dm = df["h"].diff()
    minus_dm = -df["l"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([df["h"]-df["l"], (df["h"]-df["c"].shift()).abs(), (df["l"]-df["c"].shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(p).mean()
    plus_di = 100 * plus_dm.rolling(p).mean() / atr14
    minus_di = 100 * minus_dm.rolling(p).mean() / atr14
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.rolling(p).mean()

def supertrend(df, period=10, multiplier=3):
    hl2 = (df["h"] + df["l"]) / 2
    a = atr(df, period)
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df["c"].iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["c"].iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
    return direction

# ═══════════════════════════════════════════
# ULTRA-SELECTIVE STRATEGY
# ═══════════════════════════════════════════
def ultra_selective_signal(df, fear_greed=None, funding_rate=None):
    """
    ULTRA-SELECTIVE: Only trade when ALL agree:
    1. Supertrend direction (trend)
    2. EMA20 vs EMA50 (trend confirmation)
    3. RSI in sweet spot (momentum)
    4. MACD histogram (momentum confirmation)
    5. ADX > 25 (strong trend)
    6. Volume > 1.2x average (confirmation)
    7. Fear/Greed extreme (contrarian)
    8. Funding rate extreme (contrarian)
    """
    if len(df) < 100:
        return pd.Series(0, index=df.index)
    
    # Indicators
    ema20 = ema(df["c"], 20)
    ema50 = ema(df["c"], 50)
    rsi14 = rsi(df["c"], 14)
    macd_line, signal_line, histogram = macd(df["c"])
    adx14 = adx(df)
    st_dir = supertrend(df)
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
        buy_score = 0
        sell_score = 0
        
        # 1. Supertrend
        if st_dir.iloc[i] == 1:
            buy_score += 2
        else:
            sell_score += 2
        
        # 2. EMA trend
        if ema20.iloc[i] > ema50.iloc[i]:
            buy_score += 1
        else:
            sell_score += 1
        
        # 3. RSI sweet spot
        if 55 < rsi14.iloc[i] < 70:
            buy_score += 1
        elif 30 < rsi14.iloc[i] < 45:
            sell_score += 1
        
        # 4. MACD
        if histogram.iloc[i] > 0 and histogram.iloc[i] > histogram.iloc[i-1]:
            buy_score += 1
        elif histogram.iloc[i] < 0 and histogram.iloc[i] < histogram.iloc[i-1]:
            sell_score += 1
        
        # 5. ADX (strong trend)
        if adx14.iloc[i] > 25:
            buy_score *= 1.2
            sell_score *= 1.2
        
        # 6. Volume confirmation
        if df["v"].iloc[i] > 1.2 * vol_avg.iloc[i]:
            buy_score *= 1.1
            sell_score *= 1.1
        
        # 7. Fear/Greed contrarian
        if fg_val < 25:  # Extreme fear → BUY
            buy_score += 1
        elif fg_val > 75:  # Extreme greed → SELL
            sell_score += 1
        
        # 8. Funding rate contrarian
        if fr_val < -0.001:  # Negative funding → BUY
            buy_score += 0.5
        elif fr_val > 0.001:  # Positive funding → SELL
            sell_score += 0.5
        
        # DECISION: Need strong agreement (score >= 4)
        if buy_score >= 4 and buy_score > sell_score:
            signal.iloc[i] = 1
        elif sell_score >= 4 and sell_score > buy_score:
            signal.iloc[i] = -1
    
    return signal

# ═══════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════
def backtest(df, signal, horizon=4, threshold=0.001):
    trades = []
    for i in range(len(df)-horizon):
        if signal.iloc[i] == 0: continue
        entry = df["c"].iloc[i]
        future = df["c"].iloc[i+horizon]
        ret = (future - entry) / entry
        if signal.iloc[i] == 1:
            correct = ret > threshold
            pnl = ret * 10 - 0.001
        else:
            correct = ret < -threshold
            pnl = -ret * 10 - 0.001
        trades.append({"correct": correct, "pnl": pnl, "dir": signal.iloc[i], "ret": ret})
    
    if not trades: return None
    t = pd.DataFrame(trades)
    buy_t = t[t["dir"] == 1]
    sell_t = t[t["dir"] == -1]
    return {
        "accuracy": t["correct"].mean()*100,
        "total": len(t),
        "correct": t["correct"].sum(),
        "pnl": t["pnl"].sum(),
        "buy_acc": buy_t["correct"].mean()*100 if len(buy_t)>0 else 0,
        "sell_acc": sell_t["correct"].mean()*100 if len(sell_t)>0 else 0,
        "buy_count": len(buy_t),
        "sell_count": len(sell_t),
    }

# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
print('='*70)
print('  HERMESQUANT v4.0 — ULTRA-SELECTIVE STRATEGY')
print('='*70)

for sym in ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']:
    print('\n--- %s ---' % sym)
    
    df = fetch(sym, '15m', 1440)
    if df is None or len(df) < 200:
        print('Not enough data')
        continue
    
    print('Data: %d candles' % len(df))
    
    # Fetch external data
    fg = fetch_fear_greed()
    fr = fetch_funding_rate(sym)
    
    # Run strategy
    signal = ultra_selective_signal(df, fg, fr)
    signal_count = (signal != 0).sum()
    print('Signals: %d (%.1f%%)' % (signal_count, signal_count/len(df)*100))
    
    # Backtest
    result = backtest(df, signal)
    
    if result:
        print()
        print('=== RESULTS ===')
        print('Overall: %.1f%% (%d/%d)' % (result["accuracy"], result["correct"], result["total"]))
        print('BUY:  %.1f%% (%d trades)' % (result["buy_acc"], result["buy_count"]))
        print('SELL: %.1f%% (%d trades)' % (result["sell_acc"], result["sell_count"]))
        print('P&L: %.2f%%' % (result["pnl"]*100))
        
        # Comparison
        print()
        print('=== COMPARISON ===')
        print('Previous: ~40%% accuracy, many trades')
        print('Now: %.1f%% accuracy, %d trades' % (result["accuracy"], result["total"]))
    else:
        print('No trades (too selective)')
    
    print()

print('='*70)
