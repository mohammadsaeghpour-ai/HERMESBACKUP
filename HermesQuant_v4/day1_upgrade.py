"""
HermesQuant v4.0 — 3-Day Upgrade
Day 1: Indicators + Fear/Greed + Funding Rate + Open Interest
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
# DATA FETCHER (with pagination)
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

# ═══════════════════════════════════════════
# FEAR/GREED INDEX (from alternative.me)
# ═══════════════════════════════════════════
def fetch_fear_greed():
    """Fetch Fear & Greed Index"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=30&format=json", timeout=10)
        data = r.json().get("data", [])
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except:
        return None

# ═══════════════════════════════════════════
# FUNDING RATE (from OKX)
# ═══════════════════════════════════════════
def fetch_funding_rate(symbol="BTC-USDT-SWAP"):
    """Fetch funding rate history"""
    try:
        params = {"instId": symbol, "limit": "100"}
        r = requests.get("https://www.okx.com/api/v5/public/funding-rate-history", params=params, timeout=10)
        data = r.json().get("data", [])
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["ts"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
        return df[["ts", "fundingRate"]].sort_values("ts").reset_index(drop=True)
    except:
        return None

# ═══════════════════════════════════════════
# OPEN INTEREST (from OKX)
# ═══════════════════════════════════════════
def fetch_open_interest(symbol="BTC-USDT-SWAP"):
    """Fetch open interest history"""
    try:
        params = {"instId": symbol, "period": "1H", "limit": "100"}
        r = requests.get("https://www.okx.com/api/v5/market/open-interest-history", params=params, timeout=10)
        data = r.json().get("data", [])
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df["oi"] = df["oi"].astype(float)
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
        return df[["ts", "oi"]].sort_values("ts").reset_index(drop=True)
    except:
        return None

# ═══════════════════════════════════════════
# INDICATORS (improved)
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
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(1, index=df.index)
    
    for i in range(1, len(df)):
        if df["c"].iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["c"].iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        
        if direction.iloc[i] == 1:
            st.iloc[i] = lower.iloc[i]
        else:
            st.iloc[i] = upper.iloc[i]
    
    return st, direction

# ═══════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════
def compute_features(df, fear_greed=None, funding_rate=None, open_interest=None):
    """Compute all features"""
    features = pd.DataFrame(index=df.index)
    
    # Price features
    features["close"] = df["c"]
    features["returns"] = df["c"].pct_change()
    features["log_returns"] = np.log(df["c"] / df["c"].shift(1))
    features["volatility"] = features["returns"].rolling(20).std()
    
    # Technical indicators
    features["ema20"] = ema(df["c"], 20)
    features["ema50"] = ema(df["c"], 50)
    features["ema_ratio"] = features["ema20"] / features["ema50"]
    features["rsi14"] = rsi(df["c"], 14)
    
    macd_line, signal_line, histogram = macd(df["c"])
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_hist"] = histogram
    
    features["adx"] = adx(df)
    features["atr"] = atr(df)
    features["atr_pct"] = features["atr"] / df["c"]
    
    st, st_dir = supertrend(df)
    features["supertrend"] = st
    features["supertrend_dir"] = st_dir
    
    # Volume features
    features["volume"] = df["v"]
    features["volume_sma"] = df["v"].rolling(20).mean()
    features["volume_ratio"] = features["volume"] / features["volume_sma"]
    
    # Price position
    features["high_low_range"] = (df["h"] - df["l"]) / df["c"]
    features["close_position"] = (df["c"] - df["l"]) / (df["h"] - df["l"])
    
    # Fear/Greed
    if fear_greed is not None and len(fear_greed) > 0:
        fg_value = fear_greed["value"].iloc[-1] if len(fear_greed) > 0 else 50
        features["fear_greed"] = fg_value
        features["fear_greed_signal"] = 1 if fg_value < 25 else (-1 if fg_value > 75 else 0)
    else:
        features["fear_greed"] = 50
        features["fear_greed_signal"] = 0
    
    # Funding Rate
    if funding_rate is not None and len(funding_rate) > 0:
        fr = funding_rate["fundingRate"].iloc[-1] if len(funding_rate) > 0 else 0
        features["funding_rate"] = fr
        features["funding_signal"] = -1 if fr > 0.001 else (1 if fr < -0.001 else 0)
    else:
        features["funding_rate"] = 0
        features["funding_signal"] = 0
    
    # Open Interest
    if open_interest is not None and len(open_interest) > 1:
        oi_now = open_interest["oi"].iloc[-1]
        oi_prev = open_interest["oi"].iloc[-2] if len(open_interest) > 1 else oi_now
        oi_change = (oi_now - oi_prev) / oi_prev if oi_prev > 0 else 0
        features["oi_change"] = oi_change
        features["oi_signal"] = 1 if oi_change > 0.02 else (-1 if oi_change < -0.02 else 0)
    else:
        features["oi_change"] = 0
        features["oi_signal"] = 0
    
    return features

# ═══════════════════════════════════════════
# STRATEGY
# ═══════════════════════════════════════════
def strategy(features, df):
    """
    Multi-factor strategy:
    1. Trend: EMA + Supertrend
    2. Momentum: RSI + MACD
    3. Confirmation: ADX + Volume
    4. External: Fear/Greed + Funding + OI
    """
    signal = pd.Series(0, index=features.index)
    
    for i in range(50, len(features)):
        # Trend score
        trend_buy = 0
        trend_sell = 0
        
        if features["ema_ratio"].iloc[i] > 1.002:
            trend_buy += 1
        elif features["ema_ratio"].iloc[i] < 0.998:
            trend_sell += 1
        
        if features["supertrend_dir"].iloc[i] == 1:
            trend_buy += 1
        else:
            trend_sell += 1
        
        # Momentum score
        rsi_val = features["rsi14"].iloc[i]
        if 40 < rsi_val < 65:
            trend_buy += 0.5
        elif 35 < rsi_val < 60:
            trend_sell += 0.5
        
        if features["macd_hist"].iloc[i] > 0:
            trend_buy += 0.5
        else:
            trend_sell += 0.5
        
        # Confirmation
        if features["adx"].iloc[i] > 20:
            if features["volume_ratio"].iloc[i] > 1.0:
                trend_buy *= 1.2
                trend_sell *= 1.2
        
        # External signals
        if features["fear_greed_signal"].iloc[i] == 1:
            trend_buy += 1
        elif features["fear_greed_signal"].iloc[i] == -1:
            trend_sell += 1
        
        if features["funding_signal"].iloc[i] == 1:
            trend_buy += 0.5
        elif features["funding_signal"].iloc[i] == -1:
            trend_sell += 0.5
        
        if features["oi_signal"].iloc[i] == 1:
            trend_buy += 0.5
        elif features["oi_signal"].iloc[i] == -1:
            trend_sell += 0.5
        
        # Decision
        if trend_buy > trend_sell and trend_buy >= 2:
            signal.iloc[i] = 1
        elif trend_sell > trend_buy and trend_sell >= 2:
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
print('  HERMESQUANT v4.0 — 3-DAY UPGRADE (Day 1)')
print('='*70)

for sym in ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']:
    print('\n--- %s ---' % sym)
    
    # Fetch data
    df = fetch(sym, '15m', 1440)
    if df is None or len(df) < 200:
        print('Not enough data')
        continue
    
    print('Price data: %d candles' % len(df))
    
    # Fetch external data
    fg = fetch_fear_greed()
    fr = fetch_funding_rate(sym)
    oi = fetch_open_interest(sym)
    
    print('Fear/Greed: %s' % ('OK' if fg is not None else 'FAIL'))
    print('Funding Rate: %s' % ('OK' if fr is not None else 'FAIL'))
    print('Open Interest: %s' % ('OK' if oi is not None else 'FAIL'))
    
    # Compute features
    features = compute_features(df, fg, fr, oi)
    
    # Run strategy
    signal = strategy(features, df)
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
    else:
        print('No trades')

print()
print('='*70)
print('  Day 1 Complete')
print('='*70)
