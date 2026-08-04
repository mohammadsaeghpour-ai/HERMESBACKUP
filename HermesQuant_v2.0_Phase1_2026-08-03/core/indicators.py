"""Technical Indicators — Pure Pandas"""
import numpy as np
import pandas as pd

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h - h.shift(1)
    down = l.shift(1) - l
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_val = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / (atr_val + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / (atr_val + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_val = dx.rolling(period).mean()
    return adx_val, plus_di, minus_di

def macd(df, fast=12, slow=26, signal=9):
    ema_fast = ema(df["close"], fast)
    ema_slow = ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def supertrend(df, period=10, mult=3.0):
    h, l, c = df["high"], df["low"], df["close"]
    atr_val = atr(df, period)
    hl2 = (h + l) / 2
    upper = hl2 + mult * atr_val
    lower = hl2 - mult * atr_val
    direction = pd.Series(1, index=df.index)
    st = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        if direction.iloc[i] == 1:
            st.iloc[i] = lower.iloc[i]
            if lower.iloc[i] < lower.iloc[i-1] and c.iloc[i-1] > lower.iloc[i-1]:
                lower.iloc[i] = lower.iloc[i-1]
        else:
            st.iloc[i] = upper.iloc[i]
            if upper.iloc[i] > upper.iloc[i-1] and c.iloc[i-1] < upper.iloc[i-1]:
                upper.iloc[i] = upper.iloc[i-1]
    return direction, st

def bollinger(df, period=20, std=2.0):
    mid = sma(df["close"], period)
    std_val = df["close"].rolling(period).std()
    upper = mid + std * std_val
    lower = mid - std * std_val
    return upper, mid, lower

def volume_ratio(df, period=20):
    avg_vol = df["volume"].rolling(period).mean()
    return df["volume"] / (avg_vol + 1e-10)

def find_swings(df, lookback=5):
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    for i in range(lookback, len(df) - lookback):
        if h[i] == max(h[i-lookback:i+lookback+1]):
            highs.append((i, h[i]))
        if l[i] == min(l[i-lookback:i+lookback+1]):
            lows.append((i, l[i]))
    return highs, lows
