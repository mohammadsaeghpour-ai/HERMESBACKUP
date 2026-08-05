"""
Technical Indicators — All pure pandas/numpy, no external deps
"""
import numpy as np
import pandas as pd


def ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    """Simple Moving Average"""
    return series.rolling(period).mean()


def rsi(df, period=14):
    """Relative Strength Index"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def macd(df, fast=12, slow=26, signal=9):
    """MACD"""
    ema_fast = ema(df["close"], fast)
    ema_slow = ema(df["close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df, period=14):
    """Average True Range"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bollinger(df, period=20, std_dev=2):
    """Bollinger Bands"""
    mid = sma(df["close"], period)
    std = df["close"].rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def supertrend(df, period=10, multiplier=3):
    """Supertrend"""
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1
    
    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
    
    return direction, supertrend


def adx(df, period=14):
    """Average Directional Index"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / (atr_val + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).mean() / (atr_val + 1e-10)
    
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    adx_val = dx.rolling(period).mean()
    
    return adx_val, plus_di, minus_di


def stochastic(df, k_period=14, d_period=3):
    """Stochastic Oscillator"""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def vwap(df):
    """Volume Weighted Average Price"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()


def obv(df):
    """On Balance Volume"""
    return (np.sign(df["close"].diff()) * df["volume"]).cumsum()


def volume_ratio(df, period=20):
    """Volume Ratio"""
    return df["volume"] / (df["volume"].rolling(period).mean() + 1e-10)
