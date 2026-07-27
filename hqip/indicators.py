"""HQIP Technical Indicators - All calculations in one module."""
import numpy as np, pandas as pd

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series, period=20, std_dev=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std

def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = atr(df, 1) * 14
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * ema(plus_dm, period) / atr_val.replace(0, 1e-10)
    minus_di = 100 * ema(minus_dm, period) / atr_val.replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx_val = ema(dx, period)
    return adx_val, plus_di, minus_di

def stochastic(df, k_period=14, d_period=3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
    d = sma(k, d_period)
    return k, d

def obv(df):
    obv_val = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv_val.iloc[i] = obv_val.iloc[i-1] + df["volume"].iloc[i]
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv_val.iloc[i] = obv_val.iloc[i-1] - df["volume"].iloc[i]
        else:
            obv_val.iloc[i] = obv_val.iloc[i-1]
    return obv_val

def vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, 1e-10)

def supertrend(df, period=10, multiplier=3):
    atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val
    st = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        if direction.iloc[i] == 1:
            lower.iloc[i] = max(lower.iloc[i], lower.iloc[i-1]) if direction.iloc[i-1] == 1 else lower.iloc[i]
            st.iloc[i] = lower.iloc[i]
        else:
            upper.iloc[i] = min(upper.iloc[i], upper.iloc[i-1]) if direction.iloc[i-1] == -1 else upper.iloc[i]
            st.iloc[i] = upper.iloc[i]
    return st, direction

def donchian(df, period=20):
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid = (upper + lower) / 2
    return upper, mid, lower

def volume_profile(df, bins=50):
    price_bins = np.linspace(df["low"].min(), df["high"].max(), bins)
    vol_at_price = np.zeros(bins)
    for i in range(len(df)):
        mask = (price_bins >= df["low"].iloc[i]) & (price_bins <= df["high"].iloc[i])
        vol_at_price[mask] += df["volume"].iloc[i] / max(mask.sum(), 1)
    poc_idx = np.argmax(vol_at_price)
    poc = price_bins[poc_idx]
    total_vol = vol_at_price.sum() * 0.7
    cum = np.cumsum(vol_at_price[poc_idx:])
    vah_idx = poc_idx + np.searchsorted(cum, total_vol * 0.5)
    cum_back = np.cumsum(vol_at_price[poc_idx::-1])
    val_idx = poc_idx - np.searchsorted(cum_back, total_vol * 0.5)
    vah = price_bins[min(vah_idx, bins-1)]
    val = price_bins[max(val_idx, 0)]
    return poc, vah, val

def support_resistance(df, lookback=20):
    highs = df["high"].rolling(lookback, center=True).max()
    lows = df["low"].rolling(lookback, center=True).min()
    resistance = df.loc[highs == df["high"], "high"].tail(3).values
    support = df.loc[lows == df["low"], "low"].tail(3).values
    return support, resistance

def calculate_all_indicators(df):
    if df.empty or len(df) < 30:
        return df
    df = df.copy()
    # EMAs
    for p in [9, 20, 50, 100, 200]:
        df[f"ema{p}"] = ema(df["close"], p)
    # SMA
    df["sma20"] = sma(df["close"], 20)
    # RSI
    df["rsi"] = rsi(df["close"], 14)
    df["rsi7"] = rsi(df["close"], 7)
    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    # Bollinger
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(df["close"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, 1e-10)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-10)
    # ATR
    df["atr"] = atr(df, 14)
    df["atr_pct"] = df["atr"] / df["close"].replace(0, 1e-10) * 100
    # ADX
    df["adx"], df["plus_di"], df["minus_di"] = adx(df, 14)
    # Stochastic
    df["stoch_k"], df["stoch_d"] = stochastic(df)
    # OBV
    df["obv"] = obv(df)
    df["obv_ema"] = ema(df["obv"], 20)
    # VWAP
    df["vwap"] = vwap(df)
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, 1e-10) * 100
    # SuperTrend
    df["st_line"], df["st_dir"] = supertrend(df)
    # Donchian
    df["dc_upper"], df["dc_mid"], df["dc_lower"] = donchian(df, 20)
    # Volume
    df["vol_sma"] = sma(df["volume"], 20)
    df["vol_ratio"] = df["volume"] / df["vol_sma"].replace(0, 1e-10)
    # Momentum helpers
    df["roc"] = df["close"].pct_change(10) * 100
    df["momentum"] = df["close"] - df["close"].shift(10)
    # Returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    # Body
    df["body"] = df["close"] - df["open"]
    df["body_pct"] = df["body"].abs() / df["atr"].replace(0, 1e-10)
    df["bullish_candle"] = (df["close"] > df["open"]).astype(int)
    return df
