"""
Feature Engineering Pipeline for ML Models
Generates 40+ features from OHLCV data
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from core import indicators as ind

def compute_features(df):
    """Compute all features from OHLCV dataframe"""
    if df is None or len(df) < 50:
        return None
    
    f = pd.DataFrame(index=df.index)
    
    # ── Price Features ──
    f["returns_1"] = df["close"].pct_change(1)
    f["returns_5"] = df["close"].pct_change(5)
    f["returns_10"] = df["close"].pct_change(10)
    f["returns_20"] = df["close"].pct_change(20)
    
    f["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
    f["close_open_ratio"] = (df["close"] - df["open"]) / df["open"]
    
    # ── Trend Features ──
    f["ema_8"] = ind.ema(df["close"], 8) / df["close"]
    f["ema_20"] = ind.ema(df["close"], 20) / df["close"]
    f["ema_50"] = ind.ema(df["close"], 50) / df["close"]
    f["ema_cross_8_20"] = (f["ema_8"] - f["ema_20"])
    f["ema_cross_20_50"] = (f["ema_20"] - f["ema_50"])
    
    st_dir, st_val = ind.supertrend(df)
    f["supertrend_dir"] = st_dir
    f["supertrend_dist"] = (df["close"] - st_val) / df["close"]
    
    # ── Momentum Features ──
    f["rsi_14"] = ind.rsi(df, 14) / 100
    f["rsi_7"] = ind.rsi(df, 7) / 100
    
    macd_line, signal_line, histogram = ind.macd(df)
    f["macd_hist"] = histogram / df["close"]
    f["macd_signal_diff"] = (macd_line - signal_line) / df["close"]
    
    # ── Volatility Features ──
    atr_val = ind.atr(df, 14)
    f["atr_14"] = atr_val / df["close"]
    f["atr_ratio"] = atr_val / atr_val.rolling(20).mean()
    
    upper, mid, lower = ind.bollinger(df)
    f["bb_width"] = (upper - lower) / mid
    f["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-10)
    
    adx_v, dip, dim = ind.adx(df)
    f["adx"] = adx_v / 100
    f["di_diff"] = (dip - dim) / 100
    
    # ── Volume Features ──
    f["vol_ratio"] = ind.volume_ratio(df)
    f["vol_change"] = df["volume"].pct_change()
    f["vol_price_corr"] = df["volume"].rolling(10).corr(df["close"].pct_change())
    
    # ── Pattern Features ──
    f["upper_wick"] = (df["high"] - df[["close","open"]].max(axis=1)) / df["close"]
    f["lower_wick"] = (df[["close","open"]].min(axis=1) - df["low"]) / df["close"]
    f["body_size"] = abs(df["close"] - df["open"]) / df["close"]
    
    # ── Structure Features ──
    highs, lows = ind.find_swings(df, lookback=3)
    f["swing_highs"] = 0.0
    f["swing_lows"] = 0.0
    for idx, val in highs[-10:]:
        if idx < len(f):
            f.iloc[idx, f.columns.get_loc("swing_highs")] = float(val)
    for idx, val in lows[-10:]:
        if idx < len(f):
            f.iloc[idx, f.columns.get_loc("swing_lows")] = float(val)
    
    # ── Lag Features ──
    for lag in [1, 2, 3, 5]:
        f["returns_lag_%d" % lag] = f["returns_1"].shift(lag)
        f["vol_ratio_lag_%d" % lag] = f["vol_ratio"].shift(lag)
        f["rsi_lag_%d" % lag] = f["rsi_14"].shift(lag)
    
    # Clean
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.dropna()
    
    return f

def create_labels(df, horizon=5, threshold=0.001):
    """Create classification labels: 1=BUY, 0=SELL"""
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = (future_return > threshold).astype(int)  # 1=up, 0=down
    return labels
