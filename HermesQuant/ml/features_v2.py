"""
Feature Engineering v2 — Better features for BUY signals
Focus on momentum reversal, support levels, and volume patterns
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from core import indicators as ind

def compute_features_v2(df):
    """Enhanced features with BUY-specific signals"""
    if df is None or len(df) < 50:
        return None
    
    f = pd.DataFrame(index=df.index)
    
    # ── Price Features ──
    f["returns_1"] = df["close"].pct_change(1)
    f["returns_3"] = df["close"].pct_change(3)
    f["returns_5"] = df["close"].pct_change(5)
    f["returns_10"] = df["close"].pct_change(10)
    f["returns_20"] = df["close"].pct_change(20)
    
    # ── BUY-Specific: Reversal Signals ──
    # RSI oversold bounce
    rsi = ind.rsi(df, 14)
    f["rsi_14"] = rsi / 100
    f["rsi_oversold"] = (rsi < 30).astype(float)
    f["rsi_bounce"] = ((rsi > rsi.shift(1)) & (rsi.shift(1) < 35)).astype(float)
    
    # MACD crossover
    macd_line, signal_line, histogram = ind.macd(df)
    f["macd_cross_up"] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(float)
    f["macd_hist_positive"] = (histogram > 0).astype(float)
    f["macd_hist_increasing"] = (histogram > histogram.shift(1)).astype(float)
    
    # Bollinger Band bounce
    upper, mid, lower = ind.bollinger(df)
    f["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-10)
    f["bb_oversold"] = (df["close"] < lower * 1.001).astype(float)
    f["bb_bounce"] = ((df["close"] > lower) & (df["close"].shift(1) <= lower.shift(1))).astype(float)
    
    # EMA crossover
    f["ema_8"] = ind.ema(df["close"], 8) / df["close"]
    f["ema_20"] = ind.ema(df["close"], 20) / df["close"]
    f["ema_cross_up"] = ((f["ema_8"] > f["ema_20"]) & (f["ema_8"].shift(1) <= f["ema_20"].shift(1))).astype(float)
    
    # Supertrend
    st_dir, st_val = ind.supertrend(df)
    f["supertrend_dir"] = st_dir
    f["supertrend_cross_up"] = ((st_dir == 1) & (st_dir.shift(1) == -1)).astype(float)
    
    # ── Volume Features ──
    f["vol_ratio"] = ind.volume_ratio(df)
    f["vol_spike"] = (df["volume"] > df["volume"].rolling(20).mean() * 1.5).astype(float)
    f["vol_price_divergence"] = ((df["volume"].pct_change() > 0) & (df["close"].pct_change() < 0)).astype(float)
    
    # ── Volatility Features ──
    atr = ind.atr(df, 14)
    f["atr_14"] = atr / df["close"]
    f["atr_ratio"] = atr / atr.rolling(20).mean()
    
    # ── Structure Features ──
    f["higher_low"] = ((df["low"] > df["low"].shift(1)) & (df["low"].shift(1) > df["low"].shift(2))).astype(float)
    f["higher_high"] = ((df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2))).astype(float)
    f["lower_low"] = ((df["low"] < df["low"].shift(1)) & (df["low"].shift(1) < df["low"].shift(2))).astype(float)
    f["lower_high"] = ((df["high"] < df["high"].shift(1)) & (df["high"].shift(1) < df["high"].shift(2))).astype(float)
    
    # ── Pattern Features ──
    f["bullish_engulfing"] = ((df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] > df["open"].shift(1))).astype(float)
    f["hammer"] = (((df["close"] - df["low"]) > 2 * (df["high"] - df["close"])) & ((df["high"] - df["close"]) < (df["close"] - df["low"]) * 0.3)).astype(float)
    
    # ── Lag Features ──
    for lag in [1, 2, 3, 5]:
        f["returns_lag_%d" % lag] = f["returns_1"].shift(lag)
        f["vol_ratio_lag_%d" % lag] = f["vol_ratio"].shift(lag)
        f["rsi_lag_%d" % lag] = f["rsi_14"].shift(lag)
    
    # Clean
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.dropna()
    
    return f

def create_labels_v2(df, horizon=5, threshold=0.001):
    """Create labels with lower threshold for more BUY signals"""
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = (future_return > threshold).astype(int)
    return labels
