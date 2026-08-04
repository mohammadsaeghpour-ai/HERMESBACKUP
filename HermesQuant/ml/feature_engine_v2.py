"""
Feature Engine v2 — Combines basic + advanced features (70+ total)
Based on deepalpha research
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from core import indicators as ind
from ml.advanced_features import compute_advanced_features


class FeatureEngineV2:
    """
    70+ features combining:
    - Basic technical (40+)
    - Advanced (Hurst, VPIN, Fractal, MTF)
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute(self, df):
        """Compute all features"""
        if df is None or len(df) < 50:
            return None
        
        f = pd.DataFrame(index=df.index)
        
        # ── Basic Technical Features ──
        f["ema_8"] = ind.ema(df["close"], 8) / df["close"]
        f["ema_20"] = ind.ema(df["close"], 20) / df["close"]
        f["ema_50"] = ind.ema(df["close"], 50) / df["close"] if len(df) > 50 else f["ema_20"]
        f["ema_cross_8_20"] = f["ema_8"] - f["ema_20"]
        
        st_dir, st_val = ind.supertrend(df)
        f["supertrend_dir"] = st_dir
        f["supertrend_dist"] = (df["close"] - st_val) / df["close"]
        
        adx_v, dip, dim = ind.adx(df)
        f["adx"] = adx_v / 100
        f["di_diff"] = (dip - dim) / 100
        
        f["rsi_14"] = ind.rsi(df, 14) / 100
        f["rsi_7"] = ind.rsi(df, 7) / 100
        
        macd_line, signal_line, histogram = ind.macd(df)
        f["macd_hist"] = histogram / df["close"]
        f["macd_signal_diff"] = (macd_line - signal_line) / df["close"]
        
        f["roc_5"] = df["close"].pct_change(5)
        f["roc_10"] = df["close"].pct_change(10)
        
        atr = ind.atr(df, 14)
        f["atr_14"] = atr / df["close"]
        f["atr_ratio"] = atr / atr.rolling(20).mean()
        
        upper, mid, lower = ind.bollinger(df)
        f["bb_width"] = (upper - lower) / mid
        f["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-10)
        
        f["hvol_10"] = df["close"].pct_change().rolling(10).std()
        f["hvol_20"] = df["close"].pct_change().rolling(20).std()
        f["vol_ratio"] = f["hvol_10"] / (f["hvol_20"] + 1e-10)
        
        f["vol_ratio_ind"] = ind.volume_ratio(df)
        f["vol_change"] = df["volume"].pct_change()
        f["vol_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        
        f["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
        f["close_open_ratio"] = (df["close"] - df["open"]) / df["open"]
        f["body_size"] = abs(df["close"] - df["open"]) / df["close"]
        
        f["higher_high"] = ((df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2))).astype(float)
        f["higher_low"] = ((df["low"] > df["low"].shift(1)) & (df["low"].shift(1) > df["low"].shift(2))).astype(float)
        f["lower_high"] = ((df["high"] < df["high"].shift(1)) & (df["high"].shift(1) < df["high"].shift(2))).astype(float)
        f["lower_low"] = ((df["low"] < df["low"].shift(1)) & (df["low"].shift(1) < df["low"].shift(2))).astype(float)
        
        for lag in [1, 2, 3, 5]:
            f["return_lag_%d" % lag] = f["close_open_ratio"].shift(lag)
            f["vol_lag_%d" % lag] = f["vol_ratio_ind"].shift(lag)
            f["rsi_lag_%d" % lag] = f["rsi_14"].shift(lag)
        
        f["bullish_engulfing"] = ((df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] > df["open"].shift(1))).astype(float)
        f["bearish_engulfing"] = ((df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1)) & (df["close"] < df["open"].shift(1))).astype(float)
        
        # ── Advanced Features (from deepalpha) ──
        advanced = compute_advanced_features(df)
        if advanced is not None:
            for col in advanced.columns:
                f[col] = advanced[col]
        
        # Clean
        f = f.replace([np.inf, -np.inf], np.nan)
        f = f.dropna()
        
        self.feature_names = f.columns.tolist()
        return f
    
    def compute_labels(self, df, horizon=5, threshold=0.001):
        """Create labels"""
        future_return = df["close"].pct_change(horizon).shift(-horizon)
        labels = (future_return > threshold).astype(int)
        return labels
