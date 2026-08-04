"""
Feature Engineering Engine — 40+ features for crypto trading
Based on best practices from Kaggle/GitHub research
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from core import indicators as ind


class FeatureEngine:
    """
    Comprehensive feature engineering:
    1. Trend features (EMA, Supertrend, ADX)
    2. Momentum features (RSI, MACD, ROC)
    3. Volatility features (ATR, BB, historical vol)
    4. Volume features (OBV, volume ratio, VWAP)
    5. Structure features (swing points, support/resistance)
    6. Pattern features (candlestick patterns)
    7. Cross-timeframe features
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute(self, df):
        """Compute all features"""
        if df is None or len(df) < 50:
            return None
        
        f = pd.DataFrame(index=df.index)
        
        # ── Trend Features ──
        f["ema_8"] = ind.ema(df["close"], 8) / df["close"]
        f["ema_20"] = ind.ema(df["close"], 20) / df["close"]
        f["ema_50"] = ind.ema(df["close"], 50) / df["close"] if len(df) > 50 else f["ema_20"]
        f["ema_cross_8_20"] = f["ema_8"] - f["ema_20"]
        f["ema_cross_20_50"] = f["ema_20"] - f["ema_50"]
        
        st_dir, st_val = ind.supertrend(df)
        f["supertrend_dir"] = st_dir
        f["supertrend_dist"] = (df["close"] - st_val) / df["close"]
        
        adx_v, dip, dim = ind.adx(df)
        f["adx"] = adx_v / 100
        f["di_diff"] = (dip - dim) / 100
        
        # ── Momentum Features ──
        f["rsi_14"] = ind.rsi(df, 14) / 100
        f["rsi_7"] = ind.rsi(df, 7) / 100
        f["rsi_diff"] = f["rsi_14"] - f["rsi_7"]
        
        macd_line, signal_line, histogram = ind.macd(df)
        f["macd_hist"] = histogram / df["close"]
        f["macd_signal_diff"] = (macd_line - signal_line) / df["close"]
        f["macd_cross"] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(float)
        
        # Rate of Change
        f["roc_5"] = df["close"].pct_change(5)
        f["roc_10"] = df["close"].pct_change(10)
        f["roc_20"] = df["close"].pct_change(20)
        
        # ── Volatility Features ──
        atr = ind.atr(df, 14)
        f["atr_14"] = atr / df["close"]
        f["atr_ratio"] = atr / atr.rolling(20).mean()
        
        upper, mid, lower = ind.bollinger(df)
        f["bb_width"] = (upper - lower) / mid
        f["bb_position"] = (df["close"] - lower) / (upper - lower + 1e-10)
        f["bb_pctb"] = (df["close"] - lower) / (upper - lower + 1e-10)
        
        # Historical volatility
        f["hvol_10"] = df["close"].pct_change().rolling(10).std()
        f["hvol_20"] = df["close"].pct_change().rolling(20).std()
        f["vol_ratio"] = f["hvol_10"] / (f["hvol_20"] + 1e-10)
        
        # ── Volume Features ──
        f["vol_ratio_ind"] = ind.volume_ratio(df)
        f["vol_change"] = df["volume"].pct_change()
        f["vol_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        
        # OBV approximation
        obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        f["obv_slope"] = obv.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True)
        
        # ── Structure Features ──
        f["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
        f["close_open_ratio"] = (df["close"] - df["open"]) / df["open"]
        f["body_size"] = abs(df["close"] - df["open"]) / df["close"]
        f["upper_wick"] = (df["high"] - df[["close","open"]].max(axis=1)) / df["close"]
        f["lower_wick"] = (df[["close","open"]].min(axis=1) - df["low"]) / df["close"]
        
        # Higher highs / Lower lows
        f["higher_high"] = ((df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2))).astype(float)
        f["higher_low"] = ((df["low"] > df["low"].shift(1)) & (df["low"].shift(1) > df["low"].shift(2))).astype(float)
        f["lower_high"] = ((df["high"] < df["high"].shift(1)) & (df["high"].shift(1) < df["high"].shift(2))).astype(float)
        f["lower_low"] = ((df["low"] < df["low"].shift(1)) & (df["low"].shift(1) < df["low"].shift(2))).astype(float)
        
        # ── Lag Features ──
        for lag in [1, 2, 3, 5]:
            f["return_lag_%d" % lag] = f["close_open_ratio"].shift(lag)
            f["vol_lag_%d" % lag] = f["vol_ratio_ind"].shift(lag)
            f["rsi_lag_%d" % lag] = f["rsi_14"].shift(lag)
        
        # ── Pattern Features ──
        f["bullish_engulfing"] = ((df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] > df["open"].shift(1))).astype(float)
        f["bearish_engulfing"] = ((df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1)) & (df["close"] < df["open"].shift(1))).astype(float)
        f["hammer"] = (((df["close"] - df["low"]) > 2 * (df["high"] - df["close"])) & ((df["high"] - df["close"]) < (df["close"] - df["low"]) * 0.3)).astype(float)
        
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
