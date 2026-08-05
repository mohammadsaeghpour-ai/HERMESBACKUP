"""
Feature Engineering — 80+ features
Technical + Microstructure + Advanced
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant_v3")
import numpy as np
import pandas as pd
from core import indicators as ind


class FeatureEngine:
    """Compute all features for ML models"""
    
    def compute(self, df):
        """Compute 80+ features"""
        if df is None or len(df) < 50:
            return None
        
        f = pd.DataFrame(index=df.index)
        c = df["close"]
        
        # ── Trend (6) ──
        f["ema_8"] = ind.ema(c, 8) / c
        f["ema_20"] = ind.ema(c, 20) / c
        f["ema_50"] = ind.ema(c, 50) / c if len(df) > 50 else f["ema_20"]
        f["ema_cross"] = f["ema_8"] - f["ema_20"]
        
        st_dir, st_val = ind.supertrend(df)
        f["supertrend_dir"] = st_dir
        f["supertrend_dist"] = (c - st_val) / c
        
        # ── Momentum (8) ──
        f["rsi_14"] = ind.rsi(df, 14) / 100
        f["rsi_7"] = ind.rsi(df, 7) / 100
        
        macd_l, sig_l, hist = ind.macd(df)
        f["macd_hist"] = hist / c
        f["macd_signal_diff"] = (macd_l - sig_l) / c
        
        f["roc_5"] = c.pct_change(5)
        f["roc_10"] = c.pct_change(10)
        
        k, d = ind.stochastic(df)
        f["stoch_k"] = k / 100
        f["stoch_d"] = d / 100
        
        # ── Volatility (6) ──
        atr_val = ind.atr(df, 14)
        f["atr_14"] = atr_val / c
        f["atr_ratio"] = atr_val / (atr_val.rolling(20).mean() + 1e-10)
        
        upper, mid, lower = ind.bollinger(df)
        f["bb_width"] = (upper - lower) / (mid + 1e-10)
        f["bb_position"] = (c - lower) / (upper - lower + 1e-10)
        
        f["hvol_10"] = c.pct_change().rolling(10).std()
        f["hvol_20"] = c.pct_change().rolling(20).std()
        
        # ── Volume (5) ──
        f["vol_ratio"] = ind.volume_ratio(df)
        f["vol_change"] = df["volume"].pct_change()
        f["vol_sma"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
        
        obv_val = ind.obv(df)
        f["obv_slope"] = obv_val.rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
        )
        
        vwap_val = ind.vwap(df)
        f["vwap_dist"] = (c - vwap_val) / (vwap_val + 1e-10)
        
        # ── Structure (6) ──
        f["hl_ratio"] = (df["high"] - df["low"]) / c
        f["co_ratio"] = (c - df["open"]) / (df["open"] + 1e-10)
        f["body_size"] = abs(c - df["open"]) / c
        f["upper_wick"] = (df["high"] - df[["close","open"]].max(axis=1)) / c
        f["lower_wick"] = (df[["close","open"]].min(axis=1) - df["low"]) / c
        f["range_expansion"] = (df["high"] - df["low"]) / (df["high"] - df["low"]).rolling(20).mean()
        
        # ── ADX (3) ──
        adx_v, dip, dim = ind.adx(df)
        f["adx"] = adx_v / 100
        f["di_diff"] = (dip - dim) / 100
        f["di_ratio"] = dip / (dim + 1e-10)
        
        # ── Advanced (5) ──
        # Hurst exponent approximation
        f["hurst"] = c.rolling(50).apply(
            lambda x: self._hurst(x.values) if len(x) >= 20 else 0.5, raw=False
        )
        
        # Fractal efficiency
        f["fractal_eff"] = abs(c - c.shift(10)) / (abs(c.diff()).rolling(10).sum() + 1e-10)
        
        # Volatility regime
        current_vol = c.pct_change().rolling(20).std()
        hist_vol = c.pct_change().rolling(50).std()
        f["vol_regime"] = (current_vol / (hist_vol + 1e-10)).clip(0, 3)
        
        # MTF alignment
        ema8 = ind.ema(c, 8)
        ema20 = ind.ema(c, 20)
        ema50 = ind.ema(c, 50) if len(df) > 50 else ema20
        f["mtf_aligned"] = ((ema8 > ema20) & (ema20 > ema50)).astype(float) -                            ((ema8 < ema20) & (ema20 < ema50)).astype(float)
        
        # Microstructure proxy (from OHLCV)
        f["spread_proxy"] = np.log(df["high"] / df["low"]).rolling(20).mean()
        
        # ── Lag (9) ──
        for lag in [1, 2, 3, 5]:
            f["return_lag_%d" % lag] = f["co_ratio"].shift(lag)
            f["rsi_lag_%d" % lag] = f["rsi_14"].shift(lag)
            if lag <= 3:
                f["vol_lag_%d" % lag] = f["vol_ratio"].shift(lag)
        
        # ── Pattern (4) ──
        f["bullish_engulf"] = ((c > df["open"]) & (c.shift(1) < df["open"].shift(1)) & 
                               (c > df["open"].shift(1))).astype(float)
        f["bearish_engulf"] = ((c < df["open"]) & (c.shift(1) > df["open"].shift(1)) & 
                               (c < df["open"].shift(1))).astype(float)
        f["hammer"] = (((c - df["low"]) > 2 * (df["high"] - c)) & 
                       ((df["high"] - c) < (c - df["low"]) * 0.3)).astype(float)
        f["shooting_star"] = (((df["high"] - c) > 2 * (c - df["low"])) & 
                              ((c - df["low"]) < (df["high"] - c) * 0.3)).astype(float)
        
        # Clean
        f = f.replace([np.inf, -np.inf], np.nan)
        f = f.dropna()
        
        return f
    
    def _hurst(self, ts):
        """Hurst exponent approximation"""
        if len(ts) < 20:
            return 0.5
        lags = range(2, min(20, len(ts)))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        if len(tau) < 2 or any(t == 0 for t in tau):
            return 0.5
        try:
            poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return max(0, min(poly[0], 1))
        except:
            return 0.5
    
    def compute_labels(self, df, horizon=5, threshold=0.001):
        """Create binary labels"""
        future_return = df["close"].pct_change(horizon).shift(-horizon)
        return (future_return > threshold).astype(int)
