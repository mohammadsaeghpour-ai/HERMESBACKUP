"""
Hybrid TCN (Temporal Convolutional Network) for Crypto Trading
================================================================
Mimics TCN architecture using sklearn ensemble methods.

Architecture inspiration:
- TCN uses dilated causal convolutions to capture multi-scale temporal patterns
- We simulate this by: (1) building features at multiple dilation rates via
  rolling statistics, then (2) training a GradientBoosting ensemble on the
  concatenated multi-scale feature maps
- Each "dilation level" = a different rolling window, equivalent to kernel
  dilation in a real TCN

Key advantages over plain ML:
- Multi-scale temporal features (captures patterns at 4h, 12h, 24h, 48h)
- Temporal ordering preserved through shift/lag features
- Non-linear interaction learning via GBM

Input: OHLCV DataFrame (same as existing ml_engine)
Output: (probability, direction, confidence)
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ml.features import compute_features, create_labels


class HybridTCN:
    """
    Temporal Convolutional Network approximation.
    
    How it works:
    ─────────────
    A real TCN has layers of dilated causal convolutions:
    
        Layer 0 (dilation=1):  conv([t-1, t])     → local pattern
        Layer 1 (dilation=2):  conv([t-2, t])     → wider pattern  
        Layer 2 (dilation=4):  conv([t-4, t])     → even wider
        Layer 3 (dilation=8):  conv([t-8, t])     → broad trend
    
    We simulate this by computing rolling statistics at each dilation rate,
    then concatenating them into a multi-scale feature tensor, and training
    a GradientBoosting classifier on the combined features.
    
    The "kernel size" is simulated by using mean + std + min/max over the
    dilation window, giving us the receptive field of a convolution kernel.
    """
    
    # TCN dilation schedule (powers of 2, like real TCN)
    DILATIONS = [1, 2, 4, 8, 16]
    BASE_PERIOD = 1  # candles
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = []
        self.is_trained = False
        self.train_accuracy = 0
        self.feature_names = []
    
    def _build_tcn_features(self, df):
        """
        Build multi-scale temporal convolution features.
        
        For each dilation rate d, compute:
        - Rolling mean over d periods (captures trend at this scale)
        - Rolling std over d periods (captures volatility at this scale)  
        - Rolling min/max ratio (captures range at this scale)
        - Rate of change between scales (captures acceleration)
        """
        if df is None or len(df) < max(self.DILATIONS) * 3:
            return None
        
        f = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]
        returns = close.pct_change()
        
        # ── Dilation Level 0: Raw (dilation=1) ──
        f["tcn_d0_ret"] = returns
        f["tcn_d0_vol"] = volume / volume.rolling(5).mean()
        f["tcn_d0_body"] = (df["close"] - df["open"]) / df["open"]
        f["tcn_d0_wick_up"] = (df["high"] - df[["close","open"]].max(axis=1)) / close
        f["tcn_d0_wick_dn"] = (df[["close","open"]].min(axis=1) - df["low"]) / close
        
        # ── Dilation Levels 1-4: Multi-scale convolutions ──
        prev_means = None
        for i, dilation in enumerate(self.DILATIONS):
            w = dilation * 2 + 1  # window size = 2*dilation + 1 (like kernel)
            
            # Mean convolution (dilated average)
            roll_mean = returns.rolling(w, min_periods=dilation).mean()
            roll_std = returns.rolling(w, min_periods=dilation).std()
            roll_max = returns.rolling(w, min_periods=dilation).max()
            roll_min = returns.rolling(w, min_periods=dilation).min()
            
            f[f"tcn_d{i}_mean"] = roll_mean
            f[f"tcn_d{i}_std"] = roll_std
            f[f"tcn_d{i}_range"] = roll_max - roll_min
            f[f"tcn_d{i}_skew"] = returns.rolling(w, min_periods=max(dilation, 3)).skew()
            
            # Volume-weighted version
            vw = (returns * volume).rolling(w, min_periods=dilation).sum() / \
                 (volume.rolling(w, min_periods=dilation).sum() + 1e-10)
            f[f"tcn_d{i}_vwap_dev"] = roll_mean - vw
            
            # Cross-scale features (like skip connections in TCN)
            if prev_means is not None:
                f[f"tcn_diff_{i}_prev"] = roll_mean - prev_means
            prev_means = roll_mean
        
        # ── Channel mixing: inter-channel features (like 1x1 conv) ──
        f["tcn_price_accel"] = returns.diff()  # 2nd derivative
        f["tcn_price_jerk"] = returns.diff().diff()  # 3rd derivative
        
        vol_ma = volume.rolling(20).mean()
        f["tcn_vol_regime"] = volume / vol_ma
        f["tcn_vol_trend"] = vol_ma / vol_ma.shift(20)
        
        f["tcn_pv_corr_5"] = returns.rolling(5).corr(volume.pct_change())
        f["tcn_pv_corr_20"] = returns.rolling(20).corr(volume.pct_change())
        
        f = f.replace([np.inf, -np.inf], np.nan)
        return f
    
    def train(self, df, horizon=5, threshold=0.001, train_ratio=0.7):
        """Train TCN hybrid on historical data."""
        tcn_features = self._build_tcn_features(df)
        if tcn_features is None or len(tcn_features) < 100:
            return False
        
        # Also include base features for richer signal
        base_features = compute_features(df)
        
        # Combine TCN + base features
        if base_features is not None:
            common_idx = tcn_features.index.intersection(base_features.index)
            features = pd.concat([
                tcn_features.loc[common_idx],
                base_features.loc[common_idx]
            ], axis=1)
        else:
            features = tcn_features
        
        labels = create_labels(df, horizon, threshold)
        
        # Align
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        # Clean
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            return False
        
        self.feature_names = X.columns.tolist()
        
        # Time-series aware split (no shuffling!)
        split = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # Train multi-scale ensemble (mimics TCN's hierarchical processing)
        self.models = [
            ("gb_shallow", GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42)),
            ("gb_deep", GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08,
                subsample=0.7, random_state=43)),
            ("rf_wide", RandomForestClassifier(
                n_estimators=200, max_depth=8, max_features="sqrt",
                random_state=44)),
        ]
        
        for name, model in self.models:
            model.fit(X_train_s, y_train)
        
        # Evaluate
        preds = self._ensemble_predict(X_test_s)
        self.train_accuracy = accuracy_score(y_test, preds)
        self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict direction for latest bar."""
        if not self.is_trained:
            return 0.5, "NEUTRAL", 0.5
        
        tcn_features = self._build_tcn_features(df)
        base_features = compute_features(df)
        
        if tcn_features is None or len(tcn_features) < 1:
            return 0.5, "NEUTRAL", 0.5
        
        if base_features is not None:
            idx = tcn_features.index.intersection(base_features.index)
            if len(idx) == 0:
                return 0.5, "NEUTRAL", 0.5
            features = pd.concat([
                tcn_features.loc[idx].iloc[[-1]],
                base_features.loc[idx].iloc[[-1]]
            ], axis=1)
        else:
            features = tcn_features.iloc[[-1]]
        
        if not features.notna().all(axis=1).iloc[0]:
            return 0.5, "NEUTRAL", 0.5
        
        X_s = self.scaler.transform(features)
        
        probs = []
        for name, model in self.models:
            p = model.predict_proba(X_s)[0]
            probs.append(p[1])
        
        avg_prob = np.mean(probs)
        std_prob = np.std(probs)
        
        if avg_prob > 0.55:
            direction = "BUY"
        elif avg_prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        agreement = 1.0 - std_prob * 4
        confidence = min(max(avg_prob, 1 - avg_prob) * 100 * agreement, 90)
        
        return avg_prob, direction, confidence
    
    def _ensemble_predict(self, X):
        """Majority vote prediction."""
        votes = []
        for name, model in self.models:
            votes.append(model.predict(X))
        return np.round(np.mean(votes, axis=0)).astype(int)
