"""
Meta-Model v2 — Enhanced with market context features
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from core import indicators as ind


class MetaModelV2:
    """
    Enhanced meta-model with:
    1. Agent outputs (direction, confidence, score)
    2. Market context (trend, volatility, regime)
    3. Time features (session, hour)
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.metrics = {}
    
    def _extract_market_context(self, df):
        """Extract market context features"""
        if df is None or len(df) < 50:
            return [0] * 8
        
        price = df["close"].iloc[-1]
        
        # Trend
        ema8 = ind.ema(df["close"], 8).iloc[-1]
        ema20 = ind.ema(df["close"], 20).iloc[-1]
        trend = 1 if ema8 > ema20 else (-1 if ema8 < ema20 else 0)
        
        # Volatility
        atr = ind.atr(df, 14).iloc[-1]
        atr_pct = atr / price * 100
        vol_level = 1 if atr_pct > 2 else (0 if atr_pct < 1 else 0.5)
        
        # ADX
        adx_v, _, _ = ind.adx(df)
        adx = adx_v.iloc[-1] / 100
        
        # RSI
        rsi = ind.rsi(df, 14).iloc[-1] / 100
        
        # Volume
        vol_ratio = ind.volume_ratio(df).iloc[-1]
        
        # Price position in range
        high_20 = df["high"].rolling(20).max().iloc[-1]
        low_20 = df["low"].rolling(20).min().iloc[-1]
        price_pos = (price - low_20) / (high_20 - low_20 + 1e-10)
        
        # Supertrend
        st_dir, _ = ind.supertrend(df)
        st = st_dir.iloc[-1]
        
        return [trend, vol_level, adx, rsi, vol_ratio, price_pos, st, atr_pct]
    
    def _prepare_features(self, agent_results, df):
        """Combine agent outputs with market context"""
        agent_feats = []
        for r in agent_results:
            if r.direction == "BUY":
                agent_feats.extend([1, r.confidence / 100, r.score])
            elif r.direction == "SELL":
                agent_feats.extend([-1, r.confidence / 100, r.score])
            else:
                agent_feats.extend([0, 0, 0])
        
        market_feats = self._extract_market_context(df)
        
        return np.array(agent_feats + market_feats).reshape(1, -1)
    
    def train(self, all_agent_results, all_dfs, actual_labels):
        """Train meta-model"""
        if not HAS_SKLEARN:
            return False
        
        X_list = []
        for agents_r, df in zip(all_agent_results, all_dfs):
            feat = self._prepare_features(agents_r, df)
            X_list.append(feat[0])
        
        X = np.array(X_list)
        y = np.array(actual_labels)
        
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 30:
            return False
        
        split = int(len(X) * 0.75)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        base = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        self.model.fit(X_train_s, y_train)
        
        pred = self.model.predict(X_test_s)
        proba = self.model.predict_proba(X_test_s)[:, 1]
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        
        self.is_trained = True
        return True
    
    def predict(self, agent_results, df):
        """Predict with market context"""
        if not self.is_trained:
            return True, 0.5
        
        feat = self._prepare_features(agent_results, df)
        
        if np.isnan(feat).any():
            return True, 0.5
        
        feat_s = self.scaler.transform(feat)
        prob = self.model.predict_proba(feat_s)[:, 1][0]
        
        return prob > 0.5, prob
