"""
TCN-Inspired Model — Temporal Convolutional Pattern Detection
Uses sliding window features + sklearn ensemble
Mimics TCN's ability to capture multi-scale temporal patterns
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from collections import Counter

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml.features import compute_features


class TCNHybrid:
    """
    TCN-inspired: Multi-scale sliding window features
    Captures patterns at different time horizons
    """
    name = "TCN_Hybrid"
    
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.windows = [3, 5, 10, 20]  # Multiple time scales
    
    def _create_temporal_features(self, df):
        """Create multi-scale temporal features (mimics TCN convolutions)"""
        features = {}
        
        for w in self.windows:
            if len(df) < w + 1:
                continue
            
            # Price patterns at this scale
            features["return_%d" % w] = df["close"].pct_change(w).iloc[-1]
            features["vol_change_%d" % w] = df["volume"].pct_change(w).iloc[-1]
            
            # Volatility at this scale
            features["volatility_%d" % w] = df["close"].pct_change().rolling(w).std().iloc[-1]
            
            # Trend strength at this scale
            ema = df["close"].ewm(span=w).mean()
            features["trend_%d" % w] = (df["close"].iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]
            
            # High-Low range
            features["range_%d" % w] = (df["high"].iloc[-w:].max() - df["low"].iloc[-w:].min()) / df["close"].iloc[-1]
        
        # Cross-scale features
        for i in range(len(self.windows)-1):
            w1, w2 = self.windows[i], self.windows[i+1]
            if "return_%d" % w1 in features and "return_%d" % w2 in features:
                features["momentum_%d_%d" % (w1, w2)] = features["return_%d" % w1] - features["return_%d" % w2]
        
        return features
    
    def train(self, df, horizon=5, threshold=0.001):
        """Train on historical data"""
        if not HAS_SKLEARN or df is None or len(df) < 100:
            return False
        
        # Create features for each bar
        X_list = []
        y_list = []
        
        for i in range(50, len(df) - horizon):
            window = df.iloc[i-50:i+1]
            feats = self._create_temporal_features(window)
            
            if not feats:
                continue
            
            # Label
            future_return = (df["close"].iloc[i+horizon] - df["close"].iloc[i]) / df["close"].iloc[i]
            label = 1 if future_return > threshold else 0
            
            X_list.append(feats)
            y_list.append(label)
        
        if len(X_list) < 30:
            return False
        
        # Convert to DataFrame
        X = pd.DataFrame(X_list)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        y = np.array(y_list)
        
        # Train
        self.scaler.fit(X)
        X_s = self.scaler.transform(X)
        
        self.models = [
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        ]
        
        for model in self.models:
            model.fit(X_s, y)
        
        # Evaluate
        pred = self._predict_all(X_s)
        self.train_accuracy = accuracy_score(y, pred)
        self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict using multi-scale patterns"""
        if not self.is_trained:
            return 0.5, "NEUTRAL"
        
        feats = self._create_temporal_features(df)
        if not feats:
            return 0.5, "NEUTRAL"
        
        X = pd.DataFrame([feats])
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Ensure all training columns exist
        for col in self.scaler.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X = X[self.scaler.feature_names_in_]
        
        X_s = self.scaler.transform(X)
        
        # Ensemble prediction
        probs = []
        for model in self.models:
            prob = model.predict_proba(X_s)[:, 1][0]
            probs.append(prob)
        
        avg_prob = np.mean(probs)
        
        if avg_prob > 0.55:
            direction = "BUY"
        elif avg_prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return avg_prob, direction
    
    def _predict_all(self, X):
        votes = []
        for model in self.models:
            votes.append(model.predict(X))
        return np.round(np.mean(votes, axis=0)).astype(int)
