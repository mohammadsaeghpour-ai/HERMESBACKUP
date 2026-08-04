"""
Attention-Inspired Model — Feature Importance Weighting
Mimics attention by learning which features matter most
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml.features import compute_features


class AttentionHybrid:
    """
    Attention-inspired: Learns feature importance weights
    and applies them for prediction
    """
    name = "Attention_Hybrid"
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.feature_weights = None
        self.is_trained = False
    
    def train(self, df, horizon=5, threshold=0.001):
        """Train with attention-like feature weighting"""
        if not HAS_SKLEARN or df is None or len(df) < 100:
            return False
        
        features = compute_features(df)
        if features is None or len(features) < 50:
            return False
        
        # Create labels
        future_return = df["close"].pct_change(horizon).shift(-horizon)
        labels = (future_return > threshold).astype(int)
        
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 30:
            return False
        
        # Train
        self.scaler.fit(X)
        X_s = self.scaler.transform(X)
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.model.fit(X_s, y)
        
        # Extract feature importance (attention weights)
        self.feature_weights = self.model.feature_importances_
        
        # Evaluate
        pred = self.model.predict(X_s)
        self.train_accuracy = accuracy_score(y, pred)
        self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict with attention-weighted features"""
        if not self.is_trained:
            return 0.5, "NEUTRAL"
        
        features = compute_features(df)
        if features is None or len(features) < 1:
            return 0.5, "NEUTRAL"
        
        X = features.iloc[[-1]]
        if not X.notna().all(axis=1).iloc[0]:
            return 0.5, "NEUTRAL"
        
        X_s = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_s)[:, 1][0]
        
        # Get top attention features
        if self.feature_weights is not None:
            top_idx = np.argsort(self.feature_weights)[-5:]
            attention_summary = "Top features: " + ", ".join([X.columns[i] for i in top_idx])
        else:
            attention_summary = ""
        
        if prob > 0.55:
            direction = "BUY"
        elif prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return prob, direction
