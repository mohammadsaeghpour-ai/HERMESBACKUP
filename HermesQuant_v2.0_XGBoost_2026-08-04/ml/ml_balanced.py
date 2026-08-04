"""
ML Balanced — Handles class imbalance for better BUY accuracy
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.utils.class_weight import compute_class_weight
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml.features_v2 import compute_features_v2, create_labels_v2


class MLBalanced:
    """
    Balanced ML engine that handles class imbalance
    Uses class weights to improve BUY accuracy
    """
    
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.train_accuracy = 0
    
    def train(self, df, horizon=5, threshold=0.001, train_ratio=0.7):
        """Train with class balancing"""
        if not HAS_SKLEARN or df is None or len(df) < 100:
            return False
        
        features = compute_features_v2(df)
        if features is None or len(features) < 100:
            return False
        
        labels = create_labels_v2(df, horizon, threshold)
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return False
        
        split = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # Compute class weights for balance
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        self.models = []
        
        # Model 1: RandomForest with class weights
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            class_weight="balanced"  # Auto-balance classes
        )
        rf.fit(X_train_s, y_train)
        self.models.append(("rf", rf))
        
        # Model 2: GradientBoosting with sample weights
        sample_weights = np.array([class_weights[y] for y in y_train])
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        gb.fit(X_train_s, y_train, sample_weight=sample_weights)
        self.models.append(("gb", gb))
        
        # Model 3: Another RF with different params
        rf2 = RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=123,
            class_weight="balanced_subsample"
        )
        rf2.fit(X_train_s, y_train)
        self.models.append(("rf2", rf2))
        
        # Evaluate
        predictions = self._ensemble_predict(X_test_s)
        self.train_accuracy = accuracy_score(y_test, predictions)
        self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict with balanced output"""
        if not self.is_trained or not self.models:
            return 0.5, "NEUTRAL", 0.5
        
        features = compute_features_v2(df)
        if features is None or len(features) < 1:
            return 0.5, "NEUTRAL", 0.5
        
        X = features.iloc[[-1]]
        if not X.notna().all(axis=1).iloc[0]:
            return 0.5, "NEUTRAL", 0.5
        
        X_s = self.scaler.transform(X)
        
        # Get probabilities
        probs = []
        for name, model in self.models:
            prob = model.predict_proba(X_s)[:, 1][0]
            probs.append(prob)
        
        avg_prob = np.mean(probs)
        uncertainty = np.std(probs)
        
        # Direction with lower BUY threshold
        if avg_prob > 0.52:  # Lower threshold for BUY
            direction = "BUY"
        elif avg_prob < 0.48:  # Lower threshold for SELL
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return avg_prob, direction, uncertainty
    
    def _ensemble_predict(self, X):
        votes = []
        for name, model in self.models:
            votes.append(model.predict(X))
        return np.round(np.mean(votes, axis=0)).astype(int)
