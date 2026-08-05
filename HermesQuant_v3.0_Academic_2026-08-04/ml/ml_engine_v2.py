"""
ML Engine v2 — Enhanced Ensemble with LightGBM
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, log_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    HAS_LGB = False

from ml.features import compute_features, create_labels


class MLEngineV2:
    """
    Enhanced ML engine with:
    1. RandomForest
    2. GradientBoosting
    3. LightGBM (if available)
    4. Soft voting ensemble
    """
    
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.train_accuracy = 0
        self.model_weights = []
    
    def train(self, df, horizon=5, threshold=0.001, train_ratio=0.7):
        """Train ensemble models"""
        features = compute_features(df)
        if features is None or len(features) < 100:
            return False
        
        labels = create_labels(df, horizon, threshold)
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
        
        self.models = []
        self.model_weights = []
        
        # Model 1: RandomForest
        if HAS_SKLEARN:
            rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            rf.fit(X_train_s, y_train)
            self.models.append(("rf", rf))
        
        # Model 2: GradientBoosting
        if HAS_SKLEARN:
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
            gb.fit(X_train_s, y_train)
            self.models.append(("gb", gb))
        
        # Model 3: LightGBM
        if HAS_LGB:
            train_data = lgb.Dataset(X_train_s, label=y_train)
            params = {"objective": "binary", "metric": "binary_logloss", "num_leaves": 31,
                     "learning_rate": 0.05, "feature_fraction": 0.8, "verbose": -1}
            lgb_model = lgb.train(params, train_data, num_boost_round=100)
            self.models.append(("lgb", lgb_model))
        
        # Evaluate
        if self.models:
            predictions = self._ensemble_predict(X_test_s)
            self.train_accuracy = accuracy_score(y_test, predictions)
            self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict with uncertainty estimation"""
        if not self.is_trained or not self.models:
            return 0.5, "NEUTRAL", 0.5
        
        features = compute_features(df)
        if features is None or len(features) < 1:
            return 0.5, "NEUTRAL", 0.5
        
        X = features.iloc[[-1]]
        if not X.notna().all(axis=1).iloc[0]:
            return 0.5, "NEUTRAL", 0.5
        
        X_s = self.scaler.transform(X)
        
        # Get probabilities from each model
        probs = []
        for name, model in self.models:
            if name == "lgb":
                p = model.predict(X_s)[0]
            else:
                p = model.predict_proba(X_s)[:, 1][0]
            probs.append(p)
        
        # Average probability
        avg_prob = np.mean(probs)
        
        # Uncertainty = disagreement between models
        uncertainty = np.std(probs)
        
        # Direction
        if avg_prob > 0.55:
            direction = "BUY"
        elif avg_prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return avg_prob, direction, uncertainty
    
    def _ensemble_predict(self, X):
        """Majority vote"""
        votes = []
        for name, model in self.models:
            if name == "lgb":
                pred = (model.predict(X) > 0.5).astype(int)
            else:
                pred = model.predict(X)
            votes.append(pred)
        return np.round(np.mean(votes, axis=0)).astype(int)
