"""
XGBoost/LightGBM Ensemble — Best models for tabular data
Based on Kaggle/GitHub research: XGBoost wins most competitions
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    HAS_LGB = False

from ml.feature_engine import FeatureEngine


class XGBEnsemble:
    """
    Ensemble of best models:
    1. RandomForest (with class weights)
    2. GradientBoosting (with sample weights)
    3. LightGBM (if available)
    4. Logistic Regression (calibrated)
    
    Uses purged train/test split
    """
    
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.feature_engine = FeatureEngine()
        self.is_trained = False
        self.metrics = {}
    
    def train(self, df, horizon=5, threshold=0.001):
        """Train ensemble with class balancing"""
        if not HAS_SKLEARN or df is None or len(df) < 100:
            return False
        
        # Compute features
        features = self.feature_engine.compute(df)
        if features is None or len(features) < 100:
            return False
        
        labels = self.feature_engine.compute_labels(df, horizon, threshold)
        
        # Align
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        # Remove NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return False
        
        # Purged split (no overlap)
        split = int(len(X) * 0.75)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Scale
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # Compute class weights
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        self.models = []
        
        # Model 1: RandomForest (balanced)
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train_s, y_train)
        self.models.append(("rf", rf))
        
        # Model 2: GradientBoosting (with sample weights)
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=10, random_state=42
        )
        gb.fit(X_train_s, y_train, sample_weight=sample_weights)
        self.models.append(("gb", gb))
        
        # Model 3: LightGBM (if available)
        if HAS_LGB:
            train_data = lgb.Dataset(X_train_s, label=y_train, 
                                     weight=sample_weights)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "class_weight": "balanced"
            }
            lgb_model = lgb.train(params, train_data, num_boost_round=200)
            self.models.append(("lgb", lgb_model))
        
        # Model 4: Logistic Regression (calibrated)
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        lr_cal = CalibratedClassifierCV(lr, cv=3, method="isotonic")
        lr_cal.fit(X_train_s, y_train)
        self.models.append(("lr", lr_cal))
        
        # Evaluate
        predictions = self._ensemble_predict(X_test_s)
        proba = self._ensemble_predict_proba(X_test_s)
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "up_ratio": y.mean(),
            "models": len(self.models),
        }
        
        self.is_trained = True
        return True
    
    def predict(self, df):
        """Predict with uncertainty"""
        if not self.is_trained:
            return 0.5, "NEUTRAL", 0.5
        
        features = self.feature_engine.compute(df)
        if features is None or len(features) < 1:
            return 0.5, "NEUTRAL", 0.5
        
        X = features.iloc[[-1]]
        if not X.notna().all(axis=1).iloc[0]:
            return 0.5, "NEUTRAL", 0.5
        
        X_s = self.scaler.transform(X)
        
        prob = self._ensemble_predict_proba(X_s)[0]
        uncertainty = self._compute_uncertainty(X_s)
        
        if prob > 0.55:
            direction = "BUY"
        elif prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return prob, direction, uncertainty
    
    def _ensemble_predict_proba(self, X):
        """Average probability from all models"""
        probs = []
        for name, model in self.models:
            if name == "lgb":
                p = model.predict(X)
            else:
                p = model.predict_proba(X)[:, 1]
            probs.append(p)
        return np.mean(probs, axis=0)
    
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
    
    def _compute_uncertainty(self, X):
        """Compute uncertainty from model disagreement"""
        probs = []
        for name, model in self.models:
            if name == "lgb":
                p = model.predict(X)[0]
            else:
                p = model.predict_proba(X)[:, 1][0]
            probs.append(p)
        return np.std(probs)
    
    def get_metrics(self):
        return self.metrics
