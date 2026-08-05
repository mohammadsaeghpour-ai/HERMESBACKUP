"""
MIT Ensemble Boosting — Robert Schapire (AdaBoost)
Combine weak learners into strong learner
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml.feature_engine_v2 import FeatureEngineV2


class EnsembleBoosting:
    """
    MIT-style ensemble boosting:
    1. AdaBoost (Schapire)
    2. Gradient Boosting (Friedman)
    3. Stacking with Logistic Regression
    
    Key insight: Weak learners + smart combination = strong learner
    """
    
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.meta_model = None
        self.feature_engine = FeatureEngineV2()
        self.is_trained = False
        self.metrics = {}
    
    def train(self, df, horizon=5, threshold=0.001):
        """Train ensemble boosting"""
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
        
        # Purged split
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
        
        # Model 1: AdaBoost (Schapire)
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        ada.fit(X_train_s, y_train, sample_weight=sample_weights)
        self.models.append(("ada", ada))
        
        # Model 2: Gradient Boosting (Friedman)
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train_s, y_train, sample_weight=sample_weights)
        self.models.append(("gb", gb))
        
        # Model 3: Stacking with Logistic Regression
        # Get predictions from base models
        base_preds = np.array([m.predict(X_train_s) for _, m in self.models]).T
        
        # Train meta-model on base predictions
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(base_preds, y_train)
        self.meta_model = lr
        
        # Evaluate
        base_test = np.array([m.predict(X_test_s) for _, m in self.models]).T
        meta_pred = self.meta_model.predict(base_test)
        
        # Also get probabilities for confidence
        base_proba = np.array([m.predict_proba(X_test_s)[:, 1] for _, m in self.models]).T
        meta_proba = self.meta_model.predict_proba(base_test)[:, 1]
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, meta_pred),
            "precision": precision_score(y_test, meta_pred, zero_division=0),
            "recall": recall_score(y_test, meta_pred, zero_division=0),
            "f1": f1_score(y_test, meta_pred, zero_division=0),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "models": len(self.models) + 1,
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
        
        # Base predictions
        base_preds = np.array([m.predict(X_s)[0] for _, m in self.models])
        
        # Meta prediction
        meta_pred = self.meta_model.predict(base_preds.reshape(1, -1))[0]
        meta_proba = self.meta_model.predict_proba(base_preds.reshape(1, -1))[:, 1][0]
        
        # Uncertainty from model disagreement
        uncertainty = np.std([m.predict_proba(X_s)[:, 1][0] for _, m in self.models])
        
        if meta_proba > 0.55:
            direction = "BUY"
        elif meta_proba < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return meta_proba, direction, uncertainty
    
    def get_metrics(self):
        return self.metrics
