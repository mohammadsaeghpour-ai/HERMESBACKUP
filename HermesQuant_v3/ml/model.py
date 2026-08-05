"""
Ensemble Model — RF + GB + LR with class balancing
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant_v3")
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.feature_selection import mutual_info_classif
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class EnsembleModel:
    """Ensemble of RF + GB + LR with feature selection and calibration"""
    
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.selected_features = []
        self.is_trained = False
        self.metrics = {}
    
    def _select_features(self, X, y, top_k=30):
        """Select top features using mutual information"""
        if not HAS_SKLEARN:
            return X.columns.tolist()
        
        importances = mutual_info_classif(X.fillna(0), y, random_state=42)
        top_idx = np.argsort(importances)[-top_k:]
        return X.columns[top_idx].tolist()
    
    def train(self, X, y):
        """Train ensemble with feature selection and class balancing"""
        if not HAS_SKLEARN or len(X) < 50:
            return False
        
        # Remove NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return False
        
        # Feature selection
        self.selected_features = self._select_features(X, y)
        X = X[self.selected_features]
        
        # Purged split
        split = int(len(X) * 0.75)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Scale
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # Class weights
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cw = dict(zip(classes, weights))
        sw = np.array([cw[yi] for yi in y_train])
        
        self.models = []
        
        # Model 1: Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        self.models.append(("rf", rf))
        
        # Model 2: Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                         min_samples_leaf=10, subsample=0.8, random_state=42)
        gb.fit(X_train_s, y_train, sample_weight=sw)
        self.models.append(("gb", gb))
        
        # Model 3: Calibrated Logistic Regression
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        lr_cal = CalibratedClassifierCV(lr, cv=3, method="isotonic")
        lr_cal.fit(X_train_s, y_train)
        self.models.append(("lr", lr_cal))
        
        # Evaluate
        pred = self._ensemble_predict(X_test_s)
        proba = self._ensemble_predict_proba(X_test_s)
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(self.selected_features),
        }
        
        self.is_trained = True
        return True
    
    def predict(self, X):
        """Predict with uncertainty"""
        if not self.is_trained or len(X) == 0:
            return 0.5, "NEUTRAL", 0.5
        
        X_sel = X[self.selected_features]
        
        if not X_sel.notna().all(axis=1).iloc[-1]:
            return 0.5, "NEUTRAL", 0.5
        
        X_s = self.scaler.transform(X_sel.iloc[[-1]])
        
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
        probs = []
        for _, model in self.models:
            p = model.predict_proba(X)[:, 1]
            probs.append(p)
        return np.mean(probs, axis=0)
    
    def _ensemble_predict(self, X):
        votes = []
        for _, model in self.models:
            pred = model.predict(X)
            votes.append(pred)
        return np.round(np.mean(votes, axis=0)).astype(int)
    
    def _compute_uncertainty(self, X):
        probs = []
        for _, model in self.models:
            p = model.predict_proba(X)[:, 1][0]
            probs.append(p)
        return np.std(probs)
    
    def get_metrics(self):
        return self.metrics
