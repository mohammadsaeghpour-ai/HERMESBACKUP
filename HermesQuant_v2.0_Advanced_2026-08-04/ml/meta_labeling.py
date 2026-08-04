"""
Meta-Labeling System (Marcos Lopez de Prado)
Primary model: agents suggest direction
Meta model: decides whether to trust the signal
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml.triple_barrier import triple_barrier_labels
from ml.purged_cv import purged_train_test_split


class MetaLabelingSystem:
    """
    Two-layer system:
    1. Primary: agent signals (direction)
    2. Meta: whether to execute (confidence filter)
    
    Uses Triple-Barrier labels + Purged CV
    """
    
    def __init__(self):
        self.meta_model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.metrics = {}
    
    def prepare_features(self, df, agents, symbol="", timeframe="50"):
        """
        Run agents on df and return feature matrix.
        Each row: [agent1_direction, agent1_conf, ..., agentN_direction, agentN_conf, meta_features]
        """
        if df is None or len(df) < 50:
            return None
        
        LOOKBACK = 50
        rows = []
        
        for i in range(LOOKBACK, len(df)):
            window = df.iloc[i-LOOKBACK:i+1]
            row = []
            
            for agent in agents:
                try:
                    r = agent.analyze(window, symbol, timeframe)
                    # Direction as numeric
                    if r.direction == "BUY":
                        row.extend([1, r.confidence / 100, r.score])
                    elif r.direction == "SELL":
                        row.extend([-1, r.confidence / 100, r.score])
                    else:
                        row.extend([0, 0, 0])
                except Exception:
                    row.extend([0, 0, 0])
            
            rows.append(row)
        
        return np.array(rows)
    
    def train(self, df, agents, symbol="", timeframe="15m",
              horizon=10, profit_mult=1.5, loss_mult=1.5):
        """
        Train meta-labeling system:
        1. Get agent features
        2. Create Triple-Barrier labels
        3. Train meta-model with purged split
        """
        if not HAS_SKLEARN:
            return False
        
        # Get features
        X = self.prepare_features(df, agents, symbol, timeframe)
        if X is None:
            return False
        
        # Create Triple-Barrier labels
        labels_all, distances = triple_barrier_labels(df, horizon, profit_mult, loss_mult)
        if labels_all is None:
            return False
        
        # Align features with labels
        # Features start at index 50, labels start at index 0
        # So features[i] corresponds to labels[50+i]
        offset = 50
        if offset + len(X) > len(labels_all):
            X = X[:len(labels_all) - offset]
        
        y = labels_all[offset:offset + len(X)]
        
        # Filter out TIME labels (2)
        mask = y != 2
        X_clean = X[mask]
        y_clean = (y[mask] == 1).astype(int)  # 1=UP, 0=DOWN
        
        if len(X_clean) < 50:
            return False
        
        # Purged train/test split
        train_idx, test_idx = purged_train_test_split(len(X_clean), test_size=0.25)
        
        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
        y_train, y_test = y_clean[train_idx], y_clean[test_idx]
        
        # Scale
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # Train calibrated meta-model
        base_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta_model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
        self.meta_model.fit(X_train_s, y_train)
        
        # Evaluate
        pred = self.meta_model.predict(X_test_s)
        proba = self.meta_model.predict_proba(X_test_s)[:, 1]
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "up_ratio": y_clean.mean(),
        }
        
        self.is_trained = True
        return True
    
    def predict(self, agent_results):
        """
        Predict whether to execute a signal.
        Returns: (should_execute, probability, confidence)
        """
        if not self.is_trained:
            return True, 0.5, "NO_MODEL"
        
        # Convert agent results to feature vector
        row = []
        for r in agent_results:
            if r.direction == "BUY":
                row.extend([1, r.confidence / 100, r.score])
            elif r.direction == "SELL":
                row.extend([-1, r.confidence / 100, r.score])
            else:
                row.extend([0, 0, 0])
        
        X = np.array(row).reshape(1, -1)
        
        if np.isnan(X).any():
            return True, 0.5, "NaN_FEATURES"
        
        X_s = self.scaler.transform(X)
        
        prob = self.meta_model.predict_proba(X_s)[:, 1][0]
        should_execute = prob > 0.5
        
        return should_execute, prob, "EXECUTE" if should_execute else "SKIP"
    
    def get_metrics(self):
        """Get training metrics"""
        return self.metrics
