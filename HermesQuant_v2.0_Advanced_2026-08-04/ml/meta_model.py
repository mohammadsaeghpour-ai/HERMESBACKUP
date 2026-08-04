"""
Meta-Model — Calibrated Logistic Regression
Replaces: vote + Bayesian + vector geometry
Input: 15 agent outputs → Output: calibrated probability
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, brier_score_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MetaModel:
    """
    Meta-learned combination of agent outputs.
    Replaces manual vote + Bayesian + vector geometry.
    
    Input: DataFrame with columns [agent1_score, agent1_conf, ..., agentN_score, agentN_conf]
    Output: Calibrated probability of UP direction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.train_accuracy = 0
        self.brier_score = 0
    
    def _prepare_features(self, agent_results):
        """Convert agent outputs to feature vector"""
        features = []
        for r in agent_results:
            # Direction as numeric
            if r.direction == "BUY":
                direction = 1
            elif r.direction == "SELL":
                direction = -1
            else:
                direction = 0
            
            features.extend([
                direction,                    # agent direction
                r.confidence / 100,           # normalized confidence
                r.score,                      # raw score
                r.weight,                     # weight
            ])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, all_agent_results, actual_labels):
        """
        Train meta-model on agent outputs vs actual outcomes.
        
        all_agent_results: list of lists (each inner list = agent outputs for one bar)
        actual_labels: list of 0/1 (actual direction)
        """
        if not HAS_SKLEARN:
            return False
        
        X_list = []
        for agent_results in all_agent_results:
            feat = self._prepare_features(agent_results)
            X_list.append(feat[0])
        
        X = np.array(X_list)
        y = np.array(actual_labels)
        
        # Remove NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 30:
            return False
        
        # Scale
        self.scaler.fit(X)
        X_s = self.scaler.transform(X)
        
        # Train calibrated logistic regression
        base_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
        self.model.fit(X_s, y)
        
        # Evaluate
        pred = self.model.predict(X_s)
        self.train_accuracy = accuracy_score(y, pred)
        
        proba = self.model.predict_proba(X_s)[:, 1]
        self.brier_score = brier_score_loss(y, proba)
        
        self.is_trained = True
        return True
    
    def predict(self, agent_results):
        """Predict calibrated probability from agent outputs"""
        if not self.is_trained:
            return 0.5, "NEUTRAL", 0.5
        
        feat = self._prepare_features(agent_results)
        
        if np.isnan(feat).any():
            return 0.5, "NEUTRAL", 0.5
        
        feat_s = self.scaler.transform(feat)
        
        prob_up = self.model.predict_proba(feat_s)[:, 1][0]
        
        # Uncertainty from calibration
        uncertainty = abs(prob_up - 0.5) * 2  # 0 = max uncertainty, 1 = certain
        
        if prob_up > 0.55:
            direction = "BUY"
        elif prob_up < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return prob_up, direction, 1 - uncertainty
