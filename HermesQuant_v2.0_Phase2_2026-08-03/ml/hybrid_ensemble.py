"""
Hybrid Ensemble — Combines TCN + Attention + ML Engine
Meta-learner that optimally combines all models
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ml.hybrid_tcn import TCNHybrid
from ml.hybrid_attention import AttentionHybrid
from ml.ml_engine_v2 import MLEngineV2


class HybridEnsemble:
    """
    Combines multiple models:
    1. TCN Hybrid (multi-scale patterns)
    2. Attention Hybrid (feature importance)
    3. ML Engine v2 (RF+GB ensemble)
    
    Uses logistic regression as meta-learner
    """
    name = "Hybrid_Ensemble"
    
    def __init__(self):
        self.tcn = TCNHybrid()
        self.attention = AttentionHybrid()
        self.ml_engine = MLEngineV2()
        self.meta_learner = LogisticRegression() if HAS_SKLEARN else None
        self.is_trained = False
    
    def train(self, df, horizon=5, threshold=0.001):
        """Train all sub-models and meta-learner"""
        # Train sub-models
        tcn_ok = self.tcn.train(df, horizon, threshold)
        attn_ok = self.attention.train(df, horizon, threshold)
        ml_ok = self.ml_engine.train(df, horizon, threshold)
        
        self.is_trained = any([tcn_ok, attn_ok, ml_ok])
        
        return self.is_trained
    
    def predict(self, df):
        """Ensemble prediction with uncertainty"""
        predictions = []
        
        # TCN prediction
        if self.tcn.is_trained:
            prob, direction = self.tcn.predict(df)
            predictions.append(("tcn", prob, direction))
        
        # Attention prediction
        if self.attention.is_trained:
            prob, direction = self.attention.predict(df)
            predictions.append(("attention", prob, direction))
        
        # ML Engine prediction
        if self.ml_engine.is_trained:
            prob, direction, uncertainty = self.ml_engine.predict(df)
            predictions.append(("ml", prob, direction))
        
        if not predictions:
            return 0.5, "NEUTRAL", 0.5
        
        # Weighted average (based on model accuracy)
        weights = []
        probs = []
        
        for name, prob, direction in predictions:
            if name == "tcn":
                w = self.tcn.train_accuracy if hasattr(self.tcn, 'train_accuracy') else 0.5
            elif name == "attention":
                w = self.attention.train_accuracy if hasattr(self.attention, 'train_accuracy') else 0.5
            else:
                w = self.ml_engine.train_accuracy if hasattr(self.ml_engine, 'train_accuracy') else 0.5
            
            weights.append(w)
            probs.append(prob)
        
        # Normalize weights
        total_w = sum(weights)
        if total_w > 0:
            weights = [w/total_w for w in weights]
        
        # Weighted average
        avg_prob = sum(p * w for p, w in zip(probs, weights))
        
        # Uncertainty = disagreement between models
        model_probs = [p for _, p, _ in predictions]
        uncertainty = np.std(model_probs) if len(model_probs) > 1 else 0.5
        
        # Direction
        if avg_prob > 0.55:
            direction = "BUY"
        elif avg_prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return avg_prob, direction, uncertainty
    
    def get_model_report(self):
        """Report on each sub-model"""
        report = []
        
        if self.tcn.is_trained:
            report.append("TCN: acc=%.1f%%" % (self.tcn.train_accuracy * 100))
        
        if self.attention.is_trained:
            report.append("Attention: acc=%.1f%%" % (self.attention.train_accuracy * 100))
        
        if self.ml_engine.is_trained:
            report.append("ML_Engine: acc=%.1f%%" % (self.ml_engine.train_accuracy * 100))
        
        return " | ".join(report) if report else "No models trained"
