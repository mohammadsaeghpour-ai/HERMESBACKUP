"""
Hybrid Ensemble — Master Orchestrator for Deep Learning Models
================================================================
Combines all 5 hybrid DL models into a meta-ensemble:
1. HybridTCN          — Temporal Convolutional Network
2. HybridTransformer  — Transformer with multi-head attention
3. HybridCNNLSTM      — CNN + LSTM hybrid
4. HybridGAN          — GAN-augmented training
5. HybridAttentionLSTM — Attention-weighted LSTM

Architecture:
- Each model produces a probability, direction, and confidence
- Meta-ensemble combines via weighted averaging
- Weights are learned based on recent accuracy
- Disagreement between models signals uncertainty

This module replaces the old MLEngineV2 as the ML backbone.
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from ml.hybrid_tcn import HybridTCN
from ml.hybrid_transformer import HybridTransformer
from ml.hybrid_cnn_lstm import HybridCNNLSTM
from ml.hybrid_gan import HybridGAN
from ml.hybrid_attention_lstm import HybridAttentionLSTM


class HybridEnsemble:
    """
    Meta-ensemble combining 5 hybrid DL models.
    
    Architecture:
    ─────────────
    Layer 1 (Specialist Models):
        TCN, Transformer, CNN-LSTM, GAN, Attention-LSTM
        Each produces: (probability, direction, confidence)
    
    Layer 2 (Meta-Learner):
        - Dynamic weighting based on recent performance
        - Agreement scoring
        - Uncertainty estimation
    
    Layer 3 (Decision):
        - Weighted probability → direction
        - Confidence = model agreement × individual confidence
        - Disagreement threshold → NEUTRAL (high uncertainty)
    """
    
    # Base weights (adjusted dynamically based on performance)
    BASE_WEIGHTS = {
        "tcn": 1.0,
        "transformer": 1.2,   # transformers tend to be strong
        "cnn_lstm": 1.1,
        "gan": 0.9,           # augmentation is helpful but secondary
        "attention_lstm": 1.1,
    }
    
    # Minimum accuracy threshold for a model to contribute
    MIN_ACCURACY = 0.48
    
    def __init__(self):
        self.models = {
            "tcn": HybridTCN(),
            "transformer": HybridTransformer(),
            "cnn_lstm": HybridCNNLSTM(),
            "gan": HybridGAN(),
            "attention_lstm": HybridAttentionLSTM(),
        }
        self.weights = dict(self.BASE_WEIGHTS)
        self.is_trained = False
        self.model_accuracies = {}
        self.last_predictions = {}  # track recent predictions for adaptive weighting
    
    def train(self, df, horizon=5, threshold=0.001, train_ratio=0.7):
        """
        Train all hybrid models on the same data.
        Each model may use different subsets of features internally.
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                success = model.train(df, horizon, threshold, train_ratio)
                if success:
                    results[name] = model.train_accuracy
                    self.model_accuracies[name] = model.train_accuracy
                else:
                    results[name] = 0
                    self.model_accuracies[name] = 0
            except Exception as e:
                print(f"[HybridEnsemble] Error training {name}: {e}")
                results[name] = 0
                self.model_accuracies[name] = 0
        
        # Adaptive weight adjustment based on training accuracy
        self._update_weights()
        
        trained_count = sum(1 for v in results.values() if v > 0)
        self.is_trained = trained_count >= 2  # need at least 2 models
        
        return results
    
    def _update_weights(self):
        """
        Update model weights based on training accuracy.
        Better-performing models get higher weight.
        """
        for name, acc in self.model_accuracies.items():
            if acc > 0:
                # Scale weight by accuracy relative to baseline (0.5)
                # Models above 55% get boosted, below get reduced
                acc_factor = max(0.1, (acc - 0.45) / 0.55)
                self.weights[name] = self.BASE_WEIGHTS.get(name, 1.0) * acc_factor
    
    def predict(self, df):
        """
        Get consensus prediction from all models.
        
        Returns:
            (probability, direction, confidence, details)
        """
        if not self.is_trained:
            return 0.5, "NEUTRAL", 0.5, {}
        
        predictions = {}
        weighted_probs = []
        total_weight = 0
        
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    prob, direction, confidence = model.predict(df)
                    predictions[name] = {
                        "prob": prob,
                        "direction": direction,
                        "confidence": confidence,
                        "weight": self.weights.get(name, 1.0),
                    }
                    weighted_probs.append(prob * self.weights.get(name, 1.0))
                    total_weight += self.weights.get(name, 1.0)
                except Exception as e:
                    print(f"[HybridEnsemble] Error predicting {name}: {e}")
        
        if not weighted_probs or total_weight == 0:
            return 0.5, "NEUTRAL", 0.5, predictions
        
        # ── Meta-Ensemble Decision ──
        
        # Weighted average probability
        avg_prob = np.mean(weighted_probs) / (total_weight / len(weighted_probs))
        
        # Model agreement (how much do models agree?)
        all_probs = [p["prob"] for p in predictions.values()]
        agreement = 1.0 - np.std(all_probs) * 4  # high std = low agreement
        agreement = max(0, min(1, agreement))
        
        # Direction consensus
        buy_votes = sum(1 for p in predictions.values() if p["direction"] == "BUY")
        sell_votes = sum(1 for p in predictions.values() if p["direction"] == "SELL")
        total_votes = buy_votes + sell_votes
        
        # Decision thresholds
        if avg_prob > 0.58 and agreement > 0.3:
            direction = "BUY"
        elif avg_prob < 0.42 and agreement > 0.3:
            direction = "SELL"
        elif buy_votes >= 3:
            direction = "BUY"
            avg_prob = max(avg_prob, 0.55)
        elif sell_votes >= 3:
            direction = "SELL"
            avg_prob = min(avg_prob, 0.45)
        else:
            direction = "NEUTRAL"
        
        # Confidence = accuracy × agreement × distance from 0.5
        base_confidence = max(avg_prob, 1 - avg_prob)
        confidence = min(base_confidence * agreement * 100, 90)
        
        # Store for adaptive weighting
        self.last_predictions = predictions
        
        return avg_prob, direction, confidence, predictions
    
    def get_model_report(self):
        """Get diagnostic report of all models."""
        report = {}
        for name, acc in self.model_accuracies.items():
            report[name] = {
                "accuracy": acc,
                "weight": self.weights.get(name, 1.0),
                "is_trained": self.models[name].is_trained,
            }
        return report
    
    def retrain_weak(self, df, horizon=5, threshold=0.001, min_acc=0.48):
        """Retrain only underperforming models."""
        retrained = []
        for name, model in self.models.items():
            acc = self.model_accuracies.get(name, 0)
            if acc < min_acc and acc > 0:
                try:
                    success = model.train(df, horizon, threshold)
                    if success:
                        self.model_accuracies[name] = model.train_accuracy
                        retrained.append((name, model.train_accuracy))
                except Exception:
                    pass
        
        if retrained:
            self._update_weights()
        
        return retrained
