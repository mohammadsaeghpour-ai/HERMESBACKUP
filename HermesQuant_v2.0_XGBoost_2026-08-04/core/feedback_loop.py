"""
Feedback Loop — Connects agent performance to weights
Updates weights based on actual prediction accuracy
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import json
import os
from collections import defaultdict

WEIGHTS_FILE = "/data/workspace/HermesQuant/agent_weights.json"

DEFAULT_WEIGHTS = {
    "Trend": 1.5, "Momentum": 1.3, "Volume": 1.3,
    "Volatility": 1.0, "Pattern": 1.1,
    "Regime": 1.0, "MarketStructure": 1.4, "Whale": 1.5,
    "RSI_Divergence": 1.2, "BB_Squeeze": 1.1,
    "Liquidity": 1.5, "Wyckoff": 1.4, "MathBrain": 1.4,
    "GameTheory": 1.3, "SmartAction": 1.7,
}


class FeedbackLoop:
    """
    Tracks agent predictions and updates weights based on accuracy.
    
    Rules:
    - Minimum 30 predictions before updating
    - Update every 50 predictions
    - Weight range: 0.5 to 2.5
    - High accuracy → increase weight
    - Low accuracy → decrease weight
    """
    
    def __init__(self, min_predictions=30, update_interval=50):
        self.min_predictions = min_predictions
        self.update_interval = update_interval
        self.predictions = defaultdict(list)  # agent_name -> [correct/wrong]
        self.total_predictions = 0
        self.weights = self._load_weights()
    
    def record(self, agent_name, predicted_direction, actual_direction):
        """Record a prediction for an agent"""
        correct = (predicted_direction == actual_direction)
        self.predictions[agent_name].append(1 if correct else 0)
        self.total_predictions += 1
        
        # Check if update needed
        if self.total_predictions % self.update_interval == 0:
            self._update_weights()
    
    def get_weight(self, agent_name):
        """Get current weight for an agent"""
        return self.weights.get(agent_name, 1.0)
    
    def _update_weights(self):
        """Update weights based on accumulated predictions"""
        for agent_name, preds in self.predictions.items():
            if len(preds) < self.min_predictions:
                continue
            
            # Calculate accuracy
            accuracy = sum(preds[-self.update_interval:]) / min(len(preds), self.update_interval)
            
            # Update weight based on accuracy
            current = self.weights.get(agent_name, 1.0)
            
            if accuracy > 0.6:
                # Good agent — increase weight
                new_weight = min(2.5, current * 1.1)
            elif accuracy < 0.4:
                # Bad agent — decrease weight
                new_weight = max(0.5, current * 0.9)
            else:
                # Average — no change
                new_weight = current
            
            self.weights[agent_name] = round(new_weight, 2)
        
        self._save_weights()
    
    def _load_weights(self):
        """Load weights from file"""
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_WEIGHTS.copy()
    
    def _save_weights(self):
        """Save weights to file"""
        try:
            with open(WEIGHTS_FILE, "w") as f:
                json.dump(self.weights, f, indent=2)
        except Exception:
            pass
    
    def get_report(self):
        """Get performance report for all agents"""
        report = {}
        for agent_name, preds in self.predictions.items():
            if len(preds) >= 10:
                recent = preds[-min(50, len(preds)):]
                report[agent_name] = {
                    "accuracy": sum(recent) / len(recent),
                    "predictions": len(preds),
                    "weight": self.weights.get(agent_name, 1.0),
                }
        return report
