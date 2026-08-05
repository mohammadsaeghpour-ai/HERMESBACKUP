"""
Walk-Forward Optimization
Rolling window train → test → retrain
Prevents overfitting, simulates real trading
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from ml.ml_engine import MLEngine


class WalkForwardOptimizer:
    """
    Walk-Forward Analysis:
    1. Train on window [0, train_end]
    2. Test on window [train_end, train_end + test_size]
    3. Slide window forward
    4. Repeat
    
    This simulates real trading where you train on past data
    and predict on future (unseen) data.
    """
    
    def __init__(self, train_size=200, test_size=20, retrain_every=10):
        self.train_size = train_size
        self.test_size = test_size
        self.retrain_every = retrain_every
        self.results = []
    
    def run(self, df, horizon=5, threshold=0.001):
        """Run walk-forward optimization"""
        engine = MLEngine()
        
        predictions = []
        actuals = []
        probabilities = []
        
        i = self.train_size
        model_age = 0
        
        while i + self.test_size <= len(df):
            # Retrain if needed
            if model_age == 0:
                train_window = df.iloc[i-self.train_size:i]
                engine.train(train_window, horizon, threshold)
            
            # Test on next test_size bars
            for j in range(self.test_size):
                if i + j >= len(df):
                    break
                
                test_window = df.iloc[max(0, i+j-50):i+j+1]
                prob, direction = engine.predict(test_window)
                
                # Get actual label
                if i + j + horizon < len(df):
                    actual_return = (df["close"].iloc[i+j+horizon] - df["close"].iloc[i+j]) / df["close"].iloc[i+j]
                    actual = 1 if actual_return > threshold else 0
                    
                    predictions.append(direction)
                    actuals.append(actual)
                    probabilities.append(prob)
            
            i += self.test_size
            model_age += 1
            if model_age >= self.retrain_every:
                model_age = 0
        
        # Calculate metrics
        if not predictions:
            return {"accuracy": 0, "trades": 0}
        
        # Count correct predictions
        correct = 0
        total = 0
        buy_correct = 0
        buy_total = 0
        sell_correct = 0
        sell_total = 0
        
        for pred, actual, prob in zip(predictions, actuals, probabilities):
            if pred == "NEUTRAL":
                continue
            
            total += 1
            if (pred == "BUY" and actual == 1) or (pred == "SELL" and actual == 0):
                correct += 1
            
            if pred == "BUY":
                buy_total += 1
                if actual == 1: buy_correct += 1
            elif pred == "SELL":
                sell_total += 1
                if actual == 0: sell_correct += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        buy_acc = buy_correct / buy_total * 100 if buy_total > 0 else 0
        sell_acc = sell_correct / sell_total * 100 if sell_total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "trades": total,
            "buy_accuracy": buy_acc,
            "sell_accuracy": sell_acc,
            "buy_trades": buy_total,
            "sell_trades": sell_total,
            "predictions": predictions,
            "actuals": actuals,
        }
