"""
Walk-Forward v2 — Better parameters + Class balancing
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from ml.ml_balanced import MLBalanced


class WalkForwardV2:
    """
    Improved Walk-Forward with:
    1. Class-balanced models
    2. Better window sizes
    3. Threshold optimization
    """
    
    def __init__(self, train_size=200, test_size=15, retrain_every=15):
        self.train_size = train_size
        self.test_size = test_size
        self.retrain_every = retrain_every
    
    def run(self, df, horizon=5, threshold=0.001):
        """Run improved walk-forward"""
        engine = MLBalanced()
        
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
            
            # Test
            for j in range(self.test_size):
                if i + j >= len(df):
                    break
                
                test_window = df.iloc[max(0, i+j-50):i+j+1]
                prob, direction, uncertainty = engine.predict(test_window)
                
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
        
        if not predictions:
            return {"accuracy": 0, "trades": 0}
        
        # Calculate metrics
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
        }
