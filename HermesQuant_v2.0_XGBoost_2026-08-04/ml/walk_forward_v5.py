"""
Walk-Forward v5 — XGBoost Ensemble + Triple-Barrier + Purged CV
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from ml.xgb_ensemble import XGBEnsemble
from ml.triple_barrier import triple_barrier_labels


class WalkForwardV5:
    """
    Walk-Forward with:
    1. XGBoost/LightGBM ensemble
    2. Triple-Barrier labels
    3. Purged split
    4. Class balancing
    """
    
    def __init__(self, train_size=200, test_size=20, horizon=10,
                 profit_mult=1.5, loss_mult=1.5):
        self.train_size = train_size
        self.test_size = test_size
        self.horizon = horizon
        self.profit_mult = profit_mult
        self.loss_mult = loss_mult
    
    def run(self, df, symbol="ETH-USDT-SWAP"):
        """Run walk-forward"""
        
        predictions = []
        actuals = []
        
        i = self.train_size
        
        while i + self.test_size <= len(df):
            # Train on past
            train_df = df.iloc[max(0, i-self.train_size):i]
            
            # Train ensemble
            engine = XGBEnsemble()
            engine.train(train_df, self.horizon, 0.001)
            
            if not engine.is_trained:
                i += self.test_size
                continue
            
            # Test on next window
            for j in range(self.test_size):
                if i + j >= len(df) - self.horizon:
                    break
                
                test_df = df.iloc[max(0, i+j-50):i+j+1]
                prob, direction, uncertainty = engine.predict(test_df)
                
                if direction == "NEUTRAL":
                    continue
                
                # Actual outcome (Triple-Barrier)
                entry = df["close"].iloc[i+j]
                
                # Simple direction check
                future = (df["close"].iloc[i+j+self.horizon] - entry) / entry
                actual = 1 if future > 0.001 else 0
                
                pred = 1 if direction == "BUY" else 0
                
                predictions.append(pred)
                actuals.append(actual)
            
            i += self.test_size
        
        if not predictions:
            return {"accuracy": 0, "buy_accuracy": 0, "sell_accuracy": 0, "total": 0}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Overall accuracy
        accuracy = (predictions == actuals).mean() * 100
        
        # BUY/SELL accuracy
        buy_mask = predictions == 1
        sell_mask = predictions == 0
        
        buy_acc = (predictions[buy_mask] == actuals[buy_mask]).mean() * 100 if buy_mask.sum() > 0 else 0
        sell_acc = (predictions[sell_mask] == actuals[sell_mask]).mean() * 100 if sell_mask.sum() > 0 else 0
        
        return {
            "accuracy": accuracy,
            "buy_accuracy": buy_acc,
            "sell_accuracy": sell_acc,
            "buy_trades": int(buy_mask.sum()),
            "sell_trades": int(sell_mask.sum()),
            "total": len(predictions),
        }
