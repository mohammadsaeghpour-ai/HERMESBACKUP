"""
Walk-Forward v3 — Triple-Barrier + Meta-Labeling + Purged CV
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from ml.meta_labeling import MetaLabelingSystem
from core import indicators as ind


class WalkForwardV3:
    """
    Walk-Forward with:
    1. Triple-Barrier labeling
    2. Meta-Labeling filter
    3. Purged train/test split
    """
    
    def __init__(self, train_size=200, test_size=20, horizon=10, 
                 profit_mult=1.5, loss_mult=1.5):
        self.train_size = train_size
        self.test_size = test_size
        self.horizon = horizon
        self.profit_mult = profit_mult
        self.loss_mult = loss_mult
    
    def run(self, df, agents, symbol="ETH-USDT-SWAP", timeframe="15m"):
        """
        Run walk-forward with meta-labeling.
        """
        predictions = []
        actuals = []
        meta_decisions = []
        
        i = self.train_size
        
        while i + self.test_size <= len(df):
            # Train on past window
            train_df = df.iloc[max(0, i-self.train_size):i]
            test_df = df.iloc[i:i+self.test_size]
            
            # Train meta-model
            meta = MetaLabelingSystem()
            meta.train(train_df, agents, symbol, timeframe,
                      self.horizon, self.profit_mult, self.loss_mult)
            
            # Test on next window
            if meta.is_trained:
                for j in range(self.test_size):
                    if i + j >= len(df) - self.horizon:
                        break
                    
                    # Get agents' prediction
                    window = df.iloc[max(0, i+j-50):i+j+1]
                    agent_results = []
                    for agent in agents:
                        try:
                            r = agent.analyze(window, symbol, timeframe)
                            agent_results.append(r)
                        except Exception:
                            from core.data_types import AgentOutput
                            agent_results.append(AgentOutput(name="?", direction="NEUTRAL"))
                    
                    # Meta-model decides
                    should_execute, prob, decision = meta.predict(agent_results)
                    
                    # Actual outcome
                    entry = df["close"].iloc[i+j]
                    atr = ind.atr(df, 14).iloc[i+j]
                    
                    # Check Triple-Barrier outcome
                    upper = entry + self.profit_mult * atr
                    lower = entry - self.loss_mult * atr
                    
                    actual = 2  # TIME
                    for k in range(1, self.horizon + 1):
                        if i + j + k >= len(df):
                            break
                        if df["high"].iloc[i + j + k] >= upper:
                            actual = 1  # UP
                            break
                        if df["low"].iloc[i + j + k] <= lower:
                            actual = 0  # DOWN
                            break
                    
                    if actual != 2:  # Not TIME
                        # Determine agent direction
                        buy_count = sum(1 for r in agent_results if r.direction == "BUY")
                        sell_count = sum(1 for r in agent_results if r.direction == "SELL")
                        agent_dir = 1 if buy_count > sell_count else (0 if sell_count > buy_count else -1)
                        
                        if agent_dir >= 0:
                            predictions.append(1)
                        else:
                            predictions.append(0)
                        
                        actuals.append(actual)
                        meta_decisions.append(should_execute)
            
            i += self.test_size
        
        if not predictions:
            return {"accuracy": 0, "filtered_accuracy": 0, "total": 0, "filtered": 0}
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        meta_decisions = np.array(meta_decisions)
        
        # Overall accuracy (no filter)
        total_correct = (predictions == actuals).sum()
        total_accuracy = total_correct / len(predictions) * 100
        
        # Filtered accuracy (meta-model says EXECUTE)
        filtered_mask = meta_decisions
        if filtered_mask.sum() > 0:
            filtered_correct = ((predictions[filtered_mask] == actuals[filtered_mask]).sum())
            filtered_accuracy = filtered_correct / filtered_mask.sum() * 100
        else:
            filtered_accuracy = 0
        
        return {
            "accuracy": total_accuracy,
            "filtered_accuracy": filtered_accuracy,
            "total": len(predictions),
            "filtered": int(filtered_mask.sum()),
            "filter_rate": (1 - filtered_mask.mean()) * 100,
        }
