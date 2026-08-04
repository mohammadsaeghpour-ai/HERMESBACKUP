"""
Walk-Forward v4 — Best agents + Meta-Model v2 + Triple-Barrier
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from ml.meta_model_v2 import MetaModelV2
from ml.triple_barrier import triple_barrier_labels
from core import indicators as ind
from core.data_types import AgentOutput


class WalkForwardV4:
    """
    Best agents + Enhanced meta-model + Triple-Barrier
    """
    
    def __init__(self, train_size=200, test_size=20, horizon=10,
                 profit_mult=1.5, loss_mult=1.5):
        self.train_size = train_size
        self.test_size = test_size
        self.horizon = horizon
        self.profit_mult = profit_mult
        self.loss_mult = loss_mult
    
    def run(self, df, agents, symbol="ETH-USDT-SWAP"):
        """Run walk-forward with best agents + meta-model"""
        
        # Pre-compute all agent signals
        LOOKBACK = 50
        all_agent_signals = []
        valid_start = LOOKBACK
        
        for i in range(LOOKBACK, len(df)):
            window = df.iloc[max(0, i-LOOKBACK):i+1]
            signals = []
            for agent in agents:
                try:
                    r = agent.analyze(window, symbol, "15m")
                    signals.append(r)
                except Exception:
                    signals.append(AgentOutput(name="?", direction="NEUTRAL"))
            all_agent_signals.append(signals)
        
        # Get Triple-Barrier labels
        labels, distances = triple_barrier_labels(df, self.horizon, 
                                                   self.profit_mult, self.loss_mult)
        if labels is None:
            return {"accuracy": 0, "filtered_accuracy": 0}
        
        # Align: signals[i] corresponds to labels[valid_start + i]
        predictions = []
        actuals = []
        meta_decisions = []
        
        i = 0
        while i + self.test_size <= len(all_agent_signals):
            # Train meta-model on past
            train_end = i
            train_start = max(0, train_end - self.train_size)
            
            if train_end - train_start < 50:
                i += self.test_size
                continue
            
            # Prepare training data
            X_train_feats = []
            y_train = []
            train_dfs = []
            
            for j in range(train_start, train_end):
                label_idx = valid_start + j
                if label_idx >= len(labels) or labels[label_idx] == 2:
                    continue
                
                X_train_feats.append(all_agent_signals[j])
                y_train.append(int(labels[label_idx] == 1))
                train_dfs.append(df.iloc[max(0, j-LOOKBACK):j+1])
            
            if len(y_train) < 30:
                i += self.test_size
                continue
            
            # Train meta-model
            meta = MetaModelV2()
            meta.train(X_train_feats, train_dfs, y_train)
            
            # Test on next window
            if meta.is_trained:
                for j in range(self.test_size):
                    if i + j >= len(all_agent_signals):
                        break
                    
                    label_idx = valid_start + i + j
                    if label_idx >= len(labels) or labels[label_idx] == 2:
                        continue
                    
                    # Meta prediction
                    test_df = df.iloc[max(0, (i+j)-LOOKBACK):(i+j)+1]
                    should_execute, prob = meta.predict(all_agent_signals[i+j], test_df)
                    
                    # Agent direction
                    buy_count = sum(1 for r in all_agent_signals[i+j] if r.direction == "BUY")
                    sell_count = sum(1 for r in all_agent_signals[i+j] if r.direction == "SELL")
                    
                    if buy_count > sell_count:
                        predictions.append(1)
                    elif sell_count > buy_count:
                        predictions.append(0)
                    else:
                        predictions.append(-1)  # NEUTRAL
                    
                    actuals.append(int(labels[label_idx] == 1))
                    meta_decisions.append(should_execute)
            
            i += self.test_size
        
        if not predictions:
            return {"accuracy": 0, "filtered_accuracy": 0, "total": 0, "filtered": 0}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        meta_decisions = np.array(meta_decisions)
        
        # Filter out NEUTRAL predictions
        valid = predictions != -1
        predictions = predictions[valid]
        actuals = actuals[valid]
        meta_decisions = meta_decisions[valid]
        
        if len(predictions) == 0:
            return {"accuracy": 0, "filtered_accuracy": 0, "total": 0, "filtered": 0}
        
        # Raw accuracy
        raw_acc = (predictions == actuals).mean() * 100
        
        # Filtered accuracy
        filtered_mask = meta_decisions
        if filtered_mask.sum() > 0:
            filtered_acc = (predictions[filtered_mask] == actuals[filtered_mask]).mean() * 100
        else:
            filtered_acc = 0
        
        # BUY/SELL accuracy
        buy_mask = predictions == 1
        sell_mask = predictions == 0
        
        buy_acc = (predictions[buy_mask] == actuals[buy_mask]).mean() * 100 if buy_mask.sum() > 0 else 0
        sell_acc = (predictions[sell_mask] == actuals[sell_mask]).mean() * 100 if sell_mask.sum() > 0 else 0
        
        return {
            "accuracy": raw_acc,
            "filtered_accuracy": filtered_acc,
            "buy_accuracy": buy_acc,
            "sell_accuracy": sell_acc,
            "total": len(predictions),
            "filtered": int(filtered_mask.sum()),
            "buy_trades": int(buy_mask.sum()),
            "sell_trades": int(sell_mask.sum()),
        }
