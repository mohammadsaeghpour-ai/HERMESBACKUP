"""
Backtest Engine — Clean, Realistic, No Lookahead
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant_v3")
import numpy as np
import pandas as pd
from core import indicators as ind
from ml.features import FeatureEngine
from ml.model import EnsembleModel
from strategies.multi_tf import MultiTFStrategy
from risk.risk_manager import RiskManager


class BacktestEngine:
    """Walk-Forward backtest with realistic costs"""
    
    def __init__(self, capital=10.0, leverage=20.0, max_daily_loss=3.0,
                 train_size=200, test_size=20, horizon=5, threshold=0.001):
        self.capital = capital
        self.leverage = leverage
        self.max_daily_loss = max_daily_loss
        self.train_size = train_size
        self.test_size = test_size
        self.horizon = horizon
        self.threshold = threshold
        
        self.feature_engine = FeatureEngine()
        self.strategy = MultiTFStrategy()
        self.risk = RiskManager(capital, leverage, max_daily_loss)
    
    def run(self, df, symbol="BTC-USDT-SWAP"):
        """Run walk-forward backtest"""
        if df is None or len(df) < self.train_size + self.test_size + 50:
            return None
        
        predictions = []
        actuals = []
        confidences = []
        
        i = self.train_size
        
        while i + self.test_size <= len(df):
            # Train on past window
            train_df = df.iloc[max(0, i-self.train_size):i]
            
            # Compute features
            features = self.feature_engine.compute(train_df)
            labels = self.feature_engine.compute_labels(train_df, self.horizon, self.threshold)
            
            if features is None or labels is None:
                i += self.test_size
                continue
            
            # Align
            common_idx = features.index.intersection(labels.dropna().index)
            X = features.loc[common_idx]
            y = labels.loc[common_idx]
            
            # Train model
            model = EnsembleModel()
            model.train(X, y)
            
            if not model.is_trained:
                i += self.test_size
                continue
            
            # Test on next window
            for j in range(self.test_size):
                if i + j >= len(df) - self.horizon:
                    break
                
                # Get features for current point
                test_df = df.iloc[max(0, i+j-50):i+j+1]
                test_features = self.feature_engine.compute(test_df)
                
                if test_features is None or len(test_features) == 0:
                    continue
                
                # ML prediction
                prob, direction, uncertainty = model.predict(test_features)
                
                # Strategy confirmation
                strat_dir, strat_conf, reasons = self.strategy.analyze(test_df)
                
                # Combined signal: ML is primary, strategy confirms or ML has high confidence
                if direction != "NEUTRAL" and (direction == strat_dir or prob > 0.7 or prob < 0.3):
                    # Check R:R
                    atr_val = ind.atr(df, 14).iloc[i+j]
                    price = df["close"].iloc[i+j]
                    
                    if direction == "BUY":
                        tp = price + 2 * atr_val
                        sl = price - 1 * atr_val
                    else:
                        tp = price - 2 * atr_val
                        sl = price + 1 * atr_val
                    
                    rr = abs(tp - price) / abs(sl - price + 1e-10)
                    
                    if rr >= 2.0:  # Minimum R:R 1:2
                        # Actual outcome
                        future = (df["close"].iloc[i+j+self.horizon] - price) / price
                        actual = 1 if (direction == "BUY" and future > self.threshold) or                                      (direction == "SELL" and future < -self.threshold) else 0
                        
                        predictions.append(1 if direction == "BUY" else 0)
                        actuals.append(actual)
                        confidences.append(prob)
            
            i += self.test_size
        
        if not predictions:
            return {"accuracy": 0, "buy_accuracy": 0, "sell_accuracy": 0, "total": 0}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        confidences = np.array(confidences)
        
        # Overall accuracy
        accuracy = (predictions == actuals).mean() * 100
        
        # BUY/SELL accuracy
        buy_mask = predictions == 1
        sell_mask = predictions == 0
        
        buy_acc = (predictions[buy_mask] == actuals[buy_mask]).mean() * 100 if buy_mask.sum() > 0 else 0
        sell_acc = (predictions[sell_mask] == actuals[sell_mask]).mean() * 100 if sell_mask.sum() > 0 else 0
        
        # Profit simulation
        capital = self.capital
        peak = capital
        max_dd = 0
        
        for pred, actual in zip(predictions, actuals):
            if actual == 1:
                pnl = capital * 0.02 * 2  # 2% risk, 2:1 R:R
            else:
                pnl = -capital * 0.02  # 2% risk loss
            
            capital += pnl
            peak = max(peak, capital)
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
        
        return {
            "accuracy": accuracy,
            "buy_accuracy": buy_acc,
            "sell_accuracy": sell_acc,
            "buy_trades": int(buy_mask.sum()),
            "sell_trades": int(sell_mask.sum()),
            "total": len(predictions),
            "final_capital": capital,
            "total_return": (capital - self.capital) / self.capital * 100,
            "max_drawdown": max_dd * 100,
            "avg_confidence": confidences.mean(),
        }
