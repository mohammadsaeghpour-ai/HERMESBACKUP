"""
Real ML Agent — XGBoost/LightGBM Ensemble
Replaces the fake DLForecast agent
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from ml.ml_engine import MLEngine

class MLRealAgent:
    """
    Real ML agent using sklearn ensemble models.
    Trains on window, predicts next direction.
    """
    name = "ML_Real"
    weight = 1.5
    
    def __init__(self):
        self.engine = MLEngine()
        self.trained = False
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 100:
            return AgentOutput(name=self.name, confidence=0, evidence=["Not enough data"])
        
        # Train if not trained
        if not self.trained:
            self.engine.train(df)
            self.trained = True
        
        # Predict
        prob, direction = self.engine.predict(df)
        
        if direction == "NEUTRAL":
            score = 0
            conf = 10
        elif direction == "BUY":
            score = (prob - 0.5) * 2
            conf = abs(prob - 0.5) * 200
        else:
            score = -(0.5 - prob) * 2
            conf = abs(0.5 - prob) * 200
        
        return AgentOutput(name=self.name, direction=direction, confidence=conf,
                          score=score, weight=self.weight,
                          evidence=["ML prediction: %.1f%% up, model_acc=%.0f%%" % (
                              prob*100, self.engine.train_accuracy*100)])
