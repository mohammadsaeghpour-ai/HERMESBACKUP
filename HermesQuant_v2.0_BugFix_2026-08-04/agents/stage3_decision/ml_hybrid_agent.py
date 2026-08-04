"""
ML Hybrid Agent — Uses HybridEnsemble for predictions
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from ml.hybrid_ensemble import HybridEnsemble

class MLHybridAgent:
    """
    Hybrid ML agent combining TCN + Attention + Ensemble
    """
    name = "ML_Hybrid"
    weight = 1.8  # Highest weight — most sophisticated
    
    def __init__(self):
        self.ensemble = HybridEnsemble()
        self.trained = False
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 100:
            return AgentOutput(name=self.name, confidence=0, evidence=["Not enough data"])
        
        # Train if not trained
        if not self.trained:
            self.ensemble.train(df)
            self.trained = True
        
        # Predict
        prob, direction, uncertainty = self.ensemble.predict(df)
        
        if direction == "NEUTRAL":
            score = 0
            conf = 10
        elif direction == "BUY":
            score = (prob - 0.5) * 2
            conf = abs(prob - 0.5) * 200 * (1 - uncertainty)
        else:
            score = -(0.5 - prob) * 2
            conf = abs(0.5 - prob) * 200 * (1 - uncertainty)
        
        model_report = self.ensemble.get_model_report()
        
        return AgentOutput(name=self.name, direction=direction, confidence=conf,
                          score=score, weight=self.weight,
                          evidence=["Hybrid: %.1f%% %s (unc=%.3f)" % (prob*100, direction, uncertainty),
                                   "Models: %s" % model_report])
