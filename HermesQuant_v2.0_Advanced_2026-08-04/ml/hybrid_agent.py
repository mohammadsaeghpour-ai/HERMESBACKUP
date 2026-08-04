"""
Hybrid DL Agent — Integration with 22-Agent System
=====================================================
Drop-in replacement for the old DLForecastAgent.
Uses the full HybridEnsemble (5 deep learning models) instead of
simple EMA crossover.

This agent runs at Stage 0 (independent analysis) and feeds into
the existing multi-agent decision pipeline.
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

from core.data_types import AgentOutput
from ml.hybrid_ensemble import HybridEnsemble


class HybridDLAgent:
    """
    Deep Learning agent using 5 hybrid DL models.
    
    Integration points:
    - Stage 0: Provides independent DL-based signal
    - Feeds into Stage 3 (ML ensemble vote)
    - Can override other agents when confidence is high
    - Reports model agreement as additional metadata
    
    Usage in orchestrator:
        agent = HybridDLAgent()
        result = agent.analyze(df, symbol="BTC-USDT", timeframe="5m")
        # result is an AgentOutput, compatible with existing pipeline
    """
    
    name = "HybridDL"
    weight = 1.3  # higher than original DLForecast (1.0) because more sophisticated
    
    def __init__(self):
        self.ensemble = HybridEnsemble()
        self._last_report = {}
    
    def analyze(self, df, symbol="", timeframe="", **kwargs):
        """
        Run hybrid DL analysis.
        
        Returns AgentOutput compatible with the existing pipeline.
        """
        if df is None or len(df) < 100:
            return AgentOutput(
                name=self.name,
                direction="NEUTRAL",
                confidence=0,
                weight=self.weight,
                evidence=["Insufficient data for DL analysis"]
            )
        
        # Train if not trained yet (or retrain periodically)
        if not self.ensemble.is_trained:
            try:
                self.ensemble.train(df)
            except Exception as e:
                return AgentOutput(
                    name=self.name,
                    direction="NEUTRAL",
                    confidence=0,
                    weight=self.weight,
                    evidence=[f"Training error: {str(e)[:100]}"]
                )
        
        # Predict
        try:
            result = self.ensemble.predict(df)
            if len(result) == 4:
                prob, direction, confidence, model_details = result
            else:
                prob, direction, confidence = result
                model_details = {}
        except Exception as e:
            return AgentOutput(
                name=self.name,
                direction="NEUTRAL",
                confidence=0,
                weight=self.weight,
                evidence=[f"Prediction error: {str(e)[:100]}"]
            )
        
        # Build evidence for transparency
        evidence = []
        model_agreement = 0
        for name, detail in model_details.items():
            if isinstance(detail, dict):
                d = detail.get("direction", "N/A")
                p = detail.get("prob", 0)
                evidence.append(f"{name}: {d} ({p:.3f})")
                if d == direction:
                    model_agreement += 1
        
        total_models = max(len(model_details), 1)
        
        # Build score (-1 to +1)
        if direction == "BUY":
            score = prob - 0.5
        elif direction == "SELL":
            score = prob - 0.5  # negative
        else:
            score = 0
        
        # Add model agreement to evidence
        evidence.append(
            f"Agreement: {model_agreement}/{total_models} models agree ({direction})"
        )
        
        # Create output
        output = AgentOutput(
            name=self.name,
            direction=direction,
            confidence=confidence,
            score=score,
            weight=self.weight,
            evidence=evidence,
            metadata={
                "probability": prob,
                "model_agreement": model_agreement,
                "total_models": total_models,
                "model_details": {
                    k: {"direction": v.get("direction"), "prob": v.get("prob")}
                    for k, v in model_details.items()
                    if isinstance(v, dict)
                },
            }
        )
        
        self._last_report = model_details
        return output
    
    def retrain(self, df, horizon=5, threshold=0.001):
        """Retrain weak models only."""
        return self.ensemble.retrain_weak(df, horizon, threshold)
    
    def get_status(self):
        """Get model status report."""
        return self.ensemble.get_model_report()
