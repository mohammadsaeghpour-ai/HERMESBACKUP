"""Game Theory Agent — Fixed"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class GameTheoryAgent:
    name = "GameTheory"
    weight = 1.3
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 20:
            return AgentOutput(name=self.name, confidence=0)
        
        # Bull/bear power from recent closes (not opens — more reliable)
        closes = df["close"].iloc[-10:].values
        gains = sum(max(0, closes[i] - closes[i-1]) for i in range(1, len(closes)))
        losses = sum(max(0, closes[i-1] - closes[i]) for i in range(1, len(closes)))
        total = gains + losses if gains + losses > 0 else 1
        
        bull_p = gains / total
        bear_p = losses / total
        
        # Only signal if there's a clear imbalance (>60%)
        if bull_p > 0.6:
            d = "BUY"
            score = (bull_p - 0.5) * 0.8
        elif bear_p > 0.6:
            d = "SELL"
            score = -(bear_p - 0.5) * 0.8
        else:
            d = "NO_TRADE"
            score = 0
        
        return AgentOutput(name=self.name, direction=d,
                          confidence=abs(score)*200, score=score,
                          weight=self.weight,
                          evidence=["Bull=%.0f%% Bear=%.0f%%" % (bull_p*100, bear_p*100)])
