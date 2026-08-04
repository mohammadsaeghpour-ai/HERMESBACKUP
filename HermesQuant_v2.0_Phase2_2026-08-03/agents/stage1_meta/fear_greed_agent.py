"""
Fear & Greed Index Agent — Sentiment Analysis
Source: alternative.me API (free, no key needed)
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import requests
from core.data_types import AgentOutput

class FearGreedAgent:
    """
    Fear & Greed Index analysis.
    Extreme Fear (<25) = contrarian BUY
    Extreme Greed (>75) = contrarian SELL
    """
    name = "FearGreed"
    weight = 1.2
    
    def analyze(self, df, symbol="", timeframe=""):
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=10)
            data = r.json().get("data", [])
            
            if not data:
                return AgentOutput(name=self.name, direction="NEUTRAL", confidence=10,
                                  evidence=["No Fear/Greed data"])
            
            current = int(data[0]["value"])
            classification = data[0]["value_classification"]
            
            # 7-day trend
            values = [int(d["value"]) for d in data[:7]]
            avg_7 = sum(values) / len(values)
            trend = values[0] - values[-1]  # positive = getting greedier
            
            score = 0
            ev = []
            
            # Extreme readings = contrarian signals
            if current < 20:
                score = 0.4  # Extreme Fear = BUY
                ev.append("Extreme Fear (%d) — contrarian BUY" % current)
            elif current < 30:
                score = 0.2  # Fear = mild BUY
                ev.append("Fear (%d) — mild BUY bias" % current)
            elif current > 80:
                score = -0.4  # Extreme Greed = SELL
                ev.append("Extreme Greed (%d) — contrarian SELL" % current)
            elif current > 70:
                score = -0.2  # Greed = mild SELL
                ev.append("Greed (%d) — mild SELL bias" % current)
            else:
                ev.append("Neutral (%d)" % current)
            
            # Trend bonus
            if abs(trend) > 15:
                ev.append("Strong %s trend (7d)" % ("fear" if trend < 0 else "greed"))
            
            d = "BUY" if score > 0.05 else ("SELL" if score < -0.05 else "NEUTRAL")
            return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                              score=score, weight=self.weight, evidence=ev)
        
        except Exception as e:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=0,
                              evidence=["Error: %s" % str(e)])
