"""Volume Agent — Volume Ratio"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class VolumeAgent:
    name = "Volume"
    weight = 1.3
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        vr = ind.volume_ratio(df).iloc[-1]
        trend = 1 if df["close"].iloc[-1] > df["close"].iloc[-20] else -1
        
        if vr > 1.5 and trend == 1:
            d = "BUY"
            score = min((vr - 1) / 2, 1.0)
        elif vr > 1.5 and trend == -1:
            d = "SELL"
            score = -min((vr - 1) / 2, 1.0)
        else:
            d = "NEUTRAL"
            score = 0
        
        conf = min(vr * 30, 100) if vr > 1.2 else 10
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight,
                          evidence=["VolRatio=%.1fx" % vr])
