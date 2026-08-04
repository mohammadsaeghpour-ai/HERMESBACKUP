"""Structure Agent — Market Structure (HH/HL/LH/LL)"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class StructureAgent:
    name = "MarketStructure"
    weight = 1.4
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        highs, lows = ind.find_swings(df, lookback=3)
        if len(highs) < 2 or len(lows) < 2:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=10)
        
        hh = highs[-1][1] > highs[-2][1]
        hl = lows[-1][1] > lows[-2][1]
        ll = lows[-1][1] < lows[-2][1]
        lh = highs[-1][1] < highs[-2][1]
        
        if hh and hl:
            d = "BUY"
            score = 0.3
            conf = 40
        elif ll and lh:
            d = "SELL"
            score = -0.3
            conf = 40
        else:
            d = "NEUTRAL"
            score = 0
            conf = 20
        
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight,
                          evidence=["HH=%s HL=%s LH=%s LL=%s" % (hh,hl,lh,ll)])
