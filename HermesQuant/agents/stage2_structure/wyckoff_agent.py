"""Wyckoff Agent — Spring/Upthrust Detection"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class WyckoffAgent:
    name = "Wyckoff"
    weight = 1.4
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        highs, lows = ind.find_swings(df, lookback=5)
        if len(lows) < 2:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=5)
        
        support = lows[-2][1]
        resistance = highs[-1][1] if highs else df["high"].max()
        price = df.iloc[-1]["close"]
        prev_close = df.iloc[-2]["close"]
        
        # Spring: price dipped below support then closed above
        recent_low = df.iloc[-3:]["low"].min()
        if recent_low < support and price > support:
            score = 0.4
            d = "BUY"
            ev = ["Wyckoff SPRING detected (low=%.1f < support=%.1f)" % (recent_low, support)]
        # Upthrust: price spiked above resistance then closed below
        elif df.iloc[-3:]["high"].max() > resistance and price < resistance:
            score = -0.4
            d = "SELL"
            ev = ["Wyckoff UPTHRUST detected"]
        else:
            score = 0
            d = "NEUTRAL"
            ev = ["No Wyckoff pattern"]
        
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*150,
                          score=score, weight=self.weight, evidence=ev)
