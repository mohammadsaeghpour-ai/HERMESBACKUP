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
        
        # Spring: needs volume confirmation + recent support test
        recent_low = df.iloc[-5:]["low"].min()
        recent_vol = ind.volume_ratio(df).iloc[-1]
        
        if recent_low < support * 1.002 and price > support and recent_vol > 1.3:
            score = 0.3
            d = "BUY"
            ev = ["Wyckoff SPRING (vol=%.1fx)" % recent_vol]
        elif df.iloc[-5:]["high"].max() > resistance * 0.998 and price < resistance and recent_vol > 1.3:
            score = -0.3
            d = "SELL"
            ev = ["Wyckoff UPTHRUST (vol=%.1fx)" % recent_vol]
        else:
            score = 0
            d = "NEUTRAL"
            ev = ["No Wyckoff"]
        
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*150,
                          score=score, weight=self.weight, evidence=ev)
