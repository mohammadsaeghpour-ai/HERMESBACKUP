"""DL Forecast Agent — EMA Crossover as DL Proxy"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class DLForecastAgent:
    name = "DLForecast"
    weight = 1.0
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 30:
            return AgentOutput(name=self.name, confidence=0)
        
        e5 = ind.ema(df["close"], 5)
        e13 = ind.ema(df["close"], 13)
        e21 = ind.ema(df["close"], 21)
        
        if e5.iloc[-1] > e13.iloc[-1] > e21.iloc[-1]:
            d = "BUY"
            score = 0.2
        elif e5.iloc[-1] < e13.iloc[-1] < e21.iloc[-1]:
            d = "SELL"
            score = -0.2
        else:
            d = "NEUTRAL"
            score = 0
        
        return AgentOutput(name=self.name, direction=d, confidence=abs(score)*100,
                          score=score, weight=self.weight,
                          evidence=["EMA5>13>21=%s" % (d=="BUY")])
