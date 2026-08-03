"""Trend Agent — EMA Fan + Supertrend + ADX"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class TrendAgent:
    name = "Trend"
    weight = 1.5
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 50:
            return AgentOutput(name=self.name, confidence=0)
        
        e8 = ind.ema(df["close"], 8).iloc[-1]
        e20 = ind.ema(df["close"], 20).iloc[-1]
        e50 = ind.ema(df["close"], 50).iloc[-1]
        adx_v, dip, dim = ind.adx(df)
        adx_val = adx_v.iloc[-1]
        st_dir, _ = ind.supertrend(df)
        direction = st_dir.iloc[-1]
        
        if direction == 1 and e8 > e20 > e50:
            d = "BUY"
            score = min(adx_val / 50, 1.0)
        elif direction == -1 and e8 < e20 < e50:
            d = "SELL"
            score = -min(adx_val / 50, 1.0)
        else:
            d = "NEUTRAL"
            score = 0
        
        conf = min(adx_val * 2, 100) if adx_val > 10 else 0
        ev = ["Supertrend=%s" % ("UP" if direction==1 else "DOWN"),
              "ADX=%.1f" % adx_val, "EMA: 8=%.1f 20=%.1f 50=%.1f" % (e8,e20,e50)]
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=ev)
