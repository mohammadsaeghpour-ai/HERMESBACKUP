"""Trend Agent — Fixed for both BTC and ETH"""
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
        
        # Price position relative to EMAs
        price = df["close"].iloc[-1]
        above_all = price > e8 > e20 > e50
        below_all = price < e8 < e20 < e50
        
        # Volume confirmation
        vr = ind.volume_ratio(df).iloc[-1]
        
        # Strict: need ADX>25 + EMA alignment + Supertrend agreement + Volume
        strong_trend = adx_val > 25
        
        if direction == 1 and above_all and strong_trend and vr > 0.8:
            d = "BUY"
            score = min(adx_val / 50, 1.0) * 0.7
            if vr > 1.2: score *= 1.3  # Volume boost
        elif direction == -1 and below_all and strong_trend and vr > 0.8:
            d = "SELL"
            score = -min(adx_val / 50, 1.0) * 0.7
            if vr > 1.2: score *= 1.3
        else:
            d = "NEUTRAL"
            score = 0
        
        conf = min(adx_val * 1.2, 70) if adx_val > 25 else 0
        ev = ["ST=%s ADX=%.0f VR=%.1f" % (
            "UP" if direction==1 else "DOWN", adx_val, vr)]
        
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=ev)
