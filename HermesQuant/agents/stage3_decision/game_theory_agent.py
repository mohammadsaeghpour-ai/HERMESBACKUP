"""Game Theory Agent — Nash Equilibrium"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class GameTheoryAgent:
    name = "GameTheory"
    weight = 1.3
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 20:
            return AgentOutput(name=self.name, confidence=0)
        
        # Bull/bear power from recent price action
        gains = sum(max(0, df["close"].iloc[i] - df["open"].iloc[i]) for i in range(-10, 0))
        losses = sum(max(0, df["open"].iloc[i] - df["close"].iloc[i]) for i in range(-10, 0))
        total = gains + losses if gains + losses > 0 else 1
        
        bull_p = gains / total
        bear_p = losses / total
        
        # Whale power from volume spikes
        vr = ind.volume_ratio(df).iloc[-1]
        whale_p = min(vr / 5, 0.5)
        
        # Nash check
        t = bull_p + bear_p + whale_p
        if t == 0:
            state = "EQUILIBRIUM"
        else:
            dom = max(bull_p, bear_p, whale_p) / t
            state = "DISEQUILIBRIUM" if dom > 0.6 else "EQUILIBRIUM"
        
        if state == "DISEQUILIBRIUM" and bull_p > bear_p:
            d = "BUY"
            score = (bull_p - bear_p) * 0.5
        elif state == "DISEQUILIBRIUM" and bear_p > bull_p:
            d = "SELL"
            score = -(bear_p - bull_p) * 0.5
        else:
            d = "NO_TRADE"
            score = 0
        
        return AgentOutput(name=self.name, direction=d, confidence=40 if state=="DISEQUILIBRIUM" else 10,
                          score=score, weight=self.weight,
                          evidence=["State=%s Bull=%.1f%% Bear=%.1f%% Whale=%.1f%%" % (state, bull_p*100, bear_p*100, whale_p*100)])
