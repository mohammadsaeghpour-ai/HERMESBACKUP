"""ML Agent — Ensemble Vote"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput

class MLAgent:
    name = "ML"
    weight = 1.2
    
    def analyze(self, df, symbol="", timeframe="", agent_results=None):
        if not agent_results:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=0)
        
        buy_w = sum(r.weight for r in agent_results if r.direction == "BUY")
        sell_w = sum(r.weight for r in agent_results if r.direction == "SELL")
        total = buy_w + sell_w if buy_w + sell_w > 0 else 1
        
        if buy_w > sell_w * 1.3:
            d = "BUY"
            score = (buy_w - sell_w) / total
        elif sell_w > buy_w * 1.3:
            d = "SELL"
            score = -(sell_w - buy_w) / total
        else:
            d = "NEUTRAL"
            score = 0
        
        return AgentOutput(name=self.name, direction=d,
                          confidence=abs(score)*100, score=score,
                          weight=self.weight,
                          evidence=["Ensemble: BUY_w=%.1f SELL_w=%.1f" % (buy_w, sell_w)])
