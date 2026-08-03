"""ML Agent — Ensemble Vote"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput

class MLAgent:
    name = "ML"
    weight = 1.2
    
    # Accuracy-based weights from backtest (updated)
    ACCURACY_WEIGHTS = {
        "RSI_Divergence": 1.5, "BB_Squeeze": 1.2, "Liquidity": 1.3,
        "GameTheory": 1.5, "Momentum": 1.2, "Pattern": 1.0,
        "MarketStructure": 1.0, "Trend": 0.7, "MathBrain": 0.8,
        "DLForecast": 0.8, "Volume": 0.9, "Wyckoff": 1.0,
        "Regime": 0.5, "Whale": 0.5, "SmartAction": 1.0,
    }
    
    def analyze(self, df, symbol="", timeframe="", agent_results=None):
        if not agent_results:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=0)
        
        buy_w = 0
        sell_w = 0
        for r in agent_results:
            aw = self.ACCURACY_WEIGHTS.get(r.name, 1.0)
            eff_w = r.weight * aw * (r.confidence / 100.0)
            if r.direction == "BUY":
                buy_w += eff_w
            elif r.direction == "SELL":
                sell_w += eff_w
        
        total = buy_w + sell_w if buy_w + sell_w > 0 else 1
        
        if buy_w > sell_w * 2.0:
            d = "BUY"
            score = (buy_w - sell_w) / total
        elif sell_w > buy_w * 2.0:
            d = "SELL"
            score = -(sell_w - buy_w) / total
        else:
            d = "NEUTRAL"
            score = 0
        
        return AgentOutput(name=self.name, direction=d,
                          confidence=min(abs(score)*150, 90), score=score,
                          weight=self.weight,
                          evidence=["Ensemble: BUY_w=%.1f SELL_w=%.1f" % (buy_w, sell_w)])
