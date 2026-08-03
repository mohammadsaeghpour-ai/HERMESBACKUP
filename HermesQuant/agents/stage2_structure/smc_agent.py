"""SMC Agent — Smart Money Concepts"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
from core.data_types import AgentOutput
from core import indicators as ind

class SMCAgent:
    name = "SMC"
    weight = 1.0
    
    def analyze(self, df, symbol="", timeframe=""):
        if df is None or len(df) < 20:
            return AgentOutput(name=self.name, confidence=0)
        
        # Order Blocks
        obs = []
        for i in range(2, min(len(df), 30)):
            body = abs(df.iloc[i]["close"] - df.iloc[i]["open"])
            avg_body = df.iloc[max(0,i-20):i]["close"].diff().abs().mean()
            if avg_body == 0: continue
            if body > avg_body * 2:
                curr_dir = 1 if df.iloc[i]["close"] > df.iloc[i]["open"] else -1
                prev_dir = 1 if df.iloc[i-1]["close"] > df.iloc[i-1]["open"] else -1
                if curr_dir != prev_dir:
                    obs.append({"idx": i, "dir": "bullish" if curr_dir==1 else "bearish",
                               "high": df.iloc[i-1]["high"], "low": df.iloc[i-1]["low"],
                               "strength": body / avg_body})
        
        # FVGs
        fvgs = []
        for i in range(2, min(len(df), 20)):
            if df.iloc[i]["low"] > df.iloc[i-2]["high"]:
                fvgs.append({"type": "bullish", "top": df.iloc[i]["low"], "bottom": df.iloc[i-2]["high"]})
            if df.iloc[i]["high"] < df.iloc[i-2]["low"]:
                fvgs.append({"type": "bearish", "top": df.iloc[i-2]["low"], "bottom": df.iloc[i]["high"]})
        
        price = df.iloc[-1]["close"]
        score = 0
        ev = []
        
        # Check if price is near bullish OB
        for ob in obs[-3:]:
            if ob["dir"] == "bullish" and ob["low"] <= price <= ob["high"] * 1.002:
                score += 0.3
                ev.append("Price at bullish OB (%.0f-%.0f)" % (ob["low"], ob["high"]))
        
        # Check FVGs
        for fvg in fvgs[-3:]:
            mid = (fvg["top"] + fvg["bottom"]) / 2
            if abs(price - mid) / price < 0.002:
                if fvg["type"] == "bullish":
                    score += 0.2
                else:
                    score -= 0.2
                ev.append("Price at FVG %s" % fvg["type"])
        
        d = "BUY" if score > 0.2 else ("SELL" if score < -0.2 else "NEUTRAL")
        conf = min(abs(score) * 200, 80)
        
        return AgentOutput(name=self.name, direction=d, confidence=conf,
                          score=score, weight=self.weight, evidence=ev or ["No SMC setup"])
