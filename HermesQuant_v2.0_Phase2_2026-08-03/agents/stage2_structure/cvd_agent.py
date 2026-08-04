"""
CVD (Cumulative Volume Delta) Agent
Detects buying/selling pressure from trade data
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import requests
import pandas as pd
from core.data_types import AgentOutput

class CVDAgent:
    """
    Cumulative Volume Delta analysis.
    CVD = running sum of (buy_volume - sell_volume)
    Rising CVD + Rising Price = strong trend
    Falling CVD + Rising Price = bearish divergence
    """
    name = "CVD"
    weight = 1.3
    
    def analyze(self, df, symbol="", timeframe=""):
        try:
            # Fetch recent trades
            r = requests.get("https://www.okx.com/api/v5/market/trades",
                           params={"instId": symbol, "limit": 100}, timeout=10)
            data = r.json().get("data", [])
            
            if not data or len(data) < 20:
                return AgentOutput(name=self.name, direction="NEUTRAL", confidence=10,
                                  evidence=["Insufficient trade data"])
            
            # Parse trades
            buy_vol = 0
            sell_vol = 0
            for trade in data:
                vol = float(trade.get("sz", 0))
                side = trade.get("side", "")
                if side == "buy":
                    buy_vol += vol
                else:
                    sell_vol += vol
            
            total_vol = buy_vol + sell_vol
            if total_vol == 0:
                return AgentOutput(name=self.name, direction="NEUTRAL", confidence=10)
            
            # CVD ratio
            cvd_ratio = (buy_vol - sell_vol) / total_vol  # -1 to +1
            
            # Price trend
            price_now = df["close"].iloc[-1]
            price_5ago = df["close"].iloc[-5] if len(df) >= 5 else price_now
            price_trend = (price_now - price_5ago) / price_5ago
            
            score = 0
            ev = []
            
            # CVD analysis
            if cvd_ratio > 0.15:
                score = 0.25  # Strong buying pressure
                ev.append("Strong buying pressure (CVD ratio: %.2f)" % cvd_ratio)
            elif cvd_ratio < -0.15:
                score = -0.25  # Strong selling pressure
                ev.append("Strong selling pressure (CVD ratio: %.2f)" % cvd_ratio)
            
            # Divergence detection
            if price_trend > 0.001 and cvd_ratio < -0.1:
                score -= 0.15
                ev.append("BEARISH DIVERGENCE: Price up but CVD down")
            elif price_trend < -0.001 and cvd_ratio > 0.1:
                score += 0.15
                ev.append("BULLISH DIVERGENCE: Price down but CVD up")
            
            d = "BUY" if score > 0.05 else ("SELL" if score < -0.05 else "NEUTRAL")
            return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                              score=score, weight=self.weight, evidence=ev or ["CVD neutral"])
        
        except Exception as e:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=0,
                              evidence=["Error: %s" % str(e)])
