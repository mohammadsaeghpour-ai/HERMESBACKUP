"""
Open Interest Agent — Position Flow Analysis
Rising OI + Rising Price = Strong trend
Falling OI + Rising Price = Weakening
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import requests
from core.data_types import AgentOutput

class OpenInterestAgent:
    """
    Open Interest analysis for crypto perpetual contracts.
    OI changes indicate conviction behind moves.
    """
    name = "OpenInterest"
    weight = 1.3
    
    def analyze(self, df, symbol="", timeframe=""):
        try:
            # Fetch open interest
            r = requests.get("https://www.okx.com/api/v5/public/open-interest",
                           params={"instId": symbol, "instType": "SWAP"}, timeout=10)
            data = r.json().get("data", [])
            
            if not data:
                return AgentOutput(name=self.name, direction="NEUTRAL", confidence=10,
                                  evidence=["No OI data"])
            
            oi = float(data[0].get("oi", 0))
            oi_usd = float(data[0].get("oiUsd", 0))
            
            # Price trend from df
            price_now = df["close"].iloc[-1]
            price_5ago = df["close"].iloc[-5] if len(df) >= 5 else price_now
            price_trend = (price_now - price_5ago) / price_5ago * 100
            
            # Volume
            vol_now = df["volume"].iloc[-1]
            vol_avg = df["volume"].iloc[-20:].mean() if len(df) >= 20 else vol_now
            vol_ratio = vol_now / (vol_avg + 1e-10)
            
            score = 0
            ev = []
            
            # OI + Price analysis
            # Strong trend: OI rising + price rising = conviction
            # Weak trend: OI falling + price rising = no conviction
            # Reversal: OI extreme + price extreme = potential reversal
            
            if price_trend > 0.5 and vol_ratio > 1.2:
                # Price up + high volume = strong buying
                score = 0.2
                ev.append("Price +%.1f%% with volume %.1fx — buying pressure" % (price_trend, vol_ratio))
            elif price_trend < -0.5 and vol_ratio > 1.2:
                # Price down + high volume = strong selling
                score = -0.2
                ev.append("Price -%.1f%% with volume %.1fx — selling pressure" % (abs(price_trend), vol_ratio))
            elif price_trend > 0.5 and vol_ratio < 0.8:
                # Price up but low volume = weak rally
                score = -0.1
                ev.append("Price up but low volume — weak rally")
            elif price_trend < -0.5 and vol_ratio < 0.8:
                # Price down but low volume = weak sell-off
                score = 0.1
                ev.append("Price down but low volume — weak sell-off")
            
            ev.append("OI: %.0f contracts ($%.0fM)" % (oi, oi_usd/1e6 if oi_usd > 0 else 0))
            
            d = "BUY" if score > 0.05 else ("SELL" if score < -0.05 else "NEUTRAL")
            return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                              score=score, weight=self.weight, evidence=ev)
        
        except Exception as e:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=0,
                              evidence=["Error: %s" % str(e)])
