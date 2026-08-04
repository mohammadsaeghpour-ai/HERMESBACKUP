"""
Funding Rate Agent — Crypto-Specific Signal
Extreme positive funding = overcrowded longs → SELL
Extreme negative funding = overcrowded shorts → BUY
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import requests
from core.data_types import AgentOutput

class FundingRateAgent:
    """
    Funding rate analysis for crypto perpetual contracts.
    OKX funding is paid every 8 hours.
    Extreme readings = high probability of reversal.
    """
    name = "FundingRate"
    weight = 1.4
    
    def analyze(self, df, symbol="", timeframe=""):
        try:
            # Fetch funding rate history
            r = requests.get("https://www.okx.com/api/v5/public/funding-rate-history",
                           params={"instId": symbol, "limit": 30}, timeout=10)
            data = r.json().get("data", [])
            
            if not data or len(data) < 3:
                return AgentOutput(name=self.name, direction="NEUTRAL", confidence=10,
                                  evidence=["No funding data"])
            
            rates = [float(d["fundingRate"]) for d in data]
            current = rates[0]
            avg_7 = sum(rates[:7]) / min(7, len(rates))
            avg_30 = sum(rates) / len(rates)
            
            # Extreme thresholds
            EXTREME_POS = 0.001   # 0.1% per 8h = very bullish crowding
            EXTREME_NEG = -0.001  # -0.1% = very bearish crowding
            
            score = 0
            ev = []
            
            # Current funding analysis
            if current > EXTREME_POS:
                score = -0.3  # Overcrowded longs → SELL bias
                ev.append("Extreme positive funding (%.4f%%) — longs overcrowded" % (current*100))
            elif current < EXTREME_NEG:
                score = 0.3  # Overcrowded shorts → BUY bias
                ev.append("Extreme negative funding (%.4f%%) — shorts overcrowded" % (current*100))
            
            # Acceleration (funding increasing = momentum building)
            if len(rates) >= 7:
                recent_avg = sum(rates[:3]) / 3
                older_avg = sum(rates[3:7]) / 4
                if recent_avg > older_avg * 1.5:
                    ev.append("Funding accelerating upward — caution for longs")
                    score -= 0.1
                elif recent_avg < older_avg * 0.5:
                    ev.append("Funding decelerating — possible reversal")
                    score += 0.1
            
            # Average deviation
            deviation = (current - avg_30) / (abs(avg_30) + 1e-10)
            ev.append("Current vs 30-period avg: %.1fx deviation" % deviation)
            
            d = "BUY" if score > 0.1 else ("SELL" if score < -0.1 else "NEUTRAL")
            return AgentOutput(name=self.name, direction=d, confidence=abs(score)*200,
                              score=score, weight=self.weight, evidence=ev)
        
        except Exception as e:
            return AgentOutput(name=self.name, direction="NEUTRAL", confidence=0,
                              evidence=["Error: %s" % str(e)])
