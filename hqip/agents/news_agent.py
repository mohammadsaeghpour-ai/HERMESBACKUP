"""
News Agent
==========
Fetches Fear & Greed Index, funding rates, open interest, market sentiment.
"""
from hqip.agents.base import BaseAgent
import requests, numpy as np

class NewsAgent(BaseAgent):
    name = "News"
    weight = 1.2

    def _safe_get(self, url, timeout=10):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "HQIP/1.0"})
            return r.json() if r.status_code == 200 else None
        except:
            return None

    def analyze(self, df=None, symbol="BTCUSDT", timeframe="", **kwargs):
        evidence = []
        score = 0.0
        data = {}

        # ── Fear & Greed Index ──
        fng = self._safe_get("https://api.alternative.me/fng/?limit=1")
        if fng and "data" in fng:
            fg = fng["data"][0]
            fg_value = int(fg["value"])
            fg_class = fg["value_classification"]
            data["fear_greed"] = fg_value
            data["fear_greed_class"] = fg_class
            evidence.append(f"Fear & Greed: {fg_value} ({fg_class})")

            if fg_value < 25:
                evidence.append("🟢 Extreme Fear = potential buying opportunity")
                score += 0.3
            elif fg_value < 40:
                evidence.append("Fear zone = cautious optimism")
                score += 0.15
            elif fg_value > 75:
                evidence.append("🔴 Extreme Greed = potential selling opportunity")
                score -= 0.3
            elif fg_value > 60:
                evidence.append("Greed zone = caution")
                score -= 0.15
        else:
            evidence.append("Fear & Greed data unavailable")

        # ── Funding Rate ──
        try:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})
            funding = exchange.fapiPublicGetFundingRate({"symbol": symbol, "limit": 3})
            if funding and len(funding) > 0:
                rate = float(funding[-1]["fundingRate"])
                data["funding_rate"] = rate
                rate_pct = rate * 100
                evidence.append(f"Funding Rate: {rate_pct:.4f}%")

                if rate > 0.001:
                    evidence.append("🔴 High positive funding = longs overleveraged (bearish signal)")
                    score -= 0.2
                elif rate < -0.001:
                    evidence.append("🟢 High negative funding = shorts overleveraged (bullish signal)")
                    score += 0.2
                elif rate > 0.0005:
                    evidence.append("Positive funding = slightly crowded longs")
                    score -= 0.05
                elif rate < -0.0005:
                    evidence.append("Negative funding = slightly crowded shorts")
                    score += 0.05
                else:
                    evidence.append("Neutral funding rate")
        except Exception as e:
            evidence.append(f"Funding rate unavailable: {str(e)[:30]}")

        # ── Open Interest (via Binance) ──
        try:
            oi_data = self._safe_get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}")
            if oi_data:
                oi = float(oi_data.get("openInterest", 0))
                data["open_interest"] = oi
                evidence.append(f"Open Interest: {oi:,.2f}")

                # Compare with price trend
                if df is not None and len(df) > 1:
                    price_change = df["close"].pct_change().iloc[-1]
                    if oi > 0 and price_change > 0.005:
                        evidence.append("OI rising + price rising = NEW longs opening")
                        score += 0.1
                    elif oi > 0 and price_change < -0.005:
                        evidence.append("OI rising + price falling = NEW shorts opening")
                        score -= 0.1
        except:
            evidence.append("OI data unavailable")

        # ── BTC Dominance (from CoinGecko) ──
        global_data = self._safe_get("https://api.coingecko.com/api/v3/global")
        if global_data and "data" in global_data:
            btc_dom = global_data["data"].get("market_cap_percentage", {}).get("btc", 0)
            data["btc_dominance"] = btc_dom
            evidence.append(f"BTC Dominance: {btc_dom:.1f}%")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 80 + 25)

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            data=data,
            reasoning=f"Sentiment: {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'}"
        )
