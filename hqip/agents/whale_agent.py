"""
Whale Agent
===========
Detects large order activity, buy/sell walls, institutional accumulation.
Uses Binance order book and recent trades.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class WhaleAgent(BaseAgent):
    name = "Whale"
    weight = 1.3

    def analyze(self, df=None, symbol="BTCUSDT", timeframe="", **kwargs):
        evidence = []
        score = 0.0
        data = {}

        try:
            import ccxt
            exchange = ccxt.binance({"enableRateLimit": True})

            # Fetch order book
            ob = exchange.fetch_order_book(symbol, limit=20)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if bids and asks:
                bid_vol = sum([b[1] for b in bids[:10]])
                ask_vol = sum([a[1] for a in asks[:10]])
                ratio = bid_vol / max(ask_vol, 1e-10)

                evidence.append(f"Order Book: Bid vol={bid_vol:.2f} | Ask vol={ask_vol:.2f} | Ratio={ratio:.2f}")

                if ratio > 1.5:
                    evidence.append("🟢 Strong buy wall detected - more bids than asks")
                    score += 0.3
                elif ratio < 0.67:
                    evidence.append("🔴 Strong sell wall detected - more asks than bids")
                    score -= 0.3
                else:
                    evidence.append("Balanced order book")

                # Large order detection
                large_bids = [b for b in bids if b[1] > bid_vol / 20 * 3]
                large_asks = [a for a in asks if a[1] > ask_vol / 20 * 3]
                evidence.append(f"Large bid orders: {len(large_bids)} | Large ask orders: {len(large_asks)}")

                data["bid_vol"] = bid_vol
                data["ask_vol"] = ask_vol
                data["ratio"] = ratio

            # Fetch recent trades for whale detection
            trades = exchange.fetch_trades(symbol, limit=500)
            if trades:
                amounts = [t["amount"] for t in trades]
                median_amt = np.median(amounts)
                threshold = median_amt * 5  # 5x median = whale
                whale_buys = [t for t in trades if t["amount"] > threshold and t["side"] == "buy"]
                whale_sells = [t for t in trades if t["amount"] > threshold and t["side"] == "sell"]

                total_whale_buy = sum([t["amount"] * t["price"] for t in whale_buys])
                total_whale_sell = sum([t["amount"] * t["price"] for t in whale_sells])

                evidence.append(f"Whale activity: {len(whale_buys)} buys (${total_whale_buy:,.0f}) vs {len(whale_sells)} sells (${total_whale_sell:,.0f})")

                if total_whale_buy > total_whale_sell * 1.5:
                    evidence.append("🟢 Institutional ACCUMULATION")
                    score += 0.35
                elif total_whale_sell > total_whale_buy * 1.5:
                    evidence.append("🔴 Institutional DISTRIBUTION")
                    score -= 0.35
                else:
                    evidence.append("Neutral whale activity")

                data["whale_buys"] = len(whale_buys)
                data["whale_sells"] = len(whale_sells)
                data["whale_buy_usd"] = total_whale_buy
                data["whale_sell_usd"] = total_whale_sell

        except Exception as e:
            evidence.append(f"Whale data unavailable: {str(e)[:50]}")
            return self._out(direction="NEUTRAL", confidence=30, score=0, evidence=evidence, reasoning="Whale data unavailable")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 80 + 20)

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            data=data,
            reasoning=f"Whale: {'accumulation' if score > 0 else 'distribution' if score < 0 else 'neutral'}"
        )
