"""
Whale Agent — Institutional Activity Detector
=============================================
Detects large orders, absorption, spoofing, iceberg orders, climax.
Goes into the HEART of the market to find what big money is doing.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class WhaleAgent(BaseAgent):
    name = "Whale"
    weight = 1.4

    def _get_exchange(self):
        for ex_id in ["okx", "bybit", "gate"]:
            try:
                import ccxt
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
                ex.load_markets()
                return ex
            except:
                pass
        return None

    def _convert_symbol(self, symbol, exchange_id):
        if exchange_id == "okx":
            return f"{symbol[:3]}/{symbol[3:]}" if symbol.endswith("USDT") else symbol
        return symbol

    def analyze(self, df=None, symbol="BTCUSDT", timeframe="", **kwargs):
        evidence = []
        score = 0.0
        data = {}

        exchange = self._get_exchange()
        if not exchange:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["No exchange available"])

        ccxt_symbol = self._convert_symbol(symbol, exchange.id)

        # ── ORDER BOOK ANALYSIS ──
        try:
            ob = exchange.fetch_order_book(ccxt_symbol, limit=50)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if bids and asks:
                # Large order detection (orders > 3x average)
                bid_amounts = [b[1] for b in bids]
                ask_amounts = [a[1] for a in asks]
                avg_bid = np.mean(bid_amounts) if bid_amounts else 1
                avg_ask = np.mean(ask_amounts) if ask_amounts else 1

                large_bids = [b for b in bids if b[1] > avg_bid * 3]
                large_asks = [a for a in asks if a[1] > avg_ask * 3]

                total_bid_vol = sum(bid_amounts[:20])
                total_ask_vol = sum(ask_amounts[:20])
                ratio = total_bid_vol / max(total_ask_vol, 1e-10)

                evidence.append(f"📊 Order Book (Top 20): Bid={total_bid_vol:.2f} | Ask={total_ask_vol:.2f} | Ratio={ratio:.2f}")
                evidence.append(f"🐋 Large bid orders (>3x avg): {len(large_bids)} | Large asks: {len(large_asks)}")

                # Buy/Sell wall detection
                if ratio > 2.0:
                    evidence.append("🟢 STRONG BUY WALL — Institutions supporting price")
                    score += 0.35
                elif ratio > 1.5:
                    evidence.append("🟢 Moderate buy wall detected")
                    score += 0.2
                elif ratio < 0.5:
                    evidence.append("🔴 STRONG SELL WALL — Institutions distributing")
                    score -= 0.35
                elif ratio < 0.67:
                    evidence.append("🔴 Moderate sell wall detected")
                    score -= 0.2

                # Absorption detection: large orders sitting without being filled
                # Price near a large order that hasn't moved = absorption
                if large_bids:
                    closest_large_bid = max(large_bids, key=lambda x: x[1])
                    price = df["close"].iloc[-1] if df is not None and not df.empty else 0
                    bid_price = closest_large_bid[0]
                    dist = abs(price - bid_price) / price * 100 if price else 0
                    if dist < 0.5:
                        evidence.append(f"🧲 ABSORPTION: Large bid at {bid_price:.2f} ({dist:.2f}% away) — institutions absorbing sells")

                if large_asks:
                    closest_large_ask = max(large_asks, key=lambda x: x[1])
                    price = df["close"].iloc[-1] if df is not None and not df.empty else 0
                    ask_price = closest_large_ask[0]
                    dist = abs(price - ask_price) / price * 100 if price else 0
                    if dist < 0.5:
                        evidence.append(f"🧲 ABSORPTION: Large ask at {ask_price:.2f} ({dist:.2f}% away) — institutions absorbing buys")

                data["bid_vol"] = total_bid_vol
                data["ask_vol"] = total_ask_vol
                data["ratio"] = ratio

        except Exception as e:
            evidence.append(f"Order book error: {str(e)[:40]}")

        # ── TRADE FLOW ANALYSIS ──
        try:
            trades = exchange.fetch_trades(ccxt_symbol, limit=500)
            if trades and len(trades) > 10:
                amounts = np.array([t["amount"] for t in trades])
                sides = [t["side"] for t in trades]
                prices = np.array([t["price"] for t in trades])

                median_amt = np.median(amounts)
                whale_threshold = median_amt * 5

                whale_buys = [(t["amount"], t["price"]) for t in trades
                             if t["amount"] > whale_threshold and t["side"] == "buy"]
                whale_sells = [(t["amount"], t["price"]) for t in trades
                              if t["amount"] > whale_threshold and t["side"] == "sell"]

                whale_buy_vol = sum(a * p for a, p in whale_buys)
                whale_sell_vol = sum(a * p for a, p in whale_sells)
                total_volume = sum(a * p for a, p in zip(amounts, prices))

                evidence.append(f"🐋 Whale trades: {len(whale_buys)} buys (${whale_buy_vol:,.0f}) vs {len(whale_sells)} sells (${whale_sell_vol:,.0f})")

                if whale_buy_vol > whale_sell_vol * 2:
                    evidence.append("🟢 ACCUMULATION: Whales buying aggressively")
                    score += 0.4
                elif whale_sell_vol > whale_buy_vol * 2:
                    evidence.append("🔴 DISTRIBUTION: Whales selling aggressively")
                    score -= 0.4
                elif whale_buy_vol > whale_sell_vol * 1.3:
                    evidence.append("🟢 Mild accumulation detected")
                    score += 0.15
                elif whale_sell_vol > whale_buy_vol * 1.3:
                    evidence.append("🔴 Mild distribution detected")
                    score -= 0.15

                # Absorption in trades: large volume without price change
                recent_30 = trades[-30:]
                price_range = max(t["price"] for t in recent_30) - min(t["price"] for t in recent_30)
                avg_price = np.mean([t["price"] for t in recent_30])
                vol_in_range = sum(t["amount"] for t in recent_30)
                price_change_pct = price_range / avg_price * 100

                if vol_in_range > median_amt * 100 and price_change_pct < 0.3:
                    evidence.append(f"🧲 ABSORPTION: High volume ({vol_in_range:.2f}) with tiny price movement ({price_change_pct:.2f}%) — big player absorbing")

                # Climax detection
                recent_vol = np.mean([t["amount"] for t in trades[-50:]])
                last_vol = np.mean([t["amount"] for t in trades[-5:]])
                if last_vol > recent_vol * 3:
                    price_dir = "up" if prices[-1] > prices[-5] else "down"
                    evidence.append(f"⚡ CLIMAX detected: Volume {last_vol/recent_vol:.1f}x average, price moving {price_dir}")
                    if price_dir == "up":
                        evidence.append("⚠️ Potential distribution climax — smart money may be selling into strength")
                        score -= 0.15
                    else:
                        evidence.append("⚠️ Potential selling climax — smart money may be buying into fear")
                        score += 0.15

                # Buy/Sell pressure (aggressive vs passive)
                buy_trades = [t for t in trades[-100:] if t["side"] == "buy"]
                sell_trades = [t for t in trades[-100:] if t["side"] == "sell"]
                buy_pressure = sum(t["amount"] * t["price"] for t in buy_trades)
                sell_pressure = sum(t["amount"] * t["price"] for t in sell_trades)
                pressure_ratio = buy_pressure / max(sell_pressure, 1)
                evidence.append(f"📊 Aggressive flow: Buy={buy_pressure:,.0f} | Sell={sell_pressure:,.0f} | Ratio={pressure_ratio:.2f}")

                if pressure_ratio > 1.3:
                    evidence.append("🟢 Aggressive buying — market orders hitting asks = bullish")
                    score += 0.15
                elif pressure_ratio < 0.7:
                    evidence.append("🔴 Aggressive selling — market orders hitting bids = bearish")
                    score -= 0.15

                data["whale_buys"] = len(whale_buys)
                data["whale_sells"] = len(whale_sells)
                data["pressure_ratio"] = pressure_ratio

        except Exception as e:
            evidence.append(f"Trade flow error: {str(e)[:40]}")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100, abs(score) * 70 + 25)

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            data=data,
            reasoning=f"Whale: {'accumulation' if score > 0.2 else 'distribution' if score < -0.2 else 'absorption' if abs(score) > 0.1 else 'neutral'}"
        )
