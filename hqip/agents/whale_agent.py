"""
Whale Agent — Institutional Activity Detector (Enhanced)
=========================================================
Deep order book analysis, trade flow, absorption, iceberg detection,
climax detection, and institutional accumulation/distribution patterns.

Uses ccxt for real-time order book and trade data.
Gracefully falls back if APIs are unavailable.

Weight: 1.5
"""
from hqip.agents.base import BaseAgent
import numpy as np


class WhaleAgent(BaseAgent):
    name = "Whale"
    weight = 1.5

    # ── Exchange bootstrap ──────────────────────────────────────
    def _get_exchange(self):
        """Try multiple exchanges; return the first that works."""
        try:
            import ccxt
        except ImportError:
            return None

        for ex_id in ["binance", "okx", "bybit"]:
            try:
                ex = getattr(ccxt, ex_id)({
                    "enableRateLimit": True,
                    "timeout": 10000,
                })
                ex.load_markets()
                return ex
            except Exception:
                continue
        return None

    def _ccxt_symbol(self, symbol: str, exchange_id: str) -> str:
        """Convert BTCUSDT-style to BTC/USDT for ccxt."""
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT"
        return symbol

    # ── Helpers ─────────────────────────────────────────────────
    @staticmethod
    def _safe_get_exchange_data(fetch_fn, *args, **kwargs):
        """Call an exchange method; return data or None on error."""
        try:
            return fetch_fn(*args, **kwargs)
        except Exception:
            return None

    # ── ORDER BOOK ANALYSIS ────────────────────────────────────
    def _analyze_order_book(self, ob, df, evidence, data):
        """
        Bid/ask ratio, large-order detection, buy/sell walls, absorption.
        """
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        if not bids or not asks:
            evidence.append("⚠️ Order book empty or one-sided")
            return 0.0

        bid_amounts = np.array([b[1] for b in bids])
        ask_amounts = np.array([a[1] for a in asks])

        # ── Top-N volume ──
        top_n = min(20, len(bids), len(asks))
        total_bid_vol = float(np.sum(bid_amounts[:top_n]))
        total_ask_vol = float(np.sum(ask_amounts[:top_n]))
        ratio = total_bid_vol / max(total_ask_vol, 1e-10)

        avg_bid = float(np.mean(bid_amounts))
        avg_ask = float(np.mean(ask_amounts))
        large_bid_thresh = avg_bid * 3
        large_ask_thresh = avg_ask * 3

        large_bids = [b for b in bids if b[1] > large_bid_thresh]
        large_asks = [a for a in asks if a[1] > large_ask_thresh]

        evidence.append(
            f"📊 Order Book (Top {top_n}): Bid={total_bid_vol:.2f} | "
            f"Ask={total_ask_vol:.2f} | Ratio={ratio:.2f}"
        )
        evidence.append(
            f"🐋 Large bid orders (>3x avg): {len(large_bids)} | "
            f"Large asks (>3x avg): {len(large_asks)}"
        )

        score = 0.0

        # ── Buy / Sell Wall Detection ──
        if ratio > 2.0:
            evidence.append("🟢 STRONG BUY WALL — Institutions supporting price")
            score += 0.35
        elif ratio > 1.5:
            evidence.append("🟢 Moderate buy wall detected")
            score += 0.20
        elif ratio < 0.5:
            evidence.append("🔴 STRONG SELL WALL — Institutions distributing")
            score -= 0.35
        elif ratio < 0.67:
            evidence.append("🔴 Moderate sell wall detected")
            score -= 0.20

        # ── Absorption Detection (order book) ──
        price = df["close"].iloc[-1] if df is not None and not df.empty else 0

        if large_bids and price > 0:
            biggest = max(large_bids, key=lambda x: x[1])
            dist = abs(price - biggest[0]) / price * 100
            if dist < 0.5:
                evidence.append(
                    f"🧲 ABSORPTION: Large bid ${biggest[1]:.2f} at {biggest[0]:.2f} "
                    f"({dist:.2f}% away) — absorbing sells"
                )
                score += 0.10

        if large_asks and price > 0:
            biggest = max(large_asks, key=lambda x: x[1])
            dist = abs(price - biggest[0]) / price * 100
            if dist < 0.5:
                evidence.append(
                    f"🧲 ABSORPTION: Large ask ${biggest[1]:.2f} at {biggest[0]:.2f} "
                    f"({dist:.2f}% away) — absorbing buys"
                )
                score -= 0.10

        # ── Iceberg Order Detection ──
        # Look for repeated same-size orders in the book
        bid_sizes = [b[1] for b in bids[:15]]
        ask_sizes = [a[1] for a in asks[:15]]
        iceberg_bid = self._detect_iceberg(bid_sizes)
        iceberg_ask = self._detect_iceberg(ask_sizes)

        if iceberg_bid:
            evidence.append(
                f"🧊 ICEBERG bid detected: {iceberg_bid['count']} orders "
                f"of ~{iceberg_bid['avg_size']:.4f} each — hidden accumulation"
            )
            score += 0.15

        if iceberg_ask:
            evidence.append(
                f"🧊 ICEBERG ask detected: {iceberg_ask['count']} orders "
                f"of ~{iceberg_ask['avg_size']:.4f} each — hidden distribution"
            )
            score -= 0.15

        data["bid_vol"] = total_bid_vol
        data["ask_vol"] = total_ask_vol
        data["ratio"] = round(ratio, 3)
        data["large_bids"] = len(large_bids)
        data["large_asks"] = len(large_asks)
        data["iceberg_bid"] = iceberg_bid is not None
        data["iceberg_ask"] = iceberg_ask is not None

        return score

    def _detect_iceberg(self, sizes):
        """
        Detect iceberg orders: 4+ consecutive orders within 5% size tolerance.
        Returns dict with count and avg_size, or None.
        """
        if len(sizes) < 4:
            return None

        for i in range(len(sizes)):
            cluster = [sizes[i]]
            for j in range(i + 1, len(sizes)):
                if abs(sizes[j] - sizes[i]) / max(sizes[i], 1e-10) < 0.05:
                    cluster.append(sizes[j])
                else:
                    break
            if len(cluster) >= 4:
                return {"count": len(cluster), "avg_size": float(np.mean(cluster))}
        return None

    # ── TRADE FLOW ANALYSIS ────────────────────────────────────
    def _analyze_trades(self, trades, df, evidence, data):
        """
        Whale trade tracking, buy vs sell volume, absorption,
        climax detection, institutional accumulation/distribution.
        """
        if not trades or len(trades) < 10:
            evidence.append("⚠️ Insufficient trade data")
            return 0.0

        amounts = np.array([t["amount"] for t in trades])
        sides = [t.get("side", "unknown") for t in trades]
        prices = np.array([t["price"] for t in trades])

        median_amt = float(np.median(amounts))
        whale_threshold = median_amt * 5

        # ── Whale Trades: Buy vs Sell ──
        whale_buys = [
            (t["amount"], t["price"]) for t in trades
            if t["amount"] > whale_threshold and t.get("side") == "buy"
        ]
        whale_sells = [
            (t["amount"], t["price"]) for t in trades
            if t["amount"] > whale_threshold and t.get("side") == "sell"
        ]

        whale_buy_vol = sum(a * p for a, p in whale_buys)
        whale_sell_vol = sum(a * p for a, p in whale_sells)

        evidence.append(
            f"🐋 Whale trades: {len(whale_buys)} buys (${whale_buy_vol:,.0f}) vs "
            f"{len(whale_sells)} sells (${whale_sell_vol:,.0f})"
        )

        score = 0.0

        # ── Accumulation / Distribution ──
        if whale_buy_vol > whale_sell_vol * 2:
            evidence.append("🟢 ACCUMULATION: Whales buying aggressively — 2x+")
            score += 0.40
        elif whale_sell_vol > whale_buy_vol * 2:
            evidence.append("🔴 DISTRIBUTION: Whales selling aggressively — 2x+")
            score -= 0.40
        elif whale_buy_vol > whale_sell_vol * 1.3:
            evidence.append("🟢 Mild accumulation detected")
            score += 0.15
        elif whale_sell_vol > whale_buy_vol * 1.3:
            evidence.append("🔴 Mild distribution detected")
            score -= 0.15

        # ── Absorption in Trades: High Volume, Low Price Change ──
        recent_30 = trades[-30:]
        price_range = max(t["price"] for t in recent_30) - min(t["price"] for t in recent_30)
        avg_price = float(np.mean([t["price"] for t in recent_30]))
        vol_in_range = sum(t["amount"] for t in recent_30)
        price_change_pct = price_range / max(avg_price, 1e-10) * 100

        if vol_in_range > median_amt * 100 and price_change_pct < 0.3:
            evidence.append(
                f"🧲 ABSORPTION: High volume ({vol_in_range:.2f}) with "
                f"tiny price movement ({price_change_pct:.2f}%) — big player absorbing"
            )
            score += 0.10

        # ── Climax Detection ──
        if len(trades) >= 50:
            recent_vol = float(np.mean([t["amount"] for t in trades[-50:]]))
            last_vol = float(np.mean([t["amount"] for t in trades[-5:]]))
            if recent_vol > 0 and last_vol > recent_vol * 3:
                price_dir = "up" if prices[-1] > prices[max(0, len(prices) - 5)] else "down"
                evidence.append(
                    f"⚡ CLIMAX: Volume {last_vol / recent_vol:.1f}x average, "
                    f"price moving {price_dir}"
                )
                if price_dir == "up":
                    evidence.append(
                        "⚠️ Distribution climax — smart money may sell into strength"
                    )
                    score -= 0.15
                else:
                    evidence.append(
                        "⚠️ Selling climax — smart money may buy into fear"
                    )
                    score += 0.15

        # ── Aggressive Buy/Sell Pressure (Market Orders) ──
        recent_100 = trades[-min(100, len(trades)):]
        buy_trades = [t for t in recent_100 if t.get("side") == "buy"]
        sell_trades = [t for t in recent_100 if t.get("side") == "sell"]
        buy_pressure = sum(t["amount"] * t["price"] for t in buy_trades)
        sell_pressure = sum(t["amount"] * t["price"] for t in sell_trades)
        pressure_ratio = buy_pressure / max(sell_pressure, 1)

        evidence.append(
            f"📊 Aggressive flow: Buy=${buy_pressure:,.0f} | "
            f"Sell=${sell_pressure:,.0f} | Ratio={pressure_ratio:.2f}"
        )

        if pressure_ratio > 1.3:
            evidence.append("🟢 Aggressive buying — market orders hitting asks = bullish")
            score += 0.15
        elif pressure_ratio < 0.7:
            evidence.append("🔴 Aggressive selling — market orders hitting bids = bearish")
            score -= 0.15

        data["whale_buys"] = len(whale_buys)
        data["whale_sells"] = len(whale_sells)
        data["whale_buy_vol"] = round(whale_buy_vol, 2)
        data["whale_sell_vol"] = round(whale_sell_vol, 2)
        data["pressure_ratio"] = round(pressure_ratio, 3)
        data["absorption_detected"] = vol_in_range > median_amt * 100 and price_change_pct < 0.3

        return score

    # ── MAIN ENTRY ─────────────────────────────────────────────
    def analyze(self, df=None, symbol="BTCUSDT", timeframe="", **kwargs):
        evidence = []
        score = 0.0
        data = {
            "bid_vol": 0, "ask_vol": 0, "ratio": 1.0,
            "large_bids": 0, "large_asks": 0,
            "whale_buys": 0, "whale_sells": 0,
            "pressure_ratio": 1.0,
        }

        exchange = self._get_exchange()
        if not exchange:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["⚠️ No exchange available (ccxt error)"],
                reasoning="Whale: exchange unavailable — no live data",
                data=data,
            )

        ccxt_sym = self._ccxt_symbol(symbol, exchange.id)

        # ── ORDER BOOK ──
        ob = self._safe_get_exchange_data(exchange.fetch_order_book, ccxt_sym, limit=50)
        if ob:
            score += self._analyze_order_book(ob, df, evidence, data)
        else:
            evidence.append("⚠️ Order book fetch failed")

        # ── TRADES ──
        trades = self._safe_get_exchange_data(exchange.fetch_trades, ccxt_sym, limit=500)
        if trades:
            score += self._analyze_trades(trades, df, evidence, data)
        else:
            evidence.append("⚠️ Trade history fetch failed")

        # ── AGGREGATE ──
        score = float(np.clip(score, -1.0, 1.0))
        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
        confidence = min(100.0, abs(score) * 70 + 25)

        if not evidence:
            evidence.append("No whale signals detected")

        # Reasoning
        if score > 0.2:
            reasoning = "Whale: institutional accumulation — large buyers dominating"
        elif score < -0.2:
            reasoning = "Whale: institutional distribution — large sellers dominating"
        elif abs(score) > 0.1:
            reasoning = "Whale: mild absorption detected"
        else:
            reasoning = "Whale: neutral flow — no clear institutional bias"

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=score,
            evidence=evidence,
            data=data,
            reasoning=reasoning,
        )
