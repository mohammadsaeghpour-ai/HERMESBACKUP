"""
News Agent — Market Intelligence (Enhanced)
=============================================
Fetches Fear & Greed Index, funding rates (Binance + OKX fallback),
Long/Short Ratios (Global + Top Traders), BTC Dominance.

Each data point produces a BUY/SELL/NEUTRAL signal.
Contrarian logic: high retail optimism = bearish, high funding = squeeze risk.

Weight: 1.2
"""
from hqip.agents.base import BaseAgent
import requests
import numpy as np
from datetime import datetime, timezone


class NewsAgent(BaseAgent):
    name = "News"
    weight = 1.2

    # ── HTTP helper ─────────────────────────────────────────────
    def _safe_get(self, url, timeout=10, headers=None):
        """GET JSON with timeout; returns dict or None."""
        try:
            h = {"User-Agent": "HQIP/2.0"}
            if headers:
                h.update(headers)
            r = requests.get(url, timeout=timeout, headers=h)
            if r.status_code == 200:
                return r.json()
            return None
        except Exception:
            return None

    def _signal(self, label, detail, signal_dir, signal_strength):
        """Return a formatted evidence line with emoji verdict."""
        emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "➖"}.get(signal_dir, "❓")
        return f"{emoji} {label}: {detail}  → [{signal_dir} {signal_strength:.2f}]"

    # ── MAIN ANALYSIS ──────────────────────────────────────────
    def analyze(self, df=None, symbol="BTCUSDT", timeframe="", **kwargs):
        evidence = []
        score = 0.0
        data = {}
        signals = []  # list of (direction, weight)

        # ═══════════════════════════════════════════════════════
        # 1. FEAR & GREED INDEX
        # ═══════════════════════════════════════════════════════
        fng = self._safe_get("https://api.alternative.me/fng/?limit=1")
        if fng and "data" in fng and fng["data"]:
            fg = fng["data"][0]
            fg_value = int(fg["value"])
            fg_class = fg.get("value_classification", "")
            data["fear_greed"] = fg_value
            data["fear_greed_class"] = fg_class
            data["fear_greed_timestamp"] = fg.get("timestamp", "")

            if fg_value <= 15:
                detail = (f"{fg_value} ({fg_class}) — Maximum capitulation. "
                          "Historically extreme bottoms. Retail panic-selling; smart money accumulates.")
                sig = ("BUY", 0.40)
            elif fg_value <= 25:
                detail = (f"{fg_value} ({fg_class}) — Deep fear zone. "
                          "Heavy retail selling. Contrarian buy setups forming.")
                sig = ("BUY", 0.30)
            elif fg_value <= 40:
                detail = (f"{fg_value} ({fg_class}) — Fear persists. "
                          "Market uncertain, early accumulation phase.")
                sig = ("BUY", 0.15)
            elif fg_value <= 60:
                detail = f"{fg_value} ({fg_class}) — Neutral sentiment. No strong edge."
                sig = ("NEUTRAL", 0.0)
            elif fg_value <= 75:
                detail = (f"{fg_value} ({fg_class}) — Greed building. "
                          "Retail getting confident. Caution warranted.")
                sig = ("SELL", 0.15)
            elif fg_value <= 85:
                detail = (f"{fg_value} ({fg_class}) — High greed. "
                          "Retail FOMO active. Distribution likely.")
                sig = ("SELL", 0.30)
            else:
                detail = (f"{fg_value} ({fg_class}) — Extreme greed / euphoria. "
                          "Retail all-in. Historically major tops form here.")
                sig = ("SELL", 0.40)

            evidence.append(self._signal("Fear & Greed", detail, sig[0], sig[1]))
            signals.append(sig)
        else:
            evidence.append("⚠️ Fear & Greed data unavailable")

        # ═══════════════════════════════════════════════════════
        # 2. FUNDING RATE (Binance → OKX fallback)
        # ═══════════════════════════════════════════════════════
        funding_rate = None
        funding_source = None

        # 2a. Binance
        binance_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        bd = self._safe_get(binance_url)
        if bd and "lastFundingRate" in bd:
            funding_rate = float(bd["lastFundingRate"])
            funding_source = "Binance"
            data["funding_rate"] = funding_rate
            data["funding_rate_source"] = "Binance"
            next_time = bd.get("nextFundingTime", 0)
            data["next_funding_time"] = (
                datetime.fromtimestamp(next_time / 1000, tz=timezone.utc).isoformat()
                if next_time else ""
            )

        # 2b. OKX fallback
        if funding_rate is None:
            okx_inst = symbol.replace("USDT", "-USDT-SWAP")
            okx_url = f"https://www.okx.com/api/v5/public/funding-rate?instId={okx_inst}"
            od = self._safe_get(okx_url)
            if od and od.get("data"):
                funding_rate = float(od["data"][0]["fundingRate"])
                funding_source = "OKX"
                data["funding_rate"] = funding_rate
                data["funding_rate_source"] = "OKX"

        if funding_rate is not None:
            rate_pct = funding_rate * 100
            # High positive = overleveraged longs = contrarian bearish
            if funding_rate > 0.001:
                detail = (f"{rate_pct:.4f}% ({funding_source}) — "
                          "HIGH positive: longs pay shorts. Overleveraged longs → squeeze DOWN.")
                sig = ("SELL", 0.30)
            elif funding_rate > 0.0005:
                detail = (f"{rate_pct:.4f}% ({funding_source}) — "
                          "Positive: slightly crowded longs. Mild downward pressure.")
                sig = ("SELL", 0.15)
            elif funding_rate > 0.0001:
                detail = f"{rate_pct:.4f}% ({funding_source}) — Mildly positive. Balanced market."
                sig = ("NEUTRAL", 0.0)
            elif funding_rate > -0.0001:
                detail = f"{rate_pct:.4f}% ({funding_source}) — Neutral funding."
                sig = ("NEUTRAL", 0.0)
            elif funding_rate > -0.0005:
                detail = (f"{rate_pct:.4f}% ({funding_source}) — "
                          "Mildly negative: shorts paying longs. Slight contrarian bullish.")
                sig = ("BUY", 0.10)
            elif funding_rate > -0.001:
                detail = (f"{rate_pct:.4f}% ({funding_source}) — "
                          "Negative: crowded shorts. Squeeze UP potential.")
                sig = ("BUY", 0.20)
            else:
                detail = (f"{rate_pct:.4f}% ({funding_source}) — "
                          "HIGH negative: shorts overleveraged. Short squeeze UP highly probable.")
                sig = ("BUY", 0.35)

            evidence.append(self._signal("Funding Rate", detail, sig[0], sig[1]))
            signals.append(sig)
        else:
            evidence.append("⚠️ Funding rate unavailable from Binance & OKX")

        # ═══════════════════════════════════════════════════════
        # 3. GLOBAL LONG/SHORT RATIO
        # ═══════════════════════════════════════════════════════
        ls_url = (
            f"https://fapi.binance.com/futures/data/"
            f"globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
        )
        ls_data = self._safe_get(ls_url)
        if ls_data and len(ls_data) > 0:
            row = ls_data[0]
            long_ratio = float(row["longAccountRatio"])
            short_ratio = float(row["shortAccountRatio"])
            ls_value = float(row["longShortRatio"])
            data["ls_ratio"] = ls_value
            data["ls_long_ratio"] = long_ratio
            data["ls_short_ratio"] = short_ratio

            # High L/S = retail bullish = contrarian bearish
            if long_ratio > 2.5:
                detail = (f"Long {long_ratio:.2f} / Short {short_ratio:.2f} — "
                          "EXTREME retail long bias. Contrarian SELL. Retail always wrong at extremes.")
                sig = ("SELL", 0.30)
            elif long_ratio > 1.8:
                detail = (f"Long {long_ratio:.2f} / Short {short_ratio:.2f} — "
                          "Retail heavily long. Crowd overconfident.")
                sig = ("SELL", 0.20)
            elif long_ratio > 1.3:
                detail = (f"Long {long_ratio:.2f} / Short {short_ratio:.2f} — "
                          "Mildly long-biased retail. Slight bearish tilt.")
                sig = ("SELL", 0.10)
            elif long_ratio < 0.5:
                detail = (f"Long {long_ratio:.2f} / Short {short_ratio:.2f} — "
                          "EXTREME retail short bias. Contrarian BUY. Panic-short = squeeze fuel.")
                sig = ("BUY", 0.30)
            elif long_ratio < 0.7:
                detail = (f"Long {long_ratio:.2f} / Short {short_ratio:.2f} — "
                          "Retail heavily short. Contrarian bullish.")
                sig = ("BUY", 0.20)
            elif long_ratio < 0.9:
                detail = (f"Long {long_ratio:.2f} / Short {short_ratio:.2f} — "
                          "Mildly short-biased. Slight bullish tilt.")
                sig = ("BUY", 0.10)
            else:
                detail = f"Long {long_ratio:.2f} / Short {short_ratio:.2f} — Balanced retail. No edge."
                sig = ("NEUTRAL", 0.0)

            evidence.append(self._signal("Global L/S Ratio", detail, sig[0], sig[1]))
            signals.append(sig)
        else:
            evidence.append("⚠️ Global Long/Short Ratio unavailable")

        # ═══════════════════════════════════════════════════════
        # 4. TOP TRADER LONG/SHORT RATIO (institutions — follow them)
        # ═══════════════════════════════════════════════════════
        top_url = (
            f"https://fapi.binance.com/futures/data/"
            f"topLongShortPositionRatio?symbol={symbol}&period=1h&limit=1"
        )
        top_data = self._safe_get(top_url)
        if top_data and len(top_data) > 0:
            row = top_data[0]
            t_long = float(row["longAccountRatio"])
            t_short = float(row["shortAccountRatio"])
            t_ratio = float(row["longShortRatio"])
            data["top_trader_ls_ratio"] = t_ratio
            data["top_trader_long_ratio"] = t_long
            data["top_trader_short_ratio"] = t_short

            # Top traders = institutions — follow them directly
            if t_long > 2.0:
                detail = (f"Top traders Long {t_long:.2f} / Short {t_short:.2f} — "
                          "Institutions HEAVILY long. Strong conviction buy.")
                sig = ("BUY", 0.30)
            elif t_long > 1.5:
                detail = (f"Top traders Long {t_long:.2f} / Short {t_short:.2f} — "
                          "Institutions leaning long. Smart money bullish.")
                sig = ("BUY", 0.20)
            elif t_short > 2.0:
                detail = (f"Top traders Long {t_long:.2f} / Short {t_short:.2f} — "
                          "Institutions HEAVILY short. Smart money bearish.")
                sig = ("SELL", 0.30)
            elif t_short > 1.5:
                detail = (f"Top traders Long {t_long:.2f} / Short {t_short:.2f} — "
                          "Institutions leaning short. Caution.")
                sig = ("SELL", 0.20)
            else:
                detail = (f"Top traders Long {t_long:.2f} / Short {t_short:.2f} — "
                          "Institutions neutral/balanced.")
                sig = ("NEUTRAL", 0.0)

            evidence.append(self._signal("Top Trader L/S", detail, sig[0], sig[1]))
            signals.append(sig)
        else:
            evidence.append("⚠️ Top Trader L/S Ratio unavailable")

        # ═══════════════════════════════════════════════════════
        # 5. BTC DOMINANCE
        # ═══════════════════════════════════════════════════════
        global_data = self._safe_get("https://api.coingecko.com/api/v3/global")
        if global_data and "data" in global_data:
            btc_dom = global_data["data"].get("market_cap_percentage", {}).get("btc", 0)
            data["btc_dominance"] = round(btc_dom, 2)
            data["total_market_cap"] = global_data["data"].get(
                "total_market_cap", {}
            ).get("usd", 0)
            evidence.append(f"📊 BTC Dominance: {btc_dom:.1f}%")
        else:
            evidence.append("⚠️ BTC Dominance data unavailable")

        # ═══════════════════════════════════════════════════════
        # AGGREGATION
        # ═══════════════════════════════════════════════════════
        if signals:
            buy_weight = sum(w for d, w in signals if d == "BUY")
            sell_weight = sum(w for d, w in signals if d == "SELL")
            net = buy_weight - sell_weight
            score = float(np.clip(net, -1.0, 1.0))

            # ── Retail vs Institutional Divergence ──
            retail_signals = []
            inst_signals = []
            for i, (d, w) in enumerate(signals):
                label = evidence[i] if i < len(evidence) else ""
                if any(kw in label for kw in ["Fear", "Global L/S", "Retail"]):
                    retail_signals.append((d, w))
                elif any(kw in label for kw in ["Top Trader", "Funding"]):
                    inst_signals.append((d, w))

            retail_bias = (
                sum(w for d, w in retail_signals if d == "BUY") -
                sum(w for d, w in retail_signals if d == "SELL")
            )
            institutional_bias = (
                sum(w for d, w in inst_signals if d == "BUY") -
                sum(w for d, w in inst_signals if d == "SELL")
            )
            data["retail_bias"] = round(retail_bias, 3)
            data["institutional_bias"] = round(institutional_bias, 3)

            # Divergence detection
            if retail_bias > 0.1 and institutional_bias < -0.1:
                evidence.append(
                    "⚠️ DIVERGENCE: Retail bullish but institutions bearish. "
                    "Follow institutions."
                )
            elif retail_bias < -0.1 and institutional_bias > 0.1:
                evidence.append(
                    "⚠️ DIVERGENCE: Retail bearish but institutions bullish. "
                    "Follow institutions."
                )

            direction = "BUY" if score > 0.1 else "SELL" if score < -0.1 else "NEUTRAL"
            confidence = min(100, abs(score) * 90 + 20)
        else:
            direction = "NEUTRAL"
            confidence = 0

        data["signals_summary"] = [{"direction": d, "weight": w} for d, w in signals]

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=score,
            evidence=evidence,
            data=data,
            reasoning=(
                f"Sentiment: {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'} "
                f"({len(signals)} signals, retail_bias={data.get('retail_bias', 0):.2f}, "
                f"inst_bias={data.get('institutional_bias', 0):.2f})"
            ),
        )
