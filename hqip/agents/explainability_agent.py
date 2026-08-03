"""
Explainability Agent — Enhanced Explainability
===============================================
Generates clear ~100-word Farsi explanations, institutional vs retail analysis,
contradiction detection, confidence scoring, and score waterfall.

Features:
- Farsi explanation (~100 words)
- What institutions are doing
- What retail traders are doing
- Key contradictions between agents
- Confidence score 0-100
- Score waterfall: which agents contributed most

Weight: 0 (doesn't vote)
"""
from hqip.agents.base import BaseAgent
import numpy as np


class ExplainabilityAgent(BaseAgent):
    name = "Explainability"
    weight = 0  # doesn't vote

    # ── Signal names mapping for Farsi ───────────────────────
    _SIGNAL_NAMES = {
        "Fear & Greed": "شاخص ترس و طمع",
        "Funding Rate": "نرخ تأمین مالی",
        "Global L/S Ratio": "نسبت لانگ/شورت عمومی",
        "Top Trader L/S": "نسبت لانگ/شورت معامله‌گران بزرگ",
        "OI Analysis": "تحلیل اوپن اینترست",
        "Technical": "تحلیل تکنیکال",
        "VolumeProfile": "پروفایل حجم",
        "Momentum": "مومنتوم",
        "Whale": "نهادهای بزرگ",
        "Pattern": "الگوهای کندلی",
        "MarketStructure": "ساختار بازار",
        "Trend": "روند",
    }

    def analyze(self, df=None, symbol="", timeframe="",
                all_results=None, consensus_result=None, **kwargs):
        if not consensus_result:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["No consensus to explain"],
                data={"confidence_score": 0, "farsi_explanation": ""},
            )

        evidence = []
        direction = consensus_result.direction

        # ═══════════════════════════════════════════════════════
        # COLLECT ALL AGENT RESULTS
        # ═══════════════════════════════════════════════════════
        all_agents = []
        if all_results:
            for tf, agents in all_results.items():
                for a in agents:
                    name = getattr(a, "agent_name", getattr(a, "name", "?"))
                    all_agents.append({
                        "name": name,
                        "direction": a.direction,
                        "confidence": a.confidence,
                        "score": a.score,
                        "evidence": a.evidence[:3],
                        "timeframe": tf,
                        "weight": a.weight,
                        "data": getattr(a, "data", {}),
                    })

        buy_agents = [a for a in all_agents if a["direction"] == "BUY" and a["weight"] > 0]
        sell_agents = [a for a in all_agents if a["direction"] == "SELL" and a["weight"] > 0]

        # ═══════════════════════════════════════════════════════
        # NO TRADE EXPLANATION
        # ═══════════════════════════════════════════════════════
        if direction in ("NO_TRADE", "WAIT"):
            evidence.append("═══ REASON FOR NO TRADE ═══")
            if all_results:
                for tf, agents in all_results.items():
                    buy_c = sum(1 for a in agents if a.direction == "BUY")
                    sell_c = sum(1 for a in agents if a.direction == "SELL")
                    neu_c = sum(1 for a in agents if a.direction == "NEUTRAL")
                    evidence.append(f"  {tf}: {buy_c} BUY · {sell_c} SELL · {neu_c} NEUTRAL")

            evidence.append("Agents failed to reach sufficient agreement.")
            evidence.append("System correctly protected capital by staying out.")

            farsi_explanation = (
                f"بازار {symbol} در وضعیت عدم قطعیت است. "
                f"معامله‌گر هوشمند صبر می‌کند تا سیگنال‌ها همسو شوند."
            )

            confidence_score = 30
            return self._out(
                direction="NEUTRAL",
                confidence=100,
                score=0,
                evidence=evidence,
                reasoning="NO_TRADE — agents disagree",
                data={
                    "confidence_score": confidence_score,
                    "farsi_explanation": farsi_explanation,
                },
            )

        # ═══════════════════════════════════════════════════════
        # BUY / SELL EXPLANATION
        # ═══════════════════════════════════════════════════════
        evidence.append(f"═══ WHY {direction} {symbol} ═══")

        # ── Institutional vs Retail Analysis ──
        inst_bias = self._analyze_institutional(all_agents)
        retail_bias = self._analyze_retail(all_agents)

        evidence.append("")
        evidence.append("🏛️ INSTITUTIONAL BEHAVIOR:")
        evidence.append(f"  {inst_bias['summary']}")
        evidence.append("")
        evidence.append("👥 RETAIL BEHAVIOR:")
        evidence.append(f"  {retail_bias['summary']}")

        # ── Divergence ──
        evidence.append("")
        if inst_bias["direction"] and retail_bias["direction"]:
            if inst_bias["direction"] != retail_bias["direction"]:
                evidence.append(
                    f"⚠️ DIVERGENCE: Institutions {inst_bias['direction']} "
                    f"while retail {retail_bias['direction']}"
                )
                evidence.append(
                    "  → Follow institutions — retail is wrong at turning points"
                )
            else:
                evidence.append(
                    f"✅ ALIGNMENT: Both institutions and retail are "
                    f"{inst_bias['direction']}"
                )
        else:
            evidence.append("📊 Insufficient data for divergence analysis")

        # ── Supporting / Opposing Factors ──
        supporting = [a for a in all_agents
                      if a["direction"] == direction and a["weight"] > 0]
        opposing = [a for a in all_agents
                    if a["direction"] not in (direction, "NEUTRAL") and a["weight"] > 0]
        supporting.sort(key=lambda x: -x["confidence"])
        opposing.sort(key=lambda x: -x["confidence"])

        evidence.append("")
        evidence.append("🟢 SUPPORTING FACTORS:")
        for a in supporting[:8]:
            sig_name = self._SIGNAL_NAMES.get(a["name"], a["name"])
            top_ev = a["evidence"][0] if a["evidence"] else ""
            evidence.append(
                f"  • {sig_name} [{a['timeframe']}] "
                f"({a['confidence']:.0f}%): {top_ev}"
            )

        if opposing:
            evidence.append("")
            evidence.append("🔴 OPPOSING FACTORS:")
            for a in opposing[:5]:
                sig_name = self._SIGNAL_NAMES.get(a["name"], a["name"])
                top_ev = a["evidence"][0] if a["evidence"] else ""
                evidence.append(
                    f"  • {sig_name} [{a['timeframe']}] "
                    f"({a['confidence']:.0f}%): {top_ev}"
                )

        # ── Contradiction Analysis ──
        evidence.append("")
        contradictions = self._find_contradictions(all_agents)
        if contradictions:
            evidence.append("🔀 CONTRADICTIONS BETWEEN AGENTS:")
            for c in contradictions:
                evidence.append(f"  • {c}")
        else:
            evidence.append("✅ No major contradictions between agents")

        # ── Score Waterfall ──
        evidence.append("")
        evidence.append("📊 SCORE WATERFALL (who contributed most):")
        waterfall = self._compute_waterfall(all_agents, direction)
        for item in waterfall[:5]:
            evidence.append(f"  {item}")

        # ── Confidence Breakdown ──
        evidence.append("")
        grade = consensus_result.data.get("grade", "?")
        agree_pct = consensus_result.data.get("agree_pct", 0)
        evidence.append("📈 CONFIDENCE BREAKDOWN:")
        evidence.append(f"  Grade: {grade}")
        evidence.append(f"  TF Agreement: {agree_pct:.0%}")
        evidence.append(f"  Overall: {consensus_result.confidence:.0f}%")

        # ── Risk Factors ──
        evidence.append("")
        evidence.append("⚠️ RISK FACTORS:")
        evidence.append("  • AI recommendation, not financial advice")
        evidence.append("  • Always use proper position sizing")
        evidence.append("  • Set stop loss before entering")

        # ═══════════════════════════════════════════════════════
        # CONFIDENCE SCORE (0-100)
        # ═══════════════════════════════════════════════════════
        confidence_score = self._compute_confidence(
            direction, buy_agents, sell_agents, all_agents, consensus_result
        )

        # ═══════════════════════════════════════════════════════
        # FARSI EXPLANATION (~100 words)
        # ═══════════════════════════════════════════════════════
        farsi_explanation = self._generate_farsi(
            symbol, direction, inst_bias, retail_bias,
            contradictions, len(buy_agents), len(sell_agents),
            confidence_score
        )

        return self._out(
            direction="NEUTRAL",
            confidence=100,
            score=0,
            evidence=evidence,
            reasoning=f"Full explanation for {direction} {symbol}",
            data={
                "confidence_score": confidence_score,
                "farsi_explanation": farsi_explanation,
                "institutional_bias": inst_bias,
                "retail_bias": retail_bias,
                "contradictions": contradictions,
                "waterfall": waterfall[:5],
            },
        )

    # ── Helper: Institutional analysis ───────────────────────
    def _analyze_institutional(self, agents):
        """Analyze what institutions are doing based on smart-money agents."""
        inst_agents = [
            a for a in agents
            if any(kw in a["name"].lower() for kw in
                   ["top trader", "institution", "smart", "whale"])
            or a.get("data", {}).get("institutional_bias") is not None
        ]

        if not inst_agents:
            return {"direction": None, "summary": "Insufficient institutional data"}

        buy_score = sum(a["confidence"] for a in inst_agents if a["direction"] == "BUY")
        sell_score = sum(a["confidence"] for a in inst_agents if a["direction"] == "SELL")

        if buy_score > sell_score * 1.2:
            return {
                "direction": "BUY",
                "summary": (
                    f"Institutions accumulating "
                    f"(buy: {buy_score:.0f} vs sell: {sell_score:.0f}). "
                    f"Smart money positioning for upside."
                ),
            }
        elif sell_score > buy_score * 1.2:
            return {
                "direction": "SELL",
                "summary": (
                    f"Institutions distributing "
                    f"(sell: {sell_score:.0f} vs buy: {buy_score:.0f}). "
                    f"Smart money taking profits."
                ),
            }
        return {
            "direction": "NEUTRAL",
            "summary": "Institutions show no clear directional bias.",
        }

    # ── Helper: Retail analysis ──────────────────────────────
    def _analyze_retail(self, agents):
        """Analyze what retail traders are doing based on sentiment agents."""
        retail_agents = [
            a for a in agents
            if any(kw in a["name"].lower() for kw in
                   ["fear", "greed", "global l/s", "retail"])
        ]

        if not retail_agents:
            return {"direction": None, "summary": "Insufficient retail sentiment data"}

        buy_score = sum(a["confidence"] for a in retail_agents if a["direction"] == "BUY")
        sell_score = sum(a["confidence"] for a in retail_agents if a["direction"] == "SELL")

        if buy_score > sell_score * 1.2:
            return {
                "direction": "BUY",
                "summary": (
                    f"Retail is bullish (buy: {buy_score:.0f} vs sell: {sell_score:.0f}). "
                    f"Caution: retail sentiment is often a contrarian indicator."
                ),
            }
        elif sell_score > buy_score * 1.2:
            return {
                "direction": "SELL",
                "summary": (
                    f"Retail is bearish (sell: {sell_score:.0f} vs buy: {buy_score:.0f}). "
                    f"Contrarian opportunity: fear often marks bottoms."
                ),
            }
        return {"direction": "NEUTRAL", "summary": "Retail sentiment is mixed."}

    # ── Helper: Find contradictions ──────────────────────────
    def _find_contradictions(self, agents):
        """Detect high-confidence contradictions between agent pairs."""
        contradictions = []
        buy_agents = [a for a in agents if a["direction"] == "BUY" and a["weight"] > 0]
        sell_agents = [a for a in agents if a["direction"] == "SELL" and a["weight"] > 0]

        for ba in buy_agents:
            for sa in sell_agents:
                if (ba["weight"] > 0.5 and sa["weight"] > 0.5
                        and ba["confidence"] > 50 and sa["confidence"] > 50):
                    b_name = self._SIGNAL_NAMES.get(ba["name"], ba["name"])
                    s_name = self._SIGNAL_NAMES.get(sa["name"], sa["name"])
                    contradictions.append(
                        f"{b_name} ({ba['timeframe']}, {ba['confidence']:.0f}% BUY) "
                        f"vs {s_name} ({sa['timeframe']}, {sa['confidence']:.0f}% SELL)"
                    )

        return contradictions

    # ── Helper: Score waterfall ──────────────────────────────
    def _compute_waterfall(self, agents, direction):
        """Rank agents by contribution to the final score."""
        contributions = []
        for a in agents:
            if a["direction"] == direction and a["weight"] > 0:
                contribution = a["weight"] * (a["confidence"] / 100)
                contributions.append((a["name"], a["timeframe"], contribution, a["confidence"]))

        contributions.sort(key=lambda x: -x[2])

        total = sum(c[2] for c in contributions) if contributions else 1
        result = []
        for name, tf, contrib, conf in contributions:
            pct = (contrib / total * 100) if total > 0 else 0
            result.append(f"{name} [{tf}]: {pct:.0f}% of score (conf={conf:.0f}%)")

        return result

    # ── Helper: Confidence score ─────────────────────────────
    def _compute_confidence(self, direction, buy_agents, sell_agents,
                            all_agents, consensus):
        """Compute a 0-100 confidence score for the explanation."""
        factors = []

        # 1. Agreement ratio (up to 40 pts)
        relevant = buy_agents if direction == "BUY" else sell_agents
        total_active = len(buy_agents) + len(sell_agents)
        if total_active > 0:
            agreement = len(relevant) / total_active
            factors.append(agreement * 40)

        # 2. Average confidence of agreeing agents (up to 25 pts)
        if relevant:
            avg_conf = np.mean([a["confidence"] for a in relevant])
            factors.append(avg_conf * 0.25)

        # 3. Contradiction penalty (max -20 pts)
        contradictions = self._find_contradictions(all_agents)
        penalty = min(20, len(contradictions) * 10)
        factors.append(20 - penalty)

        # 4. Consensus agreement (up to 15 pts)
        agree_pct = consensus.data.get("agree_pct", 0)
        factors.append(agree_pct * 15)

        score = sum(factors)
        return max(0, min(100, round(score)))

    # ── Helper: Farsi explanation ────────────────────────────
    def _generate_farsi(self, symbol, direction, inst_bias, retail_bias,
                        contradictions, n_buy, n_sell, confidence):
        """Generate a ~100-word Farsi explanation."""
        direction_fa = {
            "BUY": "خرید", "SELL": "فروش",
            "NEUTRAL": "خنثی", "NO_TRADE": "عدم معامله",
        }.get(direction, direction)

        parts = []

        # Overview
        parts.append(
            f"سیگنال فعلی برای {symbol} '{direction_fa}' است "
            f"با اطمینان {confidence} درصد."
        )

        # Institutions
        if inst_bias.get("direction") == "BUY":
            parts.append("نهادها در حال جمع‌آوری هستند و قیمت را بالا می‌برند.")
        elif inst_bias.get("direction") == "SELL":
            parts.append("نهادها در حال توزیع سود هستند و فشار فروش ایجاد می‌کنند.")
        else:
            parts.append("رفتار نهادها نامشخص است.")

        # Retail
        if retail_bias.get("direction") == "BUY":
            parts.append("عموم بازار صعودی‌اند ولی این سیگنال گمراه‌کننده است.")
        elif retail_bias.get("direction") == "SELL":
            parts.append("عموم بازار نزولی‌اند ولی این معمولاً فرصت خرید است.")
        else:
            parts.append("احساسات عمومی خنثی است.")

        # Contradiction
        if contradictions:
            parts.append(
                f"{len(contradictions)} تناقض بین عوامل وجود دارد "
                "که نشان‌دهنده ابهام بازار است."
            )
        else:
            parts.append("عوامل هماهنگ هستند و سیگنال قوی‌تر است.")

        # Conclusion
        if confidence >= 70:
            parts.append("سیگنال قابل اعتمادی است.")
        elif confidence >= 40:
            parts.append("سیگنال متوسط است، منتظر تأیید بیشتر باشید.")
        else:
            parts.append("سیگنال ضعیف است. معامله نکنید.")

        return " ".join(parts)
