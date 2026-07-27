"""
Explainability Agent
====================
Explains WHY a recommendation was made.
Shows top contributing factors and rejected signals.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class ExplainabilityAgent(BaseAgent):
    name = "Explainability"
    weight = 0.0

    def analyze(self, df=None, symbol="", timeframe="",
                all_results=None, consensus_result=None, **kwargs):
        if not consensus_result:
            return self._out(direction="NEUTRAL", confidence=0, evidence=["No consensus to explain"])

        evidence = []
        direction = consensus_result.direction

        if direction == "NO_TRADE":
            evidence.append("REASON FOR NO TRADE:")
            if all_results:
                for tf, agents in all_results.items():
                    buy_agree = sum(1 for a in agents if a.direction == "BUY")
                    sell_agree = sum(1 for a in agents if a.direction == "SELL")
                    evidence.append(f"  {tf}: {buy_agree} BUY vs {sell_agree} SELL agents")
                evidence.append("Agents failed to reach sufficient agreement.")
                evidence.append("System correctly protected capital by staying out.")
            return self._out(direction="NEUTRAL", confidence=100, evidence=evidence,
                           reasoning="Explaining NO_TRADE decision")

        # For BUY/SELL - explain why
        evidence.append(f"=== WHY {direction} {symbol} ===")
        evidence.append("")

        # Top supporting evidence
        supporting = []
        opposing = []
        if all_results:
            for tf, agents in all_results.items():
                for a in agents:
                    if a.weight == 0:
                        continue
                    if a.direction == direction:
                        supporting.append((a.agent_name, tf, a.confidence, a.evidence[:2]))
                    elif a.direction != "NEUTRAL":
                        opposing.append((a.agent_name, tf, a.confidence, a.evidence[:2]))

        supporting.sort(key=lambda x: -x[2])
        opposing.sort(key=lambda x: -x[2])

        evidence.append("🟢 SUPPORTING FACTORS:")
        for name, tf, conf, ev in supporting[:8]:
            evidence.append(f"  • {name} [{tf}] ({conf:.0f}%): {ev[0] if ev else ''}")

        if opposing:
            evidence.append("")
            evidence.append("🔴 OPPOSING FACTORS:")
            for name, tf, conf, ev in opposing[:5]:
                evidence.append(f"  • {name} [{tf}] ({conf:.0f}%): {ev[0] if ev else ''}")

        # Multi-TF alignment
        evidence.append("")
        if all_results:
            tf_directions = {}
            for tf, agents in all_results.items():
                buy_s = sum(a.confidence for a in agents if a.direction == "BUY")
                sell_s = sum(a.confidence for a in agents if a.direction == "SELL")
                tf_directions[tf] = "BUY" if buy_s > sell_s else "SELL" if sell_s > buy_s else "NEUTRAL"

            aligned = [tf for tf, d in tf_directions.items() if d == direction]
            evidence.append(f"📊 TIMEFRAME ALIGNMENT:")
            for tf, d in tf_directions.items():
                marker = "✅" if d == direction else "❌" if d != "NEUTRAL" else "➖"
                evidence.append(f"  {marker} {tf}: {d}")
            evidence.append(f"  Aligned: {len(aligned)}/{len(tf_directions)} timeframes")

        # Confidence factors
        evidence.append("")
        grade = consensus_result.data.get("grade", "?")
        agree = consensus_result.data.get("agree_pct", 0)
        evidence.append(f"📈 CONFIDENCE BREAKDOWN:")
        evidence.append(f"  Grade: {grade}")
        evidence.append(f"  TF Agreement: {agree:.0%}")
        evidence.append(f"  Overall: {consensus_result.confidence:.0f}%")

        # Risk factors
        evidence.append("")
        evidence.append("⚠️ RISK FACTORS:")
        evidence.append("  • This is an AI recommendation, not financial advice")
        evidence.append("  • Past performance doesn't guarantee future results")
        evidence.append("  • Always use proper position sizing")
        evidence.append("  • Set stop loss before entering any trade")

        return self._out(
            direction="NEUTRAL",
            confidence=100,
            score=0,
            evidence=evidence,
            reasoning=f"Full explanation for {direction} {symbol}"
        )
