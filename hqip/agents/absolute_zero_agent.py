"""
HQIP Absolute Zero Agent
=========================
Learns market from scratch — no assumptions.
Measures: velocity, volume intensity, range expansion,
absorption, swing structure. Session-aware.
"""
from hqip.agents.base import BaseAgent
from hqip.absolute_zero import AbsoluteZeroEngine
import numpy as np


class AbsoluteZeroAgent(BaseAgent):
    name = "AbsoluteZero"
    weight = 1.5

    def __init__(self):
        super().__init__()
        self.engine = AbsoluteZeroEngine()

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 20:
            return self._out("NEUTRAL", 0, evidence=["insufficient data"])

        session_key = kwargs.get("session_key", "off")
        obs = self.engine.observe(df, timeframe)
        if obs is None:
            return self._out("NEUTRAL", 0, evidence=["observation failed"])

        # Single-TF decision (the engine handles multi-TF via decide())
        score = 0
        reasons = []

        # Structure
        if obs["structure"] == "UP":
            score += 0.40
            reasons.append(f"🟢 ساختار صعودی (HH+HL)")
        elif obs["structure"] == "DOWN":
            score -= 0.40
            reasons.append(f"🔴 ساختار نزولی (LH+LL)")

        # Velocity
        vel = obs["velocity"]
        if vel > 0.3:
            score += 0.25
            reasons.append(f"🟢 شتاب صعودی ({vel:+.3f}%)")
        elif vel < -0.3:
            score -= 0.25
            reasons.append(f"🔴 شتاب نزولی ({vel:+.3f}%)")

        # Volume
        if obs["vol_intensity"] > 1.5 and obs["direction_bias"] > 0.2:
            score += 0.20
            reasons.append(f"🟢 حجم بالا + سبز ({obs['vol_intensity']:.1f}x)")
        elif obs["vol_intensity"] > 1.5 and obs["direction_bias"] < -0.2:
            score -= 0.20
            reasons.append(f"🔴 حجم بالا + قرمز ({obs['vol_intensity']:.1f}x)")

        # Absorption
        if obs["absorption"] > 2.0:
            reasons.append(f"🔵 جذب نهادی (vol/range={obs['absorption']:.1f})")

        # Range position
        pos = obs["position_in_range"]
        if pos > 85:
            score -= 0.15
            reasons.append(f"🔴 نزدیک مقاومت ({pos:.0f}%)")
        elif pos < 15:
            score += 0.15
            reasons.append(f"🟢 نزدیک حمایت ({pos:.0f}%)")

        # Session
        if session_key == "overlap":
            score *= 1.10
            reasons.append("⚡ سشن همپوشانی — نقدینگی بالا")
        elif session_key == "asia":
            score *= 0.85
            reasons.append("⚠️ سشن آسیا — نقدینگی پایین")

        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NO_TRADE"
        confidence = min(100, abs(score) * 150 + 20)

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=round(score, 3),
            evidence=reasons[:6],
            reasoning=f"AbsZero: structure={obs['structure']}, vel={vel:+.3f}%, vol={obs['vol_intensity']:.1f}x",
            data={
                "structure": obs["structure"],
                "velocity": round(obs["velocity"], 4),
                "vol_intensity": round(obs["vol_intensity"], 2),
                "absorption": round(obs["absorption"], 2),
                "position_in_range": round(obs["position_in_range"], 1),
            },
        )
