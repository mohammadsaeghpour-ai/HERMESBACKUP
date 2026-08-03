"""
HQIP Game Theory Agent
========================
Models the market as a multi-player game:
- Retail traders (herd behavior, predictable)
- Institutional players (strategic, contrarian)
- Market makers (liquidity providers, spread capturers)
Uses Nash Equilibrium, Expected Utility, Minimax, and
Bayesian updating to determine optimal strategy.
"""
from hqip.agents.base import BaseAgent
import numpy as np


class GameTheoryAgent(BaseAgent):
    name = "GameTheory"
    weight = 1.3

    def analyze(self, df, symbol="", timeframe="", **kwargs):
        if df is None or df.empty or len(df) < 30:
            return self._out("NEUTRAL", 0, evidence=["insufficient data"])

        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        p = closes[-1]
        evidence = []
        score = 0.0

        # ═══════════════════════════════════════════════════
        # 1. RETAIL vs INSTITUTIONAL GAME
        # ═══════════════════════════════════════════════════
        # Retail tends to: buy high (FOMO), sell low (panic)
        # Institutional tends to: buy low (accumulation), sell high (distribution)

        # Measure retail sentiment via volume + price direction
        recent_vol = np.mean(volumes[-5:])
        avg_vol = np.mean(volumes[-20:])
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

        recent_returns = np.diff(np.log(closes[-10:]))
        avg_return = np.mean(recent_returns)

        # Retail FOMO detection: high volume + rising price = retail buying
        if vol_ratio > 1.5 and avg_return > 0.001:
            evidence.append("🐟 خرده‌فروش‌ها FOMO دارن (حجم بالا + قیمت صعودی)")
            score -= 0.15  # Contrarian: sell when retail buys
        elif vol_ratio > 1.5 and avg_return < -0.001:
            evidence.append("🐟 خرده‌فروش‌ها panic selling (حجم بالا + قیمت نزولی)")
            score += 0.15  # Contrarian: buy when retail sells

        # ═══════════════════════════════════════════════════
        # 2. NASH EQUILIBRIUM — Best Response
        # ═══════════════════════════════════════════════════
        # In a 2-player game (Bull vs Bear):
        # If both play aggressively → high volatility, unpredictable
        # If one plays passively → the other wins
        # Nash Equilibrium: mix of strategies

        # Estimate market regime
        returns_20 = np.diff(np.log(closes[-20:]))
        vol_regime = np.std(returns_20) * 100
        trend_strength = abs(np.mean(returns_20)) * 100

        if trend_strength > vol_regime * 0.5:
            # Strong trend → trend-following dominates
            evidence.append("📊 نظریه بازی: روند قوی → ترند فالوینگ بهینه")
            if avg_return > 0:
                score += 0.10
            else:
                score -= 0.10
        elif vol_regime > 0.5:
            # High vol, no trend → mean reversion dominates
            evidence.append("📊 نظریه بازی: نوسان بالا → mean reversion بهینه")
            if avg_return > 0.003:
                score -= 0.10
            elif avg_return < -0.003:
                score += 0.10

        # ═══════════════════════════════════════════════════
        # 3. EXPECTED UTILITY — Kelly's criterion logic
        # ═══════════════════════════════════════════════════
        # Win probability from recent pattern
        wins = sum(1 for r in recent_returns if r > 0)
        losses = len(recent_returns) - wins
        win_prob = wins / len(recent_returns) if len(recent_returns) > 0 else 0.5

        avg_win = np.mean([r for r in recent_returns if r > 0]) if wins > 0 else 0.001
        avg_loss = abs(np.mean([r for r in recent_returns if r < 0])) if losses > 0 else 0.001

        # Kelly fraction: f* = (p*b - q) / b
        # where p = win prob, q = 1-p, b = avg_win/avg_loss
        b = avg_win / avg_loss if avg_loss > 0 else 1
        kelly = (win_prob * b - (1 - win_prob)) / b if b > 0 else 0

        if kelly > 0.1:
            evidence.append(f"🟢 Kelly={kelly:.2f} — بازی به نفع خریداران")
            score += 0.15
        elif kelly < -0.1:
            evidence.append(f"🔴 Kelly={kelly:.2f} — بازی به نفع فروشندگان")
            score -= 0.15
        else:
            evidence.append(f"🟡 Kelly={kelly:.2f} — بازی نزدیک به تعادل")

        evidence.append(f"📊 Win rate: {win_prob:.0%} | R:R واقعی: 1:{b:.1f}")

        # ═══════════════════════════════════════════════════
        # 4. MINIMAX — Worst case analysis
        # ═══════════════════════════════════════════════════
        # What's the worst that can happen?
        max_drawdown = 0
        peak = closes[-1]
        for c in closes[-20:]:
            if c > peak:
                peak = c
            dd = (peak - c) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        if max_drawdown > 0.03:
            evidence.append(f"⚠️ Drawdown اخیر: {max_drawdown*100:.1f}% — ریسک بالا")
            score *= 0.8  # Reduce conviction in high-risk environment

        # ═══════════════════════════════════════════════════
        # 5. BAYESIAN UPDATING
        # ═══════════════════════════════════════════════════
        # Prior: 50/50 buy/sell
        # Update with each new candle
        prior_buy = 0.5
        for i in range(-10, 0):
            candle_return = (closes[i] - closes[i-1]) / closes[i-1]
            if candle_return > 0:
                # Likelihood of buy given green candle
                prior_buy = min(0.9, prior_buy * 1.1)
            else:
                prior_buy = max(0.1, prior_buy * 0.9)

        posterior_buy = prior_buy
        if posterior_buy > 0.6:
            score += 0.10
            evidence.append(f"🟢 Bayesian P(buy)={posterior_buy:.0%}")
        elif posterior_buy < 0.4:
            score -= 0.10
            evidence.append(f"🔴 Bayesian P(buy)={posterior_buy:.0%}")

        # ═══════════════════════════════════════════════════
        # 6. MARKET MAKER GAME
        # ═══════════════════════════════════════════════════
        # Market makers profit from stops on both sides
        # They hunt stops before moving price in real direction
        upper_wick = (highs[-1] - max(closes[-1], df.iloc[-1]["open"])) / p * 100
        lower_wick = (min(closes[-1], df.iloc[-1]["open"]) - lows[-1]) / p * 100

        if upper_wick > 0.3 and lower_wick > 0.3:
            evidence.append(f"🎰 Market Maker بازی: سایه بالا {upper_wick:.2f}% + پایین {lower_wick:.2f}%")
            score *= 0.7  # Reduce conviction — manipulation likely

        # ═══════════════════════════════════════════════════
        # FINAL DECISION
        # ═══════════════════════════════════════════════════
        direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NO_TRADE"
        confidence = min(100, abs(score) * 140 + 20)

        return self._out(
            direction=direction,
            confidence=round(confidence, 1),
            score=round(score, 3),
            evidence=evidence[:7],
            reasoning=f"GameTheory: kelly={kelly:.2f}, bayes={posterior_buy:.2f}, retail={vol_ratio:.1f}x",
            data={
                "kelly_fraction": round(kelly, 3),
                "bayesian_buy_prob": round(posterior_buy, 3),
                "win_rate": round(win_prob, 3),
                "real_rr": round(b, 2),
                "max_drawdown": round(max_drawdown * 100, 2),
                "vol_regime": round(vol_regime, 4),
            },
        )
