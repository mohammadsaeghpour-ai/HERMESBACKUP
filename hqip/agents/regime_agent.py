"""
Market Regime Detection Agent — Adaptive Regime Classification
===============================================================
Detects and classifies the current market regime using a multi-factor
scoring system.  The 5 target regimes are:

1. **TRENDING_UP** — strong directional move upward
2. **TRENDING_DOWN** — strong directional move downward
3. **RANGING** — sideways/consolidation with no clear direction
4. **VOLATILE** — high volatility without clear direction (chaotic)
5. **CALM** — low volatility, tight consolidation

Features:
- ADX-based trend strength classification
- ATR percentile for volatility regime
- Bollinger Band width for compression/expansion
- Price vs EMA alignment for direction
- **Regime TRANSITION detection** — the most actionable signal
- Dynamic weight adjustment for other agents based on regime
- Outputs regime_info dict for consumption by other agents

Weight: 1.0
"""
from hqip.agents.base import BaseAgent
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class RegimeAgent(BaseAgent):
    """Market regime detection with transition alerting.

    Classifies the current market environment into one of five regimes and
    detects transitions between regimes (the most actionable signal).
    Also computes dynamic weight adjustments for downstream agents.

    Attributes
    ----------
    name : str
        Agent identifier.
    weight : float
        Consensus weight (1.0).
    REGIMES : list[str]
        Valid regime labels.
    """

    name = "Regime"
    weight = 1.0
    REGIMES = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "CALM"]

    # ------------------------------------------------------------------ #
    #  Internal: Percentile helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _percentile(arr, value, lookback=100):
        """Compute percentile rank of value within the last *lookback* observations.

        Parameters
        ----------
        arr : numpy.ndarray
            Historical values.
        value : float
            Value to rank.
        lookback : int
            Number of historical values to compare.

        Returns
        -------
        float
            Percentile (0–100).
        """
        n = min(lookback, len(arr))
        if n < 5:
            return 50.0
        return float(np.sum(arr[-n:] < value) / n * 100)

    # ------------------------------------------------------------------ #
    #  Internal: Regime scoring
    # ------------------------------------------------------------------ #
    def _score_regime(self, df):
        """Compute regime scores across all 5 categories.

        Parameters
        ----------
        df : pandas.DataFrame
            OHLCV + indicator data.

        Returns
        -------
        tuple[str, float, dict]
            Regime label, confidence, and detail dict with all scores.
        """
        close_arr = df["close"].values.astype(float)
        close = close_arr[-1]

        # ── ADX ──
        adx_val = float(df["adx"].iloc[-1]) if "adx" in df.columns else 20.0
        plus_di = float(df["plus_di"].iloc[-1]) if "plus_di" in df.columns else 25.0
        minus_di = float(df["minus_di"].iloc[-1]) if "minus_di" in df.columns else 25.0

        # ── ATR percentile ──
        atr_pct = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else 1.0
        atr_pctile = self._percentile(
            df["atr_pct"].values.astype(float) if "atr_pct" in df.columns else np.array([atr_pct]),
            atr_pct,
        )

        # ── BB width ──
        bb_width = float(df["bb_width"].iloc[-1]) if "bb_width" in df.columns else 0.02
        if "bb_width" in df.columns:
            bbw_pctile = self._percentile(
                df["bb_width"].values.astype(float), bb_width
            )
        else:
            bbw_pctile = 50.0

        # ── Price vs EMAs ──
        ema_vals = {}
        for p in [9, 21, 50, 200]:
            col = f"ema{p}"
            if col in df.columns:
                ema_vals[p] = float(df[col].iloc[-1])

        # EMA alignment score
        bullish_pairs = 0
        bearish_pairs = 0
        ema_list = [ema_vals.get(p, close) for p in [9, 21, 50, 200]]
        for i in range(len(ema_list) - 1):
            if ema_list[i] > ema_list[i + 1]:
                bullish_pairs += 1
            elif ema_list[i] < ema_list[i + 1]:
                bearish_pairs += 1

        ema_direction = (bullish_pairs - bearish_pairs) / 3.0  # [-1, 1]

        # Price slope (20-period)
        if len(close_arr) >= 20:
            x = np.arange(20)
            slope = np.polyfit(x, close_arr[-20:], 1)[0]
            norm_slope = slope / (np.mean(close_arr[-20:]) + 1e-10) * 100
        else:
            norm_slope = 0.0

        # ── Regime scoring ──
        scores = {
            "TRENDING_UP": 0.0,
            "TRENDING_DOWN": 0.0,
            "RANGING": 0.0,
            "VOLATILE": 0.0,
            "CALM": 0.0,
        }

        # ADX contribution
        if adx_val > 25:
            # Trending regime
            trend_weight = min(1.0, (adx_val - 25) / 25.0)
            if plus_di > minus_di:
                scores["TRENDING_UP"] += trend_weight * 3.0
            else:
                scores["TRENDING_DOWN"] += trend_weight * 3.0
        else:
            # Non-trending
            scores["RANGING"] += (1.0 - adx_val / 25.0) * 2.0

        # ATR percentile contribution
        if atr_pctile > 80:
            scores["VOLATILE"] += 2.5
        elif atr_pctile < 20:
            scores["CALM"] += 2.5
        else:
            scores["RANGING"] += 0.5

        # BB width contribution
        if bbw_pctile > 80:
            scores["VOLATILE"] += 1.5
        elif bbw_pctile < 20:
            scores["CALM"] += 1.5

        # EMA direction contribution
        if ema_direction > 0.3:
            scores["TRENDING_UP"] += abs(ema_direction) * 2.0
        elif ema_direction < -0.3:
            scores["TRENDING_DOWN"] += abs(ema_direction) * 2.0
        else:
            scores["RANGING"] += 1.0

        # Slope contribution
        if norm_slope > 0.1:
            scores["TRENDING_UP"] += min(2.0, abs(norm_slope) * 2)
        elif norm_slope < -0.1:
            scores["TRENDING_DOWN"] += min(2.0, abs(norm_slope) * 2)

        # ── Pick winning regime ──
        regime = max(scores, key=scores.get)
        max_score = scores[regime]
        total_score = sum(max(0, s) for s in scores.values())

        if total_score > 0:
            confidence = (max_score / total_score) * 100.0
        else:
            confidence = 0.0

        detail = {
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "adx": round(adx_val, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "atr_percentile": round(atr_pctile, 1),
            "bb_width_percentile": round(bbw_pctile, 1),
            "ema_direction": round(ema_direction, 3),
            "price_slope": round(norm_slope, 4),
        }

        return regime, confidence, detail

    # ------------------------------------------------------------------ #
    #  Internal: Transition Detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_transition(regime_history, current_regime):
        """Detect if we just transitioned from one regime to another.

        Parameters
        ----------
        regime_history : list[str]
            List of recent regime labels (most recent last).
        current_regime : str
            Current regime classification.

        Returns
        -------
        tuple[bool, str]
            Whether a transition occurred and a description.
        """
        if len(regime_history) < 2:
            return False, "Insufficient history"

        prev_regime = regime_history[-2]
        if current_regime != prev_regime:
            return True, f"{prev_regime} → {current_regime}"

        return False, f"Stable: {current_regime}"

    # ------------------------------------------------------------------ #
    #  Internal: Dynamic Weight Adjustments
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_dynamic_weights(regime, detail):
        """Compute dynamic weight adjustments for downstream agents.

        Parameters
        ----------
        regime : str
            Current regime.
        detail : dict
            Regime detail with scores.

        Returns
        -------
        dict
            Agent name → weight multiplier.
        """
        base = {
            "Trend": 1.0,
            "Momentum": 1.0,
            "Volume": 1.0,
            "Volatility": 1.0,
            "ML": 1.0,
            "DLForecast": 1.0,
            "SMC": 1.0,
            "Liquidity": 1.0,
            "SupplyDemand": 1.0,
            "Wyckoff": 1.0,
            "MarketStructure": 1.0,
        }

        if regime == "TRENDING_UP" or regime == "TRENDING_DOWN":
            # Boost trend-following agents
            base["Trend"] = 1.5
            base["Momentum"] = 1.3
            base["DLForecast"] = 1.2
            base["ML"] = 1.2
            base["Volatility"] = 0.8

        elif regime == "RANGING":
            # Boost mean-reversion / structural agents
            base["SMC"] = 1.5
            base["Liquidity"] = 1.5
            base["SupplyDemand"] = 1.3
            base["Wyckoff"] = 1.3
            base["MarketStructure"] = 1.2
            base["Trend"] = 0.7
            base["Momentum"] = 0.7

        elif regime == "VOLATILE":
            # Boost risk management, reduce trend confidence
            base["Volatility"] = 1.5
            base["Volume"] = 1.3
            base["Trend"] = 0.6
            base["Momentum"] = 0.7

        elif regime == "CALM":
            # Low vol = prepare for breakout
            base["Volatility"] = 1.3
            base["Trend"] = 1.0
            base["DLForecast"] = 1.1

        return base

    # ------------------------------------------------------------------ #
    #  Main Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        """Run market regime detection.

        Parameters
        ----------
        df : pandas.DataFrame or None
            OHLCV + indicator data.
        symbol : str, optional
            Trading pair symbol.
        timeframe : str, optional
            Candle timeframe.
        **kwargs : dict
            Extra parameters — may contain 'regime_history' (list[str])
            for transition detection.

        Returns
        -------
        AgentOutput
            Direction (NEUTRAL), confidence, score, evidence, reasoning,
            and data containing regime_info for downstream agents.
        """
        if df is None or df.empty or len(df) < 50:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["Insufficient data (need 50+ candles)"],
                reasoning="Regime: Insufficient data",
            )

        evidence: list = []

        try:
            # ── Classify regime ──
            regime, confidence, detail = self._score_regime(df)

            evidence.append(f"📊 Market Regime: **{regime}** "
                            f"(confidence: {confidence:.0f}%)")
            evidence.append(f"ADX: {detail['adx']:.1f} | "
                            f"+DI={detail['plus_di']:.1f} -DI={detail['minus_di']:.1f}")
            evidence.append(f"ATR percentile: {detail['atr_percentile']:.0f}% | "
                            f"BB width percentile: {detail['bb_width_percentile']:.0f}%")
            evidence.append(f"EMA direction: {detail['ema_direction']:+.3f} | "
                            f"Price slope: {detail['price_slope']:+.4f}")

            # ── Regime scores breakdown ──
            for r, s in sorted(detail["scores"].items(), key=lambda x: -x[1]):
                bar = "█" * int(s * 5) if s > 0 else ""
                marker = " ◀" if r == regime else ""
                evidence.append(f"  {r}: {s:.3f} {bar}{marker}")

            # ── Transition Detection ──
            regime_history = kwargs.get("regime_history", [])
            transitioned, transition_desc = self._detect_transition(
                regime_history, regime
            )

            if transitioned:
                confidence = min(100.0, confidence + 20.0)  # Boost confidence on transitions
                evidence.append(f"🔄 TRANSITION DETECTED: {transition_desc}")
                evidence.append("⚠️ Transitions are high-actionability signals!")
            else:
                evidence.append(f"Regime stable: {transition_desc}")

            # ── Dynamic Weight Adjustments ──
            dynamic_weights = self._compute_dynamic_weights(regime, detail)
            evidence.append("Dynamic weight adjustments:")
            for agent, mult in sorted(dynamic_weights.items(), key=lambda x: -x[1]):
                if mult != 1.0:
                    evidence.append(f"  {agent}: {mult:.1f}x")

            # ── Score (regime doesn't directly trade) ──
            # Score based on trend direction if trending
            score = 0.0
            if regime == "TRENDING_UP":
                score = 0.3
            elif regime == "TRENDING_DOWN":
                score = -0.3

            # Direction is always NEUTRAL for regime agent
            direction = "NEUTRAL"

            return self._out(
                direction=direction,
                confidence=confidence,
                score=score,
                evidence=evidence,
                data={
                    "regime": regime,
                    "transitioned": transitioned,
                    "transition": transition_desc if transitioned else None,
                    "regime_scores": detail["scores"],
                    "adx": detail["adx"],
                    "atr_percentile": detail["atr_percentile"],
                    "bb_width_percentile": detail["bb_width_percentile"],
                    "ema_direction": detail["ema_direction"],
                    "dynamic_weights": dynamic_weights,
                    "plus_di": detail["plus_di"],
                    "minus_di": detail["minus_di"],
                },
                reasoning=(
                    f"Regime: {regime} ({confidence:.0f}%) | "
                    f"ADX={detail['adx']:.0f} | ATR pct={detail['atr_percentile']:.0f}% | "
                    f"{'TRANSITION!' if transitioned else 'Stable'}"
                ),
            )

        except Exception as e:
            evidence.append(f"Regime error: {str(e)[:120]}")
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=evidence,
                reasoning=f"Regime failed: {str(e)[:80]}",
            )
