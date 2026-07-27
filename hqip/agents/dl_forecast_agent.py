"""
DL Forecast Agent
=================
Statistical forecast using linear regression + exponential smoothing.
Simplified DL substitute for environments without TensorFlow/PyTorch.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class DLForecastAgent(BaseAgent):
    name = "DLForecast"
    weight = 0.8

    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        if df is None or len(df) < 50:
            return self._out(direction="NEUTRAL", confidence=20, evidence=["Need 50+ candles"])

        evidence = []
        try:
            closes = df["close"].values[-50:]
            atr_val = df["atr"].iloc[-1]

            # Linear regression forecast
            x = np.arange(50)
            coeffs = np.polyfit(x, closes, 2)
            trend_line = np.polyval(coeffs, x)
            forecast_5 = np.polyval(coeffs, 55)  # 5 bars ahead
            forecast_10 = np.polyval(coeffs, 60)  # 10 bars ahead

            current = closes[-1]
            trend_slope = (forecast_5 - current) / current * 100
            evidence.append(f"Trend forecast: {trend_slope:+.2f}% (5 bars)")
            evidence.append(f"Current: {current:.2f} → Forecast: {forecast_5:.2f}")

            # Exponential smoothing
            alpha = 0.3
            es = closes[0]
            for v in closes:
                es = alpha * v + (1 - alpha) * es
            es_diff = (es - current) / current * 100
            evidence.append(f"Exp. Smoothing: {es_diff:+.2f}% from current")

            # R-squared of trend fit
            ss_res = np.sum((closes - trend_line) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)
            evidence.append(f"Trend fit R²: {r_squared:.3f}")

            # Bollinger bands from forecast
            std_20 = np.std(closes[-20:])
            upper_fc = forecast_5 + 2 * std_20
            lower_fc = forecast_5 - 2 * std_20
            evidence.append(f"Forecast range: [{lower_fc:.2f}, {upper_fc:.2f}]")

            # Score
            score = np.clip(trend_slope / 2, -1, 1)
            if r_squared < 0.3:
                evidence.append("⚠️ Weak trend fit - forecast unreliable")
                score *= 0.5

            direction = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "NEUTRAL"
            confidence = min(80, r_squared * 100)

        except Exception as e:
            evidence.append(f"Forecast error: {str(e)[:50]}")
            return self._out(direction="NEUTRAL", confidence=0, evidence=evidence)

        return self._out(
            direction=direction,
            confidence=confidence,
            score=score,
            evidence=evidence,
            data={"forecast_5": forecast_5, "forecast_10": forecast_10, "r_squared": r_squared},
            reasoning=f"DL Forecast: {'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'} trend | R²={r_squared:.2f}"
        )
