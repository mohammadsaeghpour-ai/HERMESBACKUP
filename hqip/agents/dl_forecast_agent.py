"""
DL Forecast Agent — Advanced Statistical Forecasting (NumPy Only)
=================================================================
Implements three independent forecasting models entirely in NumPy:

1. **Holt-Winters Exponential Smoothing** — triple exponential smoothing with
   additive trend and seasonality components for short-horizon extrapolation.
2. **ARIMA-like Model** — auto-regressive coefficients estimated via OLS with
   a simple moving-average residual correction.
3. **Prophet-like Model** — piecewise-linear trend decomposition plus Fourier
   seasonality terms for periodic pattern capture.

Forecasts are combined via a weighted average whose weights adapt based on
each model's recent in-sample fit (RMSSE).  Confidence is derived from the
agreement (inverse standard deviation) of the three forecasts.

No torch or tensorflow — pure NumPy.

Weight: 1.0
"""
from hqip.agents.base import BaseAgent
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class DLForecastAgent(BaseAgent):
    """Triple-model statistical forecaster using pure NumPy.

    Combines Holt-Winters, ARIMA-like, and Prophet-like models with adaptive
    weighting to forecast 4–12 candles ahead.

    Attributes
    ----------
    name : str
        Agent identifier.
    weight : float
        Consensus weight (1.0).
    MIN_HISTORY : int
        Minimum candles required.
    MIN_SEASON : int
        Minimum seasonality period.
    """

    name = "DLForecast"
    weight = 1.0
    MIN_HISTORY = 60
    MIN_SEASON = 10

    # ------------------------------------------------------------------ #
    #  Holt-Winters Exponential Smoothing (additive)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _holt_winters(data, h=8, season_len=12, alpha=0.4, beta=0.1, gamma=0.1):
        """Fit additive Holt-Winters and forecast *h* steps ahead.

        Parameters
        ----------
        data : numpy.ndarray
            Univariate time series.
        h : int
            Forecast horizon.
        season_len : int
            Seasonal period.
        alpha, beta, gamma : float
            Smoothing parameters for level, trend, and season.

        Returns
        -------
        tuple[numpy.ndarray, float]
            Forecast array of length *h* and in-sample RMSSE.
        """
        n = len(data)
        if n < 2 * season_len:
            # Fallback to simple exponential smoothing
            level = data[0]
            levels = []
            for v in data:
                level = alpha * v + (1 - alpha) * level
                levels.append(level)
            levels = np.array(levels)
            trend = (levels[-1] - levels[-min(20, n)]) / min(20, n)
            fc = np.array([levels[-1] + trend * (i + 1) for i in range(h)])
            rmse = np.sqrt(np.mean((data - levels) ** 2))
            return fc, rmse

        # Initialize: level from first season, trend from first two seasons
        level = np.mean(data[:season_len])
        trend = (np.mean(data[season_len:2 * season_len]) -
                 np.mean(data[:season_len])) / season_len
        seasons = np.zeros(season_len)
        for i in range(season_len):
            seasons[i] = data[i] - level

        # Fit
        fitted = np.zeros(n)
        for t in range(n):
            s_idx = t % season_len
            prev_level = level
            prev_trend = trend
            level = alpha * (data[t] - seasons[s_idx]) + (1 - alpha) * (prev_level + prev_trend)
            trend = beta * (level - prev_level) + (1 - beta) * prev_trend
            seasons[s_idx] = gamma * (data[t] - level) + (1 - gamma) * seasons[s_idx]
            fitted[t] = level + trend + seasons[s_idx]

        # Forecast
        fc = np.zeros(h)
        for i in range(h):
            s_idx = (n + i) % season_len
            fc[i] = level + trend * (i + 1) + seasons[s_idx]

        # RMSSE
        rmse = np.sqrt(np.mean((data - fitted) ** 2) + 1e-10)
        return fc, rmse

    # ------------------------------------------------------------------ #
    #  ARIMA-like Model (AR(p) + MA residual)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _arima_forecast(data, h=8, ar_order=5):
        """Auto-regressive forecast with moving-average residual correction.

        Parameters
        ----------
        data : numpy.ndarray
            Univariate time series.
        h : int
            Forecast horizon.
        ar_order : int
            Number of AR lags.

        Returns
        -------
        tuple[numpy.ndarray, float]
            Forecast array and in-sample RMSE.
        """
        n = len(data)
        ar_order = min(ar_order, n // 4, 10)
        if ar_order < 2:
            mean_val = np.mean(data[-20:]) if n >= 20 else np.mean(data)
            trend = (data[-1] - data[-min(20, n)]) / min(20, n)
            fc = np.array([mean_val + trend * (i + 1) for i in range(h)])
            return fc, np.std(data) + 1e-10

        # Build AR design matrix via OLS
        X = np.column_stack([data[ar_order - i - 1: n - i - 1] for i in range(ar_order)])
        y = data[ar_order:]

        # OLS with ridge for stability
        ridge = 1e-4 * np.eye(ar_order)
        try:
            ar_coefs = np.linalg.solve(X.T @ X + ridge, X.T @ y)
        except np.linalg.LinAlgError:
            ar_coefs = np.ones(ar_order) / ar_order

        # Residuals → MA correction (simple average of last q residuals)
        fitted = X @ ar_coefs
        residuals = y - fitted
        ma_order = min(5, len(residuals))
        ma_coef = np.mean(residuals[-ma_order:]) if ma_order > 0 else 0.0

        # Forecast iteratively
        recent = list(data[-ar_order:])
        fc = np.zeros(h)
        for i in range(h):
            val = sum(ar_coefs[j] * recent[-(ar_order - j)] for j in range(ar_order))
            fc[i] = val + ma_coef * 0.5  # dampened MA correction
            recent.append(fc[i])

        # In-sample RMSE
        in_sample_fitted = np.zeros(len(y))
        for t in range(len(y)):
            in_sample_fitted[t] = sum(
                ar_coefs[j] * data[ar_order - j - 1 + t - len(y)] if (t - len(y) + ar_order - j - 1) >= -ar_order else data[ar_order - j - 1]
                for j in range(ar_order)
            )
        rmse = np.sqrt(np.mean((y - in_sample_fitted) ** 2) + 1e-10)
        return fc, rmse

    # ------------------------------------------------------------------ #
    #  Prophet-like Model (Trend + Fourier Seasonality)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prophet_forecast(data, h=8, n_fourier=3, period=12):
        """Piecewise-linear trend + Fourier seasonality forecast.

        Parameters
        ----------
        data : numpy.ndarray
            Univariate time series.
        h : int
            Forecast horizon.
        n_fourier : int
            Number of Fourier pairs.
        period : int
            Seasonal period.

        Returns
        -------
        tuple[numpy.ndarray, float]
            Forecast array and in-sample RMSE.
        """
        n = len(data)
        t = np.arange(n, dtype=float)

        # ── Trend: linear regression ──
        A = np.column_stack([np.ones(n), t])
        try:
            trend_coefs = np.linalg.lstsq(A, data, rcond=None)[0]
        except np.linalg.LinAlgError:
            trend_coefs = np.array([data[0], 0.0])

        trend_line = A @ trend_coefs

        # ── Seasonality via Fourier terms ──
        season = np.zeros(n)
        residuals = data - trend_line
        if n >= 2 * period:
            for k in range(1, n_fourier + 1):
                cos_feat = np.cos(2 * np.pi * k * t / period)
                sin_feat = np.sin(2 * np.pi * k * t / period)
                B = np.column_stack([cos_feat, sin_feat])
                try:
                    s_coefs = np.linalg.lstsq(B, residuals, rcond=None)[0]
                except np.linalg.LinAlgError:
                    s_coefs = np.zeros(2)
                season += B @ s_coefs

        # ── Forecast ──
        t_fc = np.arange(n, n + h, dtype=float)
        A_fc = np.column_stack([np.ones(h), t_fc])
        trend_fc = A_fc @ trend_coefs

        season_fc = np.zeros(h)
        if n >= 2 * period:
            for k in range(1, n_fourier + 1):
                cos_feat = np.cos(2 * np.pi * k * t_fc / period)
                sin_feat = np.sin(2 * np.pi * k * t_fc / period)
                B_fc = np.column_stack([cos_feat, sin_feat])
                # Re-use seasonal coefficients from fit
                try:
                    s_coefs = np.linalg.lstsq(
                        np.column_stack([
                            np.cos(2 * np.pi * k * t / period),
                            np.sin(2 * np.pi * k * t / period),
                        ]),
                        residuals,
                        rcond=None,
                    )[0]
                except np.linalg.LinAlgError:
                    s_coefs = np.zeros(2)
                season_fc += B_fc @ s_coefs

        fc = trend_fc + season_fc

        # RMSE
        fitted = trend_line + season
        rmse = np.sqrt(np.mean((data - fitted) ** 2) + 1e-10)
        return fc, rmse

    # ------------------------------------------------------------------ #
    #  Main Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        """Run the triple-model forecast.

        Parameters
        ----------
        df : pandas.DataFrame or None
            OHLCV data with 'close' column.
        symbol : str, optional
            Trading pair symbol.
        timeframe : str, optional
            Candle timeframe.
        **kwargs : dict
            Extra parameters (unused).

        Returns
        -------
        AgentOutput
            Direction, confidence, score, evidence, reasoning, and
            forecast data.
        """
        if df is None or len(df) < self.MIN_HISTORY:
            return self._out(
                direction="NEUTRAL", confidence=20,
                evidence=[f"Need {self.MIN_HISTORY}+ candles"],
                reasoning="DLForecast: Insufficient data",
            )

        evidence: list = []

        try:
            closes = df["close"].values.astype(float)
            current = closes[-1]
            atr_val = df["atr"].iloc[-1] if "atr" in df.columns else (
                np.mean(np.abs(np.diff(closes[-20:]))) + 1e-10
            )

            # Determine seasonality period from data
            season_len = max(self.MIN_SEASON, min(24, len(closes) // 5))
            h = min(12, max(4, len(closes) // 15))  # 4–12 candles ahead

            # ── Model 1: Holt-Winters ──
            fc_hw, rmse_hw = self._holt_winters(
                closes, h=h, season_len=season_len,
                alpha=0.4, beta=0.1, gamma=0.1,
            )
            evidence.append(f"Holt-Winters forecast [{h}]: {fc_hw[-1]:.2f} "
                            f"(RMSE={rmse_hw:.4f})")

            # ── Model 2: ARIMA-like ──
            fc_ar, rmse_ar = self._arima_forecast(closes, h=h, ar_order=5)
            evidence.append(f"ARIMA forecast [{h}]: {fc_ar[-1]:.2f} "
                            f"(RMSE={rmse_ar:.4f})")

            # ── Model 3: Prophet-like ──
            fc_pr, rmse_pr = self._prophet_forecast(
                closes, h=h, n_fourier=3, period=season_len,
            )
            evidence.append(f"Prophet forecast [{h}]: {fc_pr[-1]:.2f} "
                            f"(RMSE={rmse_pr:.4f})")

            # ── Adaptive weighting (inverse RMSE) ──
            inv_err = np.array([
                1.0 / (rmse_hw + 1e-10),
                1.0 / (rmse_ar + 1e-10),
                1.0 / (rmse_pr + 1e-10),
            ])
            weights = inv_err / inv_err.sum()
            evidence.append(
                f"Adaptive weights: HW={weights[0]:.2f} AR={weights[1]:.2f} "
                f"PR={weights[2]:.2f}"
            )

            # ── Combined forecast ──
            combined_fc = (weights[0] * fc_hw +
                           weights[1] * fc_ar +
                           weights[2] * fc_pr)

            # Multi-horizon forecast
            mid_horizon = h // 2
            forecast_now = combined_fc[0]      # next candle
            forecast_mid = combined_fc[mid_horizon]
            forecast_end = combined_fc[-1]

            # ── Trend detection ──
            slope_pct = (combined_fc[-1] - current) / current * 100.0
            magnitude_pct = abs(slope_pct)

            # ── Forecast agreement (confidence) ──
            fc_range = np.array([fc_hw[-1], fc_ar[-1], fc_pr[-1]])
            fc_std = np.std(fc_range)
            fc_mean = np.mean(fc_range)

            # Agreement: low std relative to price → high confidence
            cv = fc_std / (abs(fc_mean) + 1e-10)
            agreement = max(0.0, min(1.0, 1.0 - cv * 50))
            confidence = agreement * 100.0

            # Direction agreement
            directions = [np.sign(f - current) for f in fc_range]
            dir_agreement = abs(sum(directions)) / len(directions)

            evidence.append(f"Combined forecast [{h}]: "
                            f"{current:.2f} → {forecast_end:.2f} "
                            f"({slope_pct:+.2f}%)")
            evidence.append(f"Forecast range: [{fc_range.min():.2f}, "
                            f"{fc_range.max():.2f}]")
            evidence.append(f"Model agreement: {dir_agreement:.0%} "
                            f"(CV={cv:.4f})")
            evidence.append(f"Horizon: {h} candles | "
                            f"Season period: {season_len}")

            # ── Score ──
            score = np.clip(slope_pct / 2.0, -1.0, 1.0) * dir_agreement
            if cv > 0.02:
                score *= 0.5  # Penalize when models disagree
                evidence.append("⚠️ Models disagree — reduced confidence")

            # ── Direction ──
            if score > 0.15:
                direction = "BUY"
            elif score < -0.15:
                direction = "SELL"
            else:
                direction = "NEUTRAL"

            confidence = float(np.clip(confidence, 0, 80))
            score = float(np.clip(score, -1.0, 1.0))

            # ── R² of linear fit ──
            x_full = np.arange(len(closes))
            coeffs = np.polyfit(x_full[-60:], closes[-60:], 2)
            trend_line_vals = np.polyval(coeffs, x_full[-60:])
            ss_res = np.sum((closes[-60:] - trend_line_vals) ** 2)
            ss_tot = np.sum((closes[-60:] - np.mean(closes[-60:])) ** 2)
            r_squared = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))

            return self._out(
                direction=direction,
                confidence=confidence,
                score=score,
                evidence=evidence,
                data={
                    "forecast_horizon": h,
                    "forecast_next": round(float(forecast_now), 2),
                    "forecast_mid": round(float(forecast_mid), 2),
                    "forecast_end": round(float(forecast_end), 2),
                    "slope_pct": round(float(slope_pct), 4),
                    "model_agreement": round(float(dir_agreement), 3),
                    "r_squared": round(float(r_squared), 4),
                    "weights": {
                        "holt_winters": round(float(weights[0]), 3),
                        "arima": round(float(weights[1]), 3),
                        "prophet": round(float(weights[2]), 3),
                    },
                },
                reasoning=(
                    f"DLForecast: 3-model ensemble → {direction} "
                    f"({confidence:.0f}%) | {h}-step horizon | "
                    f"Slope={slope_pct:+.2f}% | Agreement={dir_agreement:.0%}"
                ),
            )

        except Exception as e:
            evidence.append(f"Forecast error: {str(e)[:120]}")
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=evidence,
                reasoning=f"DLForecast failed: {str(e)[:80]}",
            )
