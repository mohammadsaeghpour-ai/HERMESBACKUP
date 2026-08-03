"""
ML Ensemble Agent — State-of-the-Art Binary Direction Classifier
================================================================
Walk-forward ensemble of RandomForestClassifier + GradientBoostingClassifier
with soft voting, StandardScaler, and 19 engineered features.

Design:
  - Binary classification: next candle UP (1) or DOWN (0)
  - Soft voting averages calibrated probabilities from both models
  - Walk-forward: retrain on last TRAIN_WINDOW candles each prediction
  - NaN/inf resilient: forward-fill, clip, drop irrecoverable rows
  - Never crashes — returns neutral on any error
  - Weight: 1.2
"""
from hqip.agents.base import BaseAgent
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class MLAgent(BaseAgent):
    """Walk-forward ML ensemble agent for binary candle direction prediction.

    Combines RandomForest (100 trees) and GradientBoosting (50 trees) via
    soft-voting to predict whether the next candle close will be above or
    below the current close.  Recomputes every call to stay adaptive.

    Attributes
    ----------
    name : str
        Agent identifier.
    weight : float
        Ensemble weight in the consensus system (1.2).
    FEATURE_COLS : list[str]
        Ordered feature names used by the classifier.
    TRAIN_WINDOW : int
        Maximum lookback for training data.
    """

    name = "ML"
    weight = 1.2

    FEATURE_COLS = [
        "rsi", "macd_hist", "adx", "bb_pct", "bb_width", "atr_pct",
        "vol_ratio", "vwap_dist", "stoch_k", "ema20_dist", "ema50_dist",
        "obv_trend", "roc", "cci", "willr", "mfi",
        "ema9_21_cross", "ema20_50_cross", "price_position_in_range",
    ]

    TRAIN_WINDOW = 200
    MIN_TRAIN_SAMPLES = 50
    BUY_THRESHOLD = 0.60
    SELL_THRESHOLD = 0.60

    # ------------------------------------------------------------------ #
    #  Feature Engineering
    # ------------------------------------------------------------------ #
    def _compute_derived_features(self, df):
        """Compute derived features from raw indicator columns.

        Parameters
        ----------
        df : pandas.DataFrame
            OHLCV + indicator data.

        Returns
        -------
        pandas.DataFrame
            Copy of *df* with derived feature columns added.
        """
        df = df.copy()
        close = df["close"]

        # ── EMA distances (% from price to EMA) ──
        for period in (9, 20, 50):
            col = f"ema{period}"
            if col in df.columns:
                ema_safe = df[col].replace(0, np.nan)
                df[f"ema{period}_dist"] = ((close / ema_safe) - 1.0) * 100.0
            else:
                df[f"ema{period}_dist"] = 0.0

        # ── EMA crossover signals ──
        if "ema9" in df.columns and "ema21" in df.columns:
            prev_diff = (df["ema9"] - df["ema21"]).shift(1)
            curr_diff = df["ema9"] - df["ema21"]
            df["ema9_21_cross"] = np.where(
                (prev_diff <= 0) & (curr_diff > 0), 1.0,
                np.where((prev_diff >= 0) & (curr_diff < 0), -1.0, 0.0),
            )
        else:
            df["ema9_21_cross"] = 0.0

        if "ema20" in df.columns and "ema50" in df.columns:
            prev_diff2 = (df["ema20"] - df["ema50"]).shift(1)
            curr_diff2 = df["ema20"] - df["ema50"]
            df["ema20_50_cross"] = np.where(
                (prev_diff2 <= 0) & (curr_diff2 > 0), 1.0,
                np.where((prev_diff2 >= 0) & (curr_diff2 < 0), -1.0, 0.0),
            )
        else:
            df["ema20_50_cross"] = 0.0

        # ── OBV trend (above EMA = bullish, else bearish) ──
        if "obv" in df.columns and "obv_ema" in df.columns:
            df["obv_trend"] = np.where(df["obv"] > df["obv_ema"], 1.0,
                              np.where(df["obv"] < df["obv_ema"], -1.0, 0.0))
        else:
            df["obv_trend"] = 0.0

        # ── Price position in rolling range (0 = at low, 1 = at high) ──
        rng_high = df["high"].rolling(20).max()
        rng_low = df["low"].rolling(20).min()
        rng_spread = (rng_high - rng_low).replace(0, np.nan)
        df["price_position_in_range"] = ((close - rng_low) / rng_spread).clip(0, 1)
        df["price_position_in_range"] = df["price_position_in_range"].fillna(0.5)

        # ── CCI (Commodity Channel Index) ──
        tp = (df["high"] + df["low"] + close) / 3.0
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        df["cci"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # ── Williams %R (14-period) ──
        high_14 = df["high"].rolling(14).max()
        low_14 = df["low"].rolling(14).min()
        hl_range = (high_14 - low_14).replace(0, np.nan)
        df["willr"] = ((high_14 - close) / hl_range) * -100.0

        # ── MFI (Money Flow Index) ──
        tp_val = (df["high"] + df["low"] + close) / 3.0
        mf = tp_val * df["volume"]
        pos_mf = mf.where(tp_val > tp_val.shift(1), 0.0).rolling(14).sum()
        neg_mf = mf.where(tp_val < tp_val.shift(1), 0.0).rolling(14).sum()
        mfi_ratio = pos_mf / neg_mf.replace(0, np.nan)
        df["mfi"] = 100.0 - (100.0 / (1.0 + mfi_ratio))

        return df

    def _prepare_features(self, df):
        """Build the feature matrix with NaN/inf handling.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw dataframe with OHLCV + indicators.

        Returns
        -------
        tuple[pandas.DataFrame | None, list[str]]
            Cleaned feature DataFrame and list of available columns,
            or (None, []) if insufficient data.
        """
        df = self._compute_derived_features(df)

        available = [c for c in self.FEATURE_COLS if c in df.columns]
        if len(available) < 8:
            return None, []

        feat = df[available].copy()

        # Replace inf with NaN, then forward-fill, then backward-fill
        feat = feat.replace([np.inf, -np.inf], np.nan)
        feat = feat.ffill().bfill()

        # Clip extreme values for robustness
        feat = feat.clip(-1000, 1000)

        # Drop any remaining NaN rows
        mask = feat.notna().all(axis=1)
        feat = feat.loc[mask]

        return feat, available

    def _build_labels(self, df, feature_index):
        """Create binary labels: 1 if next close > current close, else 0.

        Parameters
        ----------
        df : pandas.DataFrame
            Original dataframe (must contain 'close').
        feature_index : pandas.Index
            Index of the cleaned feature rows.

        Returns
        -------
        pandas.Series
            Aligned binary labels (0/1).
        """
        close = df["close"].reindex(feature_index)
        future_close = close.shift(-1)
        labels = (future_close > close).astype(int)
        return labels.dropna()

    # ------------------------------------------------------------------ #
    #  Main Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        """Run the ML ensemble prediction.

        Parameters
        ----------
        df : pandas.DataFrame or None
            OHLCV data with indicator columns.  Required.
        symbol : str, optional
            Trading pair symbol.
        timeframe : str, optional
            Candle timeframe.
        **kwargs : dict
            Extra parameters (unused).

        Returns
        -------
        AgentOutput
            Direction (BUY/SELL/NEUTRAL), confidence, score, evidence,
            reasoning, and metadata.
        """
        # ── Pre-flight ──
        if df is None or df.empty:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["No data provided"],
                reasoning="ML: No data available",
            )

        if len(df) < self.TRAIN_WINDOW:
            return self._out(
                direction="NEUTRAL", confidence=10,
                evidence=[f"Need {self.TRAIN_WINDOW}+ candles (have {len(df)})"],
                reasoning="ML: Insufficient training data",
            )

        evidence: list = []

        # ── Check sklearn availability ──
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.ensemble import VotingClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=["sklearn not installed — ML agent disabled"],
                reasoning="ML: Missing dependency",
            )

        try:
            # ── Prepare features ──
            feat_df, available_cols = self._prepare_features(df)
            if feat_df is None or len(feat_df) < self.MIN_TRAIN_SAMPLES:
                return self._out(
                    direction="NEUTRAL", confidence=10,
                    evidence=["Not enough clean features after NaN removal"],
                    reasoning="ML: Feature preparation failed",
                )

            # ── Build labels ──
            labels = self._build_labels(df, feat_df.index)
            if len(labels) < 30:
                return self._out(
                    direction="NEUTRAL", confidence=10,
                    evidence=[f"Only {len(labels)} valid labels (need 30+)"],
                    reasoning="ML: Insufficient labels",
                )

            # Align
            common_idx = feat_df.index.intersection(labels.index)
            X = feat_df.loc[common_idx, available_cols].values
            y = labels.loc[common_idx].values

            if len(X) < self.MIN_TRAIN_SAMPLES:
                return self._out(
                    direction="NEUTRAL", confidence=10,
                    evidence=["Not enough aligned data points"],
                    reasoning="ML: Data alignment issue",
                )

            # ── Walk-forward: use last TRAIN_WINDOW samples ──
            train_size = min(self.TRAIN_WINDOW, len(X))
            X_train = X[-train_size:]
            y_train = y[-train_size:]
            X_predict = X[-1:]

            # ── Class balance check ──
            n_up = int(y_train.sum())
            n_dn = len(y_train) - n_up
            if n_up < 5 or n_dn < 5:
                return self._out(
                    direction="NEUTRAL", confidence=10,
                    evidence=[f"Imbalanced training: {n_up} UP / {n_dn} DOWN"],
                    reasoning="ML: Severe class imbalance",
                )

            # ── Scale features ──
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_pred_s = scaler.transform(X_predict)

            # ── Build soft-voting ensemble ──
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            gb = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42,
            )
            ensemble = VotingClassifier(
                estimators=[("rf", rf), ("gb", gb)],
                voting="soft",
                weights=[1, 1],
            )

            ensemble.fit(X_train_s, y_train)
            proba = ensemble.predict_proba(X_pred_s[-1:])[0]

            # Map class probabilities
            classes = ensemble.classes_
            buy_prob = 0.0
            for i, c in enumerate(classes):
                if c == 1:
                    buy_prob = proba[i]
                    break
            sell_prob = 1.0 - buy_prob

            # ── Training accuracy ──
            train_acc = ensemble.score(X_train_s, y_train)

            # ── Feature importances (from RF component) ──
            rf_model = ensemble.named_estimators_["rf"]
            importances = rf_model.feature_importances_
            top_features = sorted(
                zip(available_cols, importances), key=lambda x: -x[1]
            )[:5]

            evidence.append(
                f"Ensemble Probabilities: P(UP)={buy_prob:.1%} | P(DN)={sell_prob:.1%}"
            )
            evidence.append(f"Training accuracy: {train_acc:.1%}")
            evidence.append(f"Training window: {train_size} candles ({n_up} up / {n_dn} down)")
            evidence.append("Top features:")
            for feat, imp in top_features:
                evidence.append(f"  {feat}: {imp:.4f}")

            # ── Direction decision ──
            if buy_prob > self.BUY_THRESHOLD:
                direction = "BUY"
                confidence = min(100.0, buy_prob * 100.0)
                score = (buy_prob - 0.5) * 2.0
            elif sell_prob > self.SELL_THRESHOLD:
                direction = "SELL"
                confidence = min(100.0, sell_prob * 100.0)
                score = -(sell_prob - 0.5) * 2.0
            else:
                direction = "NEUTRAL"
                confidence = max(buy_prob, sell_prob) * 100.0
                score = (buy_prob - sell_prob) * 0.5

            confidence = float(np.clip(confidence, 0.0, 100.0))
            score = float(np.clip(score, -1.0, 1.0))

            return self._out(
                direction=direction,
                confidence=confidence,
                score=score,
                evidence=evidence,
                data={
                    "buy_prob": round(buy_prob, 4),
                    "sell_prob": round(sell_prob, 4),
                    "train_acc": round(train_acc, 4),
                    "train_size": train_size,
                    "features_used": available_cols,
                    "top_features": [(f, round(float(v), 4)) for f, v in top_features],
                },
                reasoning=(
                    f"ML Ensemble: P(UP)={buy_prob:.1%} P(DN)={sell_prob:.1%} "
                    f"→ {direction} ({confidence:.0f}%) | "
                    f"Accuracy={train_acc:.1%} over {train_size} candles"
                ),
            )

        except Exception as e:
            evidence.append(f"ML Error: {str(e)[:120]}")
            return self._out(
                direction="NEUTRAL", confidence=0,
                evidence=evidence,
                reasoning=f"ML failed: {str(e)[:80]}",
            )
