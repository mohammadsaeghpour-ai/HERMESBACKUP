"""
ML Agent
========
Simple RandomForest ensemble for probability estimation.
"""
from hqip.agents.base import BaseAgent
import numpy as np

class MLAgent(BaseAgent):
    name = "ML"
    weight = 1.0

    FEATURE_COLS = [
        "rsi", "macd_hist", "adx", "bb_pct", "bb_width", "atr_pct",
        "vol_ratio", "vwap_dist", "stoch_k", "roc", "body_pct",
        "price_vs_ema20", "price_vs_ema50", "plus_di", "minus_di",
        "bullish_candle", "obv_trend", "st_dir", "momentum"
    ]

    def _prepare_features(self, df):
        df = df.copy()
        df["price_vs_ema20"] = (df["close"] / df["ema20"] - 1) * 100
        df["price_vs_ema50"] = (df["close"] / df["ema50"] - 1) * 100
        df["obv_trend"] = (df["obv"] > df["obv_ema"]).astype(float)
        df["momentum"] = df["close"] - df["close"].shift(10)
        df["bullish_candle"] = df["close"] > df["open"]
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        return df[available].dropna()

    def analyze(self, df=None, symbol="", timeframe="", **kwargs):
        if df is None or len(df) < 100:
            return self._out(direction="NEUTRAL", confidence=20, evidence=["Need 100+ candles for ML"], reasoning="Insufficient training data")

        evidence = []
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            features = self._prepare_features(df)
            if len(features) < 50:
                return self._out(direction="NEUTRAL", confidence=20, evidence=["Not enough clean features"])

            available_cols = [c for c in self.FEATURE_COLS if c in features.columns]
            X = features[available_cols].values

            # Create target: next candle direction
            close_vals = df["close"].reindex(features.index).values
            atr_vals = df["atr"].reindex(features.index).values
            target = np.zeros(len(close_vals))
            for i in range(5, len(close_vals)):
                if close_vals[i] > close_vals[i-5] + 0.3 * atr_vals[i-5]:
                    target[i] = 1  # BUY
                elif close_vals[i] < close_vals[i-5] - 0.3 * atr_vals[i-5]:
                    target[i] = 0  # SELL
                else:
                    target[i] = 0.5  # NEUTRAL

            # Use last 80% for train, predict last
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train = target[:split]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X_train_s, y_train)

            proba = model.predict_proba(X_test_s[-1:])
            classes = model.classes_

            buy_prob = 0
            sell_prob = 0
            for i, c in enumerate(classes):
                if c == 1:
                    buy_prob = proba[0][i]
                elif c == 0:
                    sell_prob = proba[0][i]

            evidence.append(f"ML Probabilities: BUY={buy_prob:.1%} | SELL={sell_prob:.1%}")
            evidence.append(f"Training accuracy: {model.score(X_train_s, y_train):.1%}")

            # Feature importance
            importances = model.feature_importances_
            top_features = sorted(zip(available_cols, importances), key=lambda x: -x[1])[:5]
            for feat, imp in top_features:
                evidence.append(f"  {feat}: {imp:.3f}")

            max_prob = max(buy_prob, sell_prob)
            if buy_prob > 0.6 and buy_prob > sell_prob * 1.5:
                direction = "BUY"
                score = (buy_prob - 0.5) * 2
            elif sell_prob > 0.6 and sell_prob > buy_prob * 1.5:
                direction = "SELL"
                score = -(sell_prob - 0.5) * 2
            else:
                direction = "NEUTRAL"
                score = (buy_prob - sell_prob) * 0.5

            confidence = max_prob * 100

        except Exception as e:
            evidence.append(f"ML Error: {str(e)[:50]}")
            return self._out(direction="NEUTRAL", confidence=0, evidence=evidence, reasoning="ML failed")

        return self._out(
            direction=direction,
            confidence=confidence,
            score=np.clip(score, -1, 1),
            evidence=evidence,
            data={"buy_prob": buy_prob, "sell_prob": sell_prob},
            reasoning=f"ML: BUY={buy_prob:.0%} SELL={sell_prob:.0%} → {direction}"
        )
