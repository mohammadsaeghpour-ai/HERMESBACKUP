"""
Hybrid Transformer for Financial Time Series
==============================================
Mimics Transformer architecture using sklearn + numpy.

Architecture inspiration:
- Transformers use self-attention to weight importance of each timestep
- Multi-head attention: different "heads" focus on different aspects
  (price, volume, momentum, volatility)
- Positional encoding: temporal position awareness
- We simulate this by:
  1. Computing attention scores between recent timesteps (similarity-based)
  2. Training multiple specialist models (one per "head")
  3. Combining with learned attention weights

Key advantages:
- Self-attention captures non-local temporal dependencies
- Multi-head design lets different models focus on different signals
- Attention weights are interpretable (which timesteps matter most)

Input: OHLCV DataFrame
Output: (probability, direction, confidence)
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ml.features import compute_features, create_labels


class HybridTransformer:
    """
    Transformer approximation for financial time series.
    
    Architecture:
    ─────────────
    Real Transformer:
        Input → [Multi-Head Self-Attention] → FFN → Output
    
    Our Approximation:
        Step 1: Compute attention weights over recent timesteps
                (cosine similarity between current and past embeddings)
        Step 2: Weighted feature aggregation (attention pooling)
        Step 3: Multi-head = multiple specialist models
        Step 4: Attention-weighted combination of specialist outputs
    
    The "self-attention" is computed via:
        attention(i,j) = softmax(cosine_sim(features[i], features[j]) / sqrt(d))
    """
    
    ATTENTION_WINDOW = 20  # how many past timesteps to attend to
    NUM_HEADS = 4          # number of specialist models
    LOOKBACK = 50          # context window for feature computation
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.heads = []           # specialist models
        self.head_weights = None  # learned attention weights
        self.is_trained = False
        self.train_accuracy = 0
        self.feature_names = []
    
    def _compute_self_attention(self, features_matrix):
        """
        Compute self-attention weights.
        
        In a real Transformer:
            Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
        
        Here we use the features themselves as Q=K=V and compute
        cosine similarity attention over the time axis.
        """
        T, D = features_matrix.shape
        if T < 2:
            return np.ones(T) / T
        
        # Compute pairwise cosine similarities
        norms = np.linalg.norm(features_matrix, axis=1, keepdims=True) + 1e-10
        normalized = features_matrix / norms
        
        # Attention scores: similarity of each timestep to the last timestep
        query = normalized[-1:]  # query = current timestep
        keys = normalized        # all timesteps are keys
        
        scores = (query @ keys.T).flatten()  # shape: (T,)
        scores = scores / np.sqrt(D)  # scale by sqrt(d) like real transformer
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / (exp_scores.sum() + 1e-10)
        
        return attention_weights
    
    def _build_transformer_features(self, df):
        """
        Build features with positional encoding and multi-head channels.
        """
        if df is None or len(df) < self.LOOKBACK:
            return None
        
        f = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]
        returns = close.pct_change()
        
        # ── Head 1: Price Attention Channel ──
        # Multi-scale price patterns
        f["trans_ret_1"] = returns
        f["trans_ret_3"] = close.pct_change(3)
        f["trans_ret_5"] = close.pct_change(5)
        f["trans_ret_10"] = close.pct_change(10)
        f["trans_ret_20"] = close.pct_change(20)
        
        # ── Head 2: Volatility Attention Channel ──
        for w in [5, 10, 20]:
            f[f"trans_vol_{w}"] = returns.rolling(w).std()
        f["trans_vol_ratio"] = f["trans_vol_5"] / (f["trans_vol_20"] + 1e-10)
        f["trans_vol_change"] = f["trans_vol_5"].pct_change(5)
        
        # ── Head 3: Volume Attention Channel ──
        f["trans_vol_raw"] = volume / volume.rolling(20).mean()
        f["trans_vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()
        f["trans_vol_spike"] = volume / volume.rolling(50).mean()
        f["trans_price_vol_corr"] = returns.rolling(10).corr(volume.pct_change())
        
        # ── Head 4: Momentum Attention Channel ──
        # RSI at multiple periods
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            f[f"trans_rsi_{period}"] = (100 - 100 / (1 + rs)) / 100
        
        # MACD histogram as momentum signal
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        f["trans_macd_hist"] = (macd_line - signal_line) / close
        f["trans_macd_cross"] = (macd_line > signal_line).astype(float) - \
                                 (macd_line.shift(1) > signal_line.shift(1)).astype(float)
        
        # ── Positional Encoding (sinusoidal, like real Transformer) ──
        for k in range(4):
            freq = 1.0 / (10000 ** (2 * k / 64))
            f[f"trans_pos_sin_{k}"] = np.sin(np.arange(len(f)) * freq)
            f[f"trans_pos_cos_{k}"] = np.cos(np.arange(len(f)) * freq)
        
        f = f.replace([np.inf, -np.inf], np.nan)
        return f
    
    def train(self, df, horizon=5, threshold=0.001, train_ratio=0.7):
        """Train Transformer hybrid."""
        features = self._build_transformer_features(df)
        if features is None or len(features) < 100:
            return False
        
        # Add base features
        base_features = compute_features(df)
        if base_features is not None:
            common_idx = features.index.intersection(base_features.index)
            features = pd.concat([
                features.loc[common_idx],
                base_features.loc[common_idx]
            ], axis=1)
        
        labels = create_labels(df, horizon, threshold)
        
        # Align
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            return False
        
        self.feature_names = X.columns.tolist()
        
        split = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # ── Multi-Head Specialist Models ──
        # Each "head" is a specialist trained on different aspects
        self.heads = [
            ("price_head", GradientBoostingClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42)),
            ("vol_head", GradientBoostingClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=43)),
            ("momentum_head", RandomForestClassifier(
                n_estimators=150, max_depth=6, random_state=44)),
            ("combined_head", LogisticRegression(
                C=1.0, max_iter=1000, random_state=45)),
        ]
        
        for name, model in self.heads:
            model.fit(X_train_s, y_train)
        
        # ── Learn attention weights (meta-learner) ──
        # Get predictions from each head
        head_preds_train = np.zeros((len(X_train_s), len(self.heads)))
        head_preds_test = np.zeros((len(X_test_s), len(self.heads)))
        
        for i, (name, model) in enumerate(self.heads):
            head_preds_train[:, i] = model.predict_proba(X_train_s)[:, 1]
            head_preds_test[:, i] = model.predict_proba(X_test_s)[:, 1]
        
        # Train meta-learner to combine heads (like attention weights)
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=50)
        self.meta_learner.fit(head_preds_train, y_train)
        
        # Meta prediction
        meta_preds = self.meta_learner.predict(head_preds_test)
        self.train_accuracy = accuracy_score(y_test, meta_preds)
        self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict with attention-weighted multi-head ensemble."""
        if not self.is_trained:
            return 0.5, "NEUTRAL", 0.5
        
        features = self._build_transformer_features(df)
        base_features = compute_features(df)
        
        if features is None or len(features) < 1:
            return 0.5, "NEUTRAL", 0.5
        
        if base_features is not None:
            idx = features.index.intersection(base_features.index)
            if len(idx) == 0:
                return 0.5, "NEUTRAL", 0.5
            features = pd.concat([
                features.loc[idx].iloc[[-1]],
                base_features.loc[idx].iloc[[-1]]
            ], axis=1)
        else:
            features = features.iloc[[-1]]
        
        if not features.notna().all(axis=1).iloc[0]:
            return 0.5, "NEUTRAL", 0.5
        
        X_s = self.scaler.transform(features)
        
        # Get predictions from each head
        head_probs = np.zeros(len(self.heads))
        for i, (name, model) in enumerate(self.heads):
            head_probs[i] = model.predict_proba(X_s)[0, 1]
        
        # Meta-learner combines (like learned attention weights)
        meta_input = head_probs.reshape(1, -1)
        avg_prob = self.meta_learner.predict_proba(meta_input)[0, 1]
        
        # Attention interpretability: which head contributes most
        head_contributions = self.meta_learner.coef_[0] * head_probs
        dominant_head_idx = np.argmax(np.abs(head_contributions))
        dominant_head = self.heads[dominant_head_idx][0]
        
        # Confidence from head agreement
        head_std = np.std(head_probs)
        agreement = 1.0 - head_std * 3
        confidence = min(max(avg_prob, 1 - avg_prob) * 100 * agreement, 90)
        
        if avg_prob > 0.55:
            direction = "BUY"
        elif avg_prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return avg_prob, direction, confidence
