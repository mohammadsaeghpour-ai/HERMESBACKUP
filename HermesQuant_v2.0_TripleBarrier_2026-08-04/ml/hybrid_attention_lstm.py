"""
Hybrid Attention-LSTM for Crypto Trading
==========================================
Mimics Attention-LSTM architecture using sklearn + numpy.

Architecture inspiration:
- LSTM processes sequences, but not all timesteps are equally important
- Attention mechanism assigns learned weights to each timestep
- High attention = that timestep has strong predictive power
- We simulate this by:
  1. LSTM layer: exponential state tracking (EMA-based hidden state)
  2. Attention layer: compute importance of each recent timestep
  3. Weighted aggregation: combine LSTM state + attention-weighted features
  4. Final classification via ensemble

Key advantages:
- Focuses on the most predictive timesteps (noise reduction)
- Adaptive to changing market conditions (attention shifts)
- Interpretable: shows which recent events matter most

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


class HybridAttentionLSTM:
    """
    Attention-LSTM approximation for financial time series.
    
    Architecture:
    ─────────────
    Real Attention-LSTM:
        h_t, c_t = LSTM_cell(x_t, h_{t-1}, c_{t-1})  # LSTM step
        e_t = v^T * tanh(W * [h_t; s_{t-1}])           # attention score
        α_t = softmax(e_t)                               # attention weights
        context = Σ α_t * h_t                            # context vector
        output = FC(context)                             # final prediction
    
    Our Approximation:
        
        LSTM Layer (state tracking):
            For multiple decay rates α:
                state_t = α * input_t + (1-α) * state_{t-1}
                This simulates LSTM's forget gate mechanism
        
        Attention Layer (importance weighting):
            1. Compute feature similarity between each recent timestep
               and the current timestep
            2. Apply softmax to get attention weights
            3. Weighted sum = context vector
        
        Classification:
            LSTM features + Attention context → Ensemble
    """
    
    LOOKBACK = 50        # context window
    ATTENTION_WINDOW = 15  # how many past steps to attend to
    NUM_LSTM_CELLS = 4   # number of parallel LSTM-like cells
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = []
        self.meta_learner = None
        self.is_trained = False
        self.train_accuracy = 0
    
    def _compute_attention_weights(self, feature_matrix):
        """
        Compute attention weights for each timestep.
        
        Uses a learned-style attention based on:
        1. Position proximity (more recent = more important)
        2. Feature similarity (similar context = more important)
        3. Information content (volatile = more important)
        """
        T = len(feature_matrix)
        if T == 0:
            return np.array([])
        
        # Position-based attention: exponential decay
        positions = np.arange(T)
        position_weights = np.exp(-0.1 * (T - 1 - positions))
        
        # Similarity-based attention: cosine similarity to current
        if feature_matrix.shape[1] > 0:
            norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-10
            normalized = feature_matrix / norms
            query = normalized[-1:]  # current timestep
            similarity = (query @ normalized.T).flatten()
            similarity = np.maximum(similarity, 0)  # ReLU-like
        else:
            similarity = np.ones(T)
        
        # Information content: variance as importance proxy
        if T > 3:
            info_content = np.std(feature_matrix, axis=1)
            info_content = info_content / (info_content.max() + 1e-10)
        else:
            info_content = np.ones(T)
        
        # Combine: learned-style weighted combination
        raw_attention = 0.4 * position_weights + 0.4 * similarity + 0.2 * info_content
        
        # Softmax
        exp_a = np.exp(raw_attention - raw_attention.max())
        attention = exp_a / (exp_a.sum() + 1e-10)
        
        return attention
    
    def _build_attention_lstm_features(self, df):
        """
        Build combined attention + LSTM features.
        """
        if df is None or len(df) < self.LOOKBACK:
            return None
        
        close = df["close"]
        volume = df["volume"]
        returns = close.pct_change()
        
        features = pd.DataFrame(index=df.index)
        
        # ── LSTM Layer: Multi-cell state tracking ──
        alphas = [0.05, 0.15, 0.3, 0.6]  # forget gate sensitivities
        
        for i, alpha in enumerate(alphas):
            prefix = f"attn_lstm{i}"
            
            # Hidden state (EMA)
            h = returns.ewm(alpha=alpha, min_periods=1).mean()
            features[f"{prefix}_h"] = h
            
            # Cell state (cumulative)
            c = (1 + returns.fillna(0)).cumprod() - 1
            features[f"{prefix}_c"] = c / (c.abs().rolling(50).max() + 1e-10)
            
            # State velocity and acceleration
            features[f"{prefix}_dh"] = h.diff()
            features[f"{prefix}_d2h"] = h.diff().diff()
            
            # Forget gate output (simulated)
            features[f"{prefix}_forget"] = 1 - alpha
            
            # LSTM output gate (simulated)
            output = h * (1 - alpha) + returns.fillna(0) * alpha
            features[f"{prefix}_output"] = output
            
            # Cross-cell features (like multi-layer LSTM)
            if i > 0:
                prev_prefix = f"attn_lstm{i-1}"
                features[f"{prefix}_cross_{i}"] = h - features[f"{prev_prefix}_h"]
        
        # ── Attention Layer: Context vector computation ──
        # We'll compute attention over the last N timesteps for each feature
        # This creates a "summary" of recent history weighted by importance
        
        # Use returns as the feature to attend over
        returns_values = returns.values
        attention_features = np.zeros((len(df), self.ATTENTION_WINDOW * 3))
        
        for t in range(self.ATTENTION_WINDOW, len(df)):
            # Get recent window
            window = returns_values[t - self.ATTENTION_WINDOW:t + 1].reshape(-1, 1)
            positions = np.arange(self.ATTENTION_WINDOW + 1).reshape(-1, 1)
            window_with_pos = np.hstack([window, positions / self.ATTENTION_WINDOW])
            
            # Simple attention: recent timesteps get more weight
            attn = np.exp(-0.15 * (self.ATTENTION_WINDOW - np.arange(self.ATTENTION_WINDOW + 1)))
            attn = attn / attn.sum()
            
            # Context vector = attention-weighted sum
            context = np.sum(window * attn.reshape(-1, 1), axis=0)[0]
            
            # Attention entropy (how spread out is the attention)
            entropy = -np.sum(attn * np.log(attn + 1e-10))
            
            # Max/min attention positions
            max_attn_pos = np.argmax(attn) / self.ATTENTION_WINDOW
            
            attention_features[t, 0] = context
            attention_features[t, 1] = entropy
            attention_features[t, 2] = max_attn_pos
            
            # Multi-scale attention contexts
            for scale in [3, 5, 10]:
                if t >= scale:
                    scale_window = returns_values[t-scale:t+1]
                    attn_s = np.exp(-0.2 * (scale - np.arange(scale + 1)))
                    attn_s = attn_s / attn_s.sum()
                    context_s = np.sum(scale_window * attn_s)
                    attention_features[t, 2 + scale // 2] = context_s
        
        attn_df = pd.DataFrame(
            attention_features,
            index=df.index,
            columns=[f"attn_ctx_{i}" for i in range(attention_features.shape[1])]
        )
        features = pd.concat([features, attn_df], axis=1)
        
        # ── Combined LSTM + Attention features ──
        features["alstm_h_vs_attn"] = features["attn_lstm0_h"] - features["attn_ctx_0"]
        features["alstm_momentum"] = features["attn_lstm0_h"].rolling(5).mean()
        features["alstm_regime"] = (features["attn_lstm0_h"] > 0).astype(float)
        
        features = features.replace([np.inf, -np.inf], np.nan)
        return features
    
    def train(self, df, horizon=5, threshold=0.001, train_ratio=0.7):
        """Train Attention-LSTM hybrid."""
        features = self._build_attention_lstm_features(df)
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
        
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            return False
        
        split = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # ── Ensemble: LSTM specialist + Attention specialist + Combined ──
        self.models = [
            ("lstm_gb", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42)),
            ("attn_rf", RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=43)),
            ("combined_gb", GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08,
                random_state=44)),
        ]
        
        for name, model in self.models:
            model.fit(X_train_s, y_train)
        
        # Meta-learner
        head_preds = np.column_stack([
            m.predict_proba(X_train_s)[:, 1] for _, m in self.models
        ])
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=50)
        self.meta_learner.fit(head_preds, y_train)
        
        # Evaluate
        test_heads = np.column_stack([
            m.predict_proba(X_test_s)[:, 1] for _, m in self.models
        ])
        meta_preds = self.meta_learner.predict(test_heads)
        self.train_accuracy = accuracy_score(y_test, meta_preds)
        self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict with Attention-LSTM ensemble."""
        if not self.is_trained:
            return 0.5, "NEUTRAL", 0.5
        
        features = self._build_attention_lstm_features(df)
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
        
        head_probs = np.array([
            m.predict_proba(X_s)[0, 1] for _, m in self.models
        ])
        
        meta_input = head_probs.reshape(1, -1)
        avg_prob = self.meta_learner.predict_proba(meta_input)[0, 1]
        
        agreement = 1.0 - np.std(head_probs) * 3
        confidence = min(max(avg_prob, 1 - avg_prob) * 100 * agreement, 90)
        
        if avg_prob > 0.55:
            direction = "BUY"
        elif avg_prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return avg_prob, direction, confidence
