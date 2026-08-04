"""
Hybrid CNN-LSTM for Crypto Trading
====================================
Mimics CNN-LSTM architecture using sklearn.

Architecture inspiration:
- CNN extracts spatial/local patterns from feature patches
- LSTM captures sequential dependencies across time
- CNN → LSTM pipeline: first detect patterns, then track their evolution

We simulate this by:
1. CNN layer: Sliding window feature extraction with pooling
   (compute statistics over rolling windows at multiple sizes)
2. LSTM layer: Sequential chain features that capture temporal dynamics
   (cumulative returns, exponential moving state, consecutive patterns)

Key advantages:
- CNN detects local candlestick patterns (multi-scale)
- LSTM tracks how patterns evolve over time
- Together they capture both what's happening and the context

Input: OHLCV DataFrame
Output: (probability, direction, confidence)
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ml.features import compute_features, create_labels


class HybridCNNLSTM:
    """
    CNN-LSTM approximation for financial time series.
    
    Architecture:
    ─────────────
    Real CNN-LSTM:
        Input → [Conv1D → Pool → Conv1D → Pool] → [LSTM → LSTM] → FC → Output
    
    Our Approximation:
        
        CNN Part (pattern detection):
            For each kernel size k in [3, 5, 7, 10]:
                conv_output = rolling_statistics(input, window=k)  
                # mean, std, max-min, trend direction
                pooled = max_pool(conv_output, pool_size=2)
                # downsample to capture dominant pattern
        
        LSTM Part (sequence modeling):
            State_0 = initial_state
            For t in timesteps:
                State_t = alpha * features[t] + (1-alpha) * State_{t-1}
                # alpha is learned (like LSTM gates)
            
            # Track: cumulative state, state velocity, state acceleration
        
        Combined: CNN features + LSTM features → GradientBoosting
    """
    
    # CNN kernel sizes (simulating different filter widths)
    KERNEL_SIZES = [3, 5, 7, 10]
    # LSTM-like exponential smoothing parameters
    LSTM_LOOKBACK = 30
    # Pool size for downsampling
    POOL_SIZE = 2
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.cnn_scaler = StandardScaler()
        self.lstm_scaler = StandardScaler()
        self.models = []
        self.lstm_alphas = []  # learned smoothing parameters
        self.is_trained = False
        self.train_accuracy = 0
    
    def _build_cnn_features(self, df):
        """
        CNN layer: multi-scale pattern detection via rolling statistics.
        
        Each kernel size acts as a convolution filter detecting patterns
        at that scale. Max-pooling downsamples to capture dominant patterns.
        """
        close = df["close"]
        volume = df["volume"]
        returns = close.pct_change()
        high = df["high"]
        low = df["low"]
        
        features = {}
        
        for k in self.KERNEL_SIZES:
            prefix = f"cnn_k{k}"
            
            # ── Convolution: rolling statistics over k-period window ──
            window_returns = returns.rolling(k, min_periods=max(k // 2, 1))
            
            features[f"{prefix}_mean"] = window_returns.mean()
            features[f"{prefix}_std"] = window_returns.std()
            features[f"{prefix}_max"] = window_returns.max()
            features[f"{prefix}_min"] = window_returns.min()
            features[f"{prefix}_range"] = features[f"{prefix}_max"] - features[f"{prefix}_min"]
            
            # Candlestick pattern detection at this scale
            features[f"{prefix}_bull_ratio"] = (returns > 0).rolling(k).sum() / k
            features[f"{prefix}_consec_up"] = self._consecutive_pattern(returns, k, up=True)
            features[f"{prefix}_consec_dn"] = self._consecutive_pattern(returns, k, up=False)
            
            # High-Low range pattern
            hl_range = (high - low) / close
            features[f"{prefix}_hl_mean"] = hl_range.rolling(k).mean()
            
            # Volume pattern
            features[f"{prefix}_vol_mean"] = volume.rolling(k).mean()
            features[f"{prefix}_vol_std"] = volume.rolling(k).std()
            
            # Max-pool: take the max of consecutive pooled values
            for feat_name in [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_range"]:
                pooled = pd.Series(features[feat_name], index=df.index)
                features[f"{feat_name}_pooled"] = pooled.rolling(
                    self.POOL_SIZE * 2, min_periods=1).max()
        
        return pd.DataFrame(features, index=df.index)
    
    def _consecutive_pattern(self, series, window, up=True):
        """Count longest consecutive run in a window."""
        result = pd.Series(0.0, index=series.index)
        arr = series.values
        for i in range(window, len(arr)):
            chunk = arr[i-window:i]
            if up:
                runs = self._longest_run(chunk > 0)
            else:
                runs = self._longest_run(chunk < 0)
            result.iloc[i] = runs / window
        return result
    
    @staticmethod
    def _longest_run(mask):
        """Find longest consecutive True run in boolean array."""
        best = 0
        current = 0
        for v in mask:
            if v:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best
    
    def _build_lstm_features(self, df):
        """
        LSTM layer: sequential state tracking via exponential smoothing.
        
        Simulates LSTM gates:
        - forget gate → controls how much past state to keep
        - input gate → controls how much new info to add
        - output gate → controls what to output
        
        We use exponential moving averages as a simplified LSTM state,
        with different alphas simulating different gate sensitivities.
        """
        close = df["close"]
        returns = close.pct_change()
        volume = df["volume"]
        
        features = {}
        alphas = [0.1, 0.2, 0.3, 0.5, 0.8]  # different gate sensitivities
        
        for i, alpha in enumerate(alphas):
            prefix = f"lstm_a{int(alpha*100)}"
            
            # ── State tracking (EMA = simplified LSTM hidden state) ──
            state = returns.ewm(alpha=alpha, min_periods=1).mean()
            features[f"{prefix}_state"] = state
            features[f"{prefix}_state_vel"] = state.diff()  # velocity
            features[f"{prefix}_state_accel"] = state.diff().diff()  # acceleration
            
            # ── Forget gate simulation ──
            # How much past info is retained?
            forget_gate = 1 - alpha  # higher alpha = more forgetting
            features[f"{prefix}_forget"] = forget_gate
            
            # ── Memory cell: cumulative returns ──
            cum_ret = (1 + returns.fillna(0)).rolling(20).apply(
                lambda x: np.prod(x) - 1, raw=True)
            features[f"{prefix}_memory"] = cum_ret
            
            # ── Output gate: filtered output ──
            features[f"{prefix}_output"] = state * forget_gate
            
            # ── Sequence patterns ──
            # Is the state trending up or down?
            features[f"{prefix}_trend"] = (state > state.shift(3)).astype(float)
            features[f"{prefix}_trend_strength"] = abs(state.rolling(5).mean())
            
            # Volume state
            vol_state = volume.ewm(alpha=alpha, min_periods=1).mean()
            features[f"{prefix}_vol_state"] = vol_state / vol_state.rolling(50).mean()
        
        return pd.DataFrame(features, index=df.index)
    
    def train(self, df, horizon=5, threshold=0.001, train_ratio=0.7):
        """Train CNN-LSTM hybrid."""
        cnn_features = self._build_cnn_features(df)
        lstm_features = self._build_lstm_features(df)
        
        if cnn_features is None or lstm_features is None:
            return False
        
        # Combine CNN + LSTM features
        common_idx = cnn_features.index.intersection(lstm_features.index)
        features = pd.concat([
            cnn_features.loc[common_idx],
            lstm_features.loc[common_idx]
        ], axis=1)
        
        # Add base features
        base_features = compute_features(df)
        if base_features is not None:
            common_idx2 = features.index.intersection(base_features.index)
            features = pd.concat([
                features.loc[common_idx2],
                base_features.loc[common_idx2]
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
        
        split = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # ── Ensemble: CNN specialist + LSTM specialist + Combined ──
        self.models = [
            ("cnn_gb", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42)),
            ("lstm_gb", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=43)),
            ("combined_rf", RandomForestClassifier(
                n_estimators=200, max_depth=7, max_features="sqrt",
                random_state=44)),
        ]
        
        for name, model in self.models:
            model.fit(X_train_s, y_train)
        
        preds = self._ensemble_predict(X_test_s)
        self.train_accuracy = accuracy_score(y_test, preds)
        self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict with CNN-LSTM ensemble."""
        if not self.is_trained:
            return 0.5, "NEUTRAL", 0.5
        
        cnn_features = self._build_cnn_features(df)
        lstm_features = self._build_lstm_features(df)
        
        if cnn_features is None or lstm_features is None:
            return 0.5, "NEUTRAL", 0.5
        
        common_idx = cnn_features.index.intersection(lstm_features.index)
        features = pd.concat([
            cnn_features.loc[common_idx],
            lstm_features.loc[common_idx]
        ], axis=1)
        
        base_features = compute_features(df)
        if base_features is not None:
            common_idx2 = features.index.intersection(base_features.index)
            features = pd.concat([
                features.loc[common_idx2].iloc[[-1]],
                base_features.loc[common_idx2].iloc[[-1]]
            ], axis=1)
        else:
            features = features.iloc[[-1]]
        
        if not features.notna().all(axis=1).iloc[0]:
            return 0.5, "NEUTRAL", 0.5
        
        X_s = self.scaler.transform(features)
        
        probs = []
        for name, model in self.models:
            p = model.predict_proba(X_s)[0, 1]
            probs.append(p)
        
        avg_prob = np.mean(probs)
        agreement = 1.0 - np.std(probs) * 4
        confidence = min(max(avg_prob, 1 - avg_prob) * 100 * agreement, 90)
        
        if avg_prob > 0.55:
            direction = "BUY"
        elif avg_prob < 0.45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return avg_prob, direction, confidence
    
    def _ensemble_predict(self, X):
        votes = []
        for name, model in self.models:
            votes.append(model.predict(X))
        return np.round(np.mean(votes, axis=0)).astype(int)
