"""
Hybrid GAN (Generative Adversarial Network) for Market Simulation
===================================================================
Mimics GAN architecture using sklearn for synthetic data augmentation.

Architecture inspiration:
- Real GAN: Generator creates synthetic samples, Discriminator tells real from fake
- We use this for data augmentation:
  1. "Generator": Train a statistical model that captures the distribution
     of market features (mean, covariance, correlations)
  2. "Discriminator": Use IsolationForest to detect and filter unrealistic
     synthetic samples (keeps only realistic ones)
  3. Combined: Augmented training data improves generalization

Key advantages:
- Addresses class imbalance (crypto data is often 60/40 or worse)
- Synthetic rare events (crashes, pumps) improve model robustness
- Prevents overfitting by expanding training distribution
- "Regime-aware" generation preserves market regime statistics

Input: OHLCV DataFrame
Output: (probability, direction, confidence)
"""
import sys
sys.path.insert(0, "/data/workspace/HermesQuant")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    IsolationForest
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

from ml.features import compute_features, create_labels


class HybridGAN:
    """
    GAN approximation for market data augmentation.
    
    Architecture:
    ─────────────
    Real GAN:
        Generator(z) → fake_data
        Discriminator(real_data) → 1, Discriminator(fake_data) → 0
        Train adversarially
    
    Our Approximation:
        
        Generator (GaussianMixture Model):
            1. Fit GMM on real feature distribution
            2. Sample from GMM to create synthetic data points
            3. Preserves multi-modal structure (bull/bear/sideways regimes)
        
        Discriminator (IsolationForest):
            1. Trained to detect outlier/anomalous patterns
            2. Filters synthetic samples that don't look like real data
            3. Only passes "realistic" synthetic samples
        
        Training:
            1. Generate synthetic features from GMM
            2. Filter through IsolationForest (keep inliers only)
            3. Create labels for synthetic samples
            4. Train final classifier on real + filtered synthetic data
    """
    
    N_COMPONENTS = 5      # number of market regimes to model
    SYNTHETIC_RATIO = 0.5  # how much synthetic data to generate
    ISOLATION_CONTAM = 0.1 # expected outlier fraction
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = []
        self.gmm = None
        self.discriminator = None
        self.is_trained = False
        self.train_accuracy = 0
    
    def _generate_synthetic(self, X_real, y_real, n_samples):
        """
        Generator: Create synthetic feature vectors from learned distribution.
        
        Uses GaussianMixtureModel to capture multi-modal feature distribution
        (different modes = different market regimes).
        """
        if self.gmm is None:
            return np.array([]), np.array([])
        
        # Sample from GMM
        synthetic_features, component_labels = self.gmm.sample(n_samples)
        
        # Create labels: use nearest-neighbor from real data
        # Find nearest real sample for each synthetic sample
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(synthetic_features, X_real)
        nearest_idx = np.argmin(distances, axis=1)
        synthetic_labels = y_real.iloc[nearest_idx].values
        
        return synthetic_features, synthetic_labels
    
    def _build_gan_features(self, df):
        """
        Build features for the GAN module.
        
        Includes regime-detection features that help the GMM
        identify distinct market modes.
        """
        if df is None or len(df) < 60:
            return None
        
        base_features = compute_features(df)
        if base_features is None:
            return None
        
        f = base_features.copy()
        
        # ── Regime features (help GMM cluster regimes) ──
        close = df["close"]
        returns = close.pct_change()
        
        # Volatility regime
        f["gan_vol_regime"] = returns.rolling(20).std() / returns.rolling(50).std()
        
        # Trend regime
        f["gan_trend_strength"] = abs(returns.rolling(20).mean()) / (
            returns.rolling(20).std() + 1e-10)
        
        # Volume regime
        vol = df["volume"]
        f["gan_vol_spike"] = vol / vol.rolling(20).mean()
        
        # Range regime
        f["gan_range_ratio"] = (df["high"] - df["low"]).rolling(10).mean() / (
            (df["high"] - df["low"]).rolling(50).mean() + 1e-10)
        
        # Correlation regime (how correlated are different timeframes)
        f["gan_corr_5_20"] = returns.rolling(5).corr(returns.rolling(20).mean())
        
        return f
    
    def train(self, df, horizon=5, threshold=0.001, train_ratio=0.7):
        """Train GAN-augmented ensemble."""
        features = self._build_gan_features(df)
        if features is None or len(features) < 100:
            return False
        
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
        
        # ── Step 1: Generator — Fit GMM ──
        n_components = min(self.N_COMPONENTS, len(X_train_s) // 10)
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=3,
            random_state=42,
            max_iter=100
        )
        self.gmm.fit(X_train_s)
        
        # ── Step 2: Discriminator — Fit IsolationForest ──
        self.discriminator = IsolationForest(
            n_estimators=100,
            contamination=self.ISOLATION_CONTAM,
            random_state=42
        )
        self.discriminator.fit(X_train_s)
        
        # ── Step 3: Generate and filter synthetic data ──
        n_synthetic = int(len(X_train_s) * self.SYNTHETIC_RATIO)
        synthetic_X, synthetic_y = self._generate_synthetic(
            X_train_s, y_train, n_synthetic
        )
        
        if len(synthetic_X) > 0:
            # Filter through discriminator (keep only "realistic" samples)
            is_real = self.discriminator.predict(synthetic_X) == 1
            synthetic_X = synthetic_X[is_real]
            synthetic_y = synthetic_y[is_real]
            
            # Combine real + synthetic
            X_augmented = np.vstack([X_train_s, synthetic_X])
            y_augmented = np.concatenate([y_train.values, synthetic_y])
        else:
            X_augmented = X_train_s
            y_augmented = y_train.values
        
        # ── Step 4: Train final classifiers on augmented data ──
        self.models = [
            ("gan_gb", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42)),
            ("gan_rf", RandomForestClassifier(
                n_estimators=200, max_depth=7, random_state=43)),
            ("gan_gb_deep", GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.08,
                subsample=0.7, random_state=44)),
        ]
        
        for name, model in self.models:
            model.fit(X_augmented, y_augmented)
        
        # Evaluate on real test data (not synthetic)
        preds = self._ensemble_predict(X_test_s)
        self.train_accuracy = accuracy_score(y_test, preds)
        self.is_trained = True
        
        return True
    
    def predict(self, df):
        """Predict with GAN-augmented ensemble."""
        if not self.is_trained:
            return 0.5, "NEUTRAL", 0.5
        
        features = self._build_gan_features(df)
        if features is None or len(features) < 1:
            return 0.5, "NEUTRAL", 0.5
        
        X = features.iloc[[-1]]
        if not X.notna().all(axis=1).iloc[0]:
            return 0.5, "NEUTRAL", 0.5
        
        X_s = self.scaler.transform(X)
        
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
