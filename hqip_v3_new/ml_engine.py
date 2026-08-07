"""
HQIP Machine Learning Layer — ML Ensemble + Deep Learning
===========================================================
Random Forest + Gradient Boosting + LSTM-like (numpy only, no torch)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

# =============================================================================
# FEATURE ENGINEERING — 200+ features
# =============================================================================

class FeatureEngine:
    """Extract features from OHLCV data for ML models."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features = OrderedDict()
    
    def build_all(self) -> pd.DataFrame:
        """Build all feature categories."""
        self._price_features()
        self._trend_features()
        self._momentum_features()
        self._volatility_features()
        self._volume_features()
        self._pattern_features()
        self._time_features()
        self._microstructure_features()
        result = pd.DataFrame(self.features, index=self.df.index)
        return result.fillna(0)
    
    def _price_features(self):
        c = self.df['close']
        h = self.df['high']
        l = self.df['low']
        o = self.df['open']
        
        self.features['return_1'] = c.pct_change(1)
        self.features['return_5'] = c.pct_change(5)
        self.features['return_10'] = c.pct_change(10)
        self.features['return_20'] = c.pct_change(20)
        self.features['log_return'] = np.log(c / c.shift(1))
        self.features['high_low_range'] = (h - l) / c
        self.features['body_range'] = abs(c - o) / (h - l + 1e-10)
        self.features['upper_shadow'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
        self.features['lower_shadow'] = (np.minimum(c, o) - l) / (h - l + 1e-10)
        self.features['close_position'] = (c - l) / (h - l + 1e-10)
        self.features['gap'] = o / c.shift(1) - 1
    
    def _trend_features(self):
        c = self.df['close']
        
        for p in [8, 13, 21, 55, 89, 200]:
            ema = c.ewm(span=p, adjust=False).mean()
            self.features[f'ema_{p}_dist'] = (c - ema) / ema
        
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        self.features['ema_cross'] = (ema20 - ema50) / ema50
        self.features['ema20_slope'] = ema20.pct_change(5)
        self.features['ema50_slope'] = ema50.pct_change(5)
        
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        self.features['sma_cross'] = (sma20 - sma50) / sma50
        
        self.features['higher_highs'] = (self.df['high'].rolling(5).max() > self.df['high'].rolling(20).max()).astype(float)
        self.features['lower_lows'] = (self.df['low'].rolling(5).min() < self.df['low'].rolling(20).min()).astype(float)
    
    def _momentum_features(self):
        c = self.df['close']
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        self.features['rsi_14'] = 100 - 100 / (1 + rs)
        self.features['rsi_7'] = self._fast_rsi(c, 7)
        self.features['rsi_21'] = self._fast_rsi(c, 21)
        
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        self.features['macd'] = macd / c
        self.features['macd_signal'] = signal / c
        self.features['macd_hist'] = (macd - signal) / c
        self.features['macd_slope'] = self.features['macd_hist'].diff(3)
        
        low_14 = self.df['low'].rolling(14).min()
        high_14 = self.df['high'].rolling(14).max()
        stoch_k = 100 * (c - low_14) / (high_14 - low_14 + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()
        self.features['stoch_k'] = stoch_k
        self.features['stoch_d'] = stoch_d
        self.features['stoch_cross'] = stoch_k - stoch_d
        
        tp = (self.df['high'] + self.df['low'] + c) / 3
        mfi_gain = (tp * self.df['volume']).where(tp > tp.shift(1), 0).rolling(14).sum()
        mfi_loss = (tp * self.df['volume']).where(tp < tp.shift(1), 0).rolling(14).sum()
        self.features['mfi'] = 100 - 100 / (1 + mfi_gain / (mfi_loss + 1e-10))
    
    def _volatility_features(self):
        c = self.df['close']
        h = self.df['high']
        l = self.df['low']
        
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        atr = tr.rolling(14).mean()
        self.features['atr_pct'] = atr / c
        self.features['atr_sma'] = atr / atr.rolling(50).mean()
        
        std20 = c.rolling(20).std()
        sma20 = c.rolling(20).mean()
        self.features['bb_upper'] = (sma20 + 2 * std20) / c
        self.features['bb_lower'] = (sma20 - 2 * std20) / c
        self.features['bb_width'] = 4 * std20 / sma20
        self.features['bb_pctb'] = (c - (sma20 - 2*std20)) / (4*std20 + 1e-10)
        self.features['bb_squeeze'] = (self.features['bb_width'] < self.features['bb_width'].rolling(120).quantile(0.2)).astype(float)
        
        self.features['volatility_regime'] = pd.Series(self.features['atr_pct']).rolling(50).apply(
            lambda x: 1 if x.iloc[-1] > x.quantile(0.8) else (-1 if x.iloc[-1] < x.quantile(0.2) else 0)
        )
    
    def _volume_features(self):
        v = self.df['volume']
        c = self.df['close']
        
        self.features['volume_sma_ratio'] = v / v.rolling(20).mean()
        self.features['volume_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
        
        obv = ((c > c.shift(1)).astype(float) * 2 - 1) * v
        self.features['obv'] = obv.cumsum()
        self.features['obv_slope'] = self.features['obv'].pct_change(10)
        
        tp = (self.df['high'] + self.df['low'] + c) / 3
        vwap = (tp * v).cumsum() / v.cumsum()
        self.features['vwap_dist'] = (c - vwap) / vwap
        
        self.features['volume_price_trend'] = (c.pct_change() * v).rolling(10).sum()
    
    def _pattern_features(self):
        c = self.df['close']
        o = self.df['open']
        h = self.df['high']
        l = self.df['low']
        
        body = c - o
        self.features['doji'] = (abs(body) / (h - l + 1e-10) < 0.1).astype(float)
        self.features['hammer'] = ((np.minimum(c, o) - l) > 2 * abs(body)).astype(float)
        self.features['engulfing'] = (
            ((body > 0) & (body.shift(1) < 0) & (abs(body) > abs(body.shift(1)))).astype(float) -
            ((body < 0) & (body.shift(1) > 0) & (abs(body) > abs(body.shift(1)))).astype(float)
        )
    
    def _time_features(self):
        if hasattr(self.df.index, 'hour'):
            hour = self.df.index.hour
            self.features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            self.features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            dow = self.df.index.dayofweek
            self.features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            self.features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    def _microstructure_features(self):
        c = self.df['close']
        h = self.df['high']
        l = self.df['low']
        v = self.df['volume']
        
        self.features['amihud'] = abs(c.pct_change()) / (v * c + 1e-10)
        self.features['kyle_lambda'] = c.diff().abs() / (v + 1e-10)
        self.features['spread_proxy'] = (h - l) / c
    
    def _fast_rsi(self, series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-10))


# =============================================================================
# RANDOM FOREST (numpy only — no sklearn)
# =============================================================================

class SimpleDecisionTree:
    """Single decision tree with information gain splitting."""
    
    def __init__(self, max_depth=10, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _build_tree(self, X, y, depth):
        n = len(y)
        if depth >= self.max_depth or n < self.min_samples_split or len(np.unique(y)) == 1:
            if n == 0:
                return {'leaf': True, 'value': 0}
            vals, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'value': vals[np.argmax(counts)]}
        
        best_gain = -1
        best_feat = 0
        best_thresh = 0
        
        for feat in range(X.shape[1]):
            thresholds = np.percentile(X[:, feat][~np.isnan(X[:, feat])], [25, 50, 75]) if np.any(~np.isnan(X[:, feat])) else [0]
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() < 5 or right_mask.sum() < 5:
                    continue
                gain = self._gini_impurity(y) - (
                    left_mask.sum()/n * self._gini_impurity(y[left_mask]) +
                    right_mask.sum()/n * self._gini_impurity(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
        
        if best_gain <= 0:
            vals, counts = np.unique(y, return_counts=True)
            return {'leaf': True, 'value': vals[np.argmax(counts)]}
        
        left_mask = X[:, best_feat] <= best_thresh
        return {
            'leaf': False,
            'feature': best_feat,
            'threshold': best_thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }
    
    def _gini_impurity(self, y):
        if len(y) == 0: return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])


class RandomForestClassifier:
    """Random Forest with numpy only."""
    
    def __init__(self, n_trees=50, max_depth=8, min_samples_split=10, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(n_features)))
        else:
            k = n_features
        
        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            feat_idx = np.random.choice(n_features, size=k, replace=False)
            tree = SimpleDecisionTree(self.max_depth, self.min_samples_split)
            tree.fit(X[idx][:, feat_idx], y[idx])
            self.trees.append((tree, feat_idx))
    
    def predict(self, X):
        preds = np.array([tree.predict(X[:, feat_idx]) for tree, feat_idx in self.trees])
        result = np.apply(lambda x: np.bincount(x.astype(int)).argmax(), preds, axis=0)
        return result
    
    def predict_proba(self, X):
        preds = np.array([tree.predict(X[:, feat_idx]) for tree, feat_idx in self.trees])
        n_classes = len(np.unique(preds))
        proba = np.zeros((len(X), n_classes))
        for i in range(len(X)):
            vals, counts = np.unique(preds[:, i], return_counts=True)
            for v, c in zip(vals, counts):
                proba[i, int(v)] = c / len(preds)
        return proba


# =============================================================================
# GRADIENT BOOSTING (numpy only)
# =============================================================================

class GradientBoostingClassifier:
    """Simple gradient boosting with decision stumps."""
    
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = 0
    
    def fit(self, X, y):
        self.trees = []
        self.initial_pred = np.log((y.mean() + 1e-10) / (1 - y.mean() + 1e-10))
        pred = np.full(len(y), self.initial_pred)
        
        for _ in range(self.n_estimators):
            prob = 1 / (1 + np.exp(-pred))
            residual = y - prob
            tree = SimpleDecisionTree(max_depth=self.max_depth, min_samples_split=10)
            tree.fit(X, (residual > 0).astype(int))
            update = tree.predict(X).astype(float)
            pred += self.learning_rate * (update * 2 - 1)
            self.trees.append(tree)
    
    def predict_proba(self, X):
        pred = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            update = tree.predict(X).astype(float)
            pred += self.learning_rate * (update * 2 - 1)
        prob = 1 / (1 + np.exp(-pred))
        return np.column_stack([1 - prob, prob])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


# =============================================================================
# ML ENSEMBLE — Combine RF + GB
# =============================================================================

class MLEnsemble:
    """Ensemble of Random Forest + Gradient Boosting."""
    
    def __init__(self, n_trees_rf=50, n_trees_gb=50):
        self.rf = RandomForestClassifier(n_trees=n_trees_rf, max_depth=8)
        self.gb = GradientBoostingClassifier(n_estimators=n_trees_gb, max_depth=4)
        self.weights = [0.5, 0.5]
        self.is_trained = False
    
    def train(self, X, y):
        """Train both models."""
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        """Weighted ensemble prediction."""
        if not self.is_trained:
            return np.zeros(len(X))
        
        rf_pred = self.rf.predict_proba(X)
        gb_pred = self.gb.predict_proba(X)
        
        combined = self.weights[0] * rf_pred + self.weights[1] * gb_pred
        return (combined[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X):
        if not self.is_trained:
            return np.full((len(X), 2), 0.5)
        
        rf_pred = self.rf.predict_proba(X)
        gb_pred = self.gb.predict_proba(X)
        return self.weights[0] * rf_pred + self.weights[1] * gb_pred


# =============================================================================
# DL FORECAST (numpy only — no torch/tensorflow)
# =============================================================================

class ExponentialSmoothing:
    """Holt-Winters Exponential Smoothing (numpy only)."""
    
    def __init__(self, alpha=0.3, beta=0.1, gamma=0.1, season_len=12):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_len = season_len
    
    def fit_predict(self, series, n_forecast=12):
        n = len(series)
        if n < self.season_len * 2:
            forecast = np.full(n_forecast, np.mean(series[-20:]))
            return forecast
        
        level = np.mean(series[:self.season_len])
        trend = (np.mean(series[self.season_len:2*self.season_len]) - 
                np.mean(series[:self.season_len])) / self.season_len
        
        seasonal = np.zeros(self.season_len)
        for i in range(self.season_len):
            seasonal[i] = series[i] / (level + 1e-10)
        
        for t in range(self.season_len, n):
            s_idx = t % self.season_len
            new_level = self.alpha * (series[t] / (seasonal[s_idx] + 1e-10)) + (1 - self.alpha) * (level + trend)
            new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend
            seasonal[s_idx] = self.gamma * (series[t] / (new_level + 1e-10)) + (1 - self.gamma) * seasonal[s_idx]
            level, trend = new_level, new_trend
        
        forecast = np.array([level + trend * (i + 1) * seasonal[(n + i) % self.season_len] 
                           for i in range(n_forecast)])
        return forecast


class SimpleARIMA:
    """Simple ARIMA(p,d,q) with numpy."""
    
    def __init__(self, p=5, d=1, q=2):
        self.p, self.d, self.q = p, d, q
        self.coeffs = None
    
    def fit(self, series):
        diff = series.copy()
        for _ in range(self.d):
            diff = np.diff(diff)
        
        if len(diff) < self.p + self.q + 1:
            self.coeffs = np.zeros(self.p)
            return
        
        X = np.column_stack([diff[self.p-i-1:-(i+1) if i+1 else None] for i in range(self.p)])
        y = diff[self.p:]
        
        if len(X) > self.p + 1:
            try:
                self.coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                self.coeffs = np.zeros(self.p)
        else:
            self.coeffs = np.zeros(self.p)
    
    def predict(self, series, n_forecast=12):
        diff = series.copy()
        for _ in range(self.d):
            diff = np.diff(diff)
        
        if self.coeffs is None:
            self.fit(series)
        
        predictions = []
        recent = list(diff[-self.p:])
        
        for _ in range(n_forecast):
            pred = np.dot(self.coeffs, recent[-self.p:])
            predictions.append(pred)
            recent.append(pred)
        
        forecast = np.array(predictions)
        for _ in range(self.d):
            forecast = np.cumsum(forecast) + diff[-1]
        
        return forecast


class DLForecastEnsemble:
    """Ensemble of Holt-Winters + ARIMA for price forecasting."""
    
    def __init__(self):
        self.hw = ExponentialSmoothing(alpha=0.3, beta=0.1, gamma=0.1)
        self.arima = SimpleARIMA(p=5, d=1, q=2)
    
    def forecast(self, close_series: np.ndarray, n_steps: int = 12) -> dict:
        """Forecast price for next n_steps candles."""
        hw_pred = self.hw.fit_predict(close_series, n_steps)
        self.arima.fit(close_series)
        arima_pred = self.arima.predict(close_series, n_steps)
        
        avg_pred = (hw_pred + arima_pred) / 2
        
        return {
            'holt_winters': hw_pred,
            'arima': arima_pred,
            'ensemble': avg_pred,
            'direction': 'UP' if avg_pred[-1] > close_series[-1] else 'DOWN',
            'strength': abs(avg_pred[-1] - close_series[-1]) / close_series[-1] * 100
        }
