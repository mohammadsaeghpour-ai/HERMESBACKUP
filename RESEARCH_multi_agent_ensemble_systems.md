# Multi-Agent Ensemble Trading Systems — Research Compilation

**Date**: 2026-08-05
**Purpose**: How to structure, combine, correlate, and weight multiple trading agents

---

## 1. How Top Quant Firms Structure Agent Systems

### Architecture Pattern: Vertical Pipeline (DAG), NOT Flat Voting

The most successful multi-agent trading systems use a **directed acyclic graph (DAG)** pipeline, not flat consensus voting. This is the architecture used in HermesQuant and observed across top quant research:

```
STAGE 0 (Independent — parallel):  Trend | Momentum | Volume | Volatility | Pattern
STAGE 1 (Meta — depends on S0):    Regime | Structure | Whale | MTF_Confirm | FundingRate | OpenInterest
STAGE 2 (Structure — depends on S1): RSI_Divergence | BB_Squeeze | Liquidity | Wyckoff | MathBrain
STAGE 3 (Decision — depends on S2):  GameTheory | SmartAction | ML
STAGE 4 (Risk — depends on S3):      Risk
```

**Key insight from HermesQuant research**: 6 agents are INDEPENDENT (run on raw data), 11+ are DEPENDENT (need other agents' output). Pipeline MUST run in stages.

### Why DAG > Flat

| Flat (WRONG) | DAG (CORRECT) |
|---|---|
| All agents run simultaneously | Staged execution with dependencies |
| Equal weight to all signals | Downstream agents condition on upstream |
| No regime awareness | Regime agent adjusts weights dynamically |
| No risk gating | Final stage is always risk management |

### Agent Design Principles (from HermesQuant v2-v3)

1. **One agent = one domain** — Never mix trend + volume in one agent
2. **Agents produce EVIDENCE, not recommendations** — Only the Consensus Manager produces final recommendations
3. **Each agent returns standardized output**:
```python
AgentOutput(
    agent_name=str,
    direction="BUY|SELL|NEUTRAL",
    confidence=0-100,        # absolute conviction
    score=float,             # -1.0 to +1.0, signed directional strength
    evidence=list[str],      # structured findings
    reasoning=str,           # 1-2 sentence summary
    data=dict,               # raw computed values for downstream
    weight=float,            # importance multiplier (1.0 default)
)
```

### Regime-Conditional Weight Adjustment

The Regime Agent classifies market state and dynamically adjusts downstream weights:
```python
# Regime → Weight Mapping
if regime == "TRENDING_UP":
    weights["Trend"] *= 1.5
    weights["Momentum"] *= 1.3
    weights["Volume"] *= 1.1
elif regime == "RANGING":
    weights["SMC"] *= 1.4       # Order blocks matter in ranges
    weights["Liquidity"] *= 1.3  # Stop hunts common
    weights["Trend"] *= 0.7      # Trend signals noise in range
```

### 7-Gate Filter (ALL must pass)

From HermesQuant's hard-won experience (3 wrong signals in a day → built this system):

```
Gate 1: Vote ≥ threshold (70-78% depending on agent count)
Gate 2: 4H trend agrees with signal direction (4H is KING)
Gate 3: ADX > 22 (trending market exists)
Gate 4: Volume > 0.8x average (volume confirms)
Gate 5: Session filter (Europe/US hours only)
Gate 6: Expected Value > 0 (mathematically profitable)
Gate 7: Not a 5/5 trap (all timeframes same = contrarian signal)
```

---

## 2. Agent Correlation Analysis Methods

### Why Correlation Analysis Is Critical

From HermesQuant research: "Don't add agents without correlation analysis — redundancy inflates confidence artificially."

### Method: Output Correlation Matrix

```python
import numpy as np
import pandas as pd

def analyze_agent_correlation(agents_outputs, actual_returns, n_candles=150):
    """
    Measure correlation between agent outputs AND correlation with actual price direction.
    
    agents_outputs: dict[str, pd.Series]  -- agent_name → direction scores (-1 to +1)
    actual_returns: pd.Series             -- actual future returns
    
    Returns:
        corr_matrix: pd.DataFrame  -- inter-agent correlation
        dir_corr: dict[str, float] -- correlation with actual direction
        redundant: list[str]       -- agents to remove (>0.7 inter-correlation)
        harmful: list[str]         -- agents with negative correlation to actual direction
    """
    # 1. Build DataFrame of all agent scores
    scores_df = pd.DataFrame(agents_outputs)
    
    # 2. Inter-agent correlation matrix
    corr_matrix = scores_df.corr()
    
    # 3. Correlation with actual future direction
    dir_corr = {}
    for agent_name in scores_df.columns:
        dir_corr[agent_name] = scores_df[agent_name].corr(actual_returns)
    
    # 4. Find redundant agents (inter-correlation > 0.7)
    redundant = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                redundant.append((corr_matrix.columns[i], corr_matrix.columns[j],
                                  corr_matrix.iloc[i, j]))
    
    # 5. Find harmful agents (negative correlation with actual direction)
    harmful = [name for name, corr in dir_corr.items() if corr < -0.05]
    
    return corr_matrix, dir_corr, redundant, harmful
```

### Real-World Findings from HermesQuant Backtest

| Agent Pair | Correlation | Action |
|---|---|---|
| GameTheory ↔ Momentum | 0.85 | **Remove one** (redundant) |
| Trend ↔ Momentum | 0.72 | **Remove one** |
| Trend ↔ Actual Direction | -0.09 | **Remove** (hurts, not helps) |
| Momentum ↔ Actual Direction | +0.19 | Keep |
| SmartAction ↔ Actual Direction | +0.11 | Keep |
| Structure ↔ Actual Direction | +0.14 | Keep |

### Redundancy Detection Algorithm

```python
def detect_redundant_agents(corr_matrix, threshold=0.7):
    """
    Remove agents with >0.7 inter-correlation.
    Priority: keep the agent with higher correlation to actual returns.
    """
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            agent_i = corr_matrix.columns[i]
            agent_j = corr_matrix.columns[j]
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Remove the one with lower directional correlation
                corr_i = abs(dir_corr.get(agent_i, 0))
                corr_j = abs(dir_corr.get(agent_j, 0))
                remove = agent_i if corr_i < corr_j else agent_j
                to_remove.add(remove)
    return to_remove
```

### Rule: Diverse Information Sources

Keep agents with **different information sources**, not different parameterizations of the same indicator. Two RSI variants (RSI + Stochastic RSI) are more correlated than RSI + Volume Profile.

---

## 3. Best Practices for Combining Multiple Trading Signals

### Method 1: Weighted Consensus Scoring (Used by HermesQuant)

Each directional agent produces a `(score, weight)` tuple. Consensus is the weighted average:

```python
def weighted_consensus(signals):
    """
    signals: list of (score, weight) tuples
    score: -1.0 to +1.0 (signed directional strength)
    weight: agent importance multiplier
    """
    total_weight = sum(w for _, w in signals)
    consensus = sum(s * w for s, w in signals) / (total_weight + 1e-10)
    
    bull_count = sum(1 for s, _ in signals if s > 0.1)
    bear_count = sum(1 for s, _ in signals if s < -0.1)
    
    return consensus, bull_count, bear_count
```

### Method 2: Bayesian Probability Combining (Pipeline Engine v2)

Log-odds weighted combination — accounts for prior probability:

```python
import numpy as np

def bayesian_combine(agent_predictions, agent_weights):
    """
    agent_predictions: list of P(UP) from each agent (0-1)
    agent_weights: list of weights per agent
    
    Uses log-odds space for combination (prevents probability explosion).
    """
    log_odds_prior = 0  # neutral prior (50/50)
    
    combined_log_odds = log_odds_prior
    for pred, weight in zip(agent_predictions, agent_weights):
        # Avoid log(0) and log(inf)
        p = np.clip(pred, 0.01, 0.99)
        log_odds = np.log(p / (1 - p))
        combined_log_odds += log_odds * weight
    
    # Convert back to probability
    p_combined = 1 / (1 + np.exp(-combined_log_odds))
    return p_combined  # P(UP)
```

### Method 3: Vector Geometry Convergence (Pipeline Engine v2)

Treats each agent as a vector in signal space, measures convergence angle:

```python
def vector_convergence(agent_scores):
    """
    agent_scores: dict[str, float] -- agent_name → score (-1 to +1)
    Returns: convergence (0-1), angle in radians
    """
    # Each agent is a point on a line [-1, +1]
    scores = list(agent_scores.values())
    
    # Mean direction
    mean_score = np.mean(scores)
    
    # Convergence = how close all agents agree (1 = perfect agreement)
    # Using cosine similarity of score vectors
    score_array = np.array(scores)
    magnitude = np.sqrt(np.sum(score_array**2))
    if magnitude < 1e-10:
        return 0.5, np.pi / 2
    
    # Angle from positive axis
    angle = np.arccos(np.clip(mean_score / (np.abs(mean_score) + 1e-10), -1, 1))
    convergence = 1 - (angle / np.pi)
    
    return convergence, angle
```

### Method 4: Nash Equilibrium Detection (GameTheory Agent)

Models market as multi-player game:

```python
def nash_equilibrium_detection(bull_power, bear_power, whale_power):
    """
    Determines which strategy dominates given market conditions.
    
    Returns: dominant strategy (trend_follow, mean_reversion, wait)
    """
    # If trend is strong → trend-following dominates Nash eq.
    if abs(bull_power - bear_power) > 0.3:
        return "trend_follow"
    
    # If volatile/ranging → mean reversion dominates
    if whale_power > 0.5:
        return "mean_reversion"  # whales manipulate range
    
    # If balanced → wait (no Nash equilibrium advantage)
    return "wait"
```

### Method 5: Meta-Labeling (Marcos Lopez de Prado — THE Proper ML Approach)

**Primary model**: existing agents suggest direction (BUY/SELL)
**Meta model**: calibrated Logistic Regression decides "execute or skip"

```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def meta_labeling(primary_signals, features, labels):
    """
    primary_signals: agent consensus direction (0 or 1)
    features: additional features (confidence, regime, volume, etc.)
    labels: did the primary signal actually work? (1=correct, 0=wrong)
    
    Meta-model learns WHICH signals to trust.
    Typical filter rate: 60-75% (most signals filtered out = correct!)
    """
    X = np.column_stack([primary_signals, features])
    
    # Calibrated Logistic Regression
    meta_model = CalibratedClassifierCV(
        LogisticRegression(C=1.0), cv=5
    )
    meta_model.fit(X, labels)
    
    # Prediction: should we execute this signal?
    execute_prob = meta_model.predict_proba(X)[:, 1]
    
    # Only execute when meta-model is confident
    return execute_prob > 0.6  # filter rate ~60-75%
```

### Method 6: Performance-Weighted Ensemble (DRW Competition Winner)

The DRW/G-Research Kaggle winner used: `0.37*RF + 0.33*LGB + 0.30*XGB`
Weights derived from cross-validation scores. **Model complementarity > individual strength.**

```python
def performance_weighted_ensemble(models, cv_scores):
    """
    models: list of trained models
    cv_scores: list of cross-validation scores (higher = better)
    
    Normalize scores to sum to 1.0 as weights.
    """
    total = sum(cv_scores)
    weights = [score / total for score in cv_scores]
    
    def predict(X):
        predictions = np.array([m.predict_proba(X)[:, 1] for m in models])
        weighted_avg = np.average(predictions, axis=0, weights=weights)
        return weighted_avg
    
    return predict, weights
```

### Method 7: Inverse-RMSE Adaptive Weighting (DL Forecast Agent)

```python
def inverse_rmse_weighting(model_predictions, actual_values):
    """
    Models with lower RMSE get higher weight.
    Recalculated on each walk-forward window.
    """
    rmses = []
    for pred in model_predictions:
        rmse = np.sqrt(np.mean((pred - actual_values) ** 2))
        rmses.append(max(rmse, 1e-10))
    
    inverse_rmses = [1.0 / r for r in rmses]
    total = sum(inverse_rmses)
    weights = [ir / total for ir in inverse_rmses]
    
    combined = np.zeros_like(actual_values)
    for pred, w in zip(model_predictions, weights):
        combined += pred * w
    
    return combined, weights
```

---

## 4. How to Weight Agents Based on Performance

### Method 1: Static Weight Tiers (Simple, Good Starting Point)

From HermesQuant empirical testing:

```python
WEIGHT_TIERS = {
    "strong": (1.4, 1.8),   # accuracy > 65%
    "medium": (1.0, 1.3),   # accuracy 50-65%
    "weak": (0.8, 1.0),     # accuracy < 50%
}

# Regime-specific weights
REGIME_WEIGHTS = {
    "TRENDING_UP": {"Trend": 1.5, "Momentum": 1.3, "SMC": 0.8},
    "RANGING": {"SMC": 1.4, "Liquidity": 1.3, "Trend": 0.7},
    "VOLATILE": {"Volatility": 1.4, "SmartAction": 1.3, "Trend": 0.6},
}
```

### Method 2: Adaptive Self-Learning Weights (JSON Persistence)

From HermesQuant v2.0:

```python
import json
from pathlib import Path

class AdaptiveWeights:
    def __init__(self, weight_file="weights.json", default=1.0, 
                 min_weight=0.3, max_weight=3.0, update_rate=0.05):
        self.weight_file = Path(weight_file)
        self.default = default
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.update_rate = update_rate
        self.weights = self._load()
    
    def _load(self):
        if self.weight_file.exists():
            return json.loads(self.weight_file.read_text())
        return {}
    
    def _save(self):
        self.weight_file.write_text(json.dumps(self.weights, indent=2))
    
    def get(self, agent_name):
        return self.weights.get(agent_name, self.default)
    
    def update(self, agent_name, correct: bool):
        """Update weight based on prediction outcome."""
        current = self.get(agent_name)
        if correct:
            new_weight = current * (1 + self.update_rate)
        else:
            new_weight = current * (1 - self.update_rate)
        
        self.weights[agent_name] = max(self.min_weight, 
                                        min(self.max_weight, new_weight))
        self._save()
    
    def get_all(self):
        return dict(self.weights)
```

### Method 3: EMA-Weighted Performance Tracking (Exponential Decay)

Recent performance matters more than old performance:

```python
def ema_weight_update(agent_accuracy_history, alpha=0.3):
    """
    EMA of agent accuracy over time.
    alpha=0.3 means recent 30% weight on latest observation.
    
    agent_accuracy_history: list of booleans (correct/wrong) over time
    Returns: smoothed accuracy score 0-1
    """
    if not agent_accuracy_history:
        return 0.5
    
    ema = agent_accuracy_history[0]
    for correct in agent_accuracy_history[1:]:
        ema = alpha * correct + (1 - alpha) * ema
    
    return ema

# Usage: convert to weight
accuracy = ema_weight_update(history, alpha=0.3)
weight = 0.5 + accuracy  # Range: 0.5 to 1.5
```

### Method 4: Kelly Criterion Based Weighting

```python
def kelly_weight(agent_win_rate, avg_win, avg_loss):
    """
    Kelly fraction determines optimal weight for each agent.
    Quarter-Kelly for safety.
    """
    if avg_loss == 0:
        return 0
    
    b = avg_win / avg_loss  # win/loss ratio
    p = agent_win_rate       # win probability
    q = 1 - p                # loss probability
    
    kelly = (p * b - q) / b
    quarter_kelly = kelly / 4  # Conservative sizing
    
    return max(0, min(quarter_kelly, 0.25))  # Cap at 25%
```

### Method 5: Confidence-Adjusted Dynamic Weights (Best Practice)

```python
def confidence_adjusted_weights(agent_scores, base_weights):
    """
    High-confidence signals get higher weight.
    Low-confidence signals get attenuated.
    
    This naturally filters weak signals without hard thresholds.
    """
    adjusted = {}
    for agent, score in agent_scores.items():
        base = base_weights.get(agent, 1.0)
        # Confidence = absolute value of score (higher = more certain)
        confidence = abs(score)
        
        # Only boost weight if confidence is above median
        if confidence > 0.3:
            adjusted[agent] = base * (1 + confidence)
        else:
            adjusted[agent] = base * 0.5  # Attenuate low-confidence
    
    return adjusted
```

### Weight Bounds and Safety Rules

From HermesQuant empirical testing:
- **Minimum weight**: 0.3 (never zero — weak agents add diversity)
- **Maximum weight**: 3.0 (cap to prevent single-agent dominance)
- **Update rate**: ±5% per trade (too aggressive = unstable)
- **Symbol-specific**: BTC uses 78% vote threshold, ETH uses 75%

---

## 5. Agent Ensemble Methods — Research & Algorithms

### Ensemble Architecture Categories

#### A. ML Model Ensemble (sklearn-based, no GPU needed)

**Best performer from HermesQuant**: RandomForest(100) + GradientBoosting(50) soft voting

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

# Soft voting: average predicted probabilities
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced')),
        ('gb', GradientBoostingClassifier(n_estimators=50)),
        ('lr', LogisticRegression(max_iter=1000)),
    ],
    voting='soft',  # CRITICAL: must be 'soft' not 'hard'
    weights=[0.37, 0.33, 0.30],  # Performance-weighted (DRW winner pattern)
)
```

**DRW/G-Research Competition Winner**: `0.37*RF + 0.33*LGB + 0.30*XGB`
- RF beat GBT as best single model (counter-intuitive but proven)
- Weight by CV score, not subjective judgment

#### B. Hybrid DL Ensemble (sklearn approximations — no PyTorch)

From HermesQuant v2/v3 — 5 models, meta-ensemble with dynamic weighting:

| Model | What It Approximates | Walk-Forward Accuracy |
|---|---|---|
| TCN | Temporal Convolutional Network | 67% train, 62% test |
| Transformer | Self-attention mechanism | 39% (overfits on small data) |
| CNN-LSTM | Multi-scale pattern recognition | 28% (needs more data) |
| GAN | Data augmentation | 31% (needs more data) |
| Attention-LSTM | Feature importance weighting | 26% (needs more data) |

**Key insight**: TCN is the best hybrid DL model for crypto with small datasets (<300 candles). Transformer overfits.

#### C. Triple-Barrier + Meta-Labeling (Marcos Lopez de Prado)

The PROPER ML approach for trading:

```python
def triple_barrier_labels(close, atr, profit_mult=1.5, loss_mult=1.5, horizon=10):
    """
    Instead of "price up/down after N candles", use THREE barriers:
    - Upper: entry + profit_mult × ATR (profit target)
    - Lower: entry - loss_mult × ATR (stop loss)  
    - Vertical: entry + horizon candles (time limit)
    
    Label = which barrier is hit first
    This matches real trading (SL/TP) unlike fixed-horizon labels.
    """
    labels = []
    for i in range(len(close)):
        entry = close[i]
        upper = entry + profit_mult * atr[i]
        lower = entry - loss_mult * atr[i]
        
        # Check which barrier is hit first in next `horizon` candles
        for j in range(1, min(horizon + 1, len(close) - i)):
            if close[i + j] >= upper:
                labels.append(1)  # UP (profit target hit)
                break
            elif close[i + j] <= lower:
                labels.append(-1)  # DOWN (stop loss hit)
                break
        else:
            labels.append(0)  # TIMEOUT (neither hit = skip)
    
    return labels
```

#### D. Purged + Embargoed Cross-Validation

Standard K-Fold CAUSES lookahead bias in time series:

```python
def purged_train_test_split(n_samples, purge_gap=5, embargo_size=2, n_splits=5):
    """
    Purging: remove training samples overlapping with test labels
    Embargo: add gap between train and test blocks
    
    NEVER use random split for financial time series.
    """
    fold_size = n_samples // n_splits
    splits = []
    
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n_samples)
        
        # Embargo: gap after test
        embargo_end = min(test_end + embargo_size, n_samples)
        
        # Train: everything except test + embargo
        train_idx = list(range(0, max(0, test_start - purge_gap))) + \
                    list(range(embargo_end, n_samples))
        test_idx = list(range(test_start, test_end))
        
        splits.append((train_idx, test_idx))
    
    return splits
```

#### E. Walk-Forward Optimization (Mandatory)

```python
def walk_forward_optimize(df, train_window=150, test_window=10, retrain_every=20):
    """
    Rolling window: train on train_window → test on test_window → retrain
    
    Rules:
    - NEVER trust simple backtest accuracy — it overfits
    - Walk-Forward is typically 10-20% lower than training accuracy
    - Report Walk-Forward accuracy as the REAL metric
    - Retrain every `retrain_every` candles to adapt to regime changes
    """
    results = []
    predictions = []
    
    for start in range(0, len(df) - train_window - test_window, retrain_every):
        train_data = df.iloc[start:start + train_window]
        test_data = df.iloc[start + train_window:start + train_window + test_window]
        
        # Train model
        model = train_ensemble(train_data)
        
        # Test
        for i in range(len(test_data)):
            features = extract_features(df.iloc[:start + train_window + i + 1])
            pred = model.predict(features)
            actual = 1 if test_data.iloc[i]['close'] > train_data.iloc[-1]['close'] else 0
            predictions.append((pred, actual))
    
    accuracy = sum(1 for p, a in predictions if p == a) / len(predictions)
    return accuracy, predictions
```

#### F. CUSUM Filter + Information-Driven Sampling

From Grądzki et al. (2025) — outperforms time-based sampling in crypto:

```python
def cusum_filter(close, threshold=0.02):
    """
    Only sample when cumulative price change > threshold.
    Reduces noise in low-volume hours.
    
    threshold: 0.02 for BTC, 0.01 for altcoins
    """
    s_pos, s_neg = 0, 0
    events = []
    
    for i in range(1, len(close)):
        s_pos = max(0, s_pos + close[i] - close[i-1])
        s_neg = max(0, s_neg - (close[i] - close[i-1]))
        
        if s_pos > threshold * close[i]:
            events.append(('buy_signal', i))
            s_pos = 0
        elif s_neg > threshold * close[i]:
            events.append(('sell_signal', i))
            s_neg = 0
    
    return events
```

---

## 6. Concrete Agent Accuracy Benchmarks (30-Day Backtest, 15m)

From HermesQuant v2.0 actual results:

### Best Agents (Keep)
| Agent | ETH WR | BTC WR | Method |
|---|---|---|---|
| Wyckoff | 100% | 67% | Accumulation/Distribution phases |
| GameTheory | 90% | 75% | Nash + Bayesian + Kelly |
| Momentum | 73% | 80% | RSI + MACD + Stochastic ensemble |
| SmartAction | 73% | 78% | Stop hunts, manipulation, absorption |
| RSI_Divergence | 71% | 69% | Bullish/bearish divergence detection |
| DLForecast | 65% | 71% | Holt-Winters + ARIMA + Prophet |
| Volume | 60% | 71% | OBV + VWAP + Volume Profile |

### Weak Agents (Help Ensemble Diversity)
| Agent | ETH WR | BTC WR | Note |
|---|---|---|---|
| MTF_Confirm | 60% | 54% | Multi-TF cascade |
| MarketStructure | 67% | 53% | BOS/CHoCH detection |
| BB_Squeeze | 33% | 100% | Volatility breakout |
| MathBrain | 45% | 50% | Fibonacci + Pivot points |

### Remove/Replace Agents
| Agent | ETH WR | BTC WR | Replaced With |
|---|---|---|---|
| SMC | 0-28% | 0% | RSI_Divergence (71%/69%) |
| Pattern | 25% | N/A | Removed entirely |
| Liquidity | 50% | 53% | Fixed (sweep reversal logic) |

---

## 7. Key Algorithms Summary

### Algorithm 1: Agent Weight Update (Post-Trade)
```python
# On each trade result:
for agent in active_agents:
    if agent.agreed_with_final_direction:
        if trade_profitable:
            adaptive_weights.update(agent.name, correct=True)  # +5%
        else:
            adaptive_weights.update(agent.name, correct=False)  # -5%
```

### Algorithm 2: Vote Threshold Scaling
```python
def optimal_vote_threshold(num_agents, asset):
    """
    More agents = higher threshold needed.
    BTC needs stricter filter than ETH.
    """
    base = 0.65
    agent_bonus = 0.005 * (num_agents - 10)  # +0.5% per agent over 10
    asset_adjustment = 0.03 if asset == "BTC" else 0.0  # BTC stricter
    return min(base + agent_bonus + asset_adjustment, 0.85)
```

### Algorithm 3: Consensus with Uncertainty
```python
def consensus_with_uncertainty(agent_outputs):
    """
    If ensemble std is high → uncertain → reduce confidence
    """
    scores = [a.score for a in agent_outputs]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Uncertainty reduces effective confidence
    confidence_penalty = std_score * 100  # std of 0.3 = 30% penalty
    
    raw_confidence = np.mean([a.confidence for a in agent_outputs])
    adjusted_confidence = max(0, raw_confidence - confidence_penalty)
    
    return mean_score, adjusted_confidence, std_score
```

### Algorithm 4: Feature Selection Pipeline
```python
from sklearn.feature_selection import mutual_info_classif

def select_features(X, y, max_features=30, correlation_threshold=0.95):
    """
    1. Compute mutual information
    2. Remove highly correlated features (>0.95)
    3. Keep top 30 by MI score
    
    From DRW/G-Research winning approach.
    """
    # Step 1: Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Step 2: Correlation filter
    corr_matrix = pd.DataFrame(X).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    X_filtered = pd.DataFrame(X).drop(columns=to_drop)
    
    # Step 3: Top N by MI
    feature_scores = pd.Series(mi_scores, index=range(len(mi_scores)))
    # Re-map after dropping correlated features
    selected = feature_scores.sort_values(ascending=False).head(max_features)
    
    return selected.index.tolist()
```

---

## 8. Academic Research References

| Paper/Source | Key Finding | Relevance |
|---|---|---|
| Lopez de Prado (2018) "Advances in Financial ML" | Triple-Barrier + Meta-Labeling + Purged CV | Foundation of proper ML for trading |
| Grądzki et al. (2025) | CUSUM + Triple Barrier + ResNet-LSTM outperforms Transformer for crypto | Event-driven sampling beats time-based |
| EFMA 2025 | Order Flow Imbalance (OFI) gives Sharpe 3.5+ | Strongest crypto alpha signal |
| Paskaleva et al. (2025) | On-Chain + Boruta + CNN-LSTM achieves 82.03% | On-chain data is underexploited |
| Andrew Lo | Adaptive Markets Hypothesis | Markets adapt, so should your agents |
| Robert Schapire | AdaBoost ensemble method | Foundation of boosting for trading |
| nateemma/strategies (GitHub ⭐433) | "diminishing returns from better models" | Feature engineering > model complexity |
| paulcpk/freqtrade-strategies (GitHub ⭐327) | EMA crossover + trend filter = 118% return | Simple strategies can match ML |

---

## 9. Implementation Checklist

### For Building a New Multi-Agent System:

1. [ ] Define agent output format (AgentOutput dataclass)
2. [ ] Build agents as independent modules (one domain each)
3. [ ] Run correlation analysis before adding agents
4. [ ] Set up Walk-Forward validation (NOT random splits)
5. [ ] Implement 7-gate filter system
6. [ ] Set up adaptive weight tracking (JSON persistence)
7. [ ] Implement regime detection + dynamic weight adjustment
8. [ ] Add Meta-Labeling layer (primary agents + meta classifier)
9. [ ] Set up Purged + Embargoed CV for ML models
10. [ ] Track per-agent accuracy and remove agents below 40% sustained
11. [ ] Use symbol-specific thresholds (BTC vs ETH vs alts)
12. [ ] Implement CUSUM event-driven sampling
13. [ ] Add Triple-Barrier labeling for proper ML training
14. [ ] Use DRW-style feature selection (MI + correlation filter + top 30)

### Weight Update Rules:
- Initial: 1.0 for all agents
- After each correct signal: +5% (cap at 3.0)
- After each wrong signal: -5% (floor at 0.3)
- Regime override: ±30-50% based on market state
- Recalculate weekly minimum
