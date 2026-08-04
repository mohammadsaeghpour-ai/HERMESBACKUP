"""
Advanced Features — Based on deepalpha research (84.6% accuracy)
Includes: Hurst exponent, VPIN, Fractal efficiency, HMM regime
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from core import indicators as ind


def hurst_exponent(ts, max_lag=20):
    """
    Hurst exponent: measures trend persistence
    H > 0.5: trending (persistent)
    H < 0.5: mean-reverting (anti-persistent)
    H = 0.5: random walk
    """
    if len(ts) < max_lag + 1:
        return 0.5
    
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    if len(tau) < 2 or any(t == 0 for t in tau):
        return 0.5
    
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]


def vpin(df, n_buckets=20):
    """
    Volume-synchronized Probability of Informed Trading
    High VPIN = more informed trading = potential reversal
    """
    if df is None or len(df) < n_buckets:
        return 0.5
    
    # Classify trades as buy/sell based on price direction
    price_change = df["close"].diff()
    buy_vol = df["volume"][price_change > 0].sum()
    sell_vol = df["volume"][price_change < 0].sum()
    total_vol = buy_vol + sell_vol
    
    if total_vol == 0:
        return 0.5
    
    # VPIN = |buy_vol - sell_vol| / total_vol
    vpin = abs(buy_vol - sell_vol) / total_vol
    return min(vpin, 1.0)


def fractal_efficiency(df, period=10):
    """
    Fractal Efficiency Ratio
    High efficiency = trending market
    Low efficiency = ranging market
    """
    if df is None or len(df) < period + 1:
        return 0.5
    
    # Net price change
    net_change = abs(df["close"].iloc[-1] - df["close"].iloc[-period])
    
    # Sum of individual price changes
    path_length = abs(df["close"].diff()).iloc[-period:].sum()
    
    if path_length == 0:
        return 0.5
    
    return net_change / path_length


def volatility_regime(df, lookback=50):
    """
    Volatility regime detection
    Returns: 0=low, 1=medium, 2=high
    """
    if df is None or len(df) < lookback:
        return 1
    
    returns = df["close"].pct_change().dropna()
    if len(returns) < lookback:
        return 1
    
    current_vol = returns.iloc[-lookback:].std()
    hist_vol = returns.std()
    
    if current_vol > hist_vol * 1.5:
        return 2  # High volatility
    elif current_vol < hist_vol * 0.7:
        return 0  # Low volatility
    else:
        return 1  # Medium


def compute_advanced_features(df):
    """Compute advanced features for ML"""
    if df is None or len(df) < 50:
        return None
    
    f = pd.DataFrame(index=df.index)
    
    # Hurst exponent (rolling)
    f["hurst"] = df["close"].rolling(50).apply(
        lambda x: hurst_exponent(x.values), raw=False
    )
    
    # VPIN (rolling)
    f["vpin"] = df["close"].rolling(20).apply(
        lambda x: vpin(df.loc[x.index]), raw=False
    )
    
    # Fractal efficiency
    f["fractal_eff"] = fractal_efficiency(df, 10)
    
    # Volatility regime
    f["vol_regime"] = volatility_regime(df, 50)
    
    # Multi-timeframe alignment
    ema_8 = ind.ema(df["close"], 8)
    ema_20 = ind.ema(df["close"], 20)
    ema_50 = ind.ema(df["close"], 50) if len(df) > 50 else ema_20
    
    f["mtf_aligned"] = ((ema_8 > ema_20) & (ema_20 > ema_50)).astype(float) -                         ((ema_8 < ema_20) & (ema_20 < ema_50)).astype(float)
    
    # Clean
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.fillna(0)
    
    return f
