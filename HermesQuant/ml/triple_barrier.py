"""
Triple-Barrier Method (Marcos Lopez de Prado)
Three barriers: profit-taking, stop-loss, time
Label = which barrier is hit first
"""
import sys; sys.path.insert(0, "/data/workspace/HermesQuant")
import numpy as np
import pandas as pd
from core import indicators as ind


def triple_barrier_labels(df, horizon=10, profit_mult=1.5, loss_mult=1.5):
    """
    Triple-Barrier labeling:
    - Upper barrier: entry + profit_mult * ATR (profit target)
    - Lower barrier: entry - loss_mult * ATR (stop loss)
    - Vertical barrier: entry + horizon candles (time limit)
    
    Returns:
    - labels: 1=UP (profit hit first), 0=DOWN (loss hit first), 2=TIME (time expired)
    - distances: how far into the future each label reaches
    """
    if df is None or len(df) < horizon + 20:
        return None, None
    
    # Calculate ATR for barriers
    atr = ind.atr(df, 14)
    
    labels = []
    distances = []
    
    for i in range(len(df) - horizon):
        entry = df["close"].iloc[i]
        atr_val = atr.iloc[i]
        
        if atr_val == 0 or np.isnan(atr_val):
            labels.append(2)  # TIME
            distances.append(horizon)
            continue
        
        upper_barrier = entry + profit_mult * atr_val
        lower_barrier = entry - loss_mult * atr_val
        
        # Check each future candle
        hit_upper = False
        hit_lower = False
        hit_idx = horizon
        
        for j in range(1, horizon + 1):
            if i + j >= len(df):
                break
            
            high = df["high"].iloc[i + j]
            low = df["low"].iloc[i + j]
            
            if high >= upper_barrier:
                hit_upper = True
                hit_idx = j
                break
            
            if low <= lower_barrier:
                hit_lower = True
                hit_idx = j
                break
        
        # Determine label
        if hit_upper and not hit_lower:
            labels.append(1)  # UP (profit hit first)
        elif hit_lower and not hit_upper:
            labels.append(0)  # DOWN (loss hit first)
        elif hit_upper and hit_lower:
            # Both hit in same candle — check which is closer
            high_dist = abs(upper_barrier - entry)
            low_dist = abs(entry - lower_barrier)
            labels.append(1 if high_dist < low_dist else 0)
        else:
            labels.append(2)  # TIME (neither hit)
        
        distances.append(hit_idx)
    
    return np.array(labels), np.array(distances)


def create_labels_triple_barrier(df, horizon=10, profit_mult=1.5, loss_mult=1.5):
    """
    Create labels using Triple-Barrier method.
    Returns binary labels: 1=UP, 0=DOWN (excludes TIME labels)
    """
    labels, distances = triple_barrier_labels(df, horizon, profit_mult, loss_mult)
    
    if labels is None:
        return None
    
    # Filter out TIME labels (2) — only keep clear UP/DOWN
    mask = labels != 2
    filtered_labels = labels[mask]
    
    return filtered_labels, mask, distances
