"""
Purged + Embargoed Cross-Validation
Prevents lookahead bias in time series
"""
import numpy as np
import pandas as pd


def purged_kfold_split(n_samples, n_splits=5, embargo_pct=0.01):
    """
    Purged K-Fold for time series.
    
    - Split data into n_splits chronological blocks
    - For each fold: train on all blocks except test
    - Purge: remove training samples that overlap with test labels
    - Embargo: add gap between train and test
    
    Yields: (train_indices, test_indices)
    """
    embargo_size = int(n_samples * embargo_pct)
    block_size = n_samples // n_splits
    
    indices = np.arange(n_samples)
    
    for i in range(n_splits):
        test_start = i * block_size
        test_end = min((i + 1) * block_size, n_samples)
        
        test_indices = indices[test_start:test_end]
        
        # Train = everything except test + embargo
        train_before = indices[:max(0, test_start - embargo_size)]
        train_after = indices[min(n_samples, test_end + embargo_size):]
        train_indices = np.concatenate([train_before, train_after])
        
        yield train_indices, test_indices


def purged_train_test_split(n_samples, test_size=0.2, embargo_pct=0.02):
    """
    Simple purged train/test split.
    """
    embargo_size = int(n_samples * embargo_pct)
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = np.arange(max(0, split_idx - embargo_size))
    test_indices = np.arange(split_idx, n_samples)
    
    return train_indices, test_indices
