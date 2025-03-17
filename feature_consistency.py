"""
Utility functions for ensuring feature consistency in RSIDTrade.
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any

logger = logging.getLogger(__name__)

def ensure_feature_consistency(data: pd.DataFrame, required_feature_count: int = 30) -> pd.DataFrame:
    """
    Ensure data has exactly the required number of feature columns.
    
    Args:
        data: DataFrame with OHLCV and feature data
        required_feature_count: Required number of feature columns
        
    Returns:
        DataFrame with exactly required_feature_count feature columns
    """
    # Separate OHLCV columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    ohlcv = data[ohlcv_cols].copy()
    
    # Get feature columns
    feature_cols = [col for col in data.columns if col not in ohlcv_cols]
    
    # Check if we need to adjust feature count
    if len(feature_cols) == required_feature_count:
        # No adjustment needed
        return data
    
    logger.warning(f"Feature count mismatch: got {len(feature_cols)}, need {required_feature_count}")
    
    # Create adjusted data with OHLCV
    adjusted_data = ohlcv.copy()
    
    if len(feature_cols) < required_feature_count:
        # Add existing features
        for col in feature_cols:
            adjusted_data[col] = data[col]
        
        # Add dummy features to reach required count
        for i in range(len(feature_cols), required_feature_count):
            # Create a dummy feature name that won't conflict
            dummy_name = f"dummy_feature_{i+1}"
            while dummy_name in adjusted_data.columns:
                dummy_name = f"dummy_feature_{i+1}_{np.random.randint(1000)}"
            
            # Add dummy feature (zeros)
            adjusted_data[dummy_name] = 0.0
            
            logger.debug(f"Added dummy feature: {dummy_name}")
    else:
        # Too many features, keep only the first required_feature_count
        for i, col in enumerate(feature_cols[:required_feature_count]):
            adjusted_data[col] = data[col]
    
    logger.info(f"Adjusted feature count from {len(feature_cols)} to {required_feature_count}")
    
    return adjusted_data

def ensure_sequence_dimensions(X: np.ndarray, expected_shape: Tuple[Optional[int], int, int]) -> np.ndarray:
    """
    Ensure that the input sequences have the expected dimensions.
    
    Args:
        X: Input sequences with shape (samples, sequence_length, features)
        expected_shape: Expected shape (None, sequence_length, features)
        
    Returns:
        Adjusted input sequences with correct dimensions
    """
    # Check if dimensions match
    if X.shape[1:] == expected_shape[1:]:
        # Dimensions already match
        return X
    
    logger.warning(f"Sequence dimension mismatch: got {X.shape}, expected {expected_shape}")
    
    # Get dimensions
    samples = X.shape[0]
    expected_seq_len = expected_shape[1]
    expected_features = expected_shape[2]
    
    # Create adjusted array
    X_adjusted = np.zeros((samples, expected_seq_len, expected_features))
    
    # Copy data for common dimensions
    common_seq_len = min(X.shape[1], expected_seq_len)
    common_features = min(X.shape[2], expected_features)
    
    X_adjusted[:, :common_seq_len, :common_features] = X[:, :common_seq_len, :common_features]
    
    logger.info(f"Adjusted sequence dimensions from {X.shape} to {X_adjusted.shape}")
    
    return X_adjusted