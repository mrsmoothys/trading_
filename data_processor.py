"""
Data processing module for loading and preparing data for the AI trading bot.
"""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR, RENAME_MAP, ESSENTIAL_COLUMNS

logger = logging.getLogger(__name__)

def extract_symbol_timeframe(filename: str) -> Tuple[str, str]:
    """
    Extract symbol and timeframe from filename.
    
    Args:
        filename: Filename in format 'SYMBOL_TF_data_2018_to_2025.csv'
        
    Returns:
        Tuple of (symbol, timeframe)
    """
    pattern = r"([A-Z0-9]+)_([0-9]+[hmd])_data_\d+_to_\d+\.csv"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename {filename} doesn't match expected pattern")
    return match.group(1), match.group(2)

def list_available_data() -> Dict[str, Dict[str, Path]]:
    """
    List all available data files in the data directory.
    
    Returns:
        Dictionary of available data files organized by symbol and timeframe
    """
    result = {}
    
    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} not found")
        return result
    
    for file in os.listdir(DATA_DIR):
        if not file.endswith('.csv'):
            continue
        
        try:
            symbol, timeframe = extract_symbol_timeframe(file)
            if symbol not in result:
                result[symbol] = {}
            result[symbol][timeframe] = DATA_DIR / file
        except ValueError as e:
            logger.warning(f"Skipping file {file}: {e}")
    
    logger.info(f"Found data for {len(result)} symbols")
    return result

def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Rename columns to standardized format
    if list(df.columns) == list(RENAME_MAP.keys()):
        df = df.rename(columns=RENAME_MAP)
    
    # Ensure essential columns exist
    missing_columns = [col for col in ESSENTIAL_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing essential columns: {missing_columns}")
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index
    df = df.set_index('timestamp')
    
    # Convert price and volume columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    df = df.fillna(method='ffill')
    
    # Sort by timestamp
    df = df.sort_index()
    
    logger.info(f"Loaded {len(df)} rows of data")
    return df

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample data to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe (e.g., '1h', '4h', '1d')
        
    Returns:
        Resampled DataFrame
    """
    # Validate timeframe format
    if not re.match(r'^\d+[hmd]$', timeframe):
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    # Map pandas resample rule
    rule_map = {
        'h': 'H',
        'm': 'T',
        'd': 'D'
    }
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    rule = f"{value}{rule_map[unit]}"
    
    # Resample data
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return resampled.dropna()

def prepare_multi_timeframe_data(symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for multiple timeframes for a given symbol.
    
    Args:
        symbol: Trading symbol
        timeframes: List of timeframes to prepare
        
    Returns:
        Dictionary of DataFrames for each timeframe
    """
    available_data = list_available_data()
    
    if symbol not in available_data:
        raise ValueError(f"No data available for symbol {symbol}")
    
    result = {}
    base_data = None
    base_timeframe = None
    
    # Find the smallest timeframe available to use as base
    for tf in sorted(available_data[symbol].keys()):
        base_data = load_data(available_data[symbol][tf])
        base_timeframe = tf
        break
    
    if base_data is None:
        raise ValueError(f"No data found for symbol {symbol}")
    
    # Prepare data for each requested timeframe
    for tf in timeframes:
        if tf in available_data[symbol]:
            # Load directly if available
            result[tf] = load_data(available_data[symbol][tf])
        else:
            # Resample from base timeframe
            result[tf] = resample_data(base_data, tf)
    
    return result

def create_training_sequences(
    df: pd.DataFrame, 
    lookback_window: int, 
    prediction_horizon: int, 
    feature_columns: List[str],
    target_column: str = 'close',
    overlap: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of data for training machine learning models.
        
    Args:
        df: DataFrame with features and target
        lookback_window: Number of time steps to look back
        prediction_horizon: Number of time steps to predict
        feature_columns: List of feature column names
        target_column: Column name for the target variable
        overlap: Whether to allow overlapping sequences
        normalize: Whether to normalize each sequence
            
    Returns:
        Tuple of (X, y) arrays for training
    """
    
    # Limit features to match model expectations
    if len(feature_columns) > 55:   # Changed from 36 to 55
        print(f"WARNING: Limiting features from {len(feature_columns)} to 55 to match model")
        feature_columns = feature_columns[:55]

    data = df[feature_columns + [target_column]].values
    X, y = [], []
    
    step = 1 if overlap else lookback_window
    
    for i in range(0, len(data) - lookback_window - prediction_horizon + 1, step):
        # Extract sequence
        sequence = data[i:(i + lookback_window), :-1]  # Features
        target = data[i + lookback_window:i + lookback_window + prediction_horizon, -1]  # Target
        
        # Normalize sequence if requested
        if normalize:
            # Z-score normalization for each feature
            mean = np.mean(sequence, axis=0)
            std = np.std(sequence, axis=0)
            std = np.where(std == 0, 1e-8, std)  # Avoid division by zero
            sequence = (sequence - mean) / std
        
        X.append(sequence)
        y.append(target)
    
    return np.array(X), np.array(y)

def train_val_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    train_size: float = 0.7, 
    val_size: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        X: Feature data
        y: Target data
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # List available data
    available_data = list_available_data()
    print(f"Available symbols: {list(available_data.keys())}")
    
    # Example: Load data for a symbol
    if 'BTCUSDT' in available_data and '1h' in available_data['BTCUSDT']:
        df = load_data(available_data['BTCUSDT']['1h'])
        print(f"Loaded {len(df)} rows of data for BTCUSDT 1h")
        print(df.head())