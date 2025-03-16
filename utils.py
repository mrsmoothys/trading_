"""
Utility functions for the AI trading bot.
"""
import logging
import os
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from config import (
    LOGS_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR,
    RUN_ID, RUN_METADATA
)

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    """
    # Create logs directory if not exists
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    else:
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_file = os.path.join(LOGS_DIR, f"rsidtrade_{RUN_ID}.log")
    
    # Set up logging configuration
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")

def save_metadata(metadata: Dict[str, Any], path: Optional[str] = None) -> str:
    """
    Save run metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
        path: Path to save the file
        
    Returns:
        Path to saved file
    """
    # Create path if not provided
    if not path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"metadata_{RUN_ID}.json")
    
    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()
    
    # Write to file
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {path}")
    return path

def load_metadata(path: str) -> Dict[str, Any]:
    """
    Load run metadata from JSON file.
    
    Args:
        path: Path to metadata file
        
    Returns:
        Metadata dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    
    # Read file
    with open(path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Metadata loaded from {path}")
    return metadata

def find_optimal_parameters(results_dir: str, metric: str = 'sharpe_ratio') -> Dict[str, Dict[str, Any]]:
    """
    Find optimal parameters from optimization results.
    
    Args:
        results_dir: Directory containing optimization results
        metric: Performance metric to optimize
        
    Returns:
        Dictionary of optimal parameters by symbol and timeframe
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    optimal_params = {}
    
    # Walk through results directory
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'summary.csv':
                # Load summary CSV
                summary_path = os.path.join(root, file)
                summary_df = pd.read_csv(summary_path)
                
                # Group by symbol and timeframe and find best parameters
                if 'symbol' in summary_df.columns and 'timeframe' in summary_df.columns:
                    for (symbol, timeframe), group in summary_df.groupby(['symbol', 'timeframe']):
                        # Find row with best metric
                        if metric in group.columns:
                            best_row = group.loc[group[metric].idxmax()]
                            
                            # Extract parameters
                            params = {col: best_row[col] for col in group.columns 
                                     if col not in ['symbol', 'timeframe', metric]}
                            
                            # Store in dictionary
                            if symbol not in optimal_params:
                                optimal_params[symbol] = {}
                            
                            optimal_params[symbol][timeframe] = params
    
    logger.info(f"Found optimal parameters for {len(optimal_params)} symbols")
    return optimal_params

def save_model_architecture(model, path: str) -> None:
    """
    Save model architecture to JSON file.
    
    Args:
        model: Keras model
        path: Path to save the file
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Get model configuration
    model_json = model.to_json()
    
    # Write to file
    with open(path, 'w') as f:
        f.write(model_json)
    
    logger.info(f"Model architecture saved to {path}")

def load_model_architecture(path: str) -> tf.keras.Model:
    """
    Load model architecture from JSON file.
    
    Args:
        path: Path to architecture file
        
    Returns:
        Keras model (without weights)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model architecture file not found: {path}")
    
    # Read file
    with open(path, 'r') as f:
        model_json = f.read()
    
    # Load model
    model = tf.keras.models.model_from_json(model_json)
    
    logger.info(f"Model architecture loaded from {path}")
    return model

def save_results_to_csv(results: List[Dict[str, Any]], path: str) -> None:
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        path: Path to save the file
    """
    if not results:
        logger.warning("No results to save")
        return
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Get fieldnames from first result
    fieldnames = list(results[0].keys())
    
    # Write to CSV
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to {path}")

def find_all_datasets() -> Dict[str, List[str]]:
    """
    Find all available datasets.
    
    Returns:
        Dictionary of available timeframes by symbol
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    datasets = {}
    
    # List files in data directory
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            # Parse filename to extract symbol and timeframe
            try:
                # Expected format: SYMBOL_TIMEFRAME_data_START_to_END.csv
                parts = file.split('_')
                if len(parts) >= 4 and parts[2] == 'data':
                    symbol = parts[0]
                    timeframe = parts[1]
                    
                    # Add to dictionary
                    if symbol not in datasets:
                        datasets[symbol] = []
                    
                    if timeframe not in datasets[symbol]:
                        datasets[symbol].append(timeframe)
            except Exception as e:
                logger.warning(f"Failed to parse filename {file}: {e}")
    
    logger.info(f"Found datasets for {len(datasets)} symbols")
    return datasets

def find_common_timeframes(datasets: Dict[str, List[str]]) -> List[str]:
    """
    Find timeframes common to all symbols.
    
    Args:
        datasets: Dictionary of available timeframes by symbol
        
    Returns:
        List of common timeframes
    """
    if not datasets:
        return []
    
    # Get all unique timeframes
    all_timeframes = set()
    for timeframes in datasets.values():
        all_timeframes.update(timeframes)
    
    # Find common timeframes
    common_timeframes = []
    for timeframe in all_timeframes:
        if all(timeframe in timeframes for timeframes in datasets.values()):
            common_timeframes.append(timeframe)
    
    return sorted(common_timeframes)

def get_resource_usage():
    """
    Get current memory and CPU usage.
    
    Returns:
        Dictionary with memory and CPU usage
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Get memory usage in MB
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / 1024 / 1024
    
    # Get CPU usage
    cpu_usage = process.cpu_percent(interval=0.1)
    
    return {
        'memory_mb': memory_usage,
        'cpu_percent': cpu_usage
    }

def clear_keras_session():
    """Clear Keras session to free up memory."""
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    Parse timeframe string to value and unit.
    
    Args:
        timeframe: Timeframe string (e.g., '1h', '15m', '1d')
        
    Returns:
        Tuple of (value, unit)
    """
    import re
    
    # Match number followed by unit
    match = re.match(r'(\d+)([mhd])', timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    return value, unit

def get_timeframe_seconds(timeframe: str) -> int:
    """
    Convert timeframe to seconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1h', '15m', '1d')
        
    Returns:
        Number of seconds
    """
    value, unit = parse_timeframe(timeframe)
    
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

def load_cached_data(cache_path: str) -> Any:
    """
    Load data from cache.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Cached data
    """
    if not os.path.exists(cache_path):
        return None
    
    # Read file
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Data loaded from cache: {cache_path}")
    return data

def save_cached_data(data: Any, cache_path: str) -> None:
    """
    Save data to cache.
    
    Args:
        data: Data to cache
        cache_path: Path to cache file
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Write to file
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Data saved to cache: {cache_path}")

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add value
            result[key] = value
    
    return result

def is_numerical_column(series: pd.Series) -> bool:
    """
    Check if a pandas Series contains numerical data.
    
    Args:
        series: Pandas Series
        
    Returns:
        True if numerical, False otherwise
    """
    return pd.api.types.is_numeric_dtype(series)

def print_system_info() -> Dict[str, Any]:
    """
    Print system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    # Get system information
    system_info = {
        'python_version': platform.python_version(),
        'system': platform.system(),
        'release': platform.release(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__
    }
    
    # Print information
    logger.info("System Information:")
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    return system_info

def find_optimal_trailing_stop(trades: List[Dict[str, Any]], profit_target: float = 0.05, step: float = 0.001) -> float:
    """
    Find optimal trailing stop percentage.
    
    Args:
        trades: List of trade dictionaries
        profit_target: Profit target percentage
        step: Step size for optimization
        
    Returns:
        Optimal trailing stop percentage
    """
    if not trades:
        logger.warning("No trades to optimize trailing stop")
        return 0.02  # Default value
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Filter out non-completed trades
    completed_trades = trades_df[trades_df['exit_price'].notnull()]
    
    # Prepare result storage
    results = []
    
    # Test different trailing stop values
    for trailing_stop in np.arange(0.005, 0.1, step):
        total_profit = 0
        
        for _, trade in completed_trades.iterrows():
            # Simulate trailing stop
            entry_price = trade['entry_price']
            side = trade['side']
            
            if side == 'buy':
                # Calculate maximum theoretical profit
                max_price = entry_price * (1 + profit_target)
                
                # Calculate trailing stop price
                stop_price = max_price * (1 - trailing_stop)
                
                # Calculate profit with trailing stop
                profit = max(entry_price, stop_price) / entry_price - 1
            else:  # sell
                # Calculate maximum theoretical profit
                min_price = entry_price * (1 - profit_target)
                
                # Calculate trailing stop price
                stop_price = min_price * (1 + trailing_stop)
                
                # Calculate profit with trailing stop
                profit = 1 - min(entry_price, stop_price) / entry_price
            
            total_profit += profit
        
        # Store result
        results.append({
            'trailing_stop': trailing_stop,
            'total_profit': total_profit,
            'avg_profit': total_profit / len(completed_trades) if completed_trades.shape[0] > 0 else 0
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal value
    optimal_row = results_df.loc[results_df['total_profit'].idxmax()]
    optimal_trailing_stop = optimal_row['trailing_stop']
    
    logger.info(f"Optimal trailing stop: {optimal_trailing_stop:.4f}")
    return optimal_trailing_stop

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Print a progress bar.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Decimal places for percentage
        length: Character length of bar
        fill: Bar fill character
        print_end: End character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    
    # Print new line on complete
    if iteration == total:
        print()

def resample_multi_timeframe(df: pd.DataFrame, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Resample a DataFrame to multiple timeframes.
    
    Args:
        df: DataFrame with OHLCV data
        timeframes: List of timeframes to resample to
        
    Returns:
        Dictionary of resampled DataFrames by timeframe
    """
    result = {}
    
    for timeframe in timeframes:
        value, unit = parse_timeframe(timeframe)
        
        # Map unit to pandas frequency
        freq_map = {'m': 'min', 'h': 'H', 'd': 'D'}
        freq = f"{value}{freq_map[unit]}"
        
        # Resample
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        result[timeframe] = resampled
    
    return result

def calculate_sharpe_ratio(returns: Union[List[float], pd.Series], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: List or Series of returns
        risk_free_rate: Risk-free rate
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, list):
        returns = pd.Series(returns)
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    
    # Annualize
    sharpe = sharpe * np.sqrt(annualization_factor)
    
    return sharpe

def calculate_sortino_ratio(returns: Union[List[float], pd.Series], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: List or Series of returns
        risk_free_rate: Risk-free rate
        annualization_factor: Annualization factor (252 for daily, 12 for monthly, etc.)
        
    Returns:
        Sortino ratio
    """
    if isinstance(returns, list):
        returns = pd.Series(returns)
    
    mean_return = returns.mean()
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std()
    
    if downside_deviation == 0 or pd.isna(downside_deviation):
        return 0.0
    
    sortino = (mean_return - risk_free_rate) / downside_deviation
    
    # Annualize
    sortino = sortino * np.sqrt(annualization_factor)
    
    return sortino

def calculate_max_drawdown(equity_curve: Union[List[float], pd.Series]) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: List or Series of equity values
        
    Returns:
        Maximum drawdown as percentage
    """
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)
    
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve / running_max - 1) * 100
    
    # Get maximum drawdown
    max_drawdown = drawdown.min()
    
    
def check_memory_usage(threshold_gb=0.8, clear_if_high=True):
    """
    Check current memory usage and optionally clear memory if usage is high.
    
    Args:
        threshold_gb: Memory threshold in GB to trigger clearing
        clear_if_high: Whether to clear memory if usage exceeds threshold
        
    Returns:
        Current memory usage in GB
    """
    import psutil
    import gc
    
    # Get current memory usage
    process = psutil.Process(os.getpid())
    memory_usage_gb = process.memory_info().rss / (1024 ** 3)
    
    logger.debug(f"Current memory usage: {memory_usage_gb:.2f} GB")
    
    # Clear memory if usage exceeds threshold
    if clear_if_high and memory_usage_gb > threshold_gb:
        logger.warning(f"Memory usage high ({memory_usage_gb:.2f} GB), clearing cache")
        gc.collect()
        clear_keras_session()
        
        # Clear matplotlib cache if exists
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Check memory again
        memory_usage_gb = process.memory_info().rss / (1024 ** 3)
        logger.info(f"Memory usage after clearing: {memory_usage_gb:.2f} GB")
    
    return memory_usage_gb

def limit_gpu_memory(memory_limit=None, allow_growth=True):
    """
    Limit TensorFlow GPU memory usage.
    
    Args:
        memory_limit: Memory limit in MB, or None for proportional limit
        allow_growth: Whether to allow memory growth
    """
    try:
        # Configure GPU memory usage
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s)")
            
            for gpu in gpus:
                if memory_limit is not None:
                    # Limit GPU memory
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    logger.info(f"Limited GPU memory to {memory_limit} MB")
                
                if allow_growth:
                    # Allow memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Enabled GPU memory growth")
    except Exception as e:
        logger.warning(f"Error configuring GPU: {e}")

def create_progress_bar(total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Create a progress bar function that can be updated.
    
    Args:
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Decimal places for percentage
        length: Character length of bar
        fill: Bar fill character
        print_end: End character
        
    Returns:
        Function to update progress bar
    """
    def update_progress_bar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
        
        # Print new line on complete
        if iteration == total:
            print()
    
    return update_progress_bar

if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Print system information
    print_system_info()
    
    # Test utility functions
    try:
        # Find datasets
        datasets = find_all_datasets()
        print(f"Found datasets: {datasets}")
        
        # Find common timeframes
        common_timeframes = find_common_timeframes(datasets)
        print(f"Common timeframes: {common_timeframes}")
    except Exception as e:
        logger.error(f"Error testing utility functions: {e}")