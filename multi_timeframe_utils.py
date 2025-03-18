"""
Utilities for multi-timeframe analysis in the trading bot.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def resample_to_higher_timeframe(df: pd.DataFrame, current_timeframe: str, target_timeframe: str) -> pd.DataFrame:
    """
    Resample data from a lower timeframe to a higher timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        current_timeframe: Current timeframe string (e.g., '1h', '15m')
        target_timeframe: Target timeframe string (e.g., '4h', '1d')
        
    Returns:
        Resampled DataFrame
    """
    # Map timeframe strings to pandas frequency strings
    timeframe_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
    }
    
    if current_timeframe not in timeframe_map or target_timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {current_timeframe} or {target_timeframe}")
    
    # Convert to pandas frequency
    freq = timeframe_map[target_timeframe]         
    
    # Resample OHLCV data
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Handle any PCA or feature columns by taking the mean
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    if feature_cols:
        for col in feature_cols:
            resampled[col] = df[col].resample(freq).mean()
    
    return resampled

def create_multi_timeframe_data(df: pd.DataFrame, base_timeframe: str) -> dict:
    """
    Create a dictionary of multiple timeframe data from base timeframe.
    
    Args:
        df: DataFrame with OHLCV data at base timeframe
        base_timeframe: Base timeframe string (e.g., '1h')
        
    Returns:
        Dictionary with data for multiple timeframes
    """
    result = {base_timeframe: df.copy()}
    
    # Determine higher timeframes to generate
    if base_timeframe == '1m':
        higher_tfs = ['5m', '15m', '1h', '4h', '1d']
    elif base_timeframe == '5m':
        higher_tfs = ['15m', '1h', '4h', '1d']
    elif base_timeframe == '15m':
        higher_tfs = ['1h', '4h', '1d']
    elif base_timeframe == '1h':
        higher_tfs = ['4h', '1d']
    elif base_timeframe == '4h':
        higher_tfs = ['1d']
    else:
        higher_tfs = []
    
    # Generate data for each higher timeframe
    for tf in higher_tfs:
        try:
            result[tf] = resample_to_higher_timeframe(df, base_timeframe, tf)
            logger.info(f"Created {tf} data with {len(result[tf])} candles from {base_timeframe}")
        except Exception as e:
            logger.error(f"Error creating {tf} data: {e}")
    
    return result

def align_timeframes_to_base(timeframe_data: dict, base_timeframe: str) -> dict:
    """
    Align all timeframe signals to the base timeframe.
    
    Args:
        timeframe_data: Dictionary with data for multiple timeframes
        base_timeframe: Base timeframe string
        
    Returns:
        Dictionary with signals aligned to base timeframe
    """
    base_df = timeframe_data[base_timeframe]
    result = {}
    
    for tf, df in timeframe_data.items():
        if tf == base_timeframe:
            result[tf] = df['signal'].copy()
        else:
            # Create a series with base_df index
            aligned = pd.Series(index=base_df.index, dtype=float)
            
            # For each higher timeframe bar, copy its signal to all corresponding base timeframe bars
            for idx, row in df.iterrows():
                if tf.endswith('h'):
                    # For hourly timeframes
                    hours = int(tf[:-1])
                    end_idx = idx + pd.Timedelta(hours=hours)
                elif tf.endswith('d'):
                    # For daily timeframes
                    days = int(tf[:-1])
                    end_idx = idx + pd.Timedelta(days=days)
                else:
                    # For minute timeframes
                    minutes = int(tf[:-1])
                    end_idx = idx + pd.Timedelta(minutes=minutes)
                
                # Assign signal to all base candles in this higher timeframe candle
                mask = (base_df.index >= idx) & (base_df.index < end_idx)
                aligned.loc[mask] = row['signal']
            
            # Forward fill any remaining NaN values
            aligned = aligned.fillna(method='ffill')
            result[tf] = aligned
    
    return result