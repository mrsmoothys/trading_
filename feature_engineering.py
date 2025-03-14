"""
Feature engineering module for the AI trading bot.
Generates technical indicators and advanced features.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import ta
from scipy.signal import argrelextrema
from statsmodels.nonparametric.kernel_regression import KernelReg

from config import FEATURE_SETS

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for generating trading features from OHLCV data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the feature engineer.
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Ensure the dataframe has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")
        
        self.df = df.copy()
        # If index is not datetime, convert it
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
            self.df.index = pd.to_datetime(self.df.index)
        
        # Clean data
        self._clean_data()
    
    def _clean_data(self):
        """Clean the data by handling missing values and outliers."""
        # Handle NaN values
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (price jumps > 30%)
        price_change = self.df['close'].pct_change().abs()
        self.df = self.df[price_change <= 0.3]
    
    def add_momentum_indicators(self) -> None:
        """Add momentum indicators to the DataFrame."""
        # RSI (Relative Strength Index)
        self.df['rsi'] = ta.momentum.RSIIndicator(self.df['close']).rsi()
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(self.df['close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_hist'] = macd.macd_diff()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(self.df['high'], self.df['low'], self.df['close'])
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        self.df['williams_r'] = ta.momentum.WilliamsRIndicator(
            self.df['high'], self.df['low'], self.df['close']
        ).williams_r()
        
        # Ultimate Oscillator
        self.df['ultimate_oscillator'] = ta.momentum.UltimateOscillator(
            self.df['high'], self.df['low'], self.df['close']
        ).ultimate_oscillator()
        
        # ROC (Rate of Change)
        self.df['price_roc'] = ta.momentum.ROCIndicator(self.df['close']).roc()
        
        # Money Flow Index
        self.df['mfi'] = ta.volume.MFIIndicator(
            self.df['high'], self.df['low'], self.df['close'], self.df['volume']
        ).money_flow_index()
    
    def add_volatility_indicators(self) -> None:
        """Add volatility indicators to the DataFrame."""
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(self.df['close'])
        self.df['bollinger_upper'] = bollinger.bollinger_hband()
        self.df['bollinger_middle'] = bollinger.bollinger_mavg()
        self.df['bollinger_lower'] = bollinger.bollinger_lband()
        self.df['bollinger_width'] = bollinger.bollinger_wband()
        
        # ATR (Average True Range)
        self.df['atr'] = ta.volatility.AverageTrueRange(
            self.df['high'], self.df['low'], self.df['close']
        ).average_true_range()
        
        # Normalized ATR (ATR divided by close price)
        self.df['natr'] = self.df['atr'] / self.df['close'] * 100
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(
            self.df['high'], self.df['low'], self.df['close']
        )
        self.df['keltner_upper'] = keltner.keltner_channel_hband()
        self.df['keltner_middle'] = keltner.keltner_channel_mband()
        self.df['keltner_lower'] = keltner.keltner_channel_lband()
        
        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(
            self.df['high'], self.df['low'], self.df['close']
        )
        self.df['donchian_upper'] = donchian.donchian_channel_hband()
        self.df['donchian_middle'] = donchian.donchian_channel_mband()
        self.df['donchian_lower'] = donchian.donchian_channel_lband()
    
    def add_trend_indicators(self) -> None:
        """Add trend indicators to the DataFrame."""
        # Moving Averages
        self.df['sma_5'] = ta.trend.SMAIndicator(self.df['close'], window=5).sma_indicator()
        self.df['sma_10'] = ta.trend.SMAIndicator(self.df['close'], window=10).sma_indicator()
        self.df['sma_20'] = ta.trend.SMAIndicator(self.df['close'], window=20).sma_indicator()
        self.df['sma_50'] = ta.trend.SMAIndicator(self.df['close'], window=50).sma_indicator()
        self.df['sma_100'] = ta.trend.SMAIndicator(self.df['close'], window=100).sma_indicator()
        self.df['sma_200'] = ta.trend.SMAIndicator(self.df['close'], window=200).sma_indicator()
        
        self.df['ema_5'] = ta.trend.EMAIndicator(self.df['close'], window=5).ema_indicator()
        self.df['ema_10'] = ta.trend.EMAIndicator(self.df['close'], window=10).ema_indicator()
        self.df['ema_20'] = ta.trend.EMAIndicator(self.df['close'], window=20).ema_indicator()
        self.df['ema_50'] = ta.trend.EMAIndicator(self.df['close'], window=50).ema_indicator()
        self.df['ema_100'] = ta.trend.EMAIndicator(self.df['close'], window=100).ema_indicator()
        self.df['ema_200'] = ta.trend.EMAIndicator(self.df['close'], window=200).ema_indicator()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(self.df['high'], self.df['low'], self.df['close'])
        self.df['adx'] = adx.adx()
        self.df['adx_pos'] = adx.adx_pos()
        self.df['adx_neg'] = adx.adx_neg()
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(self.df['high'], self.df['low'])
        self.df['ichimoku_a'] = ichimoku.ichimoku_a()
        self.df['ichimoku_b'] = ichimoku.ichimoku_b()
        self.df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
        self.df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
        
        # Parabolic SAR
        self.df['psar'] = ta.trend.PSARIndicator(
            self.df['high'], self.df['low'], self.df['close']
        ).psar()
        
        # Aroon Indicator
        aroon = ta.trend.AroonIndicator(high=self.df['high'], low=self.df['low'])
        self.df['aroon_up'] = aroon.aroon_up()
        self.df['aroon_down'] = aroon.aroon_down()
        self.df['aroon_indicator'] = aroon.aroon_indicator()

    def add_volume_indicators(self) -> None:
        """Add volume indicators to the DataFrame."""
        # On-Balance Volume
        self.df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            self.df['close'], self.df['volume']
        ).on_balance_volume()
        
        # Chaikin Money Flow
        self.df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            self.df['high'], self.df['low'], self.df['close'], self.df['volume']
        ).chaikin_money_flow()
        
        # Volume Weighted Average Price
        self.df['vwap'] = (self.df['volume'] * self.df['close']).cumsum() / self.df['volume'].cumsum()
        
        # Ease of Movement
        self.df['eom'] = ta.volume.EaseOfMovementIndicator(
            self.df['high'], self.df['low'], self.df['volume']
        ).ease_of_movement()
        
        # Volume Weighted Moving Average
        vwma_window = 20
        self.df['vwma'] = (self.df['close'] * self.df['volume']).rolling(vwma_window).sum() / self.df['volume'].rolling(vwma_window).sum()
        
        # Price-Volume Trend
        self.df['pvt'] = (self.df['close'].pct_change() * self.df['volume']).cumsum()
    
    def add_candlestick_patterns(self) -> None:
        """Add candlestick pattern recognition."""
        # Doji
        self.df['doji'] = np.where(
            (abs(self.df['open'] - self.df['close']) <= 0.05 * (self.df['high'] - self.df['low'])),
            1, 0
        )
        
        # Engulfing patterns
        self.df['bullish_engulfing'] = np.where(
            (self.df['close'].shift(1) < self.df['open'].shift(1)) &  # Previous candle is bearish
            (self.df['close'] > self.df['open']) &  # Current candle is bullish
            (self.df['open'] <= self.df['close'].shift(1)) &  # Current open below previous close
            (self.df['close'] >= self.df['open'].shift(1)),  # Current close above previous open
            1, 0
        )
        
        self.df['bearish_engulfing'] = np.where(
            (self.df['close'].shift(1) > self.df['open'].shift(1)) &  # Previous candle is bullish
            (self.df['close'] < self.df['open']) &  # Current candle is bearish
            (self.df['open'] >= self.df['close'].shift(1)) &  # Current open above previous close
            (self.df['close'] <= self.df['open'].shift(1)),  # Current close below previous open
            1, 0
        )
        
        # Hammer and Shooting Star
        self.df['upper_shadow'] = self.df['high'] - np.maximum(self.df['open'], self.df['close'])
        self.df['lower_shadow'] = np.minimum(self.df['open'], self.df['close']) - self.df['low']
        self.df['body'] = abs(self.df['open'] - self.df['close'])
        
        self.df['hammer'] = np.where(
            (self.df['lower_shadow'] >= 2 * self.df['body']) &  # Long lower shadow
            (self.df['upper_shadow'] <= 0.1 * self.df['body']) &  # Small upper shadow
            (self.df['body'] > 0),  # Not a doji
            1, 0
        )
        
        self.df['shooting_star'] = np.where(
            (self.df['upper_shadow'] >= 2 * self.df['body']) &  # Long upper shadow
            (self.df['lower_shadow'] <= 0.1 * self.df['body']) &  # Small lower shadow
            (self.df['body'] > 0),  # Not a doji
            1, 0
        )
        
        # Morning and Evening Star
        self.df['morning_star'] = np.where(
            (self.df['close'].shift(2) < self.df['open'].shift(2)) &  # First candle is bearish
            (abs(self.df['close'].shift(1) - self.df['open'].shift(1)) <= 0.1 * (self.df['high'].shift(1) - self.df['low'].shift(1))) &  # Second candle is doji
            (self.df['close'] > self.df['open']) &  # Third candle is bullish
            (self.df['close'] > (self.df['open'].shift(2) + self.df['close'].shift(2)) / 2),  # Third close above mid-point of first
            1, 0
        )
        
        self.df['evening_star'] = np.where(
            (self.df['close'].shift(2) > self.df['open'].shift(2)) &  # First candle is bullish
            (abs(self.df['close'].shift(1) - self.df['open'].shift(1)) <= 0.1 * (self.df['high'].shift(1) - self.df['low'].shift(1))) &  # Second candle is doji
            (self.df['close'] < self.df['open']) &  # Third candle is bearish
            (self.df['close'] < (self.df['open'].shift(2) + self.df['close'].shift(2)) / 2),  # Third close below mid-point of first
            1, 0
        )
        
        # Three White Soldiers and Three Black Crows
        self.df['three_white_soldiers'] = np.where(
            (self.df['close'].shift(2) > self.df['open'].shift(2)) &  # First candle bullish
            (self.df['close'].shift(1) > self.df['open'].shift(1)) &  # Second candle bullish
            (self.df['close'] > self.df['open']) &  # Third candle bullish
            (self.df['open'].shift(1) > self.df['open'].shift(2)) &  # Each open higher than previous
            (self.df['open'] > self.df['open'].shift(1)) &
            (self.df['close'].shift(1) > self.df['close'].shift(2)) &  # Each close higher than previous
            (self.df['close'] > self.df['close'].shift(1)),
            1, 0
        )
        
        self.df['three_black_crows'] = np.where(
            (self.df['close'].shift(2) < self.df['open'].shift(2)) &  # First candle bearish
            (self.df['close'].shift(1) < self.df['open'].shift(1)) &  # Second candle bearish
            (self.df['close'] < self.df['open']) &  # Third candle bearish
            (self.df['open'].shift(1) < self.df['open'].shift(2)) &  # Each open lower than previous
            (self.df['open'] < self.df['open'].shift(1)) &
            (self.df['close'].shift(1) < self.df['close'].shift(2)) &  # Each close lower than previous
            (self.df['close'] < self.df['close'].shift(1)),
            1, 0
        )
        
        # Clean up temporary columns
        self.df = self.df.drop(['upper_shadow', 'lower_shadow', 'body'], axis=1)
    
    def add_market_structure_features(self) -> None:
        """Add market structure features like swing points, trends, etc."""
        # Identify swing highs and lows
        self._identify_swing_points()
        
        # Identify trends
        self._identify_trends()
        
        # Identify support and resistance levels
        self._identify_support_resistance()
        
        # Identify consolidation ranges
        self._identify_consolidation()
        
        # Identify volatility regimes
        self._identify_volatility_regimes()
    
    def _identify_swing_points(self, window: int = 5) -> None:
        """
        Identify swing highs and lows in the price series.
        
        Args:
            window: Window size for identifying local extrema
        """
        # Get high and low series
        high_series = self.df['high']
        low_series = self.df['low']
        
        # Identify local maxima for swing highs
        swing_high_indices = argrelextrema(high_series.values, np.greater_equal, order=window)[0]
        self.df['swing_high'] = 0
        self.df.iloc[swing_high_indices, self.df.columns.get_loc('swing_high')] = 1
        
        # Identify local minima for swing lows
        swing_low_indices = argrelextrema(low_series.values, np.less_equal, order=window)[0]
        self.df['swing_low'] = 0
        self.df.iloc[swing_low_indices, self.df.columns.get_loc('swing_low')] = 1
        
        # Save swing points for use in other methods
        self.swing_high_prices = high_series.iloc[swing_high_indices].to_dict()
        self.swing_low_prices = low_series.iloc[swing_low_indices].to_dict()
    
    def _identify_trends(self, ma_fast: int = 20, ma_slow: int = 50) -> None:
        """
        Identify price trends using moving averages.
        
        Args:
            ma_fast: Fast moving average period
            ma_slow: Slow moving average period
        """
        # Simple trend identification using moving averages
        fast_ma = self.df[f'sma_{ma_fast}'] if f'sma_{ma_fast}' in self.df.columns else ta.trend.SMAIndicator(self.df['close'], window=ma_fast).sma_indicator()
        slow_ma = self.df[f'sma_{ma_slow}'] if f'sma_{ma_slow}' in self.df.columns else ta.trend.SMAIndicator(self.df['close'], window=ma_slow).sma_indicator()
        
        # Trend strength using ADX
        adx = self.df['adx'] if 'adx' in self.df.columns else ta.trend.ADXIndicator(self.df['high'], self.df['low'], self.df['close']).adx()
        
        # Determine trend direction and strength
        self.df['trend_direction'] = np.where(fast_ma > slow_ma, 1, np.where(fast_ma < slow_ma, -1, 0))
        self.df['trend_strength'] = adx / 100  # Normalize to 0-1 range
        
        # Combined trend indicator
        self.df['trend'] = self.df['trend_direction'] * self.df['trend_strength']
    
    def _identify_support_resistance(self, n_levels: int = 3, window: int = 20) -> None:
        """
        Identify support and resistance levels.
        
        Args:
            n_levels: Number of support/resistance levels to identify
            window: Window size for clustering levels
        """
        # Use swing highs and swing lows to identify potential levels
        if not hasattr(self, 'swing_high_prices') or not hasattr(self, 'swing_low_prices'):
            self._identify_swing_points()
        
        # Get recent swing highs and lows
        recent_swing_highs = sorted(self.swing_high_prices.values(), reverse=True)[:n_levels*2]
        recent_swing_lows = sorted(self.swing_low_prices.values())[:n_levels*2]
        
        # Cluster levels to find significant ones
        from sklearn.cluster import KMeans
        
        # Resistance levels from swing highs
        if len(recent_swing_highs) > n_levels:
            kmeans = KMeans(n_clusters=n_levels, random_state=42).fit([[x] for x in recent_swing_highs])
            resistance_levels = sorted([float(center[0]) for center in kmeans.cluster_centers_])
        else:
            resistance_levels = sorted(recent_swing_highs)
        
        # Support levels from swing lows
        if len(recent_swing_lows) > n_levels:
            kmeans = KMeans(n_clusters=n_levels, random_state=42).fit([[x] for x in recent_swing_lows])
            support_levels = sorted([float(center[0]) for center in kmeans.cluster_centers_])
        else:
            support_levels = sorted(recent_swing_lows)
        
        # Create proximity indicators for each level
        current_price = self.df['close'].iloc[-1]
        for i, level in enumerate(resistance_levels):
            # Proximity is normalized distance to the level
            distance = abs(current_price - level) / current_price
            proximity = np.exp(-5 * distance)  # Exponential decay
            self.df[f'resistance_{i+1}'] = level
            self.df[f'resistance_{i+1}_proximity'] = proximity
        
        for i, level in enumerate(support_levels):
            distance = abs(current_price - level) / current_price
            proximity = np.exp(-5 * distance)
            self.df[f'support_{i+1}'] = level
            self.df[f'support_{i+1}_proximity'] = proximity
        
        # Combined support and resistance features
        self.df['support'] = self.df[[f'support_{i+1}_proximity' for i in range(min(n_levels, len(support_levels)))]].max(axis=1)
        self.df['resistance'] = self.df[[f'resistance_{i+1}_proximity' for i in range(min(n_levels, len(resistance_levels)))]].max(axis=1)
    
    def _identify_consolidation(self, window: int = 20, threshold: float = 0.03) -> None:
        """
        Identify price consolidation ranges.
        
        Args:
            window: Window size for checking price range
            threshold: Maximum price range as percentage for consolidation
        """
        # Calculate the price range as percentage of the mean price
        rolling_high = self.df['high'].rolling(window=window).max()
        rolling_low = self.df['low'].rolling(window=window).min()
        rolling_mean = self.df['close'].rolling(window=window).mean()
        
        price_range = (rolling_high - rolling_low) / rolling_mean
        
        # Flag consolidation when range is below threshold
        self.df['consolidation'] = np.where(price_range < threshold, 1, 0)
        
        # Measure consolidation strength (1 - normalized range)
        self.df['consolidation_strength'] = 1 - (price_range / threshold).clip(0, 1)
    
    def _identify_volatility_regimes(self, window: int = 20, n_regimes: int = 3) -> None:
        """
        Identify volatility regimes using ATR.
        
        Args:
            window: Window size for ATR calculation
            n_regimes: Number of volatility regimes to identify
        """
        # Calculate ATR if not already present
        if 'atr' not in self.df.columns:
            self.df['atr'] = ta.volatility.AverageTrueRange(
                self.df['high'], self.df['low'], self.df['close']
            ).average_true_range()
        
        # Normalize ATR by price
        norm_atr = self.df['atr'] / self.df['close']
        
        # Categorize volatility into regimes
        from sklearn.cluster import KMeans
        
        # Skip if not enough data
        if len(norm_atr.dropna()) < n_regimes:
            self.df['volatility_regime'] = 0
            return
        
        # Fit KMeans to normalized ATR
        kmeans = KMeans(n_clusters=n_regimes, random_state=42).fit(norm_atr.dropna().values.reshape(-1, 1))
        
        # Map cluster centers to regime labels (0 = low, 1 = medium, 2 = high)
        centers = kmeans.cluster_centers_.flatten()
        regime_map = {i: sorted(range(len(centers)), key=lambda x: centers[x]).index(i) for i in range(len(centers))}
        
        # Assign volatility regime to each point
        self.df['volatility_regime'] = np.zeros(len(self.df))
        for i, atr_val in enumerate(norm_atr):
            if np.isnan(atr_val):
                continue
            cluster = kmeans.predict([[atr_val]])[0]
            self.df.iloc[i, self.df.columns.get_loc('volatility_regime')] = regime_map[cluster]
    
    def create_advanced_features(self) -> None:
        """Create advanced derived features from basic indicators."""
        # Price momentum features
        self.df['ma_cross_signal'] = np.where(
            (self.df['sma_20'] > self.df['sma_50']) & (self.df['sma_20'].shift(1) <= self.df['sma_50'].shift(1)),
            1,  # Bullish crossover
            np.where(
                (self.df['sma_20'] < self.df['sma_50']) & (self.df['sma_20'].shift(1) >= self.df['sma_50'].shift(1)),
                -1,  # Bearish crossover
                0  # No crossover
            )
        )
        
        # Relative price to moving averages
        for ma in [20, 50, 200]:
            self.df[f'price_rel_sma_{ma}'] = self.df['close'] / self.df[f'sma_{ma}'] - 1
        
        # Volatility-adjusted momentum
        self.df['vol_adj_momentum'] = self.df['price_roc'] / self.df['natr']
        
        # Acceleration features
        self.df['rsi_change'] = self.df['rsi'].diff()
        self.df['macd_accel'] = self.df['macd_hist'].diff()
        
        # Combined indicator signals
        self.df['bull_signal_strength'] = (
            (self.df['rsi'] < 30).astype(int) * 0.2 +
            (self.df['macd'] > self.df['macd_signal']).astype(int) * 0.2 +
            (self.df['stoch_k'] < 20).astype(int) * 0.2 +
            (self.df['adx'] > 25).astype(int) * 0.2 +
            (self.df['price_rel_sma_20'] > 0).astype(int) * 0.2
        )
        
        self.df['bear_signal_strength'] = (
            (self.df['rsi'] > 70).astype(int) * 0.2 +
            (self.df['macd'] < self.df['macd_signal']).astype(int) * 0.2 +
            (self.df['stoch_k'] > 80).astype(int) * 0.2 +
            (self.df['adx'] > 25).astype(int) * 0.2 +
            (self.df['price_rel_sma_20'] < 0).astype(int) * 0.2
        )
        
        # Market regime features
        if 'trend' in self.df.columns and 'volatility_regime' in self.df.columns:
            # Combined regime (trend direction + volatility)
            self.df['market_regime'] = self.df['trend_direction'] * (self.df['volatility_regime'] + 1)
    
    def generate_all_features(self) -> pd.DataFrame:
        """Generate all features and return the processed DataFrame."""
        # Standard indicators
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_trend_indicators()
        self.add_volume_indicators()
        
        # Advanced features
        self.add_candlestick_patterns()
        self.add_market_structure_features()
        self.create_advanced_features()
        
        # Return the processed DataFrame
        return self.df

def generate_features(df: pd.DataFrame, feature_sets: Dict[str, List[str]] = None, max_features: int = 36) -> pd.DataFrame:
    """
    Generate features for the given DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        feature_sets: Dictionary of feature sets to generate
        
    Returns:
        DataFrame with generated features
    """
    if feature_sets is None:
        feature_sets = FEATURE_SETS
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(df)
    
    # Generate all features
    processed_df = feature_engineer.generate_all_features()
    
    # Filter columns based on requested feature sets
    if feature_sets:
        # Flatten the list of features
        requested_features = ['timestamp', 'open', 'high', 'low', 'close', 'volume']  # Always include OHLCV
        for feature_set in feature_sets.values():
            requested_features.extend(feature_set)
        
        # Filter columns
        available_columns = processed_df.columns.tolist()
        selected_columns = [col for col in requested_features if col in available_columns]
        
        # Create a new DataFrame with only the selected columns
        processed_df = processed_df[selected_columns]
    
    # Fill any remaining NaN values
    processed_df = processed_df.fillna(method='ffill').fillna(0)
    
    return processed_df

if __name__ == "__main__":
    # Test the feature generation process
    import data_processor
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # List available data
    available_data = data_processor.list_available_data()
    
    # Example: Load data for a symbol and generate features
    if 'BTCUSDT' in available_data and '1h' in available_data['BTCUSDT']:
        # Load data
        df = data_processor.load_data(available_data['BTCUSDT']['1h'])
        
        # Generate features
        processed_df = generate_features(df)
        
        # Show the resulting DataFrame
        print(f"Generated {len(processed_df.columns)} features")
        print(processed_df.columns.tolist())
        print(processed_df.head())