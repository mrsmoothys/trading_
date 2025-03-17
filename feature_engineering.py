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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

    def apply_pca(self, n_components=30):
        """
        Apply PCA to reduce feature dimensionality.
        
        Args:
            n_components: Number of principal components to retain
            
        Returns:
            DataFrame with reduced features
        """
        reduced_df, self.pca, self.scaler = apply_pca_reduction(self.df, n_components)
        self.df = reduced_df
        return self.df
    
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
        
        # After all features are generated, before returning
        print("\n===== FEATURES GENERATED =====")
        for i, feature in enumerate(self.df.columns):
            if feature not in ['open', 'high', 'low', 'close', 'volume']:  # Skip price columns
                print(f"{i+1}. {feature}")
        print(f"Total features: {len(self.df.columns)}")
        print("=============================\n")
        

        # Return the processed DataFrame
        return self.df
    



def apply_pca_reduction(data, n_components=30):
    # Force n_components to exactly 30 for consistency
    n_components = 30
    
    # Separate OHLCV columns
    ohlcv = data[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Get feature columns only (exclude OHLCV and non-numeric columns)
    feature_cols = []
    for col in data.columns:
        if col not in ['open', 'high', 'low', 'close', 'volume']:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(data[col]):
                feature_cols.append(col)
            else:
                logging.warning(f"Excluding non-numeric column from PCA: {col}")
                # Convert to numeric if possible
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    feature_cols.append(col)
                except:
                    pass
    
    # Ensure we have features to process
    if not feature_cols:
        logging.warning("No numeric features available for PCA")
        return data, None, None
    
    # Get features and handle extreme values
    features = data[feature_cols].copy()
    
    # Replace infinity with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN values with column means
    for col in features.columns:
        mean_val = features[col].mean()
        if pd.isna(mean_val):  # If mean is NaN (all values are NaN)
            features[col] = 0  # Replace with zero
        else:
            features[col] = features[col].fillna(mean_val)
    
    # Standardize features - with explicit type checking
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    except Exception as e:
        logging.error(f"Error during standardization: {e}")
        # Check for any string values
        for col in features.columns:
            if features[col].dtype == object:
                logging.error(f"Column {col} contains string values")
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        # Try again
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(features_scaled)
    
    # Create component dataframe with exactly n_components
    component_df = pd.DataFrame(
        components, 
        columns=[f'pc_{i+1}' for i in range(n_components)],
        index=data.index
    )
    
    # Combine with OHLCV
    reduced_data = pd.concat([ohlcv, component_df], axis=1)
    
    return reduced_data, pca, scaler


def generate_features(df: pd.DataFrame, feature_sets: Dict[str, List[str]] = None, apply_pca: bool = False, n_components: int = 30) -> pd.DataFrame:
    """
    Generate features with dimensional consistency.
    """
    if feature_sets is None:
        feature_sets = FEATURE_SETS
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(df)
    
    # Generate standard indicators first
    processed_df = feature_engineer.generate_all_features()
    
    # Apply new feature functions
    processed_df = detect_candlestick_patterns(processed_df)
    processed_df = detect_market_structure(processed_df)
    processed_df = enhance_trend_features(processed_df)
    processed_df = add_volatility_features(processed_df)
    
    # Register all features to maintain consistency
    feature_registry = {}
    processed_df, feature_registry = register_features(processed_df, feature_registry)
    
    # Before PCA, ensure we have consistent handling
    if apply_pca:
        logger.info(f"Applying PCA to reduce features to {n_components} components")
        processed_df, pca_model, scaler_model = apply_pca_reduction(processed_df, n_components=n_components)
        
        # Ensure exactly n_components after PCA
        processed_df = ensure_feature_consistency(processed_df, required_feature_count=n_components)
        
        return processed_df, pca_model, scaler_model
    else:
        # Without PCA, select a consistent feature set based on importance
        feature_cols = [col for col in processed_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'signal']]
        
        # If we have too many features, select most important from each category
        if len(feature_cols) > n_components:
            selected_features = []
            # Select some features from each category
            for category, features in feature_registry.items():
                # Take up to a proportional number from each category
                category_limit = max(1, int(n_components * len(features) / len(feature_cols)))
                selected_features.extend(features[:category_limit])
            
            # If still too many, truncate
            if len(selected_features) > n_components:
                selected_features = selected_features[:n_components]
            
            # If too few, add more generic features
            while len(selected_features) < n_components:
                remaining = [f for f in feature_cols if f not in selected_features]
                if not remaining:
                    break
                selected_features.append(remaining[0])
            
            # Create a new dataframe with only selected features plus OHLCV
            columns_to_keep = ['open', 'high', 'low', 'close', 'volume'] + selected_features
            processed_df = processed_df[columns_to_keep]
        
        # Ensure exact feature count
        processed_df = ensure_feature_consistency(processed_df, required_feature_count=n_components)
        
        return processed_df

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern recognition features to the dataframe."""
    # Create copies of price data for clarity
    open = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Determine candle body and shadow sizes
    body_size = abs(close - open)
    upper_shadow = high - np.maximum(close, open)
    lower_shadow = np.minimum(close, open) - low
    
    # Doji (small body, shadows significantly larger)
    df['doji'] = (body_size < (0.1 * (high - low))).astype(int)
    
    # Hammer (small body at top, long lower shadow)
    df['hammer'] = ((body_size < 0.3 * (high - low)) & 
                   (lower_shadow > 2 * body_size) & 
                   (upper_shadow < 0.2 * lower_shadow)).astype(int)
    
    # Shooting Star (small body at bottom, long upper shadow)
    df['shooting_star'] = ((body_size < 0.3 * (high - low)) & 
                          (upper_shadow > 2 * body_size) & 
                          (lower_shadow < 0.2 * upper_shadow)).astype(int)
    
    # Engulfing patterns (multi-candle patterns)
    bullish_engulfing = ((open.shift(1) > close.shift(1)) &  # Previous candle is bearish
                         (close > open) &  # Current candle is bullish
                         (open <= close.shift(1)) &  # Current open below previous close
                         (close >= open.shift(1)))  # Current close above previous open
    
    bearish_engulfing = ((open.shift(1) < close.shift(1)) &  # Previous candle is bullish
                         (close < open) &  # Current candle is bearish
                         (open >= close.shift(1)) &  # Current open above previous close
                         (close <= open.shift(1)))  # Current close below previous open
    
    df['bullish_engulfing'] = bullish_engulfing.astype(int)
    df['bearish_engulfing'] = bearish_engulfing.astype(int)
    
    # Morning & Evening Stars (three-candle patterns)
    df['morning_star'] = ((close.shift(2) < open.shift(2)) &  # First candle bearish
                          (abs(close.shift(1) - open.shift(1)) < 0.3 * body_size.shift(2)) &  # Second candle small
                          (close > open) &  # Third candle bullish
                          (close > (open.shift(2) + close.shift(2))/2)).astype(int)  # Closed above midpoint of first
    
    df['evening_star'] = ((close.shift(2) > open.shift(2)) &  # First candle bullish
                          (abs(close.shift(1) - open.shift(1)) < 0.3 * body_size.shift(2)) &  # Second candle small
                          (close < open) &  # Third candle bearish
                          (close < (open.shift(2) + close.shift(2))/2)).astype(int)  # Closed below midpoint of first
    
    return df


def detect_market_structure(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add market structure detection features."""
    # Identify swing highs and lows
    df['swing_high'] = 0
    df['swing_low'] = 0
    
    for i in range(window, len(df) - window):
        # Check for swing high
        if all(df['high'].iloc[i] > df['high'].iloc[i-window:i]) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+1:i+window+1]):
            df.loc[df.index[i], 'swing_high'] = 1
        
        # Check for swing low
        if all(df['low'].iloc[i] < df['low'].iloc[i-window:i]) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+1:i+window+1]):
            df.loc[df.index[i], 'swing_low'] = 1
    
    # Identify trend direction based on swing points
    df['higher_highs'] = 0
    df['higher_lows'] = 0
    df['lower_highs'] = 0
    df['lower_lows'] = 0
    
    # Get arrays of swing points
    swing_high_indices = df.index[df['swing_high'] == 1]
    swing_low_indices = df.index[df['swing_low'] == 1]
    
    # Analyze sequence of swing highs
    for i in range(1, len(swing_high_indices)):
        current_idx = swing_high_indices[i]
        prev_idx = swing_high_indices[i-1]
        if df.loc[current_idx, 'high'] > df.loc[prev_idx, 'high']:
            df.loc[current_idx, 'higher_highs'] = 1
        else:
            df.loc[current_idx, 'lower_highs'] = 1
    
    # Analyze sequence of swing lows
    for i in range(1, len(swing_low_indices)):
        current_idx = swing_low_indices[i]
        prev_idx = swing_low_indices[i-1]
        if df.loc[current_idx, 'low'] > df.loc[prev_idx, 'low']:
            df.loc[current_idx, 'higher_lows'] = 1
        else:
            df.loc[current_idx, 'lower_lows'] = 1
    
    # Identify support and resistance zones
    df['at_support'] = 0
    df['at_resistance'] = 0
    
    # Create a price distance measure from recent swing points
    for i in range(window, len(df)):
        # Look back for recent swing lows for support
        for j in range(i-1, max(0, i-50), -1):
            if df.loc[df.index[j], 'swing_low'] == 1:
                support_level = df.loc[df.index[j], 'low']
                if abs(df.loc[df.index[i], 'low'] - support_level) / support_level < 0.01:
                    df.loc[df.index[i], 'at_support'] = 1
                break
        
        # Look back for recent swing highs for resistance
        for j in range(i-1, max(0, i-50), -1):
            if df.loc[df.index[j], 'swing_high'] == 1:
                resistance_level = df.loc[df.index[j], 'high']
                if abs(df.loc[df.index[i], 'high'] - resistance_level) / resistance_level < 0.01:
                    df.loc[df.index[i], 'at_resistance'] = 1
                break
    
    return df


def enhance_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced trend detection features."""
    # Multiple timeframe trend alignment
    # Short-term trend
    df['sma_20'] = df['close'].rolling(20).mean()
    # Medium-term trend
    df['sma_50'] = df['close'].rolling(50).mean()
    # Long-term trend
    df['sma_200'] = df['close'].rolling(200).mean()
    
    # Trend alignment score (+1 for each aligned trend direction)
    df['trend_alignment'] = 0
    
    # Short-term trend direction
    df['trend_short'] = (df['close'] > df['sma_20']).astype(int) * 2 - 1
    
    # Medium-term trend direction
    df['trend_medium'] = (df['close'] > df['sma_50']).astype(int) * 2 - 1
    
    # Long-term trend direction
    df['trend_long'] = (df['close'] > df['sma_200']).astype(int) * 2 - 1
    
    # Calculate alignment (values from -3 to +3)
    df['trend_alignment'] = df['trend_short'] + df['trend_medium'] + df['trend_long']
    
    # Strong uptrend indicator (all timeframes aligned up)
    df['strong_uptrend'] = (df['trend_alignment'] == 3).astype(int)
    
    # Strong downtrend indicator (all timeframes aligned down)
    df['strong_downtrend'] = (df['trend_alignment'] == -3).astype(int)
    
    # Trend reversal warning: short-term opposes longer-term trends
    df['reversal_warning'] = ((df['trend_short'] != df['trend_long']) & 
                             (df['trend_medium'] == df['trend_long'])).astype(int)
    
    # Trend strength using ADX
    from ta.trend import ADXIndicator
    adx_indicator = ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx_indicator.adx()
    df['adx_pos'] = adx_indicator.adx_pos()
    df['adx_neg'] = adx_indicator.adx_neg()
    
    # Trend strength categories
    df['strong_trend'] = (df['adx'] > 25).astype(int)
    df['very_strong_trend'] = (df['adx'] > 50).astype(int)
    df['weak_trend'] = ((df['adx'] > 10) & (df['adx'] <= 25)).astype(int)
    
    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-based features."""
    # ATR calculation
    from ta.volatility import AverageTrueRange
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Normalized ATR (as percentage of price)
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # Volatility regime detection
    df['volatility_regime'] = 0  # 0: normal, 1: high, -1: low
    
    # Calculate rolling volatility using standard deviation of returns
    df['returns'] = df['close'].pct_change()
    df['rolling_vol'] = df['returns'].rolling(20).std() * np.sqrt(20)  # Annualized
    
    # Calculate historical percentiles for volatility
    vol_75th = df['rolling_vol'].quantile(0.75)
    vol_25th = df['rolling_vol'].quantile(0.25)
    
    # Identify volatility regimes
    df.loc[df['rolling_vol'] > vol_75th, 'volatility_regime'] = 1  # High volatility
    df.loc[df['rolling_vol'] < vol_25th, 'volatility_regime'] = -1  # Low volatility
    
    # Volatility expansion/contraction
    df['vol_expansion'] = (df['rolling_vol'] > df['rolling_vol'].shift(5)).astype(int)
    df['vol_contraction'] = (df['rolling_vol'] < df['rolling_vol'].shift(5)).astype(int)
    
    # Bollinger Bands width as volatility measure
    from ta.volatility import BollingerBands
    bb = BollingerBands(df['close'])
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    # Bollinger Band squeeze (narrowing bands)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
    
    return df


def register_features(df: pd.DataFrame, feature_registry: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Register all features in the dataframe to maintain consistency.
    
    Args:
        df: DataFrame with features
        feature_registry: Registry of feature categories
        
    Returns:
        Updated dataframe and feature registry
    """
    # Current feature columns (excluding OHLCV)
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'signal']]
    
    # Categorize features
    candlestick_features = [col for col in feature_cols if col in [
        'doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing',
        'morning_star', 'evening_star'
    ]]
    
    market_structure_features = [col for col in feature_cols if col in [
        'swing_high', 'swing_low', 'higher_highs', 'higher_lows', 'lower_highs', 'lower_lows',
        'at_support', 'at_resistance'
    ]]
    
    trend_features = [col for col in feature_cols if col in [
        'trend_short', 'trend_medium', 'trend_long', 'trend_alignment',
        'strong_uptrend', 'strong_downtrend', 'reversal_warning',
        'adx', 'adx_pos', 'adx_neg', 'strong_trend', 'very_strong_trend', 'weak_trend'
    ]]
    
    volatility_features = [col for col in feature_cols if col in [
        'atr', 'atr_pct', 'volatility_regime', 'rolling_vol', 'vol_expansion', 
        'vol_contraction', 'bb_width', 'bb_squeeze'
    ]]
    
    # Update registry
    feature_registry['candlestick'] = candlestick_features
    feature_registry['market_structure'] = market_structure_features
    feature_registry['trend'] = trend_features
    feature_registry['volatility'] = volatility_features
    
    return df, feature_registry





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