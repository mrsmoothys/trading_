"""
Universal model module for training across multiple symbols and timeframes.
"""
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten
from tensorflow.keras.models import Model

from config import MODELS_DIR, LOOKBACK_WINDOW
from model import DeepLearningModel
from data_processor import load_data, create_training_sequences
from feature_engineering import generate_features, apply_pca_reduction

logger = logging.getLogger(__name__)

class UniversalModel:
    """
    Universal model class that can be trained on multiple symbols and timeframes.
    """
    
    def __init__(
        self,
        lookback_window: int = LOOKBACK_WINDOW,
        prediction_horizon: int = 5,
        feature_count: int = 30,  # PCA components or feature count
        hidden_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        model_path: Optional[str] = None
    ):
        """
        Initialize the universal model.
        
        Args:
            lookback_window: Number of candles to consider for prediction
            prediction_horizon: Number of future candles to predict
            feature_count: Number of features for each input sample
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_path: Path to load a pre-trained model
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.feature_count = feature_count
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Track which symbols and timeframes the model has been trained on
        self.trained_pairs = {}
        
        # Symbol mapping (will be populated as symbols are encountered)
        self.symbol_map = {}
        self.next_symbol_id = 0
        
        # Timeframe mapping (minutes)
        self.timeframe_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        
        # Create or load model
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._build_model()
        
        # History of training
        self.training_history = {}
    
    def _build_model(self) -> None:
        """Build the universal model architecture."""
        # Price sequence input
        price_input = Input(shape=(self.lookback_window, self.feature_count), name='price_input')
        
        # Symbol embedding input
        symbol_input = Input(shape=(1,), dtype='int32', name='symbol_input')
        symbol_embedding = Embedding(input_dim=100, output_dim=8)(symbol_input)  # Allow up to 100 symbols
        symbol_embedding = Flatten()(symbol_embedding)
        
        # Timeframe input (numeric)
        timeframe_input = Input(shape=(1,), name='timeframe_input')
        
        # Process price sequence with DeepLearningModel's LSTM
        base_model = DeepLearningModel(
            input_shape=(self.lookback_window, self.feature_count),
            output_dim=16,  # Intermediate representation
            model_type='lstm',
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate
        )
        
        # Get the base model without its output layer
        price_features = base_model.model.layers[-2].output
        
        # Concatenate with symbol and timeframe information
        combined = Concatenate()([price_features, symbol_embedding, timeframe_input])
        
        # Additional layers for combined processing
        x = Dense(64, activation='relu')(combined)
        x = Dense(32, activation='relu')(x)
        
        # Output layer for prediction
        outputs = Dense(self.prediction_horizon, activation='linear')(x)
        
        # Create model
        model = Model(
            inputs=[price_input, symbol_input, timeframe_input],
            outputs=outputs
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("Built universal model architecture")
    
    def _get_symbol_id(self, symbol: str) -> int:
        """
        Get numeric ID for a symbol, creating a new one if needed.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Numeric ID for the symbol
        """
        if symbol not in self.symbol_map:
            self.symbol_map[symbol] = self.next_symbol_id
            self.next_symbol_id += 1
        
        return self.symbol_map[symbol]
    
    def _get_timeframe_minutes(self, timeframe: str) -> float:
        """
        Convert timeframe to minutes for numeric representation.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '15m')
            
        Returns:
            Timeframe in minutes
        """
        if timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        return self.timeframe_map[timeframe]
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        feature_columns: List[str],
        apply_pca: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training the universal model.
        
        Args:
            df: DataFrame with OHLCV and features
            symbol: Trading symbol
            timeframe: Trading timeframe
            feature_columns: List of feature column names
            apply_pca: Whether to apply PCA dimension reduction
            
        Returns:
            Tuple of (X_price, X_symbol, X_timeframe, y) for model training
        """
        if apply_pca:
            # Apply PCA to reduce dimensionality
            reduced_df, _, _ = apply_pca_reduction(df, n_components=self.feature_count)
            feature_columns = [f'pc_{i+1}' for i in range(self.feature_count)]
            df = reduced_df
        
        # Create sequences
        X, y = create_training_sequences(
            df,
            lookback_window=self.lookback_window,
            prediction_horizon=self.prediction_horizon,
            feature_columns=feature_columns,
            target_column='close',
            normalize=True
        )
        
        # Create symbol and timeframe inputs
        symbol_id = self._get_symbol_id(symbol)
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        
        X_symbol = np.full((len(X), 1), symbol_id, dtype=np.int32)
        X_timeframe = np.full((len(X), 1), timeframe_minutes, dtype=np.float32)
        
        return X, X_symbol, X_timeframe, y
    
    def train(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        feature_columns: List[str],
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        apply_pca: bool = True,
        callbacks: List[tf.keras.callbacks.Callback] = None
    ) -> Dict[str, List[float]]:
        """
        Train the universal model on new data.
        
        Args:
            df: DataFrame with OHLCV and features
            symbol: Trading symbol
            timeframe: Trading timeframe
            feature_columns: List of feature column names
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Proportion of data to use for validation
            apply_pca: Whether to apply PCA dimension reduction
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        logger.info(f"Training universal model on {symbol} {timeframe}")
        
        # Prepare data
        X, X_symbol, X_timeframe, y = self.prepare_data(
            df, symbol, timeframe, feature_columns, apply_pca
        )
        
        # Split into train and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        X_symbol_train, X_symbol_val = X_symbol[:split_idx], X_symbol[split_idx:]
        X_timeframe_train, X_timeframe_val = X_timeframe[:split_idx], X_timeframe[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create default callbacks if not provided
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
                )
            ]
        
        # Train model
        history = self.model.fit(
            [X_train, X_symbol_train, X_timeframe_train], y_train,
            validation_data=([X_val, X_symbol_val, X_timeframe_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Update trained pairs
        if symbol not in self.trained_pairs:
            self.trained_pairs[symbol] = []
        
        if timeframe not in self.trained_pairs[symbol]:
            self.trained_pairs[symbol].append(timeframe)
        
        # Store training history
        pair_key = f"{symbol}_{timeframe}"
        self.training_history[pair_key] = history.history
        
        return history.history
    
    def predict(
        self,
        X: np.ndarray,
        symbol: str,
        timeframe: str
    ) -> np.ndarray:
        """
        Make predictions with the universal model.
        
        Args:
            X: Input price features with shape (samples, lookback_window, feature_count)
            symbol: Trading symbol
            timeframe: Trading timeframe
            
        Returns:
            Predicted values
        """
        # Create symbol and timeframe inputs
        symbol_id = self._get_symbol_id(symbol)
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        
        X_symbol = np.full((len(X), 1), symbol_id, dtype=np.int32)
        X_timeframe = np.full((len(X), 1), timeframe_minutes, dtype=np.float32)
        
        # Make predictions
        return self.model.predict([X, X_symbol, X_timeframe])
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the universal model to disk.
        
        Args:
            path: Path to save directory
            
        Returns:
            Path to saved model
        """
        # Create default path if not provided
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(MODELS_DIR, f"universal_model_{timestamp}")
        
        # Create directory
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, "model.h5")
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'feature_count': self.feature_count,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'trained_pairs': self.trained_pairs,
            'symbol_map': self.symbol_map,
            'next_symbol_id': self.next_symbol_id,
            'datetime_saved': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Universal model saved to {path}")
        return path
    
    def _load_model(self, path: str) -> None:
        """
        Load the universal model from disk.
        
        Args:
            path: Path to model directory or file
        """
        # Determine paths
        if os.path.isdir(path):
            model_path = os.path.join(path, "model.h5")
            metadata_path = os.path.join(path, "metadata.json")
        else:
            model_path = path
            metadata_path = f"{path.rsplit('.', 1)[0]}_metadata.json"
        
        # Load model
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Load metadata if available
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Update attributes
                self.lookback_window = metadata.get('lookback_window', self.lookback_window)
                self.prediction_horizon = metadata.get('prediction_horizon', self.prediction_horizon)
                self.feature_count = metadata.get('feature_count', self.feature_count)
                self.hidden_layers = metadata.get('hidden_layers', self.hidden_layers)
                self.dropout_rate = metadata.get('dropout_rate', self.dropout_rate)
                self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                self.trained_pairs = metadata.get('trained_pairs', {})
                self.symbol_map = metadata.get('symbol_map', {})
                self.next_symbol_id = metadata.get('next_symbol_id', 0)
                
                logger.info(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training history and model configuration.
        
        Returns:
            Dictionary with training summary
        """
        return {
            'model_config': {
                'lookback_window': self.lookback_window,
                'prediction_horizon': self.prediction_horizon,
                'feature_count': self.feature_count,
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate
            },
            'trained_symbols': list(self.trained_pairs.keys()),
            'trained_pairs': self.trained_pairs,
            'symbol_map': self.symbol_map,
            'training_history': {
                pair: {
                    'final_loss': history['loss'][-1] if 'loss' in history else None,
                    'final_val_loss': history['val_loss'][-1] if 'val_loss' in history else None,
                    'final_mae': history['mae'][-1] if 'mae' in history else None
                }
                for pair, history in self.training_history.items()
            }
        }