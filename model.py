"""
Machine learning model for price prediction in the AI trading bot.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization, Input,
    Bidirectional, Conv1D, MaxPooling1D, Flatten, Attention,
    MultiHeadAttention, LayerNormalization, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
import json  # Add this for JSON operations
from config import LOOKBACK_WINDOW, PREDICTION_HORIZON
from config import (
    MODEL_TYPE, HIDDEN_LAYERS, DROPOUT_RATE, BATCH_SIZE, EPOCHS, 
    EARLY_STOPPING_PATIENCE, LEARNING_RATE, MODELS_DIR
)

logger = logging.getLogger(__name__)

class DeepLearningModel:
    """
    Deep learning model for price prediction with improved dimension handling.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_dim: int,
        model_type: str = MODEL_TYPE,
        hidden_layers: List[int] = None,
        dropout_rate: float = DROPOUT_RATE,
        learning_rate: float = LEARNING_RATE,
        model_path: Optional[str] = None
    ):
        """
        Initialize the model with improved dimension handling.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            output_dim: Dimension of output (prediction horizon)
            model_type: Type of model to use
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_path: Path to load a pre-trained model
        """
        # Store original parameters
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model_type = model_type
        self.hidden_layers = hidden_layers or HIDDEN_LAYERS
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Store expected feature count for consistency checks
        self.expected_features = input_shape[1]
        
        # Model and training history
        self.model = None
        self.history = None
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            metadata = self.load_model(model_path)
            
            # Update expected features if available in metadata
            if metadata and 'input_shape' in metadata:
                self.expected_features = metadata['input_shape'][1]
                logger.info(f"Updated expected features to {self.expected_features} from model metadata")
        else:
            self._build_model()
    
    
    def _build_lstm_model(self) -> Model:
        """Build LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        if len(self.hidden_layers) > 1:
            # For multiple layers, first LSTM returns sequence output.
            model.add(LSTM(
                self.hidden_layers[0],
                input_shape=self.input_shape,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
            # Additional LSTM layers
            for i, units in enumerate(self.hidden_layers[1:-1], 1):
                return_sequences = i < len(self.hidden_layers) - 2
                model.add(LSTM(
                    units,
                    return_sequences=True,
                    activation='tanh',
                    recurrent_activation='sigmoid'
                ))
                model.add(BatchNormalization())
                model.add(Dropout(self.dropout_rate))


                # Final LSTM layer - do NOT return sequence, so that output shape is (batch_size, units)
            model.add(LSTM(
                self.hidden_layers[-1],
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        else:
            # If only one layer is used, set return_sequences=False directly.
            model.add(LSTM(
                self.hidden_layers[0],
                input_shape=self.input_shape,
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Final dense layers
        if len(self.hidden_layers) > 1:
            model.add(Dense(self.hidden_layers[-1], activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.output_dim, activation='linear'))
        
        return model
    
    def _build_gru_model(self) -> Model:
        """Build GRU model architecture."""
        model = Sequential()
        
        # Determine if we have multiple GRU layers
        num_layers = len(self.hidden_layers)

        # First GRU layer
        if num_layers > 1:
            model.add(GRU(
                self.hidden_layers[0],
                input_shape=self.input_shape,
                return_sequences=True,  # Change to True for multi-layer networks
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
        else:
            model.add(GRU(
                self.hidden_layers[0],
                input_shape=self.input_shape,
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        # Add intermediate GRU layers
        for idx in range(1, num_layers - 1):
            model.add(GRU(
                self.hidden_layers[idx],
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Add final GRU layer (if we have more than one layer)
        if num_layers > 1:
            model.add(GRU(
                self.hidden_layers[-1],
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Final dense layers
        model.add(Dense(self.hidden_layers[-1] // 2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.output_dim, activation='linear'))
        
        return model
    
    def _build_cnn_model(self) -> Model:
        """Build CNN model architecture."""
        model = Sequential()
        
        # CNN layers
        model.add(Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        # Flatten and dense layers
        model.add(Flatten())
        
        for units in self.hidden_layers:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.output_dim, activation='linear'))
        
        return model
    
    def _build_transformer_model(self) -> Model:
        """Build Transformer model architecture."""
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Add positional encoding
        x = inputs
        
        # Transformer blocks
        for _ in range(3):  # 3 transformer blocks
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=8, key_dim=64
            )(x, x, x)
            
            # Skip connection and layer normalization
            x = LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network
            ffn = Sequential([
                Dense(self.hidden_layers[0], activation='relu'),
                Dense(self.input_shape[1])
            ])
            
            # Skip connection and layer normalization
            x = LayerNormalization(epsilon=1e-6)(ffn(x) + x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for units in self.hidden_layers:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.output_dim, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def _build_ensemble_model(self) -> Model:
        """Build ensemble model combining LSTM, GRU, and CNN."""
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # LSTM branch
        lstm = LSTM(self.hidden_layers[0], return_sequences=False)(inputs)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(self.dropout_rate)(lstm)
        lstm_output = Dense(self.hidden_layers[-1], activation='relu')(lstm)
        
        # GRU branch
        gru = GRU(self.hidden_layers[0], return_sequences=False)(inputs)
        gru = BatchNormalization()(gru)
        gru = Dropout(self.dropout_rate)(gru)
        gru_output = Dense(self.hidden_layers[-1], activation='relu')(gru)
        
        # CNN branch
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        cnn_output = Dense(self.hidden_layers[-1], activation='relu')(cnn)
        
        # Combine outputs
        combined = Concatenate()([lstm_output, gru_output, cnn_output])
        combined = Dense(self.hidden_layers[-1], activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(self.dropout_rate)(combined)
        
        # Output layer
        outputs = Dense(self.output_dim, activation='linear')(combined)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def _build_model(self) -> None:
        """Build the model based on the selected type."""
        logger.info(f"Building {self.model_type} model with input shape {self.input_shape}")
        
        # Extract feature count from input shape
        # self.input_shape is a tuple like (lookback_window, num_features)
        num_features = self.input_shape[1]

            # Adjust hidden layer sizes based on feature count if needed
        if num_features > 50:
            # For high-dimensional inputs, use larger hidden layers
            adjusted_hidden_layers = [int(size * 1.5) for size in self.hidden_layers]
            self.hidden_layers = adjusted_hidden_layers
            logger.info(f"Adjusted hidden layers for large feature set: {self.hidden_layers}")
        elif num_features < 10:
            # For low-dimensional inputs, use smaller hidden layers
            adjusted_hidden_layers = [max(int(size * 0.5), 16) for size in self.hidden_layers]
            self.hidden_layers = adjusted_hidden_layers
            logger.info(f"Adjusted hidden layers for small feature set: {self.hidden_layers}")
            
          # Build model based on selected type

        if self.model_type == 'lstm':
            self.model = self._build_lstm_model()
        elif self.model_type == 'gru':
            self.model = self._build_gru_model()
        elif self.model_type == 'cnn':
            self.model = self._build_cnn_model()
        elif self.model_type == 'transformer':
            self.model = self._build_transformer_model()
        elif self.model_type == 'ensemble':
            self.model = self._build_ensemble_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=Huber(),
            metrics=['mae', 'mse']
        )
        
        # Summary
        self.model.summary()
    
    def _check_input_dimensions(self, X: np.ndarray) -> np.ndarray:
        """
        Check and adjust input dimensions if needed.
        
        Args:
            X: Input features with shape (samples, sequence_length, features)
            
        Returns:
            Adjusted input features
        """
        # Check if dimensions match
        if X.shape[2] != self.expected_features:
            logger.warning(f"Input feature dimension mismatch: got {X.shape[2]}, expected {self.expected_features}")
            
            # Create adjusted array
            X_adjusted = np.zeros((X.shape[0], X.shape[1], self.expected_features))
            
            # Copy data for common dimensions
            common_dims = min(X.shape[2], self.expected_features)
            X_adjusted[:, :, :common_dims] = X[:, :, :common_dims]
            
            return X_adjusted
        
        return X


    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs to train
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            save_path: Path to save the model
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")
        
        logger.info(f"Training {self.model_type} model with {len(X_train)} samples")
        
        # Create callbacks if not provided
        if callbacks is None:
            callbacks = []
            
            # Early stopping
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ))
            
            # Reduce learning rate on plateau
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=EARLY_STOPPING_PATIENCE // 2,
                min_lr=1e-6
            ))
            
            # TensorBoard logging
            log_dir = os.path.join('logs', f"{self.model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
            callbacks.append(TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            ))
            
            # Save best model
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                callbacks.append(ModelCheckpoint(
                    save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                ))
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save history
        self.history = history.history
        
        return self.history
    
    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        Make predictions with enhanced dimension checking and error handling.
        
        Args:
            X: Input features with shape (samples, sequence_length, features)
            verbose: Verbosity level for prediction
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")
        
        # Extensive dimension checking
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (samples, sequence_length, features), got shape {X.shape}")
        
        # Check sequence length (lookback window)
        expected_sequence_length = self.input_shape[0]
        if X.shape[1] != expected_sequence_length:
            logger.warning(f"Input sequence length mismatch: got {X.shape[1]}, expected {expected_sequence_length}")
            if X.shape[1] < expected_sequence_length:
                # Pad sequence if too short
                padding = np.zeros((X.shape[0], expected_sequence_length - X.shape[1], X.shape[2]))
                X_padded = np.concatenate([padding, X], axis=1)
                logger.warning(f"Input padded to shape {X_padded.shape}")
                X = X_padded
            else:
                # Truncate if too long
                X = X[:, -expected_sequence_length:, :]
                logger.warning(f"Input truncated to shape {X.shape}")
        
        # Check and adjust feature dimensions
        expected_features = self.input_shape[1]
        if X.shape[2] != expected_features:
            logger.warning(f"Feature dimension mismatch: got {X.shape[2]}, expected {expected_features}")
            X_adjusted = np.zeros((X.shape[0], X.shape[1], expected_features))
            
            # Copy over available features
            common_features = min(X.shape[2], expected_features)
            X_adjusted[:, :, :common_features] = X[:, :, :common_features]
            
            logger.warning(f"Input adjusted to shape {X_adjusted.shape}")
            X = X_adjusted
        
        # Make prediction with try/except
        try:
            return self.model.predict(X, verbose=verbose)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(f"Input shape: {X.shape}, expected shape: {self.input_shape}")
            # Return zeros as fallback
            return np.zeros((X.shape[0], self.output_dim))
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")
        
        # Evaluate model
        loss, mae, mse = self.model.evaluate(X, y, verbose=0)
        
        # Return metrics
        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
    
    def save_model(self, path: str, feature_columns: List[str] = None) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            feature_columns: Optional list of feature column names
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        
        # Change file extension from .h5 to .keras if needed
        if path.endswith('.h5'):
            keras_path = path.replace('.h5', '.keras')
        else:
            keras_path = path
        
        # Save model in new format
        self.model.save(keras_path)

        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'output_dim': self.output_dim,
            'model_type': self.model_type,
            'feature_count': self.expected_features,
            'datetime_saved': pd.Timestamp.now().isoformat()
        }
        
        # Add feature columns if provided
        if feature_columns is not None:  # Fixed: proper None check
            metadata['feature_count'] = len(feature_columns)
            metadata['feature_names'] = feature_columns
        
        # Save metadata
        metadata_path = f"{path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Model saved to {path} with metadata at {metadata_path}")
    
    def load_model(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model
        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")

                # Try to load metadata
        metadata = {}
        metadata_path = f"{path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model loaded from {path} with metadata")
            
            # Verify input shape compatibility
            if 'input_shape' in metadata:
                loaded_shape = tuple(metadata['input_shape'])
                if loaded_shape[1] != self.input_shape[1]:  # Compare feature dimensions
                    logger.warning(f"Input shape mismatch: model expects {loaded_shape[1]} features, but {self.input_shape[1]} provided.")
        else:
            logger.warning(f"No metadata found for model at {path}")
        
        return metadata

class ModelManager:
    """
    Manager class for handling multiple models.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the model manager.
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = base_dir or MODELS_DIR
        self.models = {}
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_model(
        self,
        symbol: str,
        timeframe: str,
        input_shape: Tuple[int, int],
        output_dim: int,
        model_type: str = MODEL_TYPE,
        hidden_layers: List[int] = None,
        dropout_rate: float = DROPOUT_RATE,
        learning_rate: float = LEARNING_RATE
    ) -> DeepLearningModel:
        """
        Create a new model.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the model
            input_shape: Shape of input data
            output_dim: Dimension of output
            model_type: Type of model to use
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Created model
        """
        model_id = f"{symbol}_{timeframe}"
        
        # Check if model exists
        if model_id in self.models:
            logger.warning(f"Model {model_id} already exists and will be replaced")
        
        # Create model
        model = DeepLearningModel(
            input_shape=input_shape,
            output_dim=output_dim,
            model_type=model_type,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Store model
        self.models[model_id] = model
        
        return model
    
    def get_model(self, symbol: str, timeframe: str) -> Optional[DeepLearningModel]:
        """
        Get a model by symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the model
            
        Returns:
            Model if found, None otherwise
        """
        model_id = f"{symbol}_{timeframe}"
        return self.models.get(model_id)
    
    def load_model(self, symbol: str, timeframe: str) -> Optional[DeepLearningModel]:
        """
        Load a model from disk.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the model
            
        Returns:
            Loaded model if found, None otherwise
        """
        model_id = f"{symbol}_{timeframe}"
        model_path = os.path.join(self.base_dir, f"{model_id}.h5")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        # Create model with dummy shape (will be replaced by loaded model)
        model = DeepLearningModel(
            input_shape=(1, 1),
            output_dim=1,
            model_path=model_path
        )
        
        # Store model
        self.models[model_id] = model
        
        return model
    
    def save_model(self, symbol: str, timeframe: str) -> None:
        """
        Save a model to disk.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the model
        """
        model_id = f"{symbol}_{timeframe}"
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Create save path
        model_path = os.path.join(self.base_dir, f"{model_id}.h5")
        
        # Save model
        self.models[model_id].save_model(model_path)
    
    def save_all_models(self) -> None:
        """Save all models to disk."""
        for model_id in self.models:
            symbol, timeframe = model_id.split('_')
            self.save_model(symbol, timeframe)
    
    def load_all_models(self) -> None:
        """Load all models from disk."""
        # Find all model files
        if not os.path.exists(self.base_dir):
            logger.warning(f"Model directory not found: {self.base_dir}")
            return
        
        for filename in os.listdir(self.base_dir):
            if filename.endswith('.h5'):
                model_id = filename[:-3]  # Remove .h5 extension
                
                try:
                    symbol, timeframe = model_id.split('_')
                    self.load_model(symbol, timeframe)
                except ValueError:
                    logger.warning(f"Invalid model filename: {filename}")
    
    def delete_model(self, symbol: str, timeframe: str) -> None:
        """
        Delete a model.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the model
        """
        model_id = f"{symbol}_{timeframe}"
        
        # Remove from memory
        if model_id in self.models:
            del self.models[model_id]
        
        # Remove from disk
        model_path = os.path.join(self.base_dir, f"{model_id}.h5")
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Model file deleted: {model_path}")

if __name__ == "__main__":
    # Test the model
    import numpy as np
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    X = np.random.random((100, 50, 10))  # 100 samples, 50 time steps, 10 features
    y = np.random.random((100, 5))  # 100 samples, 5 prediction horizon
    
    # Split data
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    # Create model
    model = DeepLearningModel(
        input_shape=(LOOKBACK_WINDOW, 36),  # 55 features instead of 36
        output_dim=PREDICTION_HORIZON,
        model_type='lstm',
        hidden_layers=[128, 64],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # Train model
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=5,
        batch_size=16
    )
    
    # Evaluate model
    metrics = model.evaluate(X_val, y_val)
    print(f"Evaluation metrics: {metrics}")
    
    # Make predictions
    predictions = model.predict(X_val)
    print(f"Predictions shape: {predictions.shape}")