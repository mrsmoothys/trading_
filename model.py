"""
Machine learning model for price prediction in the AI trading bot.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
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

from config import (
    MODEL_TYPE, HIDDEN_LAYERS, DROPOUT_RATE, BATCH_SIZE, EPOCHS, 
    EARLY_STOPPING_PATIENCE, LEARNING_RATE, MODELS_DIR
)

logger = logging.getLogger(__name__)

class DeepLearningModel:
    """
    Deep learning model for price prediction.
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
        Initialize the model.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            output_dim: Dimension of output (prediction horizon)
            model_type: Type of model to use
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            model_path: Path to load a pre-trained model
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model_type = model_type
        self.hidden_layers = hidden_layers or HIDDEN_LAYERS
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Model and training history
        self.model = None
        self.history = None
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_lstm_model(self) -> Model:
        """Build LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.hidden_layers[0],
            input_shape=self.input_shape,  # Use dynamic input shape from parameters
            return_sequences=len(self.hidden_layers) > 1,
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
                return_sequences=return_sequences,
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
        
        # First GRU layer
        model.add(GRU(
            self.hidden_layers[0],
            input_shape=self.input_shape,
            return_sequences=False,  # Force this to False for now
            activation='tanh',
            recurrent_activation='sigmoid'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Additional GRU layers
        for i, units in enumerate(self.hidden_layers[1:-1], 1):
            return_sequences = i < len(self.hidden_layers) - 2
            model.add(GRU(
                units,
                return_sequences=return_sequences,
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")
        
        return self.model.predict(X)
    
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
    
    def save_model(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
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
        input_shape=(50, 10),
        output_dim=5,
        model_type='lstm',
        hidden_layers=[64, 32],
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