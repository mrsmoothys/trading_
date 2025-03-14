import tensorflow as tf
import numpy as np
import time

# Import Keras submodules from tensorflow.keras
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError



# Print TensorFlow version and device info
print(f"TensorFlow version: {tf.__version__}")
print(f"Physical devices: {tf.config.list_physical_devices()}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data
def generate_data(n_samples=1000):
    X = np.random.rand(n_samples, 10).astype(np.float32)
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, n_samples).astype(np.float32)
    return X, y

# Custom callback for epoch timing
class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimingCallback, self).__init__()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1} took {epoch_time:.3f} seconds")

# Build and train the model
def build_and_train_model():
    X, y = generate_data()
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Define model (slightly smaller for CPU efficiency)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),  # Reduced from 64
        Dense(16, activation='relu'),                    # Reduced from 32
        Dense(1)
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=['mae']
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    timing_callback = TimingCallback()

    # Train model
    print("Starting model training (CPU only)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, timing_callback],
        verbose=1
    )

    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    return history, timing_callback.epoch_times, test_loss, test_mae

# Print results
def print_results(history, epoch_times, test_loss, test_mae):
    print("\n=== Training Results ===")
    print(f"Total epochs run: {len(history.history['loss'])}")
    print(f"Average epoch time: {np.mean(epoch_times):.3f} seconds")
    print(f"Minimum epoch time: {np.min(epoch_times):.3f} seconds")
    print(f"Maximum epoch time: {np.max(epoch_times):.3f} seconds")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

if __name__ == "__main__":
    # Confirm CPU-only execution
    if not tf.config.list_physical_devices('GPU'):
        print("Running on CPU (No Metal GPU detected)")
    
    # Run the test
    history, epoch_times, test_loss, test_mae = build_and_train_model()
    print_results(history, epoch_times, test_loss, test_mae)