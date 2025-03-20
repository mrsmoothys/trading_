import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

import numpy as np
import pandas as pd
import logging
import os
from universal_model import UniversalModel
from data_processor import load_data 
from feature_engineering import generate_features, apply_pca_reduction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
model_path = "/Users/mrsmoothy/Desktop/rsidtrade/trading_/models/universal_model/universal_model_20250319_131443"
data_path = "/Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv"
symbol = "BTCUSDT"
timeframe = "1h"
lookback_window = 60

# Load data and prepare features
logger.info("Loading data...")
df = load_data(data_path)
df_features = generate_features(df)
df_pca, _, _ = apply_pca_reduction(df_features, n_components=30)

# Load model
logger.info(f"Loading model from {model_path}...")
if os.path.exists(model_path):
    model = UniversalModel(model_path=model_path)
else:
    logger.error(f"Model path not found: {model_path}")
    exit(1)

# Prepare a small batch of data for testing
logger.info("Preparing test data...")
X_test = df_pca.values[-100:, :]
if len(X_test.shape) == 2:
    X_test = X_test.reshape(-1, lookback_window, X_test.shape[-1] // lookback_window)

# Try prediction
logger.info("Testing prediction...")
try:
    preds = model.predict(X_test, symbol, timeframe, verbose=1)
    logger.info(f"Prediction successful! Shape: {preds.shape}")
    logger.info(f"First few predictions: {preds[:3]}")
except Exception as e:
    logger.error(f"Prediction failed: {str(e)}")