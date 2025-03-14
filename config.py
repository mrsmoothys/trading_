"""
Configuration module for the AI trading bot.
Contains all configurable parameters for the system.
"""
import os
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/mrsmoothy/Desktop/rsidtrade/trading_")
DATA_DIR = Path("/Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets")
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Data processing
TIMEFRAMES = ["1h", "4h", "1d"]  # Supported timeframes
COLUMNS = [
    "Open time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
]
RENAME_MAP = {
    "Open time": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Close time": "close_time",
    "Quote asset volume": "quote_volume",
    "Number of trades": "trades",
    "Taker buy base asset volume": "taker_buy_base",
    "Taker buy quote asset volume": "taker_buy_quote",
    "Ignore": "ignore"
}
ESSENTIAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Trading parameters
TRADING_FEE = 0.0002  # 0.02%
SLIPPAGE = 0.002      # 0.2%
INITIAL_CAPITAL = 10000  # Starting capital for backtests
POSITION_SIZE = 0.1   # Percentage of capital to risk per trade
MAX_POSITIONS = 5     # Maximum number of simultaneous positions

# Model parameters
LOOKBACK_WINDOW = 100  # Number of candles to consider for prediction
PREDICTION_HORIZON = 5  # Number of future candles to predict
FEATURE_SETS = {
    "price": ["open", "high", "low", "close", "volume"],
    "indicators": [
        "rsi", "macd", "macd_signal", "macd_hist", "bollinger_upper", 
        "bollinger_middle", "bollinger_lower", "atr", "sma_5", "sma_20", 
        "sma_50", "ema_5", "ema_20", "ema_50", "adx", "cci", "obv", 
        "mfi", "stoch_k", "stoch_d", "williams_r", "ultimate_oscillator"
    ],
    "patterns": [
        "engulfing", "hammer", "shooting_star", "doji", "morning_star", 
        "evening_star", "three_white_soldiers", "three_black_crows"
    ],
    "market_structure": [
        "swing_high", "swing_low", "trend", "support", "resistance", 
        "consolidation", "volatility_regime"
    ]
}

# Model architecture
MODEL_TYPE = "lstm"  # Options: "lstm", "gru", "cnn", "transformer", "ensemble"
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
EPOCHS = 1
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001

# Optimization
OPTIMIZATION_METRIC = "sharpe_ratio"  # Options: "sharpe_ratio", "sortino_ratio", "profit", "win_rate"
OPTIMIZATION_TRIALS = 100
CROSS_VALIDATION_FOLDS = 5

# Stop loss/Take profit
ADAPTIVE_SL_TP = True  # Use adaptive stop loss and take profit
TRAILING_STOP = True   # Use trailing stop
FIXED_SL_PERCENTAGE = 0.03  # 3% stop loss if adaptive is disabled
FIXED_TP_PERCENTAGE = 0.05  # 5% take profit if adaptive is disabled
ATR_SL_MULTIPLIER = 2.0  # Stop loss at 2x ATR if using ATR-based stops
MIN_RISK_REWARD_RATIO = 1.5  # Minimum risk-reward ratio for trades

# Logging and visualization
LOG_LEVEL = "INFO"
SAVE_PLOTS = True
PLOT_TYPES = ["equity_curve", "drawdown", "trade_distribution", "monthly_returns", "learning_progress"]

# Current run metadata
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_METADATA = {
    "id": RUN_ID,
    "timestamp": datetime.now().isoformat(),
    "description": "AI Trading Bot Run"
}

# Load user-specific config if exists
USER_CONFIG_PATH = BASE_DIR / "user_config.py"
if USER_CONFIG_PATH.exists():
    try:
        from user_config import *  # noqa
    except ImportError:
        pass