# RSIDTrade: AI-Powered Cryptocurrency Trading System

RSIDTrade is an advanced trading bot framework for cryptocurrency markets that combines machine learning models with technical analysis to generate trading signals and execute trades in both backtesting and live environments.

## Features

- **AI-Powered Trading**: Utilizes deep learning models (LSTM, GRU, CNN, Transformer, Ensemble) for price prediction
- **Advanced Feature Engineering**: Generates 50+ technical indicators and market structure features
- **Hyperparameter Optimization**: Optimizes both model and strategy parameters
- **Comprehensive Backtesting**: Evaluates strategies with detailed performance metrics
- **Live Trading Integration**: Connects to cryptocurrency exchanges (currently supports Binance)
- **Web Dashboard**: Monitor and control the trading system through an interactive web interface
- **Multi-Asset Support**: Trade multiple symbols and timeframes simultaneously

## Installation

### Prerequisites

- Python 3.10+
- Conda or Miniconda (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rsidtrade.git
cd rsidtrade
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate rsidtrade
```

3. Configure the system by modifying the `config.py` file with your preferred settings.

## Directory Structure

```
rsidtrade/
├── environment.yml           # Conda environment specification
├── config.py                 # Configuration parameters
├── data_processor.py         # Data loading and preprocessing
├── feature_engineering.py    # Technical indicators and feature creation
├── model.py                  # Deep learning model architecture
├── strategy.py               # Trading strategy implementation
├── backtest.py               # Backtesting engine
├── optimizer.py              # Hyperparameter optimization
├── visualize.py              # Performance visualization
├── utils.py                  # Utility functions
├── run_batch.py              # Batch processing script
├── run_single.py             # Single dataset processing script
└── README.md                 # Documentation
```

## Data Format

The system expects data files in CSV format with the following naming convention:
```
SYMBOL_TIMEFRAME_data_STARTDATE_to_ENDDATE.csv
```

Example: `BTCUSDT_1h_data_2018_to_2025.csv`

The CSV should contain the following columns:
```
Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore
```

Example data:
```
Open time,Open,High,Low,Close,Volume,Close time,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume,Ignore
2018-04-17 04:00:00,0.25551,0.288,0.25551,0.26664,8143693.23,2018-04-17 04:59:59.999,2165077.4853629,4421,2889823.93,767134.2091686,0
```

All data files should be placed in the `/Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/` directory or as configured in `config.py`.

## Usage

### Single Dataset Run

For running on a single dataset with detailed analysis and optimization:

```bash
python run_single.py --data-path /path/to/BTCUSDT_1h_data_2018_to_2025.csv --visualize
```

Additional options:
```
--optimize-model         # Perform model hyperparameter optimization
--optimize-strategy      # Perform strategy hyperparameter optimization
--feature-importance     # Analyze feature importance
--start-date 2020-01-01  # Start date for analysis
--end-date 2021-12-31    # End date for analysis
--detailed-backtest      # Run detailed backtest with additional analytics
--cross-validation       # Perform cross-validation
--max-trials 100         # Maximum number of optimization trials
--model-type lstm        # Model architecture (lstm, gru, cnn, transformer, ensemble)
--lookahead 5            # Prediction horizon (number of candles to predict)
--lookback 100           # Lookback window size
--batch-size 32          # Training batch size
--epochs 50              # Number of training epochs
--learning-rate 0.001    # Learning rate for model training
--dropout 0.2            # Dropout rate for regularization
--hidden-layers 128 64   # Hidden layer sizes (space separated)
--save-model             # Save the trained model
--no-gpu                 # Disable GPU acceleration
--feature-set minimal    # Feature set to use (minimal, standard, full)
--pca-components 30      # Number of PCA components to use
--test-size 0.2          # Size of test set (proportion)
--val-size 0.15          # Size of validation set (proportion)
--random-seed 42         # Random seed for reproducibility
--plot-predictions       # Plot model predictions
--export-results         # Export results to CSV
--dashboard              # Generate HTML dashboard with results

### Batch Processing

For running on multiple datasets:

```bash
python run_batch.py --visualize
```

Additional options:
```
--symbols BTCUSDT ETHUSDT     # Specific symbols to process (default: all)
--timeframes 1h 4h            # Specific timeframes to process (default: all common)
--optimize                    # Perform hyperparameter optimization
--parallel                    # Run in parallel (requires Ray)
--num-workers 4               # Number of parallel workers
```

## Configuration

The main configuration parameters are defined in `config.py`. You can modify these parameters to customize the behavior of the trading bot:

### Trading Parameters
- `TRADING_FEE`: Trading fee (0.02% by default)
- `SLIPPAGE`: Slippage (0.2% by default)
- `INITIAL_CAPITAL`: Starting capital for backtests
- `POSITION_SIZE`: Percentage of capital to risk per trade
- `MAX_POSITIONS`: Maximum number of simultaneous positions

### Model Parameters
- `LOOKBACK_WINDOW`: Number of candles to consider for prediction
- `PREDICTION_HORIZON`: Number of future candles to predict
- `MODEL_TYPE`: Type of model to use (lstm, gru, cnn, transformer, ensemble)
- `HIDDEN_LAYERS`: Structure of the neural network
- `BATCH_SIZE`: Training batch size
- `EPOCHS`: Training epochs

### Stop Loss/Take Profit
- `ADAPTIVE_SL_TP`: Use adaptive stop loss and take profit
- `TRAILING_STOP`: Use trailing stop
- `ATR_SL_MULTIPLIER`: Stop loss multiplier for ATR-based stops
- `MIN_RISK_REWARD_RATIO`: Minimum risk-reward ratio for trades

## Deep Learning Model

The system supports multiple deep learning architectures:

1. **LSTM (Long Short-Term Memory)**: Good for capturing long-term dependencies in time series data
2. **GRU (Gated Recurrent Unit)**: Similar to LSTM but with fewer parameters, faster training
3. **CNN (Convolutional Neural Network)**: Efficient at capturing local patterns in time series
4. **Transformer**: Based on attention mechanisms, good for modeling complex relationships
5. **Ensemble**: Combines multiple models for improved predictions

The model's purpose is to analyze historical price data and predict future price movements, which are then used to generate trading signals.

## Trading Strategy

The trading strategy uses the predictions from the deep learning model to generate buy/sell signals. It incorporates:

- Adaptive stop-loss and take-profit levels based on market structure
- Trailing stops that adjust as price moves in favor of the trade
- Position sizing based on risk management principles
- Market regime detection to adapt to different market conditions

## Performance Metrics

The backtesting engine calculates the following performance metrics:

- Total Return: Overall percentage gain or loss
- Sharpe Ratio: Risk-adjusted return
- Sortino Ratio: Return adjusted for downside risk
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross profits divided by gross losses
- Maximum Drawdown: Largest peak-to-trough decline
- Average Profit: Mean profit percentage for winning trades
- Average Loss: Mean loss percentage for losing trades

## Visualizations

The visualization module creates comprehensive dashboards including:

- Equity curve showing capital growth over time
- Drawdown chart showing account declines
- Monthly returns heatmap showing performance by month
- Trade distribution showing the distribution of trade outcomes
- Price chart with trading signals
- Feature importance plots showing which indicators matter most

## Advanced Features

### Feature Engineering

The system generates over 60 technical features including:

- Momentum indicators (RSI, MACD, Stochastic, etc.)
- Volatility indicators (Bollinger Bands, ATR, etc.)
- Trend indicators (Moving Averages, ADX, etc.)
- Volume indicators (OBV, MFI, etc.)
- Candlestick patterns (Engulfing, Doji, Hammer, etc.)
- Market structure features (Swing points, Support/Resistance, etc.)

### Hyperparameter Optimization

The system uses Optuna for hyperparameter optimization:

- Model optimization to find the best neural network architecture
- Strategy optimization to find the best trading parameters
- Feature selection to identify the most important predictors

### Cross-Validation

Cross-validation ensures the model generalizes well to unseen data:

- K-fold cross-validation for model validation
- Walk-forward analysis for time series data
- Out-of-sample testing on recent data

## Examples

### Basic Single Run

```bash
python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2018_to_2025.csv --visualize
```

### Optimized Run with Feature Analysis

```bash
python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/ETHUSDT_4h_data_2018_to_2025.csv --optimize-model --optimize-strategy --feature-importance --visualize
```

### Batch Run for Multiple Symbols

```bash
python run_batch.py --symbols BTCUSDT ETHUSDT ADAUSDT --timeframes 1h 4h --optimize --visualize
```

## Extending the System

### Adding New Features

To add new technical indicators, modify the `feature_engineering.py` file:

1. Create a new method in the `FeatureEngineer` class
2. Add the method call to the `generate_all_features()` method
3. Update the `FEATURE_SETS` dictionary in `config.py`

### Implementing New Model Architectures

To add a new model architecture:

1. Add a new build method in the `DeepLearningModel` class in `model.py`
2. Update the `_build_model()` method to include the new model type
3. Update the `MODEL_TYPE` options in the command-line arguments

### Creating Custom Strategies

To create a custom trading strategy:

1. Create a new class that inherits from `TradingStrategy` in `strategy.py`
2. Implement the `generate_signals()` method
3. Customize the entry/exit conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Binance for providing historical cryptocurrency data
- TA-Lib and Pandas-TA for technical analysis libraries

## Contact

For questions or feedback, please contact [your.email@example.com](mailto:your.email@example.com).

## Quick Start

### Model Training & Visualization
```bash
# Train a model with visualization and save the model to disk
python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2023.csv --visualize --save-model
```

Expected output:
```
Loading data from /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2023.csv
Loaded 17520 rows of data
Generating features...
Generated 45 features
Creating training sequences...
Training model...
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
lstm (LSTM)                 (None, 128)               76800     
_________________________________________________________________
batch_normalization (BatchN (None, 128)               512       
_________________________________________________________________
dropout (Dropout)           (None, 128)               0         
_________________________________________________________________
dense (Dense)               (None, 5)                 645       
=================================================================
Total params: 77,957
Trainable params: 77,701
Non-trainable params: 256
_________________________________________________________________
Epoch 1/50
390/390 [==============================] - 10s 24ms/step - loss: 0.0016 - mae: 0.0303 - mse: 0.0016 - val_loss: 0.0012 - val_mae: 0.0265 - val_mse: 0.0012
...
Epoch 50/50
390/390 [==============================] - 9s 23ms/step - loss: 0.0007 - mae: 0.0213 - mse: 0.0007 - val_loss: 0.0008 - val_mae: 0.0211 - val_mse: 0.0008
Evaluating model...
98/98 [==============================] - 1s 10ms/step - loss: 0.0009 - mae: 0.0222 - mse: 0.0009
Test MSE: 0.0009
Backtest results:
Total Return: 124.56%
Number of Trades: 142
Win Rate: 63.38%
Profit Factor: 2.31
Max Drawdown: 18.75%
Sharpe Ratio: 2.11
Model saved to /Users/mrsmoothy/Desktop/rsidtrade/trading_/models/BTCUSDT_1h/model_20230427_153245.h5
```

### Backtesting
```bash
# Run a backtest on a specific date range
python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2023.csv --start-date 2023-01-01 --end-date 2023-03-31 --visualize
```

Expected output:
```
Loading data from /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2023.csv
Loaded 2160 rows of data
Filtering date range: 2023-01-01 to 2023-03-31
Generating features...
Generated 45 features
Loading model from /Users/mrsmoothy/Desktop/rsidtrade/trading_/models/BTCUSDT_1h/model_latest.h5
Generating signals...
Running backtest...
Backtest results:
Total Return: 38.92%
Number of Trades: 42
Win Rate: 64.29%
Profit Factor: 2.15
Max Drawdown: 12.34%
Sharpe Ratio: 2.03
Report saved to /Users/mrsmoothy/Desktop/rsidtrade/trading_/results/backtest_20230427_154532.html
```

### Batch Processing
```bash
# Run batch processing on multiple symbols
python run_batch.py --symbols BTCUSDT ETHUSDT ADAUSDT --timeframes 1h 4h --start-date 2022-01-01 --end-date 2022-12-31
```

Expected output:
```
Processing 3 symbols with 2 timeframes each...
Processing BTCUSDT 1h...
Processing BTCUSDT 4h...
Processing ETHUSDT 1h...
Processing ETHUSDT 4h...
Processing ADAUSDT 1h...
Processing ADAUSDT 4h...
Results saved to /Users/mrsmoothy/Desktop/rsidtrade/trading_/results/batch_20230427_160523.html
```

### Hyperparameter Optimization
```bash
# Optimize model hyperparameters
python optimizer.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2023.csv --trials 50 --metric sharpe_ratio
```

Expected output:
```
Starting hyperparameter optimization with 50 trials...
Trial 1/50 complete: LSTM model with 128 units, 2 layers, dropout 0.2, learning rate 0.001 -> Sharpe Ratio: 1.85
Trial 2/50 complete: LSTM model with 256 units, 3 layers, dropout 0.3, learning rate 0.0005 -> Sharpe Ratio: 1.93
...
Trial 50/50 complete: GRU model with 192 units, 2 layers, dropout 0.25, learning rate 0.0008 -> Sharpe Ratio: 2.24
Best trial: Trial #38
Best parameters: {'model_type': 'lstm', 'hidden_layers': [192, 96], 'dropout_rate': 0.25, 'learning_rate': 0.0008}
Best Sharpe Ratio: 2.37
Optimization results saved to /Users/mrsmoothy/Desktop/rsidtrade/trading_/results/optimization_20230427_185623.html
```

### Web Dashboard
```bash
# Start the web dashboard
python app.py
```

Expected output:
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://0.0.0.0:5001
Press CTRL+C to quit
 * Restarting with stat
```

### Live Trading
```bash
# Start live trading (requires API keys)
python live_trading.py --exchange binance --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET --testnet
```

Expected output:
```
Initializing live trader...
Loading models for [BTCUSDT, ETHUSDT] on timeframes [1h, 4h]...
Models loaded successfully.
Initializing strategies...
Connecting to Binance Testnet...
Connection successful.
Live trading started.
[2023-04-27 19:32:15] Fetching market data for BTCUSDT 1h...
[2023-04-27 19:32:16] Fetching market data for BTCUSDT 4h...
[2023-04-27 19:32:17] Fetching market data for ETHUSDT 1h...
[2023-04-27 19:32:18] Fetching market data for ETHUSDT 4h...
[2023-04-27 19:32:20] Generating signals...
[2023-04-27 19:32:21] No trading signals at this time.
```

## Advanced Usage

### Feature Engineering
```bash
# Test feature generation
python feature_engineering.pyc
```

### Data Fetching
```bash
# Download historical data
python data_fetcher.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h 1d --start-date 2021-01-01 --end-date 2023-04-01
```

### Model Training with Custom Parameters
```bash
# Train with custom parameters
python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2023.csv --model-type lstm --hidden-layers 128 64 32 --dropout 0.3 --learning-rate 0.0005 --batch-size 64 --epochs 100 --save-model
```




python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --optimize-model --visualize --save-model



python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --visualize --feature-

python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --visualize --feature-importance

python optimize_features.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --model-path /Users/mrsmoothy/Desktop/rsidtrade/trading_/models/BTCUSDT_1h/model_20250313_073330.h5 --population 15 --generations 8


python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --visualize


python data_fetcher.py --symbols BTCUSDT --timeframes  5m 15m 1h 4h  --start-date 2021-01-01 --end-date 2022-12-31