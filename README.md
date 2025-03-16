# RSID Trading Bot

An AI-powered cryptocurrency trading bot that uses machine learning to predict price movements and execute automated trades.

## Features

- **Machine Learning Models**: LSTM, GRU, CNN, and Transformer architectures for price prediction
- **Feature Engineering**: Advanced technical indicators and market structure analysis
- **PCA Dimension Reduction**: Optimize feature space for better model performance
- **Multi-Timeframe Analysis**: Combine signals from multiple timeframes
- **Backtesting Engine**: Test strategies on historical data
- **Live Trading**: Connect to exchanges for automated trading
- **Web Dashboard**: Monitor performance and control the trading system
- **Optimization**: Hyperparameter tuning with Optuna

## Installation

### Prerequisites

- Python 3.10 or higher
- Conda package manager (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rsidtrade.git
   cd rsidtrade/trading_
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate rsidtrade
   ```

3. Configure API keys and settings in `config.py` or create a `user_config.py` file.

## Usage

### Data Collection

Fetch historical data from exchanges:

```bash
python data_fetcher.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --start-date 2023-01-01 --end-date 2024-01-01
```

Options:
- `--symbols`: List of trading pairs to fetch (e.g., BTCUSDT ETHUSDT)
- `--timeframes`: List of timeframes (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format
- `--data-dir`: Directory to save data files
- `--workers`: Number of parallel workers for downloading data (default: 4)
- `--list-symbols`: Show available symbols and exit

### Model Training

Train a machine learning model:

```bash
python model.py --symbol BTCUSDT --timeframe 1h --model-type lstm --epochs 100
```

Options:
- `--symbol`: Trading symbol to train on
- `--timeframe`: Timeframe to use
- `--model-type`: Model architecture (lstm, gru, cnn, transformer, ensemble)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate for optimizer
- `--dropout-rate`: Dropout rate for regularization
- `--apply-pca`: Apply PCA dimension reduction
- `--pca-components`: Number of PCA components to use

### Backtesting

Backtest a trading strategy:

```bash
python run_single.py --symbol BTCUSDT --timeframe 1h --start-date 2023-01-01 --end-date 2024-01-01 --visualize
```

Options:
- `--symbol`: Trading symbol to backtest
- `--timeframe`: Timeframe to use
- `--start-date`: Start date for backtest
- `--end-date`: End date for backtest
- `--strategy`: Strategy type (ml, ensemble, rule_based)
- `--adaptive-sl-tp`: Use adaptive stop loss and take profit
- `--trailing-stop`: Use trailing stop
- `--visualize`: Generate visual reports
- `--initial-capital`: Starting capital for backtest
- `--position-size`: Position size as percentage of capital

### Batch Backtesting

Run backtests on multiple symbols and timeframes:

```bash
python run_batch.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframes 1h 4h --start-date 2023-01-01 --end-date 2024-01-01
```

Additional options:
- `--parallel`: Run backtests in parallel processes
- `--max-workers`: Maximum number of parallel workers
- `--comparison-report`: Generate a comparison report across all backtests

### Hyperparameter Optimization

Optimize strategy parameters:

```bash
python optimizer.py --symbol BTCUSDT --timeframe 1h --trials 100 --metric sharpe_ratio
```

Options:
- `--symbol`: Trading symbol to optimize for
- `--timeframe`: Timeframe to use
- `--trials`: Number of optimization trials to run
- `--metric`: Optimization metric (sharpe_ratio, sortino_ratio, profit, win_rate)
- `--cv-folds`: Number of cross-validation folds

### Live Trading

Run live trading:

```bash
python live_trading.py --exchange binance --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET --symbols BTCUSDT ETHUSDT --timeframes 1h 4h
```

Options:
- `--exchange`: Exchange to use (currently supports binance)
- `--api-key`: API key for authentication
- `--api-secret`: API secret for authentication
- `--testnet`: Use testnet/sandbox mode
- `--symbols`: List of symbols to trade
- `--timeframes`: List of timeframes to trade
- `--config`: Path to custom configuration file
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Web Dashboard

Run the web dashboard for monitoring and control:

```bash
python app.py
```

Access the dashboard in your browser at http://localhost:5001

## Project Structure

- `config.py`: Configuration parameters
- `data_fetcher.py`: Download historical market data
- `data_processor.py`: Process and prepare data
- `feature_engineering.py`: Generate technical indicators
- `model.py`: Machine learning models for price prediction
- `strategy.py`: Trading strategies
- `backtest.py`: Backtesting engine
- `optimizer.py`: Hyperparameter optimization
- `live_trading.py`: Live trading execution
- `app.py`: Web dashboard
- `utils.py`: Utility functions
- `models/`: Saved models directory
- `data/`: Data directory
- `results/`: Results directory
- `logs/`: Log files

## Advanced Features

### Advanced Feature Engineering

The system includes sophisticated feature engineering capabilities:

- **Market Structure Analysis**: Identifies swing points, trends, support/resistance levels
- **Volatility Regimes**: Categorizes market into volatility regimes using clustering
- **Candlestick Patterns**: Recognizes common candlestick patterns automatically
- **Combined Indicators**: Creates derived features from multiple base indicators

### PCA Integration

PCA support reduces dimensionality while preserving important information:

1. Select a subset of features most relevant to price movements
2. Apply PCA to reduce dimensionality and remove correlation
3. Train models on the transformed feature space
4. Transform features in real-time during live trading

### Multi-Timeframe Analysis

Strategies can combine signals from multiple timeframes:

- **Hierarchical Analysis**: Use higher timeframes for trend identification, lower timeframes for entry/exit
- **Signal Confirmation**: Require alignment across multiple timeframes for trade execution
- **Timeframe-specific Models**: Different models for different timeframes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.