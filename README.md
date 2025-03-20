# RSIDTrade - AI Cryptocurrency Trading System

An advanced machine learning-based trading system designed for cryptocurrency markets, with a focus on technical analysis and deep learning prediction models.

## Features

- **Multi-timeframe analysis**: Support for 5m, 15m, 1h, 4h timeframes
- **Universal model architecture**: Train once, apply to multiple market conditions
- **Feature importance visualization**: Understand what drives model decisions
- **Adaptive stop-loss and take-profit**: Dynamic risk management based on market volatility
- **PCA dimensionality reduction**: Extract significant patterns from market data
- **Backtesting engine**: Comprehensive performance evaluation with visualizations
- **Genetic algorithm optimization**: Fine-tune feature selection for improved performance

## Quick Start

### Data Collection

```bash
# Fetch historical data for a specific symbol
python data_fetcher.py --symbols ETHUSDT --timeframes 5m 15m 1h 4h --start-date 2021-01-01 --end-date 2022-12-31
```

### Single Model Training & Backtest

```bash
# Train and backtest a single model with visualization
python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --visualize
```

### Feature Importance Analysis

```bash
# Analyze feature importance
python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --visualize --feature-importance
```

### Universal Model Training

```bash
# Train a universal model on multiple symbols and timeframes
python run_universal.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --start-date 2021-01-01 --end-date 2023-12-31 --save-model --visualize
```

### Model Optimization

```bash
# Optimize features using genetic algorithm
python optimize_features.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --model-path /Users/mrsmoothy/Desktop/rsidtrade/trading_/models/BTCUSDT_1h/model_20250313_073330.h5 --population 15 --generations 8
```

### Backtest Using Universal Model

```bash
# Backtest using a pre-trained universal model
python run_single.py --data-path /Users/mrsmoothy/Desktop/rsidtrade/binance_data_sets/BTCUSDT_1h_data_2021_to_2022.csv --universal-model-path /Users/mrsmoothy/Desktop/rsidtrade/trading_/models/universal_model/universal_model_20250319_145156/model.h5 --backtest-only --visualize
```

## System Architecture

The system consists of several key components:

1. **Data Processing Pipeline**: Fetches, cleans, and prepares market data
2. **Feature Engineering**: Extracts meaningful features using technical indicators and PCA
3. **Model Training**: LSTM-based deep learning for price movement prediction
4. **Backtesting Engine**: Simulates trading strategies on historical data
5. **Optimization Layer**: Genetic algorithms for feature and parameter tuning

## Model Types

- **LSTM (default)**: Long Short-Term Memory neural networks for sequence prediction
- **Universal Model**: Cross-market and cross-timeframe model that generalizes well

## Performance Metrics

The system evaluates performance using various metrics:
- Total Return
- Win Rate
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Number of Trades

## Requirements

See `environment.yml` for the complete list of dependencies.

## License

Copyright © 2024