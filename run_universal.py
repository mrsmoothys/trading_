#!/usr/bin/env python
"""
Script for training a universal model on multiple symbols and timeframes.
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR,
    LOOKBACK_WINDOW, PREDICTION_HORIZON, EPOCHS, BATCH_SIZE
)
from data_processor import load_data, list_available_data
from feature_engineering import generate_features, apply_pca_reduction
from universal_model import UniversalModel
from utils import setup_logging, print_system_info
from backtest import Backtester
from strategy import MLTradingStrategy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a universal model on multiple datasets.')
    
    # Dataset selection
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Specific symbols to train on (default: all available)')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1h', '4h', '1d'],
                        help='Specific timeframes to train on (default: 1h, 4h, 1d)')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing dataset files')
    
    # Universal model params
    parser.add_argument('--lookback', type=int, default=LOOKBACK_WINDOW,
                        help='Lookback window size (default: from config)')
    parser.add_argument('--horizon', type=int, default=PREDICTION_HORIZON,
                        help='Prediction horizon (default: from config)')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden layer sizes (default: 256 128 64)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--model-path', type=str,
                        help='Path to existing universal model (if continuing training)')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Training epochs per dataset (default: from config)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Training batch size (default: from config)')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum number of samples per dataset (default: 10000)')
    
    # Feature processing
    parser.add_argument('--pca-components', type=int, default=30,
                        help='Number of PCA components (default: 30)')
    
    # Output options
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR),
                        help='Directory to save results')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    
    # Execution options
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after training')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization dashboards')
    parser.add_argument('--parallel', action='store_true',
                        help='Process datasets in parallel')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save the trained model')
    
    return parser.parse_args()

def load_and_process_data(file_path: str, pca_components: int = 30) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and process data with feature engineering and PCA.
    
    Args:
        file_path: Path to dataset file
        pca_components: Number of PCA components
        
    Returns:
        Tuple of (processed_data, feature_columns)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load data
        data = load_data(file_path)
        
        # Generate features
        data = generate_features(data)
        
        # Apply PCA
        reduced_data, _, _ = apply_pca_reduction(data, n_components=pca_components)
        
        # Get feature columns
        feature_columns = [f'pc_{i+1}' for i in range(pca_components)]
        
        return reduced_data, feature_columns
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None, []

def train_universal_model(args):
    """
    Train a universal model on multiple datasets.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Trained universal model
    """
    logger = logging.getLogger(__name__)
    
    # Create or load universal model
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading existing universal model from {args.model_path}")
        universal_model = UniversalModel(model_path=args.model_path)
    else:
        logger.info("Creating new universal model")
        universal_model = UniversalModel(
            lookback_window=args.lookback,
            prediction_horizon=args.horizon,
            feature_count=args.pca_components,
            hidden_layers=args.hidden_layers,
            dropout_rate=args.dropout,
            learning_rate=args.learning_rate
        )
    
    # List available data
    available_data = list_available_data()
    
    # Determine symbols to process
    if args.symbols:
        symbols = [symbol for symbol in args.symbols if symbol in available_data]
        if len(symbols) != len(args.symbols):
            missing = set(args.symbols) - set(symbols)
            logger.warning(f"Some requested symbols not found: {missing}")
    else:
        symbols = list(available_data.keys())
    
    logger.info(f"Processing {len(symbols)} symbols with {len(args.timeframes)} timeframes")
    
    # Process datasets
    total_datasets = 0
    for symbol in symbols:
        for timeframe in args.timeframes:
            if timeframe in available_data[symbol]:
                total_datasets += 1
    
    with tqdm(total=total_datasets, desc="Training Universal Model") as pbar:
        for symbol in symbols:
            for timeframe in args.timeframes:
                if timeframe in available_data[symbol]:
                    try:
                        file_path = available_data[symbol][timeframe]
                        logger.info(f"Processing {symbol} {timeframe} from {file_path}")
                        
                        # Load and process data
                        data, feature_columns = load_and_process_data(file_path, args.pca_components)
                        
                        if data is None:
                            logger.warning(f"Skipping {symbol} {timeframe} due to processing error")
                            pbar.update(1)
                            continue
                        
                        # Limit data size if needed
                        if len(data) > args.max_samples:
                            logger.info(f"Limiting data from {len(data)} to {args.max_samples} samples")
                            # Take most recent data
                            data = data.iloc[-args.max_samples:]
                        
                        # Train model
                        logger.info(f"Training universal model on {symbol} {timeframe} with {len(data)} samples")
                        history = universal_model.train(
                            df=data,
                            symbol=symbol,
                            timeframe=timeframe,
                            feature_columns=feature_columns,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            validation_split=0.2,
                            apply_pca=False  # Already applied in load_and_process_data
                        )
                        
                        logger.info(f"Completed training on {symbol} {timeframe}")
                        pbar.update(1)
                    
                    except Exception as e:
                        logger.error(f"Error training on {symbol} {timeframe}: {e}")
                        pbar.update(1)
    
    # Save model if requested
    if args.save_model:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(MODELS_DIR, 'universal_model', f"universal_model_{timestamp}")
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        universal_model.save(save_dir)
        logger.info(f"Universal model saved to {save_dir}")
    
    return universal_model

def run_universal_backtest(universal_model, args):
    """
    Run backtests using the universal model.
    
    Args:
        universal_model: Trained universal model
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    
    # List available data
    available_data = list_available_data()
    
    # Determine symbols to process
    if args.symbols:
        symbols = [symbol for symbol in args.symbols if symbol in available_data]
    else:
        symbols = list(available_data.keys())
    
    # Store backtest results
    backtest_results = {}
    
    # Create wrapper for universal model
    class UniversalModelWrapper:
        def __init__(self, universal_model, symbol, timeframe):
            self.universal_model = universal_model
            self.symbol = symbol
            self.timeframe = timeframe
        
        def predict(self, X, verbose=0):
            return self.universal_model.predict(X, self.symbol, self.timeframe)
    
    for symbol in tqdm(symbols, desc="Running Backtests"):
        backtest_results[symbol] = {}
        
        for timeframe in args.timeframes:
            if timeframe in available_data[symbol]:
                try:
                    file_path = available_data[symbol][timeframe]
                    logger.info(f"Backtesting {symbol} {timeframe} from {file_path}")
                    
                    # Load and process data
                    data, feature_columns = load_and_process_data(file_path, args.pca_components)
                    
                    if data is None:
                        logger.warning(f"Skipping {symbol} {timeframe} due to processing error")
                        continue
                    
                    # Create wrapped model for strategy
                    model = UniversalModelWrapper(universal_model, symbol, timeframe)
                    
                    # Create strategy
                    strategy = MLTradingStrategy(
                        symbol=symbol,
                        timeframe=timeframe,
                        model=model,
                        lookback_window=args.lookback,
                        prediction_horizon=args.horizon,
                        threshold=0.005,  # Default threshold
                        position_size=0.1,  # Default position size
                        adaptive_sl_tp=True,
                        trailing_stop=True
                    )
                    
                    # Set data and run backtest
                    strategy.set_data(data)
                    results = strategy.backtest(data)
                    
                    # Store results
                    backtest_results[symbol][timeframe] = results['performance']
                    
                    logger.info(f"Backtest completed for {symbol} {timeframe}: "
                              f"Return: {results['performance']['total_return']:.2f}%, "
                              f"Sharpe: {results['performance']['sharpe_ratio']:.2f}")
                    
                    # Generate visualizations if requested
                    if args.visualize:
                        from visualize import Visualizer
                        vis_dir = os.path.join(args.results_dir, f"universal_{symbol}_{timeframe}")
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        visualizer = Visualizer(result_dir=vis_dir)
                        visualizer.generate_backtest_dashboard(
                            results,
                            title=f"{symbol} {timeframe} Universal Model Backtest"
                        )
                
                except Exception as e:
                    logger.error(f"Error backtesting {symbol} {timeframe}: {e}")
    
    # Save summary results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(args.results_dir, f"universal_backtest_summary_{timestamp}.json")
    
    # Create summary data
    summary_data = []
    for symbol in backtest_results:
        for timeframe in backtest_results[symbol]:
            performance = backtest_results[symbol][timeframe]
            summary_data.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'total_return': performance.get('total_return', 0),
                'num_trades': performance.get('num_trades', 0),
                'win_rate': performance.get('win_rate', 0),
                'profit_factor': performance.get('profit_factor', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'sortino_ratio': performance.get('sortino_ratio', 0),
                'max_drawdown': performance.get('max_drawdown', 0)
            })
    
    # Create DataFrame and save to CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(args.results_dir, f"universal_backtest_summary_{timestamp}.csv")
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Backtest summary saved to {summary_csv}")
    
    # Save full results as JSON
    with open(results_path, 'w') as f:
        json.dump(backtest_results, f, default=str, indent=2)
    
    logger.info(f"Backtest results saved to {results_path}")
    
    # Create performance visualizations
    if summary_data and args.visualize:
        try:
            plt.figure(figsize=(12, 8))
            summary_df_sorted = summary_df.sort_values('total_return', ascending=False)
            plt.bar(
                [f"{row['symbol']}_{row['timeframe']}" for _, row in summary_df_sorted.iterrows()],
                summary_df_sorted['total_return']
            )
            plt.xticks(rotation=90)
            plt.ylabel('Total Return (%)')
            plt.title('Universal Model Performance Across Assets')
            plt.tight_layout()
            
            performance_plot = os.path.join(args.results_dir, f"universal_performance_{timestamp}.png")
            plt.savefig(performance_plot)
            plt.close()
            
            logger.info(f"Performance visualization saved to {performance_plot}")
        except Exception as e:
            logger.error(f"Error creating performance visualization: {e}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Universal Model Training")
    
    # Print system information
    print_system_info()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'universal_model'), exist_ok=True)
    
    try:
        # Train universal model
        universal_model = train_universal_model(args)
        
        # Get training summary
        summary = universal_model.get_training_summary()
        logger.info("Universal Model Training Summary:")
        logger.info(f"Trained on symbols: {', '.join(summary['trained_symbols'])}")
        
        # Run backtests if requested
        if args.backtest:
            logger.info("Running backtests with universal model")
            run_universal_backtest(universal_model, args)
        
        logger.info("Universal model training completed successfully")
        
        return 0
    
    except Exception as e:
        import traceback
        logger.error(f"Error in universal model training: {e}")
        logger.debug(traceback.format_exc())
        
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())