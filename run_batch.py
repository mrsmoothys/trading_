#!/usr/bin/env python
"""
Batch processing script for running the AI trading bot on multiple datasets.
"""
import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR,
    LOOKBACK_WINDOW, PREDICTION_HORIZON, MODEL_TYPE, HIDDEN_LAYERS,
    DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE, EPOCHS,
    TRADING_FEE, SLIPPAGE, POSITION_SIZE, INITIAL_CAPITAL
)
from data_processor import load_data, prepare_multi_timeframe_data, create_training_sequences, train_val_test_split
from feature_engineering import generate_features
from model import DeepLearningModel, ModelManager
from strategy import MLTradingStrategy
from backtest import Backtester
from optimizer import ModelOptimizer, StrategyOptimizer
from visualize import Visualizer
from utils import (
    setup_logging, save_metadata, find_all_datasets, find_common_timeframes,
    get_resource_usage, clear_keras_session, format_time, print_progress_bar
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the AI trading bot in batch mode.')
    
    # Dataset selection
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Directory containing dataset files')
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Specific symbols to process (default: all)')
    parser.add_argument('--timeframes', type=str, nargs='+',
                        help='Specific timeframes to process (default: all common)')
    
    # Training params
    parser.add_argument('--train-size', type=float, default=0.7,
                        help='Training data proportion (default: 0.7)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation data proportion (default: 0.15)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Training epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Training batch size (default: from config)')
    
    # Model params
    parser.add_argument('--model-type', type=str, default=MODEL_TYPE,
                        choices=['lstm', 'gru', 'cnn', 'transformer', 'ensemble'],
                        help='Model type (default: from config)')
    parser.add_argument('--lookback', type=int, default=LOOKBACK_WINDOW,
                        help='Lookback window (default: from config)')
    parser.add_argument('--horizon', type=int, default=PREDICTION_HORIZON,
                        help='Prediction horizon (default: from config)')
    
    # Strategy params
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL,
                        help='Initial capital (default: from config)')
    parser.add_argument('--position-size', type=float, default=POSITION_SIZE,
                        help='Position size as fraction of capital (default: from config)')
    parser.add_argument('--fee', type=float, default=TRADING_FEE,
                        help='Trading fee (default: from config)')
    parser.add_argument('--slippage', type=float, default=SLIPPAGE,
                        help='Slippage (default: from config)')
    
    # Optimization
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    parser.add_argument('--opt-trials', type=int, default=50,
                        help='Number of optimization trials (default: 50)')
    
    # Output options
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Directory to save results')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization dashboards')
    
    # Other options
    parser.add_argument('--use-cached-models', action='store_true',
                        help='Use cached models if available')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip symbols/timeframes with existing results')
    parser.add_argument('--parallel', action='store_true',
                        help='Run in parallel (requires Ray)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue processing on error')
    
    return parser.parse_args()

def process_symbol_timeframe(
    symbol: str,
    timeframe: str,
    args: argparse.Namespace,
    model_manager: ModelManager
) -> Dict[str, Any]:
    """
    Process a single symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Trading timeframe
        args: Command-line arguments
        model_manager: Model manager instance
        
    Returns:
        Dictionary of results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {symbol} {timeframe}")
    
    start_time = time.time()
    results = {
        'symbol': symbol,
        'timeframe': timeframe,
        'status': 'started',
        'start_time': datetime.now().isoformat()
    }
    
    try:
        # Check if results already exist and should be skipped
        if args.skip_existing:
            result_dir = os.path.join(args.results_dir, f"{symbol}_{timeframe}")
            if os.path.exists(result_dir) and len(os.listdir(result_dir)) > 0:
                logger.info(f"Skipping {symbol} {timeframe} - results already exist")
                results['status'] = 'skipped'
                return results
        
        # Step 1: Load and prepare data
        logger.info(f"Loading data for {symbol} {timeframe}")
        data_path = os.path.join(args.data_dir, f"{symbol}_{timeframe}_data_2018_to_2025.csv")
        
        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            results['status'] = 'error'
            results['error'] = f"Data file not found: {data_path}"
            return results
        
        # Load data
        data = load_data(data_path)
        
        # Generate features
        logger.info(f"Generating features for {symbol} {timeframe}")
        data = generate_features(data)
        
        # Step 2: Split data for training and validation
        feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Create sequences
        X, y = create_training_sequences(
            data,
            lookback_window=args.lookback,
            prediction_horizon=args.horizon,
            feature_columns=feature_columns,
            target_column='close',
            normalize=True
        )
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, train_size=args.train_size, val_size=args.val_size
        )
        
        # Step 3: Create or load model
        model = None
        
        if args.use_cached_models:
            # Try to load cached model
            model = model_manager.load_model(symbol, timeframe)
        
        if model is None:
            if args.optimize:
                # Perform model optimization
                logger.info(f"Optimizing model for {symbol} {timeframe}")
                
                optimizer = ModelOptimizer(
                    data_path=data_path,
                    result_dir=os.path.join(args.results_dir, 'optimization', f"{symbol}_{timeframe}"),
                    n_trials=args.opt_trials
                )
                
                best_params = optimizer.optimize()
                
                # Create model with optimized parameters
                model = DeepLearningModel(
                    input_shape=(X.shape[1], X.shape[2]),
                    output_dim=y.shape[1],
                    model_type=best_params.get('model_type', args.model_type),
                    hidden_layers=[best_params.get(f'units_layer_{i}', HIDDEN_LAYERS[i]) 
                                  for i in range(best_params.get('n_layers', len(HIDDEN_LAYERS)))],
                    dropout_rate=best_params.get('dropout_rate', DROPOUT_RATE),
                    learning_rate=best_params.get('learning_rate', LEARNING_RATE)
                )
            else:
                # Create model with default parameters
                logger.info(f"Creating model for {symbol} {timeframe}")
                
                model = DeepLearningModel(
                    input_shape=(X.shape[1], X.shape[2]),
                    output_dim=y.shape[1],
                    model_type=args.model_type,
                    hidden_layers=HIDDEN_LAYERS,
                    dropout_rate=DROPOUT_RATE,
                    learning_rate=LEARNING_RATE
                )
        
        # Step 4: Train model
        logger.info(f"Training model for {symbol} {timeframe}")
        
        # Create save path
        model_dir = os.path.join(MODELS_DIR, f"{symbol}_{timeframe}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.h5")
        
        # Train model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=model_path
        )
        
        # Save trained model to model manager
        model_manager.models[f"{symbol}_{timeframe}"] = model
        
        # Step 5: Create and optimize trading strategy
        logger.info(f"Creating trading strategy for {symbol} {timeframe}")
        
        if args.optimize:
            # Optimize strategy parameters
            logger.info(f"Optimizing strategy for {symbol} {timeframe}")
            
            optimizer = StrategyOptimizer(
                symbol=symbol,
                timeframe=timeframe,
                data_path=data_path,
                model=model,
                result_dir=os.path.join(args.results_dir, 'optimization', f"{symbol}_{timeframe}_strategy"),
                n_trials=args.opt_trials
            )
            
            best_params = optimizer.optimize()
            
            # Create strategy with optimized parameters
            strategy = MLTradingStrategy(
                symbol=symbol,
                timeframe=timeframe,
                model=model,
                lookback_window=best_params.get('lookback_window', args.lookback),
                prediction_horizon=args.horizon,
                threshold=best_params.get('threshold', 0.005),
                position_size=best_params.get('position_size', args.position_size),
                initial_capital=args.initial_capital,
                trading_fee=args.fee,
                slippage=args.slippage,
                adaptive_sl_tp=True,
                trailing_stop=True
            )
            
            # Update config with optimized parameters
            import config
            config.ATR_SL_MULTIPLIER = best_params.get('atr_sl_multiplier', config.ATR_SL_MULTIPLIER)
            config.MIN_RISK_REWARD_RATIO = best_params.get('min_risk_reward_ratio', config.MIN_RISK_REWARD_RATIO)
        else:
            # Create strategy with default parameters
            strategy = MLTradingStrategy(
                symbol=symbol,
                timeframe=timeframe,
                model=model,
                lookback_window=args.lookback,
                prediction_horizon=args.horizon,
                threshold=0.005,  # Default threshold
                position_size=args.position_size,
                initial_capital=args.initial_capital,
                trading_fee=args.fee,
                slippage=args.slippage,
                adaptive_sl_tp=True,
                trailing_stop=True
            )
        
        # Step 6: Backtest strategy
        logger.info(f"Backtesting strategy for {symbol} {timeframe}")
        
        # Create backtester
        backtester = Backtester(
            strategy=strategy,
            data_path=data_path,
            results_dir=os.path.join(args.results_dir, f"{symbol}_{timeframe}")
        )
        
        # Run backtest
        backtest_results = backtester.run()
        
        # Step 7: Generate visualizations
        if args.visualize:
            logger.info(f"Generating visualizations for {symbol} {timeframe}")
            
            # Create visualizer
            visualizer = Visualizer(result_dir=os.path.join(args.results_dir, f"{symbol}_{timeframe}"))
            
            # Generate dashboard
            visualizer.generate_backtest_dashboard(
                backtest_results,
                title=f"{symbol} {timeframe} Backtest Results"
            )
        
        # Step 8: Evaluate on test data
        logger.info(f"Evaluating model on test data for {symbol} {timeframe}")
        test_metrics = model.evaluate(X_test, y_test)
        
        # Step 9: Gather results
        elapsed_time = time.time() - start_time
        
        results.update({
            'status': 'completed',
            'elapsed_time': elapsed_time,
            'elapsed_time_str': format_time(elapsed_time),
            'end_time': datetime.now().isoformat(),
            'test_metrics': test_metrics,
            'performance': backtest_results['performance']
        })
        
        # Log resource usage
        resources = get_resource_usage()
        results['resources'] = resources
        
        logger.info(f"Completed {symbol} {timeframe} in {format_time(elapsed_time)}")
        
        # Clear session to free up memory
        clear_keras_session()
        
        return results
    
    except Exception as e:
        import traceback
        logger.error(f"Error processing {symbol} {timeframe}: {e}")
        logger.debug(traceback.format_exc())
        
        # Update results
        elapsed_time = time.time() - start_time
        results.update({
            'status': 'error',
            'error': str(e),
            'elapsed_time': elapsed_time,
            'elapsed_time_str': format_time(elapsed_time),
            'end_time': datetime.now().isoformat()
        })
        
        # Clear session to free up memory
        clear_keras_session()
        
        return results

def process_batch(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process all symbols and timeframes in batch mode.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of batch results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting batch processing")
    
    start_time = time.time()
    batch_results = {
        'start_time': datetime.now().isoformat(),
        'status': 'started',
        'symbols_processed': 0,
        'timeframes_processed': 0,
        'errors': 0,
        'completed': 0,
        'skipped': 0,
        'results': []
    }
    
    # Find available datasets
    available_datasets = find_all_datasets()
    
    # Determine symbols to process
    if args.symbols:
        symbols = [symbol for symbol in args.symbols if symbol in available_datasets]
        if len(symbols) != len(args.symbols):
            missing = set(args.symbols) - set(symbols)
            logger.warning(f"Some requested symbols not found: {missing}")
    else:
        symbols = list(available_datasets.keys())
    
    # Determine timeframes to process
    if args.timeframes:
        # Use specified timeframes if available
        timeframes = args.timeframes
    else:
        # Use common timeframes across all symbols
        timeframes = find_common_timeframes(available_datasets)
    
    logger.info(f"Processing {len(symbols)} symbols with {len(timeframes)} timeframes")
    
    # Create model manager
    model_manager = ModelManager()
    
    # Process in parallel if requested
    if args.parallel:
        try:
            import ray
            from ray.util import ActorPool
            
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(num_cpus=args.num_workers)
            
            # Define remote function
            @ray.remote
            def process_remote(symbol, timeframe, args_dict):
                # Convert args_dict back to Namespace
                args_obj = argparse.Namespace(**args_dict)
                
                # Create model manager
                local_model_manager = ModelManager()
                
                # Process symbol and timeframe
                return process_symbol_timeframe(symbol, timeframe, args_obj, local_model_manager)
            
            # Create tasks
            tasks = []
            args_dict = vars(args)  # Convert Namespace to dict for serialization
            
            for symbol in symbols:
                for timeframe in timeframes:
                    if timeframe in available_datasets.get(symbol, []):
                        tasks.append(process_remote.remote(symbol, timeframe, args_dict))
            
            # Process tasks
            for i, result in enumerate(ray.get(tasks)):
                batch_results['results'].append(result)
                
                # Update counts
                batch_results['symbols_processed'] += 1
                batch_results['timeframes_processed'] += 1
                
                if result['status'] == 'completed':
                    batch_results['completed'] += 1
                elif result['status'] == 'error':
                    batch_results['errors'] += 1
                elif result['status'] == 'skipped':
                    batch_results['skipped'] += 1
                
                # Print progress
                print_progress_bar(
                    i + 1, len(tasks),
                    prefix=f"Progress:",
                    suffix=f"Complete ({i+1}/{len(tasks)})"
                )
            
        except ImportError:
            logger.warning("Ray not installed. Running in sequential mode instead.")
            args.parallel = False
    
    # Process sequentially if not parallel
    if not args.parallel:
        total_combinations = sum(1 for symbol in symbols 
                               for timeframe in timeframes 
                               if timeframe in available_datasets.get(symbol, []))
        
        # Process each symbol and timeframe
        processed = 0
        for symbol in symbols:
            for timeframe in timeframes:
                if timeframe in available_datasets.get(symbol, []):
                    try:
                        result = process_symbol_timeframe(symbol, timeframe, args, model_manager)
                        batch_results['results'].append(result)
                        
                        # Update counts
                        batch_results['symbols_processed'] += 1
                        batch_results['timeframes_processed'] += 1
                        
                        if result['status'] == 'completed':
                            batch_results['completed'] += 1
                        elif result['status'] == 'error':
                            batch_results['errors'] += 1
                        elif result['status'] == 'skipped':
                            batch_results['skipped'] += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol} {timeframe}: {e}")
                        batch_results['errors'] += 1
                        
                        if not args.continue_on_error:
                            raise
                    
                    processed += 1
                    print_progress_bar(
                        processed, total_combinations,
                        prefix=f"Progress:",
                        suffix=f"Complete ({processed}/{total_combinations})"
                    )
    
    # Complete batch results
    elapsed_time = time.time() - start_time
    batch_results.update({
        'end_time': datetime.now().isoformat(),
        'elapsed_time': elapsed_time,
        'elapsed_time_str': format_time(elapsed_time),
        'status': 'completed'
    })
    
    # Save batch results
    batch_results_path = os.path.join(args.results_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(batch_results_path, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    logger.info(f"Batch processing completed in {format_time(elapsed_time)}")
    logger.info(f"Processed {batch_results['symbols_processed']} symbols with {batch_results['timeframes_processed']} timeframes")
    logger.info(f"Completed: {batch_results['completed']}, Errors: {batch_results['errors']}, Skipped: {batch_results['skipped']}")
    logger.info(f"Results saved to {batch_results_path}")
    
    return batch_results

def generate_summary_report(batch_results: Dict[str, Any], args: argparse.Namespace) -> str:
    """
    Generate a summary report from batch results.
    
    Args:
        batch_results: Dictionary of batch results
        args: Command-line arguments
        
    Returns:
        Path to summary report
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating summary report")
    
    # Extract performance metrics
    performance_data = []
    
    for result in batch_results['results']:
        if result['status'] == 'completed' and 'performance' in result:
            performance_data.append({
                'symbol': result['symbol'],
                'timeframe': result['timeframe'],
                'total_return': result['performance']['total_return'],
                'sharpe_ratio': result['performance']['sharpe_ratio'],
                'sortino_ratio': result['performance']['sortino_ratio'],
                'win_rate': result['performance']['win_rate'],
                'profit_factor': result['performance']['profit_factor'],
                'max_drawdown': result['performance']['max_drawdown'],
                'num_trades': result['performance']['num_trades']
            })
    
    # Create DataFrame
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        
        # Save to CSV
        summary_path = os.path.join(args.results_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        performance_df.to_csv(summary_path, index=False)
        
        # Create visualizations if many results
        if len(performance_df) > 5 and args.visualize:
            try:
                # Create visualizer
                visualizer = Visualizer(result_dir=args.results_dir)
                
                # Create pivot tables
                returns_pivot = performance_df.pivot(index='timeframe', columns='symbol', values='total_return')
                sharpe_pivot = performance_df.pivot(index='timeframe', columns='symbol', values='sharpe_ratio')
                
                # Heatmap of returns
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(returns_pivot, annot=True, cmap='RdYlGn', center=0)
                plt.title('Total Return (%)')
                plt.tight_layout()
                
                returns_heatmap_path = os.path.join(args.results_dir, f"returns_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(returns_heatmap_path)
                plt.close()
                
                # Heatmap of Sharpe ratios
                plt.figure(figsize=(12, 8))
                sns.heatmap(sharpe_pivot, annot=True, cmap='RdYlGn', center=0)
                plt.title('Sharpe Ratio')
                plt.tight_layout()
                
                sharpe_heatmap_path = os.path.join(args.results_dir, f"sharpe_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(sharpe_heatmap_path)
                plt.close()
                
                # Bar chart of top performers
                top_performers = performance_df.sort_values('total_return', ascending=False).head(10)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='total_return', y='symbol', hue='timeframe', data=top_performers)
                plt.title('Top Performers by Total Return')
                plt.tight_layout()
                
                top_performers_path = os.path.join(args.results_dir, f"top_performers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(top_performers_path)
                plt.close()
                
                logger.info(f"Summary visualizations saved to {args.results_dir}")
            
            except Exception as e:
                logger.warning(f"Error creating summary visualizations: {e}")
        
        logger.info(f"Summary report saved to {summary_path}")
        return summary_path
    else:
        logger.warning("No performance data to generate summary report")
        return ""

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AI Trading Bot Batch Run")
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save run configuration
    config_path = os.path.join(args.results_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
    
    try:
        # Process all symbols and timeframes
        batch_results = process_batch(args)
        
        # Generate summary report
        summary_path = generate_summary_report(batch_results, args)
        
        # Log completion
        logger.info("Batch run completed successfully")
        
        return 0
    
    except Exception as e:
        import traceback
        logger.error(f"Error in batch run: {e}")
        logger.debug(traceback.format_exc())
        
        return 1

if __name__ == "__main__":
    import logging
    sys.exit(main())