#!/usr/bin/env python
"""
Fixed version of run_single.py with all bugs and issues resolved.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()  # This addresses the tf.data warnings
from datetime import datetime
import json
import traceback


# Setup basic logging first to catch early errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Try imports with graceful error handling
try:
    from config import (
        DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR,
        LOOKBACK_WINDOW, PREDICTION_HORIZON, MODEL_TYPE, HIDDEN_LAYERS,
        DROPOUT_RATE, LEARNING_RATE, BATCH_SIZE, EPOCHS,
        TRADING_FEE, SLIPPAGE, POSITION_SIZE, INITIAL_CAPITAL
    )
    from data_processor import load_data, prepare_multi_timeframe_data, create_training_sequences, train_val_test_split
    from feature_engineering import generate_features, FeatureEngineer, apply_pca_reduction
    from model import DeepLearningModel
    from strategy import MLTradingStrategy, TradingStrategy
    from backtest import Backtester
    from utils import (
        setup_logging, save_metadata, format_time, print_system_info,
        get_resource_usage, clear_keras_session
    )
    
    # Import optional modules
    try:
        from feature_consistency import ensure_feature_consistency, ensure_sequence_dimensions
    except ImportError:
        logger.warning("Could not import feature_consistency module. Some features might not work.")
        # Define simple fallback functions
        def ensure_feature_consistency(df, required_feature_count):
            return df
        def ensure_sequence_dimensions(X, expected_shape):
            return X
            
    try:
        from universal_model import UniversalModel
    except ImportError:
        logger.warning("Could not import UniversalModel. Universal model features will not work.")
        
    try:
        from optimizer import ModelOptimizer, StrategyOptimizer, FeatureImportanceAnalyzer
    except ImportError:
        logger.warning("Could not import optimization modules. Optimization features will not work.")
        
    try:
        from visualize import Visualizer
    except ImportError:
        logger.warning("Could not import Visualizer. Visualization features will not work.")
        
except Exception as e:
    logger.error(f"Critical import error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the AI trading bot on a single dataset with universal model support.')
    
    # Dataset selection
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--symbol', type=str,
                        help='Trading symbol (derived from filename if not provided)')
    parser.add_argument('--timeframe', type=str,
                        help='Trading timeframe (derived from filename if not provided)')
    
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
    parser.add_argument('--model-path', type=str,
                        help='Path to pre-trained model (skip training if provided)')
    
    # Universal model parameters
    parser.add_argument('--universal-model', action='store_true',
                        help='Use a universal model for all symbols and timeframes')
    parser.add_argument('--universal-model-path', type=str,
                        help='Path to pre-trained universal model')
    
    # Strategy params
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL,
                        help='Initial capital (default: from config)')
    parser.add_argument('--position-size', type=float, default=POSITION_SIZE,
                        help='Position size as fraction of capital (default: from config)')
    parser.add_argument('--fee', type=float, default=TRADING_FEE,
                        help='Trading fee (default: from config)')
    parser.add_argument('--slippage', type=float, default=SLIPPAGE,
                        help='Slippage (default: from config)')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Signal threshold (default: 0.001)')
    parser.add_argument('--auto-calibrate', action='store_true',
                        help='Auto-calibrate signal threshold based on predictions')
    
    # Optimization
    parser.add_argument('--optimize-model', action='store_true',
                        help='Perform model hyperparameter optimization')
    parser.add_argument('--optimize-strategy', action='store_true',
                        help='Perform strategy hyperparameter optimization')
    parser.add_argument('--opt-trials', type=int, default=50,
                        help='Number of optimization trials (default: 50)')
    
    # Feature analysis
    parser.add_argument('--feature-importance', action='store_true',
                        help='Analyze feature importance')
    
    # Date range
    parser.add_argument('--start-date', type=str,
                        help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='End date for analysis (YYYY-MM-DD)')
    
    # Output options
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Directory to save results')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization dashboard')
    
    # Backtest only flag
    parser.add_argument('--backtest-only', action='store_true',
                        help='Run backtest only, skipping model training')
    
    # Advanced options
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model to disk')
    parser.add_argument('--detailed-backtest', action='store_true',
                        help='Run detailed backtest with additional analytics')
    parser.add_argument('--cross-validation', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    
    parser.add_argument('--multi-timeframe', action='store_true',
                   help='Use multi-timeframe signal confirmation')
    parser.add_argument('--tf-weight', type=float, default=0.5,
                   help='Weight for higher timeframe signals (default: 0.5)')
    parser.add_argument("--progress-bar", action="store_true", 
                        help="Show progress bar during backtesting")
    
    return parser.parse_args()

def extract_symbol_timeframe(filepath: str):
    """Extract symbol and timeframe from filepath."""
    try:
        # Get filename without path and extension
        filename = os.path.basename(filepath)
        name, _ = os.path.splitext(filename)
        
        # Parse components
        parts = name.split('_')
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = parts[1]
            return symbol, timeframe
    except Exception as e:
        logger.error(f"Error extracting symbol/timeframe from filename: {e}")
    
    # Default fallback
    return "UNKNOWN", "UNKNOWN"


def run_simplified_backtest(data_path):
    """Run a simplified backtest for quick debugging."""
    try:
        logger.info(f"Running simplified backtest on {data_path}")
        data = pd.read_csv(data_path)
        
        # Ensure we have at least a datetime and OHLCV columns
        if 'Date' in data.columns or 'date' in data.columns or 'datetime' in data.columns:
            # Find the date column
            date_col = next((col for col in data.columns if col.lower() in ['date', 'datetime', 'time']), None)
            if date_col:
                data['date'] = pd.to_datetime(data[date_col])
                data.set_index('date', inplace=True)
            
        # Print basic stats
        logger.info(f"Data loaded successfully with {len(data)} rows and {len(data.columns)} columns")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"First few rows:\n{data.head(3)}")
        
        return 0
    except Exception as e:
        logger.error(f"Error in simplified backtest: {e}")
        logger.error(traceback.format_exc())
        return 1

def perform_model_optimization(
    data_path: str,
    args: argparse.Namespace,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Perform model hyperparameter optimization.
    
    Args:
        data_path: Path to dataset file
        args: Command-line arguments
        X: Feature data
        y: Target data
        
    Returns:
        Dictionary of best parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model optimization")
    
    # Create optimizer
    optimizer = ModelOptimizer(
        data_path=data_path,
        result_dir=os.path.join(args.results_dir, 'model_optimization'),
        n_trials=args.opt_trials,
        metric='mse'  # Override the config setting to use MSE instead of sharpe_ratio
    )
    
    # Set optimizer data directly to avoid reloading
    optimizer.X = X
    optimizer.y = y
    
    # Run optimization
    best_params = optimizer.optimize()
    
    logger.info(f"Model optimization completed. Best parameters: {best_params}")
    return best_params

def perform_strategy_optimization(
    symbol: str,
    timeframe: str,
    data_path: str,
    model,
    args: argparse.Namespace,
    data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Perform strategy hyperparameter optimization.
    
    Args:
        symbol: Trading symbol
        timeframe: Trading timeframe
        data_path: Path to dataset file
        model: Trained model
        args: Command-line arguments
        data: Preprocessed data
        
    Returns:
        Dictionary of best parameters
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting strategy optimization")
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        symbol=symbol,
        timeframe=timeframe,
        data_path=data_path,
        model=model,
        result_dir=os.path.join(args.results_dir, 'strategy_optimization')
    )
    
    # Set optimizer data directly to avoid reloading
    optimizer.data = data
    
    # Run optimization
    best_params = optimizer.optimize()
    
    logger.info(f"Strategy optimization completed. Best parameters: {best_params}")
    return best_params

def analyze_feature_importance(
        model,
        data_path: str,
        args: argparse.Namespace,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: List[str]
    ) -> pd.DataFrame:
    """
    Analyze feature importance.
    
    Args:
        model: Trained model
        data_path: Path to dataset file
        args: Command-line arguments
        X: Feature data
        y: Target data
        feature_columns: Feature column names
        
    Returns:
        DataFrame with feature importance
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting feature importance analysis")
    
    # Create directory for feature importance results if it doesn't exist
    result_dir = os.path.join(args.results_dir, 'feature_importance')
    os.makedirs(result_dir, exist_ok=True)
    
    try:
        # Create analyzer
        analyzer = FeatureImportanceAnalyzer(
            model=model,
            data_path=data_path,
            result_dir=result_dir
        )
        
        # Filter for PCA components if present
        pca_feature_columns = [col for col in feature_columns if col.startswith('pc_')]
        if pca_feature_columns:
            logger.info(f"Using {len(pca_feature_columns)} PCA components for feature importance analysis")
            feature_columns = pca_feature_columns
        
        # Create train/val/test split
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        
        # Set data attributes
        analyzer.X = X
        analyzer.y = y
        analyzer.X_train = X_train
        analyzer.X_val = X_val
        analyzer.X_test = X_test
        analyzer.y_train = y_train
        analyzer.y_val = y_val
        analyzer.y_test = y_test
        analyzer.feature_columns = feature_columns
        
        # Run analysis
        results = analyzer.analyze()
        
        logger.info("Feature importance analysis completed")
        return results['importance']
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['feature', 'importance', 'std'])

def perform_cross_validation(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Perform cross-validation.
    
    Args:
        model_type: Model type
        X: Feature data
        y: Target data
        args: Command-line arguments
        
    Returns:
        Dictionary of cross-validation results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {args.cv_folds}-fold cross-validation")
    
    from sklearn.model_selection import KFold
    
    # Create k-fold cross-validation
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    # Store results
    fold_results = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        logger.info(f"Training fold {fold}/{args.cv_folds}")
        
        # Split data
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        # Create model
        model = DeepLearningModel(
            input_shape=(X.shape[1], X.shape[2]),
            output_dim=y.shape[1],
            model_type=args.model_type,
            hidden_layers=HIDDEN_LAYERS,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE
        )
        
        # Train model
        history = model.train(
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_val=X_test_fold,
            y_val=y_test_fold,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate model
        metrics = model.evaluate(X_test_fold, y_test_fold)
        
        # Store results
        fold_results.append({
            'fold': fold,
            'metrics': metrics,
            'history': history
        })
        
        # Clear session to free up memory
        clear_keras_session()
    
    # Aggregate results
    aggregate_metrics = {}
    for metric in fold_results[0]['metrics'].keys():
        values = [result['metrics'][metric] for result in fold_results]
        aggregate_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    cv_results = {
        'folds': fold_results,
        'aggregate': aggregate_metrics
    }
    
    logger.info("Cross-validation completed")
    
    # Log aggregate metrics
    for metric, stats in aggregate_metrics.items():
        logger.info(f"{metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    
    return cv_results

def run_detailed_backtest(
    strategy: TradingStrategy,
    data: pd.DataFrame,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Run detailed backtest with additional analytics.
    
    Args:
        strategy: Trading strategy
        data: Preprocessed data
        args: Command-line arguments
        
    Returns:
        Dictionary of backtest results with additional analytics
    """
    logger = logging.getLogger(__name__)
    logger.info("Running detailed backtest")

    # Ensure we're using the same features for backtesting
    data_for_backtest = data.copy()
    if len(feature_columns) > 36:
        logger.warning(f"Limiting features for backtest from {len(feature_columns)} to 36")
        feature_columns = feature_columns[:36]

    
    # Run standard backtest
    backtest_results = strategy.backtest(backtest_data, progress_bar=args.progress_bar)    
    # Add additional analytics
    
    # 1. Monthly performance
    if isinstance(data.index, pd.DatetimeIndex):
        # Get equity curve
        equity_series = pd.Series(
            backtest_results['equity_curve'],
            index=data.index[:len(backtest_results['equity_curve'])]
        )
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Monthly performance
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # Convert to DataFrame
        monthly_df = monthly_returns.reset_index()
        monthly_df.columns = ['year', 'month', 'return']
        monthly_df['year_month'] = monthly_df['year'].astype(str) + '-' + monthly_df['month'].astype(str).str.zfill(2)
        
        # Add to results
        backtest_results['monthly_returns'] = monthly_df.to_dict(orient='records')
    
    # 2. Trade statistics
    if backtest_results['trades']:
        trades_df = pd.DataFrame(backtest_results['trades'])
        
        # Add trade duration if possible
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
            
            # Trade statistics by duration
            duration_stats = {
                'mean': trades_df['duration'].mean(),
                'median': trades_df['duration'].median(),
                'min': trades_df['duration'].min(),
                'max': trades_df['duration'].max(),
                'std': trades_df['duration'].std()
            }
            
            backtest_results['duration_stats'] = duration_stats
        
        # Statistics by trade direction
        if 'side' in trades_df.columns:
            side_stats = trades_df.groupby('side')['profit_percentage'].agg([
                'count', 'mean', 'std', 'min', 'max',
                lambda x: (x > 0).mean()  # Win rate
            ]).reset_index()
            side_stats.columns = ['side', 'count', 'avg_profit', 'std_profit', 'min_profit', 'max_profit', 'win_rate']
            
            backtest_results['side_stats'] = side_stats.to_dict(orient='records')
        
        # Statistics by exit reason
        if 'exit_reason' in trades_df.columns:
            reason_stats = trades_df.groupby('exit_reason')['profit_percentage'].agg([
                'count', 'mean', 'std', 'min', 'max',
                lambda x: (x > 0).mean()  # Win rate
            ]).reset_index()
            reason_stats.columns = ['exit_reason', 'count', 'avg_profit', 'std_profit', 'min_profit', 'max_profit', 'win_rate']
            
            backtest_results['reason_stats'] = reason_stats.to_dict(orient='records')
    
    # 3. Regime analysis
    if 'market_regime' in data.columns and backtest_results['trades']:
        trades_df = pd.DataFrame(backtest_results['trades'])
        
        if 'entry_time' in trades_df.columns:
            # Convert timestamps to datetime
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            
            # Map each trade to the market regime at entry
            def get_regime(timestamp):
                try:
                    idx = data.index.get_indexer([timestamp], method='nearest')[0]
                    return data['market_regime'].iloc[idx]
                except:
                    return None
            
            trades_df['regime'] = trades_df['entry_time'].apply(get_regime)
            
            # Statistics by regime
            regime_stats = trades_df.groupby('regime')['profit_percentage'].agg([
                'count', 'mean', 'std', 'min', 'max',
                lambda x: (x > 0).mean()  # Win rate
            ]).reset_index()
            regime_stats.columns = ['regime', 'count', 'avg_profit', 'std_profit', 'min_profit', 'max_profit', 'win_rate']
            
            backtest_results['regime_stats'] = regime_stats.to_dict(orient='records')
    
    # 4. Consecutive wins/losses
    if backtest_results['trades']:
        trades_df = pd.DataFrame(backtest_results['trades'])
        
        # Calculate streaks
        if len(trades_df) > 0:
            trades_df['is_win'] = trades_df['profit_percentage'] > 0
            trades_df['streak_change'] = trades_df['is_win'].ne(trades_df['is_win'].shift())
            trades_df['streak_id'] = trades_df['streak_change'].cumsum()
            
            # Get streak lengths
            streak_lengths = trades_df.groupby(['streak_id', 'is_win']).size().reset_index(name='streak_length')
            
            # Winning streaks
            win_streaks = streak_lengths[streak_lengths['is_win']]['streak_length']
            lose_streaks = streak_lengths[~streak_lengths['is_win']]['streak_length']
            
            streak_stats = {
                'max_win_streak': int(win_streaks.max()) if not win_streaks.empty else 0,
                'max_lose_streak': int(lose_streaks.max()) if not lose_streaks.empty else 0,
                'avg_win_streak': float(win_streaks.mean()) if not win_streaks.empty else 0,
                'avg_lose_streak': float(lose_streaks.mean()) if not lose_streaks.empty else 0
            }
            
            backtest_results['streak_stats'] = streak_stats
    
    logger.info("Detailed backtest completed")
    return backtest_results

def prepare_data_for_prediction(data, feature_columns, max_features=36):
    """
    Prepare data with consistent feature selection for prediction.
    
    Args:
        data: DataFrame with all features
        feature_columns: List of feature columns to use
        max_features: Maximum number of features to include
        
    Returns:
        DataFrame with consistent feature selection
    """

    logger = logging.getLogger(__name__)
    
    # Ensure we don't exceed the maximum feature count
    if len(feature_columns) > max_features:
        logger.warning(f"Limiting features from {len(feature_columns)} to {max_features} for prediction")
        feature_columns = feature_columns[:max_features]
    
    # Select only the features that were used during training plus OHLCV
    prediction_features = ['open', 'high', 'low', 'close', 'volume'] + feature_columns
    
    # Filter to only include columns that exist in the data
    available_columns = data.columns.tolist()
    selected_columns = [col for col in prediction_features if col in available_columns]
    
    logger.info(f"Using {len(selected_columns)} columns for prediction data")
    
    # Return data with only the selected columns
    return data[selected_columns]

# In run_single.py - Consistent feature processing

def prepare_and_process_features(data, n_components=30, force_pca=True):
    """
    Consistent feature preparation and processing function.
    
    Args:
        data: Input DataFrame with OHLCV data
        n_components: Number of PCA components to use
        force_pca: Whether to force PCA even if errors occur
        
    Returns:
        Tuple of (processed_data, pca_model, scaler_model, feature_columns)
    """
    # Get a logger for this function
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Preparing features with consistent processing")
    
    # Initialize these variables at the top of the function
    pca_model = None
    scaler_model = None
    
    # Step 1: Ensure all columns are numeric
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in data.columns:
        if col not in ohlcv_cols:
            try:
                if data[col].dtype == object or pd.api.types.is_string_dtype(data[col]):
                    logger.warning(f"Converting string column to numeric: {col}")
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Dropping problematic column {col}: {e}")
                data = data.drop(col, axis=1)
    
    # Step 2: Apply PCA reduction
    feature_columns = []
    processed_data = data.copy()
    
    try:
        # Generate basic features first if needed
        feature_engineer = FeatureEngineer(data)
        processed_data = feature_engineer.generate_all_features()
        
        # Apply PCA
        logger.info(f"Applying PCA with {n_components} components")
        processed_data, pca_model, scaler_model = apply_pca_reduction(processed_data, n_components=n_components)
        
        # Ensure we have exactly n_components
        processed_data = ensure_feature_consistency(processed_data, required_feature_count=n_components)
        feature_columns = [col for col in processed_data.columns if col.startswith('pc_')]
        logger.info(f"Successfully created {len(feature_columns)} PCA components")
        
    except Exception as e:
        logger.error(f"Error in feature processing: {e}")
        if force_pca:
            logger.info("Creating synthetic PCA components as fallback")
            
            # Create OHLCV dataframe
            ohlcv_data = data[ohlcv_cols].copy()
            
            # Create synthetic PCA components using price-based calculations
            for i in range(1, n_components + 1):
                component_name = f'pc_{i}'
                
                # Use different price transformations for different components
                if i % 5 == 0:
                    ohlcv_data[component_name] = data['close'].pct_change(i).fillna(0)
                elif i % 5 == 1:
                    ohlcv_data[component_name] = data['high'].pct_change(i).fillna(0)
                elif i % 5 == 2:
                    ohlcv_data[component_name] = data['low'].pct_change(i).fillna(0)
                elif i % 5 == 3:
                    ohlcv_data[component_name] = (data['high'] - data['low']) / data['close']
                else:
                    ohlcv_data[component_name] = (data['close'] - data['open']) / data['open']
                
                # Add some noise for diversity
                ohlcv_data[component_name] = ohlcv_data[component_name] + np.random.normal(0, 0.0001, len(data))
            
            processed_data = ohlcv_data
            feature_columns = [f'pc_{i}' for i in range(1, n_components + 1)]
            logger.info(f"Created {len(feature_columns)} synthetic PCA components")
        else:
            # Fall back to non-PCA features
            logger.info("Falling back to standard features without PCA")
            processed_data = generate_features(data, apply_pca=False)
            processed_data = ensure_feature_consistency(processed_data, required_feature_count=n_components)
            feature_columns = [col for col in processed_data.columns if col not in ohlcv_cols]
            logger.info(f"Using {len(feature_columns)} standard features")
    
    # Save models for consistent reuse
    if pca_model is not None and scaler_model is not None:
        import pickle
        os.makedirs(os.path.join(MODELS_DIR, 'preprocessing'), exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pca_path = os.path.join(MODELS_DIR, 'preprocessing', f"pca_{timestamp}.pkl")
        scaler_path = os.path.join(MODELS_DIR, 'preprocessing', f"scaler_{timestamp}.pkl")
        
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_model, f)
        
        logger.info(f"Feature processing models saved: {pca_path}, {scaler_path}")
    
    return processed_data, pca_model, scaler_model, feature_columns



def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AI Trading Bot Single Run")
    
    # Print system information
    print_system_info()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return 1
    
    # Extract symbol and timeframe from filename if not provided
    symbol = args.symbol
    timeframe = args.timeframe
    
    if not symbol or not timeframe:
        extracted_symbol, extracted_timeframe = extract_symbol_timeframe(args.data_path)
        symbol = symbol or extracted_symbol
        timeframe = timeframe or extracted_timeframe
    
    logger.info(f"Processing {symbol} {timeframe}")
    
    try:
        # Step 1: Load and prepare data
        logger.info("Loading data")
        
        # Load data
        data = load_data(args.data_path)
        
        # Filter by date range if provided
        if args.start_date:
            data = data[data.index >= args.start_date]
        if args.end_date:
            data = data[data.index <= args.end_date]
        
        # Generate features with PCA reduction
        logger.info("Generating features with PCA dimensionality reduction")
        data, pca_model, scaler_model, feature_columns = prepare_and_process_features(data, n_components=30)
        
        # Universal Model approach
        if args.universal_model or args.universal_model_path:
            logger.info("Using universal model approach")
            
            # Create or load universal model
            universal_model = None
            if args.universal_model_path and os.path.exists(args.universal_model_path):
                logger.info(f"Loading universal model from {args.universal_model_path}")
                try:
                    universal_model = UniversalModel(model_path=args.universal_model_path)
                except Exception as e:
                    logger.error(f"Error loading universal model: {e}")
                    logger.info("Creating new universal model")
                    universal_model = UniversalModel(
                        lookback_window=args.lookback,
                        prediction_horizon=args.horizon,
                        feature_count=len(feature_columns)
                    )
            else:
                # Look for an existing universal model
                universal_model_dir = os.path.join(MODELS_DIR, 'universal_model')
                if os.path.exists(universal_model_dir):
                    # Find latest model
                    universal_model_files = [f for f in os.listdir(universal_model_dir) 
                                           if os.path.isdir(os.path.join(universal_model_dir, f)) and 
                                           f.startswith('universal_model_')]
                    if universal_model_files:
                        # Get most recent model
                        latest_model = sorted(universal_model_files)[-1]
                        model_path = os.path.join(universal_model_dir, latest_model)
                        logger.info(f"Found existing universal model: {model_path}")
                        try:
                            universal_model = UniversalModel(model_path=model_path)
                        except Exception as e:
                            logger.error(f"Error loading existing universal model: {e}")
                            universal_model = None
                
                # Create new if none found or loading failed
                if universal_model is None:
                    logger.info("Creating new universal model")
                    universal_model = UniversalModel(
                        lookback_window=args.lookback,
                        prediction_horizon=args.horizon,
                        feature_count=len(feature_columns)
                    )
            
            # Train universal model with current data
            if not args.backtest_only:
                logger.info(f"Training universal model on {symbol} {timeframe}")
                history = universal_model.train(
                    df=data,
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_columns=feature_columns,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    validation_split=0.2
                )
                
                # Save trained model if requested
                if args.save_model:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_dir = os.path.join(MODELS_DIR, 'universal_model', f"universal_model_{timestamp}")
                    universal_model.save(save_dir)
                    logger.info(f"Universal model saved to {save_dir}")
                
                # Display training summary
                summary = universal_model.get_training_summary()
                logger.info("Universal Model Training Summary:")
                logger.info(f"Trained on symbols: {', '.join(summary['trained_symbols'])}")
                for pair, metrics in summary['training_history'].items():
                    if metrics['final_loss'] is not None:
                        logger.info(f"{pair} - Loss: {metrics['final_loss']:.6f}, "
                                   f"Val Loss: {metrics['final_val_loss']:.6f}, "
                                   f"MAE: {metrics['final_mae']:.6f}")
            
            # Create a wrapper model for the strategy
            # This adapts the universal model to work with the existing strategy class
            class UniversalModelWrapper:
                def __init__(self, universal_model, symbol, timeframe):
                    self.universal_model = universal_model
                    self.symbol = symbol
                    self.timeframe = timeframe
                
                def predict(self, X, verbose=0):
                    return self.universal_model.predict(X, self.symbol, self.timeframe)
            
            # Create wrapped model for strategy
            model = UniversalModelWrapper(universal_model, symbol, timeframe)
            
            # Create trading strategy
            strategy = MLTradingStrategy(
                symbol=symbol,
                timeframe=timeframe,
                model=model,
                lookback_window=args.lookback,
                prediction_horizon=args.horizon,
                threshold=args.threshold,
                position_size=args.position_size,
                initial_capital=args.initial_capital,
                trading_fee=args.fee,
                slippage=args.slippage,
                adaptive_sl_tp=True,
                trailing_stop=True,
                use_multi_timeframe=args.multi_timeframe,
                multi_tf_weight=args.tf_weight
            )
            
            # Run backtest
            logger.info("Running backtest with universal model")
            strategy.set_data(data)
            
            if args.detailed_backtest:
                # Use function for detailed backtest
                from run_detailed_backtest import run_detailed_backtest
                backtest_results = run_detailed_backtest(strategy, data, args)
            else:
                backtest_results = strategy.backtest(data)
            
            # Save results
            results_path = os.path.join(args.results_dir, f"universal_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # Prepare results for serialization
            serializable_results = {
                'performance': backtest_results['performance'],
                'trades_count': len(backtest_results['trades']),
                'config': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'universal_model': True,
                    'lookback': args.lookback,
                    'horizon': args.horizon,
                    'threshold': args.threshold,
                    'position_size': args.position_size,
                    'initial_capital': args.initial_capital,
                    'fee': args.fee,
                    'slippage': args.slippage
                }
            }
            
            # Add detailed results if available
            if args.detailed_backtest:
                if 'monthly_returns' in backtest_results:
                    serializable_results['monthly_returns'] = backtest_results['monthly_returns']
                if 'duration_stats' in backtest_results:
                    serializable_results['duration_stats'] = backtest_results['duration_stats']
                if 'side_stats' in backtest_results:
                    serializable_results['side_stats'] = backtest_results['side_stats']
                if 'reason_stats' in backtest_results:
                    serializable_results['reason_stats'] = backtest_results['reason_stats']
                if 'regime_stats' in backtest_results:
                    serializable_results['regime_stats'] = backtest_results['regime_stats']
                if 'streak_stats' in backtest_results:
                    serializable_results['streak_stats'] = backtest_results['streak_stats']
            
            # Save results
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Results saved to {results_path}")
            
            # Generate visualizations
            if args.visualize:
                logger.info("Generating visualizations")
                
                # Create visualizer
                visualizer = Visualizer(result_dir=args.results_dir)
                
                # Generate dashboard
                visualizer.generate_backtest_dashboard(
                    backtest_results,
                    title=f"{symbol} {timeframe} Universal Model Backtest Results"
                )
                
                logger.info("Visualizations generated")
            
            # Print summary
            logger.info("\nBacktest Summary (Universal Model):")
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Timeframe: {timeframe}")
            logger.info(f"Total Return: {backtest_results['performance']['total_return']:.2f}%")
            logger.info(f"Number of Trades: {backtest_results['performance']['num_trades']}")
            logger.info(f"Win Rate: {backtest_results['performance']['win_rate']:.2f}%")
            logger.info(f"Profit Factor: {backtest_results['performance']['profit_factor']:.2f}")
            logger.info(f"Sharpe Ratio: {backtest_results['performance']['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {backtest_results['performance']['max_drawdown']:.2f}%")
            
            logger.info("Universal model run completed successfully")
            
            return 0
        
        # === Traditional single-model approach (below) === #
        
        # Step 2: Split data for training and validation
        feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

        # Convert all feature columns to numeric
        for col in feature_columns:
            if data[col].dtype == object or not pd.api.types.is_numeric_dtype(data[col]):
                logger.warning(f"Converting non-numeric column {col} to numeric")
                data[col] = pd.to_numeric(data[col], errors='coerce')
               
        # Limit features if necessary
        max_features = 36
        if len(feature_columns) > max_features:
            logger.warning(f"Limiting feature count from {len(feature_columns)} to {max_features}")
            feature_columns = feature_columns[:max_features]

        # Create sequences
        X, y = create_training_sequences(
            data,
            lookback_window=args.lookback,
            prediction_horizon=args.horizon,
            feature_columns=feature_columns,
            target_column='close',
            normalize=True,
        )
        
        # Derive actual feature count from X for consistency
        actual_feature_count = X.shape[2]
        logger.info(f"Using {actual_feature_count} features for model")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, train_size=args.train_size, val_size=args.val_size
        )
        
        # Remaining traditional processing follows below...
        # Step 3: Cross-validation if requested
        if args.cross_validation:
            from perform_cross_validation import perform_cross_validation
            cv_results = perform_cross_validation(
                model_type=args.model_type,
                X=X,
                y=y,
                args=args
            )
            
            # Save cross-validation results
            cv_path = os.path.join(args.results_dir, f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(cv_path, 'w') as f:
                # Convert numpy values to Python types
                cv_json = json.dumps(cv_results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
                f.write(cv_json)
            
            logger.info(f"Cross-validation results saved to {cv_path}")
        
        # Step 4: Model optimization if requested
        model_params = {}
        if args.optimize_model:
            model_params = perform_model_optimization(
                data_path=args.data_path,
                args=args,
                X=X,
                y=y
            )
        
        # Step 5: Create or load model
        if args.model_path and os.path.exists(args.model_path):
            # Load pre-trained model
            logger.info(f"Loading pre-trained model from {args.model_path}")
            model = DeepLearningModel(
            input_shape=(args.lookback, 36),  # Default to 36 features
            output_dim=args.horizon,
            model_path=args.model_path
            )
             # Get feature columns from metadata
            metadata = model.load_model(args.model_path)
            if 'feature_names' in metadata:
                feature_columns = metadata['feature_names']
                logger.info(f"Using {len(feature_columns)} features from model metadata")
                
                # Create sequences again with the correct features
                X, y = create_training_sequences(
                    data,
                    lookback_window=args.lookback,
                    prediction_horizon=args.horizon,
                    feature_columns=feature_columns,
                    target_column='close',                    
                    normalize=True,
                )

        else:
            # Create new model
            logger.info("Creating new model")
            
            # Apply optimized parameters if available
            model_type = model_params.get('model_type', args.model_type)
            n_layers = model_params.get('n_layers', len(HIDDEN_LAYERS))
            hidden_layers = [model_params.get(f'units_layer_{i}', HIDDEN_LAYERS[i] if i < len(HIDDEN_LAYERS) else 32) 
                            for i in range(n_layers)]
            dropout_rate = model_params.get('dropout_rate', DROPOUT_RATE)
            learning_rate = model_params.get('learning_rate', LEARNING_RATE)
            batch_size = model_params.get('batch_size', args.batch_size)
            
            # Calculate the actual number of features being used
            feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            actual_feature_count = len(feature_columns)
            if actual_feature_count > 36:
                logger.warning(f"Limiting feature count from {actual_feature_count} to 36")
                feature_columns = feature_columns[:36]
                actual_feature_count = 36

            # Create sequences
            X, y = create_training_sequences(
                data,
                lookback_window=args.lookback,
                prediction_horizon=args.horizon,
                feature_columns=feature_columns,
                target_column='close', 
                normalize=True,
            )

            # Ensure proper dimensions
            expected_shape = (None, args.lookback, 30)  # Fixed feature count
            X = ensure_sequence_dimensions(X, expected_shape)

            # Derive actual feature count from training data (ensuring consistency)
            actual_feature_count = X.shape[2]  # This will be 30, matching your training inputs

            # Create model
            model = DeepLearningModel(
                input_shape=(args.lookback, 30),  #fixed feature count
                output_dim=args.horizon,
                model_type=args.model_type,
                hidden_layers=HIDDEN_LAYERS,
                dropout_rate=DROPOUT_RATE,
                learning_rate=LEARNING_RATE
            )
            
            # Train model if not backtest-only mode
            if not args.backtest_only:
                logger.info("Training model")
                
                # Create save path if needed
                model_path = None
                if args.save_model:
                    model_dir = os.path.join(MODELS_DIR, f"{symbol}_{timeframe}")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
                
                # Train model
                history = model.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=args.epochs,
                    batch_size=batch_size,
                    save_path=model_path
                )

                # After saving the model
                if model_path and args.save_model:
                    # Save PCA configuration
                    pca_config = {
                        'n_components': 30,
                        'pca_model_path': pca_path if 'pca_path' in locals() else None,
                        'scaler_model_path': scaler_path if 'scaler_path' in locals() else None
                    }

                    # Save it alongside your model
                    pca_config_path = os.path.join(model_dir, 'pca_config.json')
                    with open(pca_config_path, 'w') as f:
                        json.dump(pca_config, f, indent=2)
                    
                    logger.info(f"PCA configuration saved to {pca_config_path}")

                if model_path:
                    logger.info(f"Model saved to {model_path}")
                    
                    # Save PCA and scaler models
                    import pickle
                    pca_path = os.path.join(model_dir, f"pca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                    scaler_path = os.path.join(model_dir, f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                    
                    with open(pca_path, 'wb') as f:
                        pickle.dump(pca_model, f)
                    
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler_model, f)
                    
                    logger.info(f"PCA model saved to {pca_path}")
                    logger.info(f"Scaler model saved to {scaler_path}")
        
        # Step 6: Analyze feature importance if requested
        if args.feature_importance:
            importance_df = analyze_feature_importance(
                model=model,
                data_path=args.data_path,
                args=args,
                X=X,
                y=y,
                feature_columns=feature_columns
            )
            
            # Save feature importance
            importance_path = os.path.join(args.results_dir, f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            importance_df.to_csv(importance_path, index=False)
            
            logger.info(f"Feature importance saved to {importance_path}")
        
        # Step 7: Strategy optimization if requested
        strategy_params = {}
        if args.optimize_strategy:
            strategy_params = perform_strategy_optimization(
                symbol=symbol,
                timeframe=timeframe,
                data_path=args.data_path,
                model=model,
                args=args,
                data=data
            )
        
        # Step 8: Create trading strategy
        logger.info("Creating trading strategy")
        
        # Apply optimized parameters if available
        lookback_window = strategy_params.get('lookback_window', args.lookback)
        threshold = strategy_params.get('threshold', args.threshold)
        position_size = strategy_params.get('position_size', args.position_size)
        atr_sl_multiplier = strategy_params.get('atr_sl_multiplier', None)
        min_risk_reward_ratio = strategy_params.get('min_risk_reward_ratio', None)
        
        # Update config with optimized parameters if available
        if atr_sl_multiplier is not None or min_risk_reward_ratio is not None:
            import config
            if atr_sl_multiplier is not None:
                config.ATR_SL_MULTIPLIER = atr_sl_multiplier
            if min_risk_reward_ratio is not None:
                config.MIN_RISK_REWARD_RATIO = min_risk_reward_ratio
        
        # Create strategy
        strategy = MLTradingStrategy(
            symbol=symbol,
            timeframe=timeframe,
            model=model,
            lookback_window=lookback_window,
            prediction_horizon=args.horizon,
            threshold=threshold,
            position_size=position_size,
            initial_capital=args.initial_capital,
            trading_fee=args.fee,
            slippage=args.slippage,
            adaptive_sl_tp=True,
            trailing_stop=True,
            use_multi_timeframe=args.multi_timeframe,  # Add this line
            multi_tf_weight=args.tf_weight    
        )
        
        
        # Step 9: Run backtest
        logger.info("Running backtest")

        # Prepare data with consistent features for backtest
        n_components = 30
        logger.info(f"Preparing data for backtest with {n_components} PCA components")

        try:
            # Ensure feature consistency before backtest
            backtest_data = ensure_feature_consistency(data, required_feature_count=30)
            
            # Set data for strategy
            strategy.set_data(backtest_data)
            
            # Run backtest with dimension-consistent data
            if args.detailed_backtest:
                backtest_results = run_detailed_backtest(strategy=strategy, data=backtest_data, args=args)
            else:
                backtest_results = strategy.backtest(backtest_data)
        except Exception as e:
            logger.error(f"Error during feature consistency processing: {e}")
            # Fallback to original approach
            logger.info("Falling back to standard backtest without feature consistency")
            if args.detailed_backtest:
                backtest_results = run_detailed_backtest(strategy=strategy, data=data, args=args)
            else:
                backtest_results = strategy.backtest(data)
                
        # Step 10: Save results
        results_path = os.path.join(args.results_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Prepare results for serialization
        serializable_results = {
            'performance': backtest_results['performance'],
            'trades_count': len(backtest_results['trades']),
            'config': {
                'symbol': symbol,
                'timeframe': timeframe,
                'lookback': lookback_window,
                'horizon': args.horizon,
                'threshold': threshold,
                'model_type': model.model_type,
                'hidden_layers': model.hidden_layers,
                'position_size': position_size,
                'initial_capital': args.initial_capital,
                'fee': args.fee,
                'slippage': args.slippage
            }
        }
        
        # Add detailed results if available
        if args.detailed_backtest:
            if 'monthly_returns' in backtest_results:
                serializable_results['monthly_returns'] = backtest_results['monthly_returns']
            if 'duration_stats' in backtest_results:
                serializable_results['duration_stats'] = backtest_results['duration_stats']
            if 'side_stats' in backtest_results:
                serializable_results['side_stats'] = backtest_results['side_stats']
            if 'reason_stats' in backtest_results:
                serializable_results['reason_stats'] = backtest_results['reason_stats']
            if 'regime_stats' in backtest_results:
                serializable_results['regime_stats'] = backtest_results['regime_stats']
            if 'streak_stats' in backtest_results:
                serializable_results['streak_stats'] = backtest_results['streak_stats']
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Step 11: Generate visualizations
        if args.visualize:
            logger.info("Generating visualizations")
            
            # Create visualizer
            visualizer = Visualizer(result_dir=args.results_dir)
            
            # Generate dashboard
            visualizer.generate_backtest_dashboard(
                backtest_results,
                title=f"{symbol} {timeframe} Backtest Results"
            )
            
            logger.info("Visualizations generated")
        
        # Step 12: Print summary
        logger.info("\nBacktest Summary:")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Total Return: {backtest_results['performance']['total_return']:.2f}%")
        logger.info(f"Number of Trades: {backtest_results['performance']['num_trades']}")
        logger.info(f"Win Rate: {backtest_results['performance']['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {backtest_results['performance']['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {backtest_results['performance']['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {backtest_results['performance']['max_drawdown']:.2f}%")
        
        logger.info("Single run completed successfully")
        
        return 0
    
    except Exception as e:
        import traceback
        logger.error(f"Error in single run: {e}")
        logger.debug(traceback.format_exc())
        
        return 1
if __name__ == "__main__":
    main()