"""
Hyperparameter optimization module for the AI trading bot.
"""
import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
from tqdm import tqdm

from config import (
    RESULTS_DIR, OPTIMIZATION_METRIC, OPTIMIZATION_TRIALS, CROSS_VALIDATION_FOLDS,
    LOOKBACK_WINDOW, PREDICTION_HORIZON, HIDDEN_LAYERS, DROPOUT_RATE, LEARNING_RATE
)
from data_processor import load_data, create_training_sequences, train_val_test_split
from feature_engineering import generate_features
from model import DeepLearningModel
from strategy import MLTradingStrategy
from backtest import Backtester

logger = logging.getLogger(__name__)

class OptimizationProgressCallback:
    """Callback to track and display optimization progress."""
    
    def __init__(self, n_trials, log_interval=1):
        """
        Initialize the progress tracker.
        
        Args:
            n_trials: Total number of optimization trials
            log_interval: Interval for logging progress (in seconds)
        """
        self.n_trials = n_trials
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.trials_completed = 0
        self.best_value = None
        self.trial_times = []
        
    def __call__(self, study, trial):
        """
        Update progress after each trial.
        
        Args:
            study: Optuna study object
            trial: Completed trial
        """
        # Update counters
        self.trials_completed += 1
        current_time = time.time()
        trial_time = current_time - (self.last_log_time if self.trials_completed > 1 else self.start_time)
        self.trial_times.append(trial_time)
        self.last_log_time = current_time
        
        # Update best value
        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
        
        # Log progress if interval passed or on first/last trial
        if (current_time - self.start_time >= self.log_interval or 
            self.trials_completed == 1 or 
            self.trials_completed == self.n_trials):
            
            # Calculate progress
            progress_pct = self.trials_completed / self.n_trials * 100
            elapsed_time = current_time - self.start_time
            
            # Estimate remaining time
            if self.trials_completed > 0:
                avg_time_per_trial = elapsed_time / self.trials_completed
                remaining_trials = self.n_trials - self.trials_completed
                estimated_remaining = avg_time_per_trial * remaining_trials
            else:
                estimated_remaining = 0
            
            # Format times
            elapsed_str = self.format_time(elapsed_time)
            remaining_str = self.format_time(estimated_remaining)
            
            # Log progress
            logger.info(
                f"Optimization progress: {self.trials_completed}/{self.n_trials} trials "
                f"({progress_pct:.1f}%) - Elapsed: {elapsed_str}, Remaining: {remaining_str}"
            )
            logger.info(f"Best value so far: {self.best_value:.6f} - Last trial time: {trial_time:.2f}s")
            
            # Reset log time
            self.last_log_time = current_time
    
    def format_time(self, seconds):
        """Format seconds into a readable time string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"



class ModelOptimizer:
    """
    Optimizer for deep learning model hyperparameters.
    """
    
    def __init__(
        self,
        data_path: str,
        result_dir: Optional[str] = None,
        n_trials: int = OPTIMIZATION_TRIALS,
        metric: str = OPTIMIZATION_METRIC,
        cv_folds: int = CROSS_VALIDATION_FOLDS
    ):
        """
        Initialize the model optimizer.
        
        Args:
            data_path: Path to data file
            result_dir: Directory to save results
            n_trials: Number of optimization trials
            metric: Metric to optimize
            cv_folds: Number of cross-validation folds
        """
        self.data_path = data_path
        self.result_dir = result_dir or os.path.join(RESULTS_DIR, 'optimization')
        self.n_trials = n_trials
        self.metric = metric
        self.cv_folds = cv_folds
        
        # Ensure result directory exists
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Load and prepare data
        self.data = None
        self.X = None
        self.y = None
        self.feature_columns = None
    
    def prepare_data(self) -> None:
        """
        Load and prepare data for optimization.
        """
        # Load data
        self.data = load_data(self.data_path)
        
        # Generate features
        self.data = generate_features(self.data)
        
        # Select feature columns (exclude NaN columns)
        self.feature_columns = [col for col in self.data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        non_nan_cols = self.data[self.feature_columns].columns[~self.data[self.feature_columns].isna().any()]
        self.feature_columns = list(non_nan_cols)
        
        # Create sequences
        self.X, self.y = create_training_sequences(
            self.data,
            lookback_window=LOOKBACK_WINDOW,
            prediction_horizon=PREDICTION_HORIZON,
            feature_columns=self.feature_columns,
            target_column='close',
            normalize=True
        )
        
        logger.info(f"Prepared {len(self.X)} sequences with {self.X.shape[1]} time steps and {self.X.shape[2]} features")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization with performance enhancements and progress tracking.
        
        Returns:
            Dictionary of best parameters
        """
        # Prepare data if not already prepared
        if self.data is None:
            self.prepare_data()
        
        # Use TPE sampler which is more efficient than random sampling
        from optuna.samplers import TPESampler
        sampler = TPESampler(n_startup_trials=min(10, self.n_trials // 2))
        
        # Create study with pruning
        from optuna.pruners import MedianPruner
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Create progress callback
        progress_callback = OptimizationProgressCallback(n_trials=self.n_trials, log_interval=60)  # Log every minute
        
        # Define simplified objective function
        def simplified_objective(trial):
            """Simplified objective for quicker optimization."""
            start_time = time.time()
            
            # Use a smaller subset of hyperparameters
            model_type = trial.suggest_categorical('model_type', ['lstm', 'gru'])
            
            # Simplified architecture
            n_layers = 2  # Fixed to 2 layers for simplicity
            hidden_layers = []
            for i in range(n_layers):
                # More restricted range
                units = trial.suggest_int(f'units_layer_{i}', 32, 128, log=True)
                hidden_layers.append(units)
            
            # Regularization
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            
            # Optimization
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_int('batch_size', 16, 64, log=True)
            
            # Only use 2-fold CV for faster evaluation
            cv_folds = min(self.cv_folds, 2)
            
            # Log current trial parameters
            trial_info = (
                f"Trial {trial.number + 1}/{self.n_trials}: {model_type}, "
                f"layers={hidden_layers}, dropout={dropout_rate:.2f}, "
                f"lr={learning_rate:.6f}, batch_size={batch_size}"
            )
            logger.info(f"Starting {trial_info}")
            
            # Create model and evaluate
            model = DeepLearningModel(
                input_shape=(self.X.shape[1], self.X.shape[2]),
                output_dim=self.y.shape[1],
                model_type=model_type,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            # Use smaller subset of data for initial trials
            use_subset = trial.number < self.n_trials // 2
            if use_subset:
                subset_size = len(self.X) // 2
                X_subset = self.X[:subset_size]
                y_subset = self.y[:subset_size]
                logger.info(f"Using data subset of size {subset_size} for trial {trial.number + 1}")
            else:
                X_subset = self.X
                y_subset = self.y
            
            # Single train/val split for speed
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_subset, y_subset, test_size=0.2, random_state=42
            )
            
            # Train with early stopping and fewer epochs
            history = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=20,  # Reduced epochs for faster optimization
                batch_size=batch_size,
                optimize_for_hardware=True  # Use hardware optimization
            )
            
            # Evaluate model
            metrics = model.evaluate(X_val, y_val)
            
            # Log trial results
            trial_time = time.time() - start_time
            logger.info(
                f"Completed {trial_info} in {trial_time:.2f}s - "
                f"MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}"
            )
            
            # Clear memory
            from utils import check_memory_usage, clear_keras_session
            check_memory_usage(threshold_gb=0.8, clear_if_high=True)
            clear_keras_session()
            
            # Return negative MSE (to maximize)
            return -metrics['mse']
        
        # Run optimization with progress callback
        logger.info(f"Starting optimization with {self.n_trials} trials")
        start_time = time.time()
        
        study.optimize(simplified_objective, n_trials=self.n_trials, callbacks=[progress_callback])
        
        # Log total optimization time
        total_time = time.time() - start_time
        hours, remainder = divmod(int(total_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Optimization completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Log final results
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info("======= Optimization Results =======")
        logger.info(f"Best MSE: {-best_value:.6f}")
        logger.info(f"Best parameters: {json.dumps(best_params, indent=2)}")
        logger.info("===================================")
        
        # Save results
        self.save_results(study)
        
        return best_params


    def save_results(self, study: optuna.Study) -> str:
        """
        Save optimization results.
        
        Args:
            study: Optuna study
            
        Returns:
            Path to saved results
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create result directory
        result_path = os.path.join(self.result_dir, f"model_opt_{timestamp}")
        os.makedirs(result_path, exist_ok=True)
        
        # Save best parameters
        params_path = os.path.join(result_path, "best_params.json")
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save study values
        values_path = os.path.join(result_path, "study_values.csv")
        values_df = pd.DataFrame(
            [{"trial": t.number, "value": t.value, **t.params} for t in study.trials]
        )
        values_df.to_csv(values_path, index=False)
        
        # Create visualization plots
        try:
            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(result_path, "optimization_history.png"))
            
            # Parameter importance
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(result_path, "param_importances.png"))
            
            # Parameter slices
            for param in study.best_params:
                try:
                    fig = plot_slice(study, params=param)
                    fig.write_image(os.path.join(result_path, f"slice_{param}.png"))
                except:
                    pass
            
            # Contour plots for selected params
            if 'dropout_rate' in study.best_params and 'learning_rate' in study.best_params:
                fig = plot_contour(study, params=['dropout_rate', 'learning_rate'])
                fig.write_image(os.path.join(result_path, "contour_dropout_lr.png"))
        except Exception as e:
            logger.warning(f"Error creating plots: {e}")
        
        logger.info(f"Optimization results saved to {result_path}")
        return result_path

class StrategyOptimizer:
    """
    Optimizer for trading strategy hyperparameters.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        data_path: str,
        model,
        result_dir: Optional[str] = None,
        n_trials: int = OPTIMIZATION_TRIALS,
        metric: str = OPTIMIZATION_METRIC
    ):
        """
        Initialize the strategy optimizer.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            data_path: Path to data file
            model: Trained ML model
            result_dir: Directory to save results
            n_trials: Number of optimization trials
            metric: Metric to optimize
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_path = data_path
        self.model = model
        self.result_dir = result_dir or os.path.join(RESULTS_DIR, 'optimization')
        self.n_trials = n_trials
        self.metric = metric
        
        # Ensure result directory exists
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Data
        self.data = None
    
    def prepare_data(self) -> None:
        """
        Load and prepare data for optimization.
        """
        # Load data
        self.data = load_data(self.data_path)
        
        # Generate features
        self.data = generate_features(self.data)
        
        logger.info(f"Prepared data with {len(self.data)} rows and {len(self.data.columns)} columns")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Performance metric value
        """
        # Define hyperparameters to optimize
        lookback_window = trial.suggest_int('lookback_window', 20, 200)
        threshold = trial.suggest_float('threshold', 0.001, 0.02)
        position_size = trial.suggest_float('position_size', 0.05, 0.5)
        
        # Adaptive SL/TP parameters
        atr_sl_multiplier = trial.suggest_float('atr_sl_multiplier', 1.0, 5.0)
        min_risk_reward_ratio = trial.suggest_float('min_risk_reward_ratio', 1.0, 3.0)
        
        # Create strategy
        strategy = MLTradingStrategy(
            symbol=self.symbol,
            timeframe=self.timeframe,
            model=self.model,
            lookback_window=lookback_window,
            prediction_horizon=PREDICTION_HORIZON,
            threshold=threshold,
            position_size=position_size,
            adaptive_sl_tp=True,
            trailing_stop=True
        )
        
        # Hack to set ATR_SL_MULTIPLIER and MIN_RISK_REWARD_RATIO
        import config
        config.ATR_SL_MULTIPLIER = atr_sl_multiplier
        config.MIN_RISK_REWARD_RATIO = min_risk_reward_ratio
        
        # Create backtester
        backtester = Backtester(
            strategy=strategy,
            data=self.data
        )
        
        # Run backtest
        results = strategy.backtest(self.data)
        
        # Get performance metric
        performance = results['performance']
        
        if self.metric == 'total_return':
            score = performance['total_return']
        elif self.metric == 'sharpe_ratio':
            score = performance['sharpe_ratio']
        elif self.metric == 'sortino_ratio':
            score = performance['sortino_ratio']
        elif self.metric == 'profit_factor':
            score = performance['profit_factor']
        elif self.metric == 'win_rate':
            score = performance['win_rate']
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Additional constraints
        # Penalize too few trades
        if performance['num_trades'] < 10:
            score = -100
        
        # Penalize high drawdown
        if performance['max_drawdown'] > 30:
            score *= 0.5
        
        return score
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary of best parameters
        """
        # Prepare data if not already prepared
        if self.data is None:
            self.prepare_data()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = study.best_params
        
        # Save results
        self.save_results(study)
        
        return best_params
    
    def save_results(self, study: optuna.Study) -> str:
        """
        Save optimization results.
        
        Args:
            study: Optuna study
            
        Returns:
            Path to saved results
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create result directory
        result_path = os.path.join(self.result_dir, f"strategy_opt_{timestamp}")
        os.makedirs(result_path, exist_ok=True)
        
        # Save best parameters
        params_path = os.path.join(result_path, "best_params.json")
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save study values
        values_path = os.path.join(result_path, "study_values.csv")
        values_df = pd.DataFrame(
            [{"trial": t.number, "value": t.value, **t.params} for t in study.trials]
        )
        values_df.to_csv(values_path, index=False)
        
        # Create visualization plots
        try:
            # Optimization history
            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(result_path, "optimization_history.png"))
            
            # Parameter importance
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(result_path, "param_importances.png"))
            
            # Parameter slices
            for param in study.best_params:
                try:
                    fig = plot_slice(study, params=param)
                    fig.write_image(os.path.join(result_path, f"slice_{param}.png"))
                except:
                    pass
            
            # Contour plots for selected params
            if 'threshold' in study.best_params and 'position_size' in study.best_params:
                fig = plot_contour(study, params=['threshold', 'position_size'])
                fig.write_image(os.path.join(result_path, "contour_threshold_position.png"))
        except Exception as e:
            logger.warning(f"Error creating plots: {e}")
        
        # Create best strategy and backtest it
        best_params = study.best_params
        
        # Create strategy with best parameters
        strategy = MLTradingStrategy(
            symbol=self.symbol,
            timeframe=self.timeframe,
            model=self.model,
            lookback_window=best_params['lookback_window'],
            prediction_horizon=PREDICTION_HORIZON,
            threshold=best_params['threshold'],
            position_size=best_params['position_size'],
            adaptive_sl_tp=True,
            trailing_stop=True
        )
        
        # Set optimized config parameters
        import config
        config.ATR_SL_MULTIPLIER = best_params['atr_sl_multiplier']
        config.MIN_RISK_REWARD_RATIO = best_params['min_risk_reward_ratio']
        
        # Create backtester
        backtester = Backtester(
            strategy=strategy,
            data=self.data
        )
        
        # Run backtest
        results = strategy.backtest(self.data)
        
        # Save performance summary
        performance_path = os.path.join(result_path, "best_performance.json")
        with open(performance_path, 'w') as f:
            json.dump(results['performance'], f, indent=2)
        
        # Generate detailed report
        backtester = Backtester(strategy=strategy, data_path=self.data_path)
        backtester.results = results
        report_path = backtester.generate_report(show_plots=False)
        
        logger.info(f"Optimization results saved to {result_path}")
        return result_path

class FeatureImportanceAnalyzer:
    """
    Analyzer for feature importance in ML trading models.
    """
    
    def __init__(
        self,
        model,
        data_path: str,
        result_dir: Optional[str] = None
    ):
        """
        Initialize the feature importance analyzer.
        
        Args:
            model: Trained ML model
            data_path: Path to data file
            result_dir: Directory to save results
        """
        self.model = model
        self.data_path = data_path
        self.result_dir = result_dir or os.path.join(RESULTS_DIR, 'feature_importance')
        
        # Ensure result directory exists
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Data
        self.data = None
        self.X = None
        self.y = None
        self.feature_columns = None
    
    def prepare_data(self) -> None:
        """
        Load and prepare data for analysis.
        """
        # Load data
        self.data = load_data(self.data_path)
        
        # Generate features
        self.data = generate_features(self.data)
        
        # Select feature columns (exclude NaN columns)
        self.feature_columns = [col for col in self.data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        non_nan_cols = self.data[self.feature_columns].columns[~self.data[self.feature_columns].isna().any()]
        self.feature_columns = list(non_nan_cols)
        
        # Create sequences
        self.X, self.y = create_training_sequences(
            self.data,
            lookback_window=LOOKBACK_WINDOW,
            prediction_horizon=PREDICTION_HORIZON,
            feature_columns=self.feature_columns,
            target_column='close',
            normalize=True
        )
        
        # Split data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = train_val_test_split(self.X, self.y)
        
        logger.info(f"Prepared {len(self.X)} sequences with {self.X.shape[1]} time steps and {self.X.shape[2]} features")
    
    def permutation_importance(self, n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation importance for features.
        
        Args:
            n_repeats: Number of permutation repeats
            
        Returns:
            DataFrame with feature importances
        """
        # Prepare data if not already prepared
        if self.data is None:
            self.prepare_data()
        
        # Get baseline performance
        baseline_metrics = self.model.evaluate(self.X_test, self.y_test)
        baseline_error = baseline_metrics['mae']
        
        # Calculate feature importances
        importances = []
        
        for i, feature_name in enumerate(tqdm(self.feature_columns, desc="Calculating feature importance")):
            feature_importance = []
            
            for _ in range(n_repeats):
                # Create a copy of the test data
                X_permuted = self.X_test.copy()
                
                # Permute feature across all time steps
                for t in range(X_permuted.shape[1]):
                    # Shuffle the feature values across samples
                    permuted_values = X_permuted[:, t, i].copy()
                    np.random.shuffle(permuted_values)
                    X_permuted[:, t, i] = permuted_values
                
                # Evaluate with permuted feature
                permuted_metrics = self.model.evaluate(X_permuted, self.y_test)
                permuted_error = permuted_metrics['mae']
                
                # Calculate importance (increase in error)
                importance = permuted_error - baseline_error
                feature_importance.append(importance)
            
            # Average importance across repeats
            avg_importance = np.mean(feature_importance)
            std_importance = np.std(feature_importance)
            
            importances.append({
                'feature': feature_name,
                'importance': avg_importance,
                'std': std_importance
            })
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame(importances)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run feature importance analysis.
        
        Returns:
            Dictionary of analysis results
        """
        # Calculate permutation importance
        importance_df = self.permutation_importance()
        
        # Save results
        result_path = self.save_results(importance_df)
        
        return {
            'importance': importance_df,
            'result_path': result_path
        }
    
    def save_results(self, importance_df: pd.DataFrame) -> str:
        """
        Save analysis results.
        
        Args:
            importance_df: DataFrame with feature importances
            
        Returns:
            Path to saved results
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create result directory
        result_path = os.path.join(self.result_dir, f"feature_importance_{timestamp}")
        os.makedirs(result_path, exist_ok=True)
        
        # Save importance table
        importance_path = os.path.join(result_path, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        
        # Create visualization plots
        
        # Feature importance bar plot
        fig, ax = plt.figure(figsize=(12, max(8, len(importance_df) * 0.3))), plt.gca()
        
        # Sort by importance
        sorted_df = importance_df.sort_values('importance')
        
        # Plot top 20 features
        plot_df = sorted_df.tail(20)
        
        # Create bar plot
        bars = ax.barh(plot_df['feature'], plot_df['importance'])
        
        # Add error bars
        ax.errorbar(
            plot_df['importance'],
            plot_df['feature'],
            xerr=plot_df['std'],
            fmt='none',
            ecolor='black',
            capsize=5
        )
        
        # Color bars by importance
        for i, bar in enumerate(bars):
            if plot_df['importance'].iloc[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.title('Feature Importance (Permutation)')
        plt.xlabel('Increase in MAE')
        plt.grid(True, axis='x')
        plt.tight_layout()
        
        # Save plot
        importance_plot_path = os.path.join(result_path, "feature_importance.png")
        plt.savefig(importance_plot_path)
        plt.close()
        
        # Generate HTML report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Importance Analysis</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2 {{ color: #333; }}
                .table {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Feature Importance Analysis</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Feature Importance</h2>
                <p>Features ranked by importance (increase in MAE when permuted):</p>
                
                <img src="{importance_plot_path}" alt="Feature Importance" style="max-width:100%;">
                
                <h2>Importance Table</h2>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature</th>
                            <th>Importance</th>
                            <th>Std</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add feature rows
        for i, (_, row) in enumerate(importance_df.iterrows()):
            report_html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{row['feature']}</td>
                    <td>{row['importance']:.6f}</td>
                    <td>{row['std']:.6f}</td>
                </tr>
            """
        
        report_html += """
                    </tbody>
                </table>
                
                <h2>Interpretation</h2>
                <p>
                    Features with higher importance values are more critical for the model's predictive performance.
                    These values represent the increase in Mean Absolute Error (MAE) when the feature is randomly permuted.
                </p>
                <p>
                    <strong>Key findings:</strong>
                </p>
                <ul>
        """
        
        # Add key findings
        top_features = importance_df.head(5)['feature'].tolist()
        report_html += f"""
                    <li>The most important features are: {', '.join(top_features)}</li>
                    <li>A total of {sum(importance_df['importance'] > 0)} features have positive importance</li>
                    <li>The top 5 features account for {top_features['importance'].sum() / importance_df['importance'].sum() * 100:.1f}% of total importance</li>
        """
        
        report_html += """
                </ul>
                
                <h2>Recommendations</h2>
                <p>
                    Based on this analysis, consider the following recommendations:
                </p>
                <ul>
                    <li>Focus on engineering and improving the top features</li>
                    <li>Consider removing features with negative or very low importance</li>
                    <li>Explore combinations or transformations of the most important features</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Write report to file
        report_path = os.path.join(result_path, "feature_importance_report.html")
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Feature importance analysis saved to {result_path}")
        return result_path

if __name__ == "__main__":
    # Test the optimizer
    import numpy as np
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy model for testing
    class DummyModel:
        def predict(self, X):
            return np.random.random((len(X), 5))
        
        def evaluate(self, X, y):
            return {'mae': np.random.random(), 'mse': np.random.random(), 'rmse': np.random.random()}
    
    # Create optimizer
    from config import DATA_DIR
    data_path = os.path.join(DATA_DIR, "BTCUSDT_1h_data_2018_to_2025.csv")
    
    if os.path.exists(data_path):
        # Test strategy optimizer
        dummy_model = DummyModel()
        optimizer = StrategyOptimizer(
            symbol="BTCUSDT",
            timeframe="1h",
            data_path=data_path,
            model=dummy_model,
            n_trials=5  # Reduced for testing
        )
        
        # Run optimization
        best_params = optimizer.optimize()
        
        print(f"Best parameters: {best_params}")
    else:
        print(f"Data file not found: {data_path}")
        print("Please download or create the data file before running the optimizer.")