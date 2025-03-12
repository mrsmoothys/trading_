"""
Hyperparameter optimization module for the AI trading bot.
"""
import logging
import os
import json
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
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Performance metric value
        """
        # Define hyperparameters to optimize
        model_type = trial.suggest_categorical('model_type', ['lstm', 'gru', 'cnn'])
        
        # Number of layers and units
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layers = []
        for i in range(n_layers):
            units = trial.suggest_int(f'units_layer_{i}', 16, 256, log=True)
            hidden_layers.append(units)
        
        # Regularization
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        # Optimization
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
        
        # Cross-validation
        cv_scores = []
        
        # Split data into folds
        fold_size = len(self.X) // self.cv_folds
        
        for fold in range(self.cv_folds):
            # Create validation indices
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            
            # Split data
            train_indices = list(range(0, val_start)) + list(range(val_end, len(self.X)))
            val_indices = list(range(val_start, val_end))
            
            X_train, y_train = self.X[train_indices], self.y[train_indices]
            X_val, y_val = self.X[val_indices], self.y[val_indices]
            
            # Create and train model
            model = DeepLearningModel(
                input_shape=(self.X.shape[1], self.X.shape[2]),
                output_dim=self.y.shape[1],
                model_type=model_type,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            # Train model
            history = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=20,  # Reduced epochs for faster optimization
                batch_size=batch_size,
                callbacks=None,
                save_path=None
            )
            
            # Evaluate model
            val_metrics = model.evaluate(X_val, y_val)
            
            # Get performance metric
            if self.metric == 'mae':
                score = -val_metrics['mae']  # Negative because we want to maximize
            elif self.metric == 'mse':
                score = -val_metrics['mse']
            elif self.metric == 'rmse':
                score = -val_metrics['rmse']
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
            
            cv_scores.append(score)
        
        # Return average score across folds
        avg_score = np.mean(cv_scores)
        
        return avg_score
    
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