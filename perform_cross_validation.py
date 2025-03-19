"""
Cross-validation module for the AI trading bot.
Provides functionality to evaluate model performance across multiple data splits.
"""
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf

from model import DeepLearningModel
from utils import format_time
from config import (
    CROSS_VALIDATION_FOLDS, EPOCHS, BATCH_SIZE,
    RESULTS_DIR
)

logger = logging.getLogger(__name__)

def perform_cross_validation(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    args: Any,
    n_splits: int = None,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Perform time series cross-validation on financial data.
    
    Args:
        model_type: Type of model to evaluate (lstm, gru, cnn, transformer, ensemble)
        X: Input features array with shape (samples, lookback, features)
        y: Target array with shape (samples, prediction_horizon)
        args: Command line arguments with model configuration
        n_splits: Number of cross-validation folds (default: from config)
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary with cross-validation results
    """
    start_time = datetime.now()
    logger.info(f"Starting {n_splits or CROSS_VALIDATION_FOLDS}-fold cross-validation for {model_type} model")
    
    # Use provided number of splits or default from config
    n_splits = n_splits or CROSS_VALIDATION_FOLDS
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Store results
    cv_results = {
        "fold_metrics": [],
        "train_sizes": [],
        "test_sizes": [],
        "training_times": [],
        "fold_histories": []
    }
    
    # Create directory for CV results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = os.path.join(RESULTS_DIR, f"cv_results_{timestamp}")
    os.makedirs(cv_dir, exist_ok=True)
    
    # Track overall metrics
    all_maes = []
    all_mses = []
    all_rmses = []
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        fold_start_time = datetime.now()
        logger.info(f"Training fold {i+1}/{n_splits}")
        
        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Further split training data to get validation set
        val_size = int(len(X_train) * 0.15)  # 15% for validation
        X_train, X_val = X_train[:-val_size], X_train[-val_size:]
        y_train, y_val = y_train[:-val_size], y_train[-val_size:]
        
        # Log sizes
        logger.info(f"Fold {i+1} - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        cv_results["train_sizes"].append(X_train.shape[0])
        cv_results["test_sizes"].append(X_test.shape[0])
        
        # Create and train model
        input_shape = (args.lookback, X.shape[2])
        hidden_layers = args.hidden_layers if hasattr(args, 'hidden_layers') else None
        dropout_rate = args.dropout_rate if hasattr(args, 'dropout_rate') else 0.2
        learning_rate = args.learning_rate if hasattr(args, 'learning_rate') else 0.001
        
        model = DeepLearningModel(
            input_shape=input_shape,
            output_dim=args.horizon,
            model_type=model_type,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Custom callbacks for each fold to prevent early stopping from using previous fold's metrics
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping]
        )
        
        # Evaluate model on test set
        metrics = model.evaluate(X_test, y_test)
        
        # Calculate fold training time
        fold_train_time = (datetime.now() - fold_start_time).total_seconds()
        cv_results["training_times"].append(fold_train_time)
        
        # Log fold results
        logger.info(f"Fold {i+1} results: MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}")
        
        # Store fold metrics
        fold_metrics = {
            "fold": i+1,
            "mae": float(metrics['mae']),
            "mse": float(metrics['mse']),
            "rmse": float(np.sqrt(metrics['mse'])),
            "train_size": int(X_train.shape[0]),
            "val_size": int(X_val.shape[0]),
            "test_size": int(X_test.shape[0]),
            "training_time": float(fold_train_time)
        }
        cv_results["fold_metrics"].append(fold_metrics)
        
        # Store fold history (convert from numpy to Python native types for JSON serialization)
        fold_history = {
            "loss": [float(x) for x in history["loss"]],
            "val_loss": [float(x) for x in history["val_loss"]],
            "mae": [float(x) for x in history["mae"]],
            "val_mae": [float(x) for x in history["val_mae"]]
        }
        cv_results["fold_histories"].append(fold_history)
        
        # Track metrics for overall stats
        all_maes.append(metrics['mae'])
        all_mses.append(metrics['mse'])
        all_rmses.append(np.sqrt(metrics['mse']))
        
        # Clear session to free up memory
        tf.keras.backend.clear_session()
    
    # Calculate overall stats
    overall_stats = {
        "mae_mean": float(np.mean(all_maes)),
        "mae_std": float(np.std(all_maes)),
        "mse_mean": float(np.mean(all_mses)),
        "mse_std": float(np.std(all_mses)),
        "rmse_mean": float(np.mean(all_rmses)),
        "rmse_std": float(np.std(all_rmses)),
        "total_folds": n_splits,
        "total_samples": X.shape[0],
        "feature_count": X.shape[2],
        "lookback_window": args.lookback,
        "prediction_horizon": args.horizon,
        "model_type": model_type,
        "total_time": (datetime.now() - start_time).total_seconds()
    }
    
    cv_results["overall_stats"] = overall_stats
    
    # Generate visualizations if requested
    if visualize:
        generate_cv_visualizations(cv_results, cv_dir)
    
    # Log overall results
    logger.info(f"Cross-validation completed in {format_time(overall_stats['total_time'])}")
    logger.info(f"Average MAE: {overall_stats['mae_mean']:.4f} ± {overall_stats['mae_std']:.4f}")
    logger.info(f"Average RMSE: {overall_stats['rmse_mean']:.4f} ± {overall_stats['rmse_std']:.4f}")
    
    return cv_results


def generate_cv_visualizations(cv_results: Dict[str, Any], output_dir: str) -> None:
    """
    Generate visualizations for cross-validation results.
    
    Args:
        cv_results: Dictionary with cross-validation results
        output_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance metrics across folds
    plt.figure(figsize=(12, 8))
    folds = [m['fold'] for m in cv_results['fold_metrics']]
    maes = [m['mae'] for m in cv_results['fold_metrics']]
    rmses = [m['rmse'] for m in cv_results['fold_metrics']]
    
    plt.subplot(2, 1, 1)
    plt.bar(folds, maes, color='skyblue')
    plt.axhline(y=cv_results['overall_stats']['mae_mean'], color='red', linestyle='--', 
                label=f'Mean: {cv_results["overall_stats"]["mae_mean"]:.4f}')
    plt.title('MAE by Fold')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.bar(folds, rmses, color='lightgreen')
    plt.axhline(y=cv_results['overall_stats']['rmse_mean'], color='red', linestyle='--',
                label=f'Mean: {cv_results["overall_stats"]["rmse_mean"]:.4f}')
    plt.title('RMSE by Fold')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_metrics_by_fold.png'))
    
    # 2. Learning curves for each fold
    plt.figure(figsize=(12, 10))
    
    for i, history in enumerate(cv_results['fold_histories']):
        plt.subplot(len(cv_results['fold_histories']), 2, 2*i+1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'Fold {i+1} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(len(cv_results['fold_histories']), 2, 2*i+2)
        plt.plot(history['mae'], label='Train MAE')
        plt.plot(history['val_mae'], label='Val MAE')
        plt.title(f'Fold {i+1} MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_learning_curves.png'))
    
    # 3. Training times by fold
    plt.figure(figsize=(10, 6))
    plt.bar(folds, cv_results['training_times'], color='salmon')
    plt.axhline(y=np.mean(cv_results['training_times']), color='red', linestyle='--',
                label=f'Mean: {np.mean(cv_results["training_times"]):.2f}s')
    plt.title('Training Time by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'cv_training_times.png'))
    
    # 4. Summary table as image
    plt.figure(figsize=(8, 6))
    plt.axis('tight')
    plt.axis('off')
    
    table_data = [
        ["Metric", "Mean", "Std Dev"],
        ["MAE", f"{cv_results['overall_stats']['mae_mean']:.4f}", 
              f"{cv_results['overall_stats']['mae_std']:.4f}"],
        ["MSE", f"{cv_results['overall_stats']['mse_mean']:.4f}", 
              f"{cv_results['overall_stats']['mse_std']:.4f}"],
        ["RMSE", f"{cv_results['overall_stats']['rmse_mean']:.4f}", 
              f"{cv_results['overall_stats']['rmse_std']:.4f}"],
        ["Training Time", f"{np.mean(cv_results['training_times']):.2f}s", 
              f"{np.std(cv_results['training_times']):.2f}s"]
    ]
    
    plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
    plt.title('Cross-Validation Summary')
    plt.savefig(os.path.join(output_dir, 'cv_summary.png'))


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data for testing
    np.random.seed(42)
    X_dummy = np.random.random((1000, 50, 36))  # 1000 samples, lookback=50, features=36
    y_dummy = np.random.random((1000, 5))       # 1000 samples, horizon=5
    
    # Create dummy args
    class DummyArgs:
        lookback = 50
        horizon = 5
        model_type = "lstm"
        hidden_layers = [128, 64]
        dropout_rate = 0.2
        learning_rate = 0.001
    
    # Run cross-validation
    results = perform_cross_validation(
        model_type="lstm",
        X=X_dummy,
        y=y_dummy,
        args=DummyArgs(),
        n_splits=3,
        visualize=True
    )
    
    print("Cross-validation completed successfully!")
    print(f"Mean MAE: {results['overall_stats']['mae_mean']}")
    print(f"Mean RMSE: {results['overall_stats']['rmse_mean']}")