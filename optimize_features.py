"""
Feature selection optimization for RSIDTrade.
Systematically tests different feature combinations to find the best performing set.
"""
import os
import sys
import time
import logging
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple
from itertools import combinations
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, MODELS_DIR, LOOKBACK_WINDOW, PREDICTION_HORIZON
from data_processor import load_data, create_training_sequences, train_val_test_split
from feature_engineering import FeatureEngineer
from model import DeepLearningModel, ModelManager
from strategy import MLTradingStrategy
from utils import setup_logging
from feature_consistency import ensure_feature_consistency, ensure_sequence_dimensions

def optimize_feature_selection(data_path: str, 
                              required_features: int = 36, 
                              population_size: int = 20,
                              generations: int = 10,
                              mutation_rate: float = 0.1,
                              test_size: float = 0.3,
                              threshold: float = 0.005,
                              model_path: str = None) -> Dict[str, Any]:
    """
    Use a genetic algorithm to find the optimal feature combination.
    
    Args:
        data_path: Path to data file
        required_features: Exact number of features required (always 36)
        population_size: Size of the population in the genetic algorithm
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation
        test_size: Proportion of data to use for testing
        threshold: Signal threshold for the strategy
        model_path: Path to existing model file to use for evaluation
        
    Returns:
        Dictionary with optimization results
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting feature selection optimization for {data_path}")
    
    start_time = time.time()
    
    # Load and prepare data
    data = load_data(data_path)
    
    # Generate all features
    feature_engineer = FeatureEngineer(data)
    full_data = feature_engineer.generate_all_features()
    
    # OHLCV columns - these are always included and not part of the feature selection
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Get all available features (excluding OHLCV)
    all_features = [col for col in full_data.columns if col not in ohlcv_columns]
    
    # Check if we have enough features
    if len(all_features) < required_features:
        logger.error(f"Not enough features available. Need {required_features}, but only have {len(all_features)}")
        return {"error": f"Not enough features available. Need {required_features}, but only have {len(all_features)}"}
    
    # Define feature categories for more intelligent selection
    feature_categories = {
        'momentum': ['rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d', 'mfi', 'price_roc', 'williams_r', 'ultimate_oscillator'],
        'trend': ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200', 'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'adx', 'adx_pos', 'adx_neg', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base_line', 'ichimoku_conversion_line', 'aroon_up', 'aroon_down', 'aroon_indicator', 'psar'],
        'volatility': ['bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'bollinger_width', 'atr', 'natr', 'keltner_upper', 'keltner_middle', 'keltner_lower', 'donchian_upper', 'donchian_middle', 'donchian_lower'],
        'volume': ['obv', 'cmf', 'vwap', 'eom', 'vwma', 'pvt'],
        'patterns': ['doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows'],
        'structure': ['swing_high', 'swing_low', 'higher_highs', 'higher_lows', 'lower_highs', 'lower_lows', 'trend_direction', 'trend_strength', 'trend', 'consolidation', 'consolidation_strength', 'volatility_regime', 'support', 'resistance']
    }
    
    # Filter available features based on categories
    categorized_features = []
    for category, features in feature_categories.items():
        for feature in features:
            if feature in all_features:
                categorized_features.append(feature)
    
    # Ensure we have features to work with
    if not categorized_features:
        logger.error("No recognized features found in the data")
        return {"error": "No recognized features found"}
    
    # Use categorized features if available, otherwise use all features
    if len(categorized_features) >= required_features:
        available_features = categorized_features
    else:
        available_features = all_features
    
    logger.info(f"Found {len(available_features)} features available for optimization")
    
    # Load existing model if provided
    custom_model = None
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        try:
            # Extract symbol and timeframe from filename if possible
            model_filename = os.path.basename(model_path)
            symbol = "BTCUSDT"  # Default
            timeframe = "1h"    # Default
            
            if "_" in model_filename:
                parts = model_filename.split("_")
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1].split(".")[0] if "." in parts[1] else parts[1]
            
            # Create model manager
            model_manager = ModelManager(base_dir=os.path.dirname(model_path))
            
            # Load model
            custom_model = DeepLearningModel(
                input_shape=(LOOKBACK_WINDOW, required_features),
                output_dim=PREDICTION_HORIZON,
                model_path=model_path
            )
            
            logger.info(f"Successfully loaded model for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Will create new models for feature evaluation")
            custom_model = None
    
    # Initialize genetic algorithm
    
    def generate_individual() -> List[str]:
        """Generate a random individual with exactly 36 features."""
        # Ensure some variety across categories
        selected_features = []
        
        # Select at least one feature from each category if possible
        for category, features in feature_categories.items():
            available = [f for f in features if f in available_features]
            if available:
                selected_features.append(random.choice(available))
        
        # Fill remaining slots to reach exactly 36 features
        remaining_slots = required_features - len(selected_features)
        if remaining_slots > 0:
            remaining_features = [f for f in available_features if f not in selected_features]
            
            # If we don't have enough remaining features, it's an error
            if len(remaining_features) < remaining_slots:
                logger.error(f"Not enough features to fill slots. Need {remaining_slots} more but only have {len(remaining_features)}")
                # Use duplicates if necessary (not ideal but better than failing)
                while len(remaining_features) < remaining_slots:
                    remaining_features.extend(available_features)
            
            selected_features.extend(random.sample(remaining_features, remaining_slots))
        
        # If we have too many features, trim down to exactly 36
        if len(selected_features) > required_features:
            selected_features = selected_features[:required_features]
        
        return selected_features
    
    def evaluate_feature_set(features: List[str]) -> Tuple[float, Dict[str, float]]:
        """Evaluate a set of features using the trading strategy."""
        try:
            # Ensure exactly 36 features
            if len(features) != required_features:
                logger.warning(f"Feature set has {len(features)} features, expected {required_features}")
                # Fill or trim as needed
                if len(features) < required_features:
                    extra_features = [f for f in available_features if f not in features]
                    features = features + random.sample(extra_features, required_features - len(features))
                else:
                    features = features[:required_features]
            
            # Select data with chosen features - OHLCV columns are always included
            selected_data = full_data[ohlcv_columns + features].copy()
            
            # Determine test data (last portion of the dataset)
            test_start_idx = int(len(selected_data) * (1 - test_size))
            train_data = selected_data.iloc[:test_start_idx]
            test_data = selected_data.iloc[test_start_idx-LOOKBACK_WINDOW:]  # Include lookback window
            
            if custom_model:
                # Use existing model
                model = custom_model
                
                # Reuse input layers but with the current feature set
                # This is a crucial step when evaluating different feature combinations
                # with an existing model architecture
                
                # Create trading strategy
                strategy = MLTradingStrategy(
                    symbol="BTCUSDT",
                    timeframe="1h",
                    model=model,
                    lookback_window=LOOKBACK_WINDOW,
                    prediction_horizon=PREDICTION_HORIZON,
                    threshold=threshold
                )
                
                # Set test data and run backtest
                strategy.set_data(test_data)
                
                # Generate signals
                backtest_results = strategy.backtest(test_data)
                
            else:
                # Create sequences for training a new model
                X, y = create_training_sequences(
                    selected_data,
                    lookback_window=LOOKBACK_WINDOW,
                    prediction_horizon=PREDICTION_HORIZON,
                    feature_columns=features,
                    target_column='close',
                    normalize=True
                )
                
                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                    X, y, train_size=1-test_size-0.1, val_size=0.1
                )
                
                # Create and train model
                model = DeepLearningModel(
                    input_shape=(X.shape[1], X.shape[2]),
                    output_dim=y.shape[1],
                    model_type='lstm',  # Use LSTM for consistency
                    hidden_layers=[128, 64]  # Simple architecture
                )
                
                model.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=10,  # Limited epochs for speed
                    batch_size=32,
                    save_path=None
                )
                
                # Create strategy
                strategy = MLTradingStrategy(
                    symbol="BTCUSDT",
                    timeframe="1h",
                    model=model,
                    lookback_window=LOOKBACK_WINDOW,
                    prediction_horizon=PREDICTION_HORIZON,
                    threshold=threshold
                )
                
                # Set test data and run backtest
                strategy.set_data(test_data)
                backtest_results = strategy.backtest(test_data)
            
            # Get performance metrics
            metrics = backtest_results['performance']
            
            # Calculate fitness (weighted combination of metrics)
            fitness = (
                metrics['sharpe_ratio'] * 0.4 +  # Prioritize risk-adjusted returns
                metrics['total_return'] * 0.3 +  # Consider total returns
                metrics['profit_factor'] * 0.2 +  # Consider consistency
                metrics['win_rate'] * 0.1        # Consider win rate
            )
            
            # Apply penalties for extreme metrics
            
            # Heavy penalty for extreme drawdowns
            if metrics['max_drawdown'] > 25:
                fitness *= 0.8
            if metrics['max_drawdown'] > 35:
                fitness *= 0.5
            
            # Penalty for too few trades
            if metrics['num_trades'] < 10:
                fitness *= 0.7
            
            # Penalty for extremely low win rate
            if metrics['win_rate'] < 30:
                fitness *= 0.8
            
            # Ensure non-negative fitness
            fitness = max(fitness, 0.001)
            
            return fitness, metrics
            
        except Exception as e:
            logger.error(f"Error evaluating feature set: {e}")
            return 0.001, {'error': str(e)}  # Very low fitness for errors
    
    def crossover(parent1: List[str], parent2: List[str]) -> List[str]:
        """Create a child by combining features from two parents while maintaining exactly 36 features."""
        # Determine crossover point
        crossover_point = random.randint(1, required_features - 1)
        
        # Create child starting with features from parent1
        child = parent1[:crossover_point]
        
        # Add features from parent2 that aren't already in the child
        for feature in parent2:
            if feature not in child and len(child) < required_features:
                child.append(feature)
        
        # If we still don't have enough features, fill with random features
        if len(child) < required_features:
            remaining = [f for f in available_features if f not in child]
            if len(remaining) < (required_features - len(child)):
                # Not enough unique features, allow duplicates from available_features
                remaining = available_features
            
            child.extend(random.sample(remaining, required_features - len(child)))
        
        # If we have too many features, trim to exactly 36
        if len(child) > required_features:
            child = child[:required_features]
        
        return child
    
    def mutate(individual: List[str]) -> List[str]:
        """Randomly mutate an individual while maintaining exactly 36 features."""
        mutated = individual.copy()
        
        # Number of mutations to apply
        num_mutations = max(1, int(required_features * mutation_rate))
        
        for _ in range(num_mutations):
            # Select a random feature to replace
            idx = random.randrange(len(mutated))
            
            # Select a new feature not already in the set
            available = [f for f in available_features if f not in mutated]
            
            # If we have unique features to choose from, use one
            if available:
                mutated[idx] = random.choice(available)
            else:
                # Otherwise, just use any available feature
                mutated[idx] = random.choice(available_features)
        
        return mutated
    
    # Initialize population
    population = [generate_individual() for _ in range(population_size)]
    
    # Track best solution and history
    best_individual = None
    best_fitness = float('-inf')
    best_metrics = {}
    history = []
    
    # Run genetic algorithm
    for generation in range(generations):
        logger.info(f"Generation {generation+1}/{generations}")
        
        # Evaluate population
        fitness_results = []
        for individual in tqdm(population, desc=f"Evaluating Generation {generation+1}"):
            fitness, metrics = evaluate_feature_set(individual)
            fitness_results.append((individual, fitness, metrics))
        
        # Sort by fitness (descending)
        fitness_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update best solution
        if fitness_results[0][1] > best_fitness:
            best_individual = fitness_results[0][0]
            best_fitness = fitness_results[0][1]
            best_metrics = fitness_results[0][2]
            logger.info(f"New best solution: Fitness={best_fitness:.4f}, "
                       f"Return={best_metrics.get('total_return', 0):.2f}%, "
                       f"Sharpe={best_metrics.get('sharpe_ratio', 0):.2f}")
        
        # Record history
        gen_stats = {
            'generation': generation + 1,
            'best_fitness': fitness_results[0][1],
            'avg_fitness': sum(r[1] for r in fitness_results) / len(fitness_results),
            'best_return': fitness_results[0][2].get('total_return', 0),
            'best_sharpe': fitness_results[0][2].get('sharpe_ratio', 0),
            'best_features': fitness_results[0][0]
        }
        history.append(gen_stats)
        
        # Create new population through selection, crossover and mutation
        new_population = []
        
        # Elitism: Keep top performers
        elite_count = max(1, population_size // 10)
        new_population.extend([res[0] for res in fitness_results[:elite_count]])
        
        # Fill rest of population with children
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            parent1 = max(random.sample(fitness_results, tournament_size), key=lambda x: x[1])[0]
            parent2 = max(random.sample(fitness_results, tournament_size), key=lambda x: x[1])[0]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            if random.random() < mutation_rate:
                child = mutate(child)
            
            # Validate child has exactly 36 features
            if len(child) != required_features:
                logger.warning(f"Child has {len(child)} features, fixing...")
                if len(child) < required_features:
                    # Add random features
                    remaining = [f for f in available_features if f not in child]
                    if len(remaining) < (required_features - len(child)):
                        remaining = available_features
                    child.extend(random.sample(remaining, required_features - len(child)))
                else:
                    # Trim excess features
                    child = child[:required_features]
            
            new_population.append(child)
        
        # Update population
        population = new_population
    
    # Prepare results
    elapsed_time = time.time() - start_time
    
    results = {
        'best_features': best_individual,
        'best_fitness': best_fitness,
        'best_metrics': best_metrics,
        'optimization_history': history,
        'elapsed_time': elapsed_time,
        'parameters': {
            'required_features': required_features,
            'population_size': population_size,
            'generations': generations,
            'mutation_rate': mutation_rate,
            'test_size': test_size,
            'threshold': threshold,
            'used_existing_model': custom_model is not None
        }
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(RESULTS_DIR, f"feature_selection_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save JSON results
    result_path = os.path.join(result_dir, "feature_selection_results.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save best feature set as text file for easy usage
    with open(os.path.join(result_dir, "best_features.txt"), 'w') as f:
        f.write(','.join(best_individual))
    
    # Create HTML report
    report_path = os.path.join(result_dir, "feature_selection_report.html")
    
    # Generate optimization progress plots using matplotlib
    import matplotlib.pyplot as plt
    
    # Fitness progression
    plt.figure(figsize=(10, 6))
    generations_x = [h['generation'] for h in history]
    plt.plot(generations_x, [h['best_fitness'] for h in history], 'b-', label='Best Fitness')
    plt.plot(generations_x, [h['avg_fitness'] for h in history], 'r--', label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression')
    plt.legend()
    plt.grid(True)
    fitness_plot_path = os.path.join(result_dir, "fitness_progression.png")
    plt.savefig(fitness_plot_path)
    
    # Performance metrics
    plt.figure(figsize=(10, 6))
    plt.plot(generations_x, [h['best_return'] for h in history], 'g-', label='Return (%)')
    plt.plot(generations_x, [h['best_sharpe'] for h in history], 'm--', label='Sharpe Ratio')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('Performance Metrics Progression')
    plt.legend()
    plt.grid(True)
    metrics_plot_path = os.path.join(result_dir, "metrics_progression.png")
    plt.savefig(metrics_plot_path)
    
    # Feature category breakdown for best solution
    category_breakdown = {}
    for category, features in feature_categories.items():
        category_breakdown[category] = len([f for f in best_individual if f in features])
    
    plt.figure(figsize=(10, 6))
    plt.bar(category_breakdown.keys(), category_breakdown.values())
    plt.xlabel('Feature Category')
    plt.ylabel('Count')
    plt.title('Feature Categories in Best Solution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    category_plot_path = os.path.join(result_dir, "feature_categories.png")
    plt.savefig(category_plot_path)
    
    # Create HTML report content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Selection Optimization Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-value {{ font-weight: bold; }}
            .feature-list {{ columns: 3; column-gap: 20px; }}
            .model-info {{ background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Feature Selection Optimization Results</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="model-info">
            <h3>Model Information</h3>
            <p>{"Used existing model: " + os.path.basename(model_path) if custom_model else "Used freshly trained models for each evaluation"}</p>
        </div>
        
        <h2>Best Feature Combination (Exactly {required_features} Features)</h2>
        <p>The optimization process found the following optimal feature set:</p>
        <div class="feature-list">
        <ul>
    """
    
    # Add each feature
    for feature in sorted(best_individual):
        html_content += f"    <li>{feature}</li>\n"
    
    html_content += f"""
        </ul>
        </div>
        
        <h2>Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Fitness Score</td>
                <td class="metric-value">{best_fitness:.4f}</td>
            </tr>
            <tr>
                <td>Total Return</td>
                <td class="metric-value">{best_metrics.get('total_return', 'N/A'):.2f}%</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td class="metric-value">{best_metrics.get('sharpe_ratio', 'N/A'):.2f}</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td class="metric-value">{best_metrics.get('profit_factor', 'N/A'):.2f}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td class="metric-value">{best_metrics.get('win_rate', 'N/A'):.2f}%</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td class="metric-value">{best_metrics.get('max_drawdown', 'N/A'):.2f}%</td>
            </tr>
            <tr>
                <td>Number of Trades</td>
                <td class="metric-value">{best_metrics.get('num_trades', 'N/A')}</td>
            </tr>
        </table>
        
        <h2>Feature Category Distribution</h2>
        <img src="feature_categories.png" alt="Feature Categories" style="max-width:100%;">
        
        <h2>Optimization Progress</h2>
        <img src="fitness_progression.png" alt="Fitness Progression" style="max-width:100%;">
        <img src="metrics_progression.png" alt="Metrics Progression" style="max-width:100%;">
        
        <h2>Optimization Parameters</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Required Features</td>
                <td>{required_features}</td>
            </tr>
            <tr>
                <td>Population Size</td>
                <td>{population_size}</td>
            </tr>
            <tr>
                <td>Generations</td>
                <td>{generations}</td>
            </tr>
            <tr>
                <td>Mutation Rate</td>
                <td>{mutation_rate}</td>
            </tr>
            <tr>
                <td>Test Size</td>
                <td>{test_size}</td>
            </tr>
            <tr>
                <td>Signal Threshold</td>
                <td>{threshold}</td>
            </tr>
            <tr>
                <td>Used Existing Model</td>
                <td>{"Yes - " + os.path.basename(model_path) if custom_model else "No"}</td>
            </tr>
            <tr>
                <td>Execution Time</td>
                <td>{elapsed_time:.2f} seconds</td>
            </tr>
        </table>
        
        <h2>How to Use These Features</h2>
        <p>
            To use these optimized features in your trading strategy, run this command:
        </p>
        <pre>
python run_single.py --data-path {data_path} --custom-features {','.join(best_individual)} --visualize
        </pre>
        
        <p>
            The feature list has also been saved to <code>best_features.txt</code> in this directory.
        </p>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Feature selection results saved to {result_dir}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize feature selection for trading strategy")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data file")
    parser.add_argument("--population", type=int, default=10, help="Population size for genetic algorithm")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations to evolve")
    parser.add_argument("--mutation-rate", type=float, default=0.2, help="Mutation rate")
    parser.add_argument("--test-size", type=float, default=0.3, help="Proportion of data for testing")
    parser.add_argument("--threshold", type=float, default=0.005, help="Signal threshold")
    parser.add_argument("--model-path", type=str, help="Path to existing model to use for evaluation")
    
    args = parser.parse_args()
    
    optimize_feature_selection(
        data_path=args.data_path,
        required_features=36,  # Always use exactly 36 features
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        test_size=args.test_size,
        threshold=args.threshold,
        model_path=args.model_path
    )