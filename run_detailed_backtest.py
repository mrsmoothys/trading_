
"""
Detailed backtesting module for the AI trading bot.
Provides comprehensive backtesting with additional metrics and analysis.
"""
import os
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from strategy import TradingStrategy
from backtest import Backtester
from visualize import Visualizer

logger = logging.getLogger(__name__)

def run_detailed_backtest(
    strategy: TradingStrategy, 
    data: pd.DataFrame, 
    args: Any
) -> Dict[str, Any]:
    """
    Run a detailed backtest with extended metrics and analysis.
    
    Args:
        strategy: Trading strategy instance to backtest
        data: Market data DataFrame
        args: Command-line arguments with configuration options
        
    Returns:
        Dictionary with comprehensive backtest results
    """
    logger.info("Starting detailed backtest")
    
    # Create a backtester instance
    backtester = Backtester(
        strategy=strategy,
        data_path=None,  # We're providing data directly
        start_date=args.start_date,
        end_date=args.end_date,
        results_dir=args.results_dir
    )
    
    # Set the data directly instead of loading from file
    backtester.data = data
    
    # Run the basic backtest
    results = backtester.run()
    
    # Add additional detailed metrics
    
    # 1. Monthly returns analysis
    if 'equity_curve' in results and len(results['equity_curve']) > 0:
        # Create DataFrame with dates
        equity_df = pd.DataFrame({
            'equity': results['equity_curve']
        }, index=data.index[:len(results['equity_curve'])])
        
        # Calculate monthly returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        monthly_returns = equity_df['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ).to_dict()
        
        # Convert to serializable format
        results['monthly_returns'] = {
            k.strftime('%Y-%m'): float(v) 
            for k, v in monthly_returns.items() if not np.isnan(v)
        }
    
    # 2. Trade duration statistics
    if 'trades' in results and len(results['trades']) > 0:
        durations = []
        for trade in results['trades']:
            if 'exit_time' in trade and 'entry_time' in trade:
                try:
                    entry = pd.to_datetime(trade['entry_time'])
                    exit = pd.to_datetime(trade['exit_time'])
                    duration = (exit - entry).total_seconds() / 3600  # hours
                    durations.append(duration)
                except:
                    pass
        
        if durations:
            results['duration_stats'] = {
                'mean': float(np.mean(durations)),
                'median': float(np.median(durations)),
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
                'std': float(np.std(durations))
            }
    
    # 3. Trade side statistics (long vs short)
    if 'trades' in results and len(results['trades']) > 0:
        long_trades = [t for t in results['trades'] if t.get('side', '').lower() == 'long']
        short_trades = [t for t in results['trades'] if t.get('side', '').lower() == 'short']
        
        long_profits = [t.get('profit_pct', 0) for t in long_trades]
        short_profits = [t.get('profit_pct', 0) for t in short_trades]
        
        results['side_stats'] = {
            'long': {
                'count': len(long_trades),
                'win_rate': float(sum(1 for p in long_profits if p > 0) / len(long_profits)) if long_profits else 0,
                'avg_profit': float(np.mean(long_profits)) if long_profits else 0
            },
            'short': {
                'count': len(short_trades),
                'win_rate': float(sum(1 for p in short_profits if p > 0) / len(short_profits)) if short_profits else 0,
                'avg_profit': float(np.mean(short_profits)) if short_profits else 0
            }
        }
    
    # 4. Exit reason statistics
    if 'trades' in results and len(results['trades']) > 0:
        exit_reasons = {}
        for trade in results['trades']:
            reason = trade.get('exit_reason', 'unknown')
            if reason not in exit_reasons:
                exit_reasons[reason] = 0
            exit_reasons[reason] += 1
        
        results['reason_stats'] = exit_reasons
    
    # 5. Generate visualizations if requested
    if args.visualize:
        visualizer = Visualizer()
        visualizer.plot_detailed_backtest(results, show_plots=True)
    
    logger.info("Detailed backtest completed")
    return results

if __name__ == "__main__":
    # Test the function
    logging.basicConfig(level=logging.INFO)
    
    # This is just for testing in isolation - not meant to be run directly
    print("This module is not intended to be run directly.")
    print("Import and use the run_detailed_backtest function in your scripts.")