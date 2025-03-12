"""
Backtesting engine for the AI trading bot.
"""
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import (
    RESULTS_DIR, INITIAL_CAPITAL, TRADING_FEE, SLIPPAGE, 
    ADAPTIVE_SL_TP, TRAILING_STOP, OPTIMIZATION_METRIC
)
from data_processor import load_data, prepare_multi_timeframe_data
from feature_engineering import generate_features
from strategy import TradingStrategy, MLTradingStrategy, EnsembleTradingStrategy

logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    """
    
    def __init__(
        self,
        strategy: TradingStrategy,
        data_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        results_dir: Optional[str] = None
    ):
        """
        Initialize the backtester.
        
        Args:
            strategy: Trading strategy to test
            data_path: Path to data file
            start_date: Start date for backtest
            end_date: End date for backtest
            results_dir: Directory to save results
        """
        self.strategy = strategy
        self.data_path = data_path
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.end_date = pd.Timestamp(end_date) if end_date else None
        self.results_dir = results_dir or RESULTS_DIR
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Results storage
        self.results = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare data for backtesting.
        
        Returns:
            Prepared DataFrame
        """
        if not self.data_path:
            raise ValueError("Data path not provided")
        
        # Load data
        data = load_data(self.data_path)
        
        # Filter by date range
        if self.start_date:
            data = data[data.index >= self.start_date]
        if self.end_date:
            data = data[data.index <= self.end_date]
        
        # Generate features
        data = generate_features(data)
        
        return data
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dictionary of backtest results
        """
        # Load data
        data = self.load_data()
        
        # Run backtest
        self.results = self.strategy.backtest(data)
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self) -> str:
        """
        Save backtest results to disk.
        
        Returns:
            Path to saved results file
        """
        if not self.results:
            raise ValueError("No results to save. Run backtest first.")
        
        # Create result ID based on strategy and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{self.strategy.symbol}_{self.strategy.timeframe}_{timestamp}"
        
        # Prepare result directory
        result_dir = os.path.join(self.results_dir, result_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save performance metrics
        performance_path = os.path.join(result_dir, "performance.json")
        with open(performance_path, 'w') as f:
            json.dump(self.results['performance'], f, indent=2)
        
        # Save trades
        trades_path = os.path.join(result_dir, "trades.csv")
        trades_df = pd.DataFrame(self.results['trades'])
        if not trades_df.empty:
            trades_df.to_csv(trades_path, index=False)
        
        # Save equity curve
        equity_path = os.path.join(result_dir, "equity_curve.csv")
        equity_df = pd.DataFrame({
            'equity': self.results['equity_curve']
        }, index=self.results['signals'].index[:len(self.results['equity_curve'])])
        equity_df.to_csv(equity_path)
        
        # Save signals
        signals_path = os.path.join(result_dir, "signals.csv")
        self.results['signals'].to_csv(signals_path)
        
        logger.info(f"Results saved to {result_dir}")
        return result_dir
    
    def generate_report(self, show_plots: bool = False) -> str:
        """
        Generate a comprehensive backtest report.
        
        Args:
            show_plots: Whether to display plots
            
        Returns:
            Path to saved report file
        """
        if not self.results:
            raise ValueError("No results to report. Run backtest first.")
        
        # Create report timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"report_{timestamp}.html")
        
        # Create report sections
        sections = []
        
        # 1. Summary section
        performance = self.results['performance']
        summary = f"""
        <h2>Backtest Summary</h2>
        <table class="table table-striped">
            <tr><th>Symbol</th><td>{self.strategy.symbol}</td></tr>
            <tr><th>Timeframe</th><td>{self.strategy.timeframe}</td></tr>
            <tr><th>Period</th><td>{self.results['signals'].index[0]} to {self.results['signals'].index[-1]}</td></tr>
            <tr><th>Initial Capital</th><td>${performance['initial_capital']:.2f}</td></tr>
            <tr><th>Final Capital</th><td>${performance['final_capital']:.2f}</td></tr>
            <tr><th>Total Return</th><td>{performance['total_return']:.2f}%</td></tr>
            <tr><th>Number of Trades</th><td>{performance['num_trades']}</td></tr>
            <tr><th>Win Rate</th><td>{performance['win_rate']:.2f}%</td></tr>
            <tr><th>Average Profit</th><td>{performance['avg_profit']:.2f}%</td></tr>
            <tr><th>Average Loss</th><td>{performance['avg_loss']:.2f}%</td></tr>
            <tr><th>Profit Factor</th><td>{performance['profit_factor']:.2f}</td></tr>
            <tr><th>Max Drawdown</th><td>{performance['max_drawdown']:.2f}%</td></tr>
            <tr><th>Sharpe Ratio</th><td>{performance['sharpe_ratio']:.2f}</td></tr>
            <tr><th>Sortino Ratio</th><td>{performance['sortino_ratio']:.2f}</td></tr>
        </table>
        """
        sections.append(summary)
        
        # 2. Equity curve plot
        if len(self.results['equity_curve']) > 0:
            fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
            
            # Plot equity curve
            equity_series = pd.Series(
                self.results['equity_curve'],
                index=self.results['signals'].index[:len(self.results['equity_curve'])]
            )
            equity_series.plot(ax=ax, label='Equity')
            
            # Add initial capital as horizontal line
            ax.axhline(y=performance['initial_capital'], color='r', linestyle='--', label='Initial Capital')
            
            plt.title('Equity Curve')
            plt.ylabel('Capital ($)')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            equity_plot_path = os.path.join(self.results_dir, f"equity_curve_{timestamp}.png")
            plt.savefig(equity_plot_path)
            
            if show_plots:
                plt.show()
            plt.close()
            
            # Add to report
            sections.append(f"""
            <h2>Equity Curve</h2>
            <img src="{equity_plot_path}" alt="Equity Curve" style="max-width:100%;">
            """)
        
        # 3. Drawdown plot
        if len(self.results['equity_curve']) > 0:
            fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
            
            # Calculate drawdown series
            equity_series = pd.Series(
                self.results['equity_curve'],
                index=self.results['signals'].index[:len(self.results['equity_curve'])]
            )
            drawdown_series = (equity_series / equity_series.cummax() - 1) * 100
            drawdown_series.plot(ax=ax)
            
            plt.title('Drawdown')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            
            # Save plot
            drawdown_plot_path = os.path.join(self.results_dir, f"drawdown_{timestamp}.png")
            plt.savefig(drawdown_plot_path)
            
            if show_plots:
                plt.show()
            plt.close()
            
            # Add to report
            sections.append(f"""
            <h2>Drawdown</h2>
            <img src="{drawdown_plot_path}" alt="Drawdown" style="max-width:100%;">
            """)
        
        # 4. Monthly returns
        if len(self.results['equity_curve']) > 0:
            equity_series = pd.Series(
                self.results['equity_curve'],
                index=self.results['signals'].index[:len(self.results['equity_curve'])]
            )
            
            # Calculate returns
            returns_series = equity_series.pct_change().dropna()
            
            # Group by month and calculate cumulative return
            monthly_returns = returns_series.groupby([returns_series.index.year, returns_series.index.month]).apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
            
            # Convert to DataFrame for heatmap
            years = monthly_returns.index.get_level_values(0).unique()
            months = range(1, 13)
            
            returns_matrix = []
            for year in years:
                year_returns = []
                for month in months:
                    try:
                        ret = monthly_returns.loc[(year, month)]
                        year_returns.append(ret)
                    except KeyError:
                        year_returns.append(np.nan)
                returns_matrix.append(year_returns)
            
            monthly_returns_df = pd.DataFrame(
                returns_matrix,
                index=years,
                columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
            
            # Plot heatmap
            fig, ax = plt.figure(figsize=(12, len(years) * 0.6 + 2)), plt.gca()
            
            sns.heatmap(
                monthly_returns_df,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                vmin=-10,
                vmax=10,
                ax=ax
            )
            
            plt.title('Monthly Returns (%)')
            
            # Save plot
            monthly_returns_path = os.path.join(self.results_dir, f"monthly_returns_{timestamp}.png")
            plt.savefig(monthly_returns_path)
            
            if show_plots:
                plt.show()
            plt.close()
            
            # Add to report
            sections.append(f"""
            <h2>Monthly Returns</h2>
            <img src="{monthly_returns_path}" alt="Monthly Returns" style="max-width:100%;">
            """)
        
        # 5. Trade distribution
        if len(self.results['trades']) > 0:
            trades_df = pd.DataFrame(self.results['trades'])
            
            # Plot profit distribution
            fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
            
            sns.histplot(trades_df['profit_percentage'], kde=True, ax=ax)
            ax.axvline(x=0, color='r', linestyle='--')
            
            plt.title('Profit Distribution')
            plt.xlabel('Profit (%)')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            # Save plot
            profit_dist_path = os.path.join(self.results_dir, f"profit_distribution_{timestamp}.png")
            plt.savefig(profit_dist_path)
            
            if show_plots:
                plt.show()
            plt.close()
            
            # Add to report
            sections.append(f"""
            <h2>Profit Distribution</h2>
            <img src="{profit_dist_path}" alt="Profit Distribution" style="max-width:100%;">
            """)
            
            # Trade duration
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
                trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
                
                fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
                
                sns.histplot(trades_df['duration_hours'], kde=True, ax=ax)
                
                plt.title('Trade Duration')
                plt.xlabel('Duration (hours)')
                plt.ylabel('Frequency')
                plt.grid(True)
                
                # Save plot
                duration_path = os.path.join(self.results_dir, f"trade_duration_{timestamp}.png")
                plt.savefig(duration_path)
                
                if show_plots:
                    plt.show()
                plt.close()
                
                # Add to report
                sections.append(f"""
                <h2>Trade Duration</h2>
                <img src="{duration_path}" alt="Trade Duration" style="max-width:100%;">
                """)
        
        # 6. Sample trades table
        if len(self.results['trades']) > 0:
            trades_df = pd.DataFrame(self.results['trades'])
            sample_trades = trades_df.sort_values('profit_percentage', ascending=False).head(10)
            
            trades_table = """
            <h2>Best Trades</h2>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Quantity</th>
                        <th>Profit (%)</th>
                        <th>Profit ($)</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for _, trade in sample_trades.iterrows():
                trades_table += f"""
                <tr>
                    <td>{trade['symbol']}</td>
                    <td>{trade['side']}</td>
                    <td>{trade['entry_price']:.4f}</td>
                    <td>{trade['exit_price']:.4f}</td>
                    <td>{trade['quantity']:.4f}</td>
                    <td>{trade['profit_percentage']:.2f}%</td>
                    <td>${trade['profit_amount']:.2f}</td>
                    <td>{trade['entry_time']}</td>
                    <td>{trade['exit_time']}</td>
                    <td>{trade['exit_reason']}</td>
                </tr>
                """
            
            trades_table += """
                </tbody>
            </table>
            """
            
            sections.append(trades_table)
            
            # Worst trades
            sample_trades = trades_df.sort_values('profit_percentage').head(10)
            
            trades_table = """
            <h2>Worst Trades</h2>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Quantity</th>
                        <th>Profit (%)</th>
                        <th>Profit ($)</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for _, trade in sample_trades.iterrows():
                trades_table += f"""
                <tr>
                    <td>{trade['symbol']}</td>
                    <td>{trade['side']}</td>
                    <td>{trade['entry_price']:.4f}</td>
                    <td>{trade['exit_price']:.4f}</td>
                    <td>{trade['quantity']:.4f}</td>
                    <td>{trade['profit_percentage']:.2f}%</td>
                    <td>${trade['profit_amount']:.2f}</td>
                    <td>{trade['entry_time']}</td>
                    <td>{trade['exit_time']}</td>
                    <td>{trade['exit_reason']}</td>
                </tr>
                """
            
            trades_table += """
                </tbody>
            </table>
            """
            
            sections.append(trades_table)
        
        # 7. Build the full report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {self.strategy.symbol} {self.strategy.timeframe}</title>
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
                <h1>Backtest Report - {self.strategy.symbol} {self.strategy.timeframe}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                {"".join(sections)}
            </div>
        </body>
        </html>
        """
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Report saved to {report_path}")
        return report_path

class MultiAssetBacktester:
    """
    Backtester for multiple assets and timeframes.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        strategy_class,
        strategy_params: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        results_dir: Optional[str] = None
    ):
        """
        Initialize the multi-asset backtester.
        
        Args:
            symbols: List of symbols to backtest
            timeframes: List of timeframes to backtest
            strategy_class: Class of strategy to use
            strategy_params: Parameters for strategy initialization
            start_date: Start date for backtest
            end_date: End date for backtest
            results_dir: Directory to save results
        """
        self.symbols = symbols
        self.timeframes = timeframes
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.start_date = start_date
        self.end_date = end_date
        self.results_dir = results_dir or RESULTS_DIR
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Results storage
        self.results = {}
    
    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run backtests for all symbols and timeframes.
        
        Returns:
            Nested dictionary of backtest results by symbol and timeframe
        """
        for symbol in tqdm(self.symbols, desc="Symbols"):
            self.results[symbol] = {}
            
            for timeframe in tqdm(self.timeframes, desc=f"Timeframes for {symbol}", leave=False):
                # Create strategy
                strategy = self.strategy_class(
                    symbol=symbol,
                    timeframe=timeframe,
                    **self.strategy_params
                )
                
                # Prepare data path
                from config import DATA_DIR
                data_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}_data_2018_to_2025.csv")
                
                if not os.path.exists(data_path):
                    logger.warning(f"Data file not found: {data_path}")
                    continue
                
                # Create backtester
                backtester = Backtester(
                    strategy=strategy,
                    data_path=data_path,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    results_dir=os.path.join(self.results_dir, f"{symbol}_{timeframe}")
                )
                
                try:
                    # Run backtest
                    results = backtester.run()
                    
                    # Generate report
                    backtester.generate_report()
                    
                    # Store results
                    self.results[symbol][timeframe] = results['performance']
                except Exception as e:
                    logger.error(f"Error backtesting {symbol} {timeframe}: {e}")
                    self.results[symbol][timeframe] = {"error": str(e)}
        
        # Save summary results
        self.save_summary()
        
        return self.results
    
    def save_summary(self) -> str:
        """
        Save a summary of all backtest results.
        
        Returns:
            Path to summary file
        """
        # Create result ID based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.results_dir, f"summary_{timestamp}.csv")
        
        # Prepare summary data
        summary_data = []
        
        for symbol in self.results:
            for timeframe in self.results[symbol]:
                performance = self.results[symbol][timeframe]
                
                if "error" in performance:
                    continue
                
                summary_data.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "total_return": performance.get("total_return", 0),
                    "num_trades": performance.get("num_trades", 0),
                    "win_rate": performance.get("win_rate", 0),
                    "profit_factor": performance.get("profit_factor", 0),
                    "sharpe_ratio": performance.get("sharpe_ratio", 0),
                    "max_drawdown": performance.get("max_drawdown", 0)
                })
        
        # Create DataFrame and save to CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")
        else:
            logger.warning("No valid results to summarize")
        
        return summary_path
    
    def generate_comparison_report(self) -> str:
        """
        Generate a report comparing results across symbols and timeframes.
        
        Returns:
            Path to comparison report
        """
        # Prepare summary data
        summary_data = []
        
        for symbol in self.results:
            for timeframe in self.results[symbol]:
                performance = self.results[symbol][timeframe]
                
                if "error" in performance:
                    continue
                
                summary_data.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "total_return": performance.get("total_return", 0),
                    "num_trades": performance.get("num_trades", 0),
                    "win_rate": performance.get("win_rate", 0),
                    "profit_factor": performance.get("profit_factor", 0),
                    "sharpe_ratio": performance.get("sharpe_ratio", 0),
                    "max_drawdown": performance.get("max_drawdown", 0)
                })
        
        if not summary_data:
            logger.warning("No valid results to compare")
            return ""
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by performance metric
        summary_df = summary_df.sort_values(OPTIMIZATION_METRIC, ascending=False)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, f"comparison_{timestamp}.html")
        
        # Generate plots
        
        # 1. Returns by symbol
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        
        returns_by_symbol = summary_df.pivot(index='timeframe', columns='symbol', values='total_return')
        returns_by_symbol.plot(kind='bar', ax=ax)
        
        plt.title('Total Return by Symbol and Timeframe')
        plt.ylabel('Total Return (%)')
        plt.legend(title='Symbol')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        returns_plot_path = os.path.join(self.results_dir, f"returns_comparison_{timestamp}.png")
        plt.savefig(returns_plot_path)
        plt.close()
        
        # 2. Sharpe ratio by symbol
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        
        sharpe_by_symbol = summary_df.pivot(index='timeframe', columns='symbol', values='sharpe_ratio')
        sharpe_by_symbol.plot(kind='bar', ax=ax)
        
        plt.title('Sharpe Ratio by Symbol and Timeframe')
        plt.ylabel('Sharpe Ratio')
        plt.legend(title='Symbol')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        sharpe_plot_path = os.path.join(self.results_dir, f"sharpe_comparison_{timestamp}.png")
        plt.savefig(sharpe_plot_path)
        plt.close()
        
        # 3. Win rate by symbol
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        
        winrate_by_symbol = summary_df.pivot(index='timeframe', columns='symbol', values='win_rate')
        winrate_by_symbol.plot(kind='bar', ax=ax)
        
        plt.title('Win Rate by Symbol and Timeframe')
        plt.ylabel('Win Rate (%)')
        plt.legend(title='Symbol')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        winrate_plot_path = os.path.join(self.results_dir, f"winrate_comparison_{timestamp}.png")
        plt.savefig(winrate_plot_path)
        plt.close()
        
        # 4. Drawdown by symbol
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        
        drawdown_by_symbol = summary_df.pivot(index='timeframe', columns='symbol', values='max_drawdown')
        drawdown_by_symbol.plot(kind='bar', ax=ax)
        
        plt.title('Maximum Drawdown by Symbol and Timeframe')
        plt.ylabel('Max Drawdown (%)')
        plt.legend(title='Symbol')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        drawdown_plot_path = os.path.join(self.results_dir, f"drawdown_comparison_{timestamp}.png")
        plt.savefig(drawdown_plot_path)
        plt.close()
        
        # Create heat map of returns
        fig, ax = plt.figure(figsize=(max(6, len(self.symbols) * 0.8), max(6, len(self.timeframes) * 0.8))), plt.gca()
        
        returns_pivot = summary_df.pivot(index='timeframe', columns='symbol', values='total_return')
        sns.heatmap(
            returns_pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax
        )
        
        plt.title('Total Return (%) by Symbol and Timeframe')
        plt.tight_layout()
        
        heatmap_path = os.path.join(self.results_dir, f"returns_heatmap_{timestamp}.png")
        plt.savefig(heatmap_path)
        plt.close()
        
        # Build the report
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Comparison Report</title>
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
                <h1>Backtest Comparison Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Performance Summary</h2>
                <p>Top performers based on {OPTIMIZATION_METRIC}:</p>
                
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Timeframe</th>
                            <th>Total Return (%)</th>
                            <th>Trades</th>
                            <th>Win Rate (%)</th>
                            <th>Profit Factor</th>
                            <th>Sharpe Ratio</th>
                            <th>Max Drawdown (%)</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add top 10 results
        for _, row in summary_df.head(10).iterrows():
            report_html += f"""
                <tr>
                    <td>{row['symbol']}</td>
                    <td>{row['timeframe']}</td>
                    <td>{row['total_return']:.2f}</td>
                    <td>{row['num_trades']}</td>
                    <td>{row['win_rate']:.2f}</td>
                    <td>{row['profit_factor']:.2f}</td>
                    <td>{row['sharpe_ratio']:.2f}</td>
                    <td>{row['max_drawdown']:.2f}</td>
                </tr>
            """
        
        report_html += """
                    </tbody>
                </table>
                
                <h2>Return Comparison</h2>
                <img src="{}" alt="Returns Comparison" style="max-width:100%;">
                
                <h2>Return Heatmap</h2>
                <img src="{}" alt="Returns Heatmap" style="max-width:100%;">
                
                <h2>Sharpe Ratio Comparison</h2>
                <img src="{}" alt="Sharpe Ratio Comparison" style="max-width:100%;">
                
                <h2>Win Rate Comparison</h2>
                <img src="{}" alt="Win Rate Comparison" style="max-width:100%;">
                
                <h2>Max Drawdown Comparison</h2>
                <img src="{}" alt="Max Drawdown Comparison" style="max-width:100%;">
            </div>
        </body>
        </html>
        """.format(
            returns_plot_path, 
            heatmap_path, 
            sharpe_plot_path, 
            winrate_plot_path, 
            drawdown_plot_path
        )
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Comparison report saved to {report_path}")
        return report_path

if __name__ == "__main__":
    # Test the backtester
    from strategy import TradingStrategy
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy strategy for testing
    class DummyStrategy(TradingStrategy):
        def generate_signals(self):
            signals = self.data.copy()
            signals['signal'] = 0
            
            # Buy signal every 10 candles, sell every 5 candles
            for i in range(len(signals)):
                if i % 10 == 0:
                    signals.loc[signals.index[i], 'signal'] = 1
                elif i % 5 == 0:
                    signals.loc[signals.index[i], 'signal'] = -1
            
            return signals
    
    # Create strategy
    strategy = DummyStrategy("BTCUSDT", "1h", adaptive_sl_tp=True, trailing_stop=True)
    
    # Create backtester
    from config import DATA_DIR
    data_path = os.path.join(DATA_DIR, "BTCUSDT_1h_data_2018_to_2025.csv")
    
    if os.path.exists(data_path):
        backtester = Backtester(
            strategy=strategy,
            data_path=data_path,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Run backtest
        results = backtester.run()
        
        # Generate report
        report_path = backtester.generate_report(show_plots=True)
        
        print(f"Report saved to {report_path}")
    else:
        print(f"Data file not found: {data_path}")
        print("Please download or create the data file before running the backtest.")