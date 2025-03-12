"""
Visualization module for the AI trading bot.
"""
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

from config import RESULTS_DIR, PLOT_TYPES, SAVE_PLOTS

logger = logging.getLogger(__name__)

# Styling
plt.style.use('ggplot')
BLUE = '#1f77b4'
GREEN = '#2ca02c'
RED = '#d62728'
ORANGE = '#ff7f0e'
PURPLE = '#9467bd'

class Visualizer:
    """
    Class for creating visualizations of trading results and model performance.
    """
    
    def __init__(self, result_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            result_dir: Directory to save visualizations
        """
        self.result_dir = result_dir or RESULTS_DIR
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Set larger font sizes
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
    
    def plot_equity_curve(self, equity_data: Union[List[float], pd.Series], initial_capital: float = None, title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot equity curve.
        
        Args:
            equity_data: Equity data as list or pandas Series
            initial_capital: Initial capital for reference line
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()
        
        # Convert to pandas Series if needed
        if isinstance(equity_data, list):
            equity_series = pd.Series(equity_data)
        else:
            equity_series = equity_data
        
        # Plot equity curve
        if isinstance(equity_series.index, pd.DatetimeIndex):
            equity_series.plot(ax=ax, color=BLUE, linewidth=2)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
        else:
            ax.plot(equity_series.values, color=BLUE, linewidth=2)
        
        # Add initial capital reference line
        if initial_capital is not None:
            ax.axhline(y=initial_capital, color=RED, linestyle='--', label='Initial Capital')
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Equity ($)')
        ax.set_title(title or 'Equity Curve')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")
        
        return fig
    
    def plot_drawdown(self, equity_data: Union[List[float], pd.Series], title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot drawdown over time.
        
        Args:
            equity_data: Equity data as list or pandas Series
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()
        
        # Convert to pandas Series if needed
        if isinstance(equity_data, list):
            equity_series = pd.Series(equity_data)
        else:
            equity_series = equity_data
        
        # Calculate drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series / rolling_max - 1) * 100
        
        # Plot drawdown
        if isinstance(equity_series.index, pd.DatetimeIndex):
            drawdown.plot(ax=ax, color=RED, linewidth=2)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
        else:
            ax.plot(drawdown.values, color=RED, linewidth=2)
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title or 'Drawdown')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdown plot saved to {save_path}")
        
        return fig
    
    def plot_monthly_returns(self, equity_data: Union[List[float], pd.Series], title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_data: Equity data as list or pandas Series
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Convert to pandas Series if needed
        if isinstance(equity_data, list):
            # Need datetime index for this plot
            logger.warning("Monthly returns plot requires datetime index. Using range index instead.")
            equity_series = pd.Series(equity_data)
        else:
            equity_series = equity_data
        
        # Skip if not datetime index
        if not isinstance(equity_series.index, pd.DatetimeIndex):
            logger.warning("Monthly returns plot requires datetime index. Skipping plot.")
            return None
        
        # Calculate returns
        returns_series = equity_series.pct_change().dropna()
        
        # Group by month and calculate cumulative return
        monthly_returns = returns_series.groupby([returns_series.index.year, returns_series.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # Convert to DataFrame for heatmap
        years = sorted(monthly_returns.index.get_level_values(0).unique())
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, len(years) * 0.6 + 2))
        
        # Plot heatmap
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
        
        # Add title
        ax.set_title(title or 'Monthly Returns (%)')
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"monthly_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Monthly returns plot saved to {save_path}")
        
        return fig
    
    def plot_trade_distribution(self, trades: List[Dict[str, Any]], title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of trade profits.
        
        Args:
            trades: List of trade dictionaries
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not trades:
            logger.warning("No trades to plot distribution.")
            return None
        
        # Create DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot histogram
        sns.histplot(trades_df['profit_percentage'], kde=True, ax=ax, color=BLUE)
        
        # Add vertical line at zero
        ax.axvline(x=0, color=RED, linestyle='--')
        
        # Add labels and title
        ax.set_xlabel('Profit (%)')
        ax.set_ylabel('Number of Trades')
        ax.set_title(title or 'Trade Profit Distribution')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"trade_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trade distribution plot saved to {save_path}")
        
        return fig
    
    def plot_learning_progress(self, history: Dict[str, List[float]], title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning progress from training history.
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not history:
            logger.warning("No history to plot learning progress.")
            return None
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots()
        
        # Plot loss on left y-axis
        if 'loss' in history:
            ax1.plot(history['loss'], color=BLUE, label='Train Loss')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], color=ORANGE, label='Val Loss')
        
        # Add labels
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y', labelcolor=BLUE)
        
        # Create second y-axis for metrics
        ax2 = ax1.twinx()
        
        # Plot metrics on right y-axis
        metrics = [key for key in history.keys() if key not in ['loss', 'val_loss']]
        if metrics:
            for i, metric in enumerate(metrics):
                color = plt.cm.tab10(i % 10)
                ax2.plot(history[metric], color=color, linestyle='--', label=metric)
        
        ax2.set_ylabel('Metric')
        ax2.tick_params(axis='y', labelcolor=ORANGE)
        
        # Add title
        fig.suptitle(title or 'Learning Progress')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"learning_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning progress plot saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance values
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if importance_df.empty:
            logger.warning("No feature importance data to plot.")
            return None
        
        # Sort by importance
        sorted_df = importance_df.sort_values('importance')
        
        # Plot top 20 features or fewer if less available
        n_features = min(20, len(sorted_df))
        plot_df = sorted_df.tail(n_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, n_features * 0.4 + 2))
        
        # Create bar plot
        bars = ax.barh(plot_df['feature'], plot_df['importance'])
        
        # Add error bars if std column exists
        if 'std' in plot_df.columns:
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
                bar.set_color(GREEN)
            else:
                bar.set_color(RED)
        
        # Add labels and title
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(title or 'Feature Importance')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def plot_price_with_signals(self, data: pd.DataFrame, trades: List[Dict[str, Any]], title: str = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot price chart with entry and exit signals.
        
        Args:
            data: DataFrame with price data
            trades: List of trade dictionaries
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if data.empty:
            logger.warning("No price data to plot.")
            return None
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot price
        data['close'].plot(ax=ax, color=BLUE, linewidth=1.5)
        
        # Plot entry and exit points
        if trades:
            # Convert to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Convert timestamps to datetime
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Plot long trades
            long_trades = trades_df[trades_df['side'] == 'buy']
            if not long_trades.empty:
                for _, trade in long_trades.iterrows():
                    # Entry point
                    ax.scatter(trade['entry_time'], trade['entry_price'], 
                              marker='^', color=GREEN, s=100, zorder=5)
                    
                    # Exit point
                    ax.scatter(trade['exit_time'], trade['exit_price'], 
                              marker='v', color=RED, s=100, zorder=5)
                    
                    # Connect with line
                    ax.plot([trade['entry_time'], trade['exit_time']], 
                           [trade['entry_price'], trade['exit_price']], 
                           color=GREEN if trade['profit_percentage'] > 0 else RED, 
                           linewidth=1, alpha=0.5)
            
            # Plot short trades
            short_trades = trades_df[trades_df['side'] == 'sell']
            if not short_trades.empty:
                for _, trade in short_trades.iterrows():
                    # Entry point
                    ax.scatter(trade['entry_time'], trade['entry_price'], 
                              marker='v', color=RED, s=100, zorder=5)
                    
                    # Exit point
                    ax.scatter(trade['exit_time'], trade['exit_price'], 
                              marker='^', color=GREEN, s=100, zorder=5)
                    
                    # Connect with line
                    ax.plot([trade['entry_time'], trade['exit_time']], 
                           [trade['entry_price'], trade['exit_price']], 
                           color=GREEN if trade['profit_percentage'] > 0 else RED, 
                           linewidth=1, alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(title or 'Price Chart with Signals')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=BLUE, lw=2, label='Close Price'),
            Line2D([0], [0], marker='^', color=GREEN, lw=0, label='Long Entry', markersize=10),
            Line2D([0], [0], marker='v', color=RED, lw=0, label='Long Exit', markersize=10),
            Line2D([0], [0], marker='v', color=RED, lw=0, label='Short Entry', markersize=10),
            Line2D([0], [0], marker='^', color=GREEN, lw=0, label='Short Exit', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"price_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Price with signals plot saved to {save_path}")
        
        return fig
    
    def plot_interactive_equity_curve(self, equity_data: Union[List[float], pd.Series], trades: List[Dict[str, Any]] = None, title: str = None, save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive equity curve with Plotly.
        
        Args:
            equity_data: Equity data as list or pandas Series
            trades: List of trade dictionaries
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        # Convert to pandas Series if needed
        if isinstance(equity_data, list):
            equity_series = pd.Series(equity_data)
        else:
            equity_series = equity_data
        
        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=equity_series.index if isinstance(equity_series.index, pd.DatetimeIndex) else list(range(len(equity_series))),
            y=equity_series.values,
            mode='lines',
            name='Equity',
            line=dict(color=BLUE, width=2)
        ))
        
        # Add trade markers if provided
        if trades:
            # Convert to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Convert timestamps to datetime
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Calculate equity at each trade exit
            trade_equity = []
            current_equity = equity_series.iloc[0]
            
            for _, trade in trades_df.iterrows():
                profit = trade['profit_amount']
                current_equity += profit
                
                trade_equity.append({
                    'time': trade['exit_time'],
                    'equity': current_equity,
                    'profit': profit,
                    'profit_pct': trade['profit_percentage'],
                    'side': trade['side']
                })
            
            # Separate winning and losing trades
            winning_trades = [t for t in trade_equity if t['profit'] > 0]
            losing_trades = [t for t in trade_equity if t['profit'] <= 0]
            
            # Add winning trade markers
            if winning_trades:
                win_times = [t['time'] for t in winning_trades]
                win_equities = [t['equity'] for t in winning_trades]
                win_texts = [f"Profit: ${t['profit']:.2f} ({t['profit_pct']:.2f}%)<br>Side: {t['side']}" for t in winning_trades]
                
                fig.add_trace(go.Scatter(
                    x=win_times,
                    y=win_equities,
                    mode='markers',
                    name='Winning Trades',
                    marker=dict(color=GREEN, size=8, symbol='circle'),
                    text=win_texts,
                    hoverinfo='text+x+y'
                ))
            
            # Add losing trade markers
            if losing_trades:
                loss_times = [t['time'] for t in losing_trades]
                loss_equities = [t['equity'] for t in losing_trades]
                loss_texts = [f"Loss: ${t['profit']:.2f} ({t['profit_pct']:.2f}%)<br>Side: {t['side']}" for t in losing_trades]
                
                fig.add_trace(go.Scatter(
                    x=loss_times,
                    y=loss_equities,
                    mode='markers',
                    name='Losing Trades',
                    marker=dict(color=RED, size=8, symbol='circle'),
                    text=loss_texts,
                    hoverinfo='text+x+y'
                ))
        
        # Update layout
        fig.update_layout(
            title=title or 'Interactive Equity Curve',
            xaxis_title='Time',
            yaxis_title='Equity ($)',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        
        # Format y-axis as currency
        fig.update_yaxes(tickprefix='$')
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"interactive_equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            pio.write_html(fig, save_path)
            logger.info(f"Interactive equity curve saved to {save_path}")
        
        return fig
    
    def plot_interactive_trade_analysis(self, trades: List[Dict[str, Any]], title: str = None, save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive trade analysis with Plotly.
        
        Args:
            trades: List of trade dictionaries
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        if not trades:
            logger.warning("No trades to analyze.")
            return None
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Convert timestamps to datetime
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Calculate duration in hours
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Profit Distribution',
                'Profit vs Duration',
                'Win Rate by Exit Reason',
                'Profit by Side'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Profit distribution histogram
        fig.add_trace(
            go.Histogram(
                x=trades_df['profit_percentage'],
                nbinsx=20,
                marker_color=BLUE,
                name='Profit Distribution'
            ),
            row=1, col=1
        )
        
        # Add vertical line at zero
        fig.add_vline(
            x=0,
            line_width=2,
            line_dash="dash",
            line_color=RED,
            row=1, col=1
        )
        
        # 2. Profit vs duration scatter plot
        fig.add_trace(
            go.Scatter(
                x=trades_df['duration'],
                y=trades_df['profit_percentage'],
                mode='markers',
                marker=dict(
                    color=trades_df['profit_percentage'],
                    colorscale='RdYlGn',
                    cmin=-10,
                    cmax=10,
                    size=8,
                    showscale=True
                ),
                text=[f"Side: {side}<br>Entry: {entry_price:.4f}<br>Exit: {exit_price:.4f}" 
                     for side, entry_price, exit_price in zip(trades_df['side'], trades_df['entry_price'], trades_df['exit_price'])],
                hoverinfo='text+x+y',
                name='Profit vs Duration'
            ),
            row=1, col=2
        )
        
        # 3. Win rate by exit reason
        win_rate_by_reason = trades_df.groupby('exit_reason')['profit_percentage'].apply(
            lambda x: (x > 0).mean() * 100
        ).reset_index()
        win_rate_by_reason.columns = ['exit_reason', 'win_rate']
        
        fig.add_trace(
            go.Bar(
                x=win_rate_by_reason['exit_reason'],
                y=win_rate_by_reason['win_rate'],
                marker_color=ORANGE,
                name='Win Rate by Reason'
            ),
            row=2, col=1
        )
        
        # 4. Profit by side
        profit_by_side = trades_df.groupby('side')['profit_percentage'].mean().reset_index()
        
        fig.add_trace(
            go.Bar(
                x=profit_by_side['side'],
                y=profit_by_side['profit_percentage'],
                marker_color=[GREEN if x > 0 else RED for x in profit_by_side['profit_percentage']],
                name='Avg Profit by Side'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title or 'Trade Analysis',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update x and y axis labels
        fig.update_xaxes(title_text="Profit (%)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
        fig.update_xaxes(title_text="Exit Reason", row=2, col=1)
        fig.update_xaxes(title_text="Side", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Profit (%)", row=1, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Avg Profit (%)", row=2, col=2)
        
        # Add reference line at y=0 for profit by side
        fig.add_hline(
            y=0,
            line_width=2,
            line_dash="dash",
            line_color="black",
            row=2, col=2
        )
        
        # Save if requested
        if save_path or SAVE_PLOTS:
            save_path = save_path or os.path.join(self.result_dir, f"trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            pio.write_html(fig, save_path)
            logger.info(f"Interactive trade analysis saved to {save_path}")
        
        return fig
    
    def generate_backtest_dashboard(self, backtest_results: Dict[str, Any], title: str = None, save_path: Optional[str] = None) -> None:
        """
        Generate a comprehensive backtest dashboard with multiple plots.
        
        Args:
            backtest_results: Dictionary of backtest results
            title: Dashboard title
            save_path: Path to save the dashboard
        """
        if not backtest_results:
            logger.warning("No backtest results for dashboard.")
            return
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory for dashboard
        dashboard_dir = save_path or os.path.join(self.result_dir, f"dashboard_{timestamp}")
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Extract data from results
        performance = backtest_results.get('performance', {})
        equity_curve = backtest_results.get('equity_curve', [])
        trades = backtest_results.get('trades', [])
        signals = backtest_results.get('signals', pd.DataFrame())
        
        # Create equity curve with signals
        if not signals.empty and equity_curve:
            equity_series = pd.Series(
                equity_curve,
                index=signals.index[:len(equity_curve)]
            )
            self.plot_price_with_signals(
                signals[['close']], 
                trades, 
                title="Price Chart with Trading Signals",
                save_path=os.path.join(dashboard_dir, "price_signals.png")
            )
        
        # Create equity curve
        if equity_curve:
            self.plot_equity_curve(
                equity_curve,
                initial_capital=performance.get('initial_capital'),
                title="Equity Curve",
                save_path=os.path.join(dashboard_dir, "equity_curve.png")
            )
            
            # Drawdown
            self.plot_drawdown(
                equity_curve,
                title="Drawdown",
                save_path=os.path.join(dashboard_dir, "drawdown.png")
            )
            
            # Monthly returns
            if isinstance(signals.index, pd.DatetimeIndex) and len(equity_curve) > 0:
                equity_series = pd.Series(
                    equity_curve,
                    index=signals.index[:len(equity_curve)]
                )
                self.plot_monthly_returns(
                    equity_series,
                    title="Monthly Returns",
                    save_path=os.path.join(dashboard_dir, "monthly_returns.png")
                )
        
        # Trade distribution
        if trades:
            self.plot_trade_distribution(
                trades,
                title="Trade Profit Distribution",
                save_path=os.path.join(dashboard_dir, "trade_distribution.png")
            )
        
        # Interactive plots
        if equity_curve:
            try:
                # Interactive equity curve
                if isinstance(signals.index, pd.DatetimeIndex) and len(equity_curve) > 0:
                    equity_series = pd.Series(
                        equity_curve,
                        index=signals.index[:len(equity_curve)]
                    )
                    self.plot_interactive_equity_curve(
                        equity_series,
                        trades,
                        title="Interactive Equity Curve",
                        save_path=os.path.join(dashboard_dir, "interactive_equity.html")
                    )
            except Exception as e:
                logger.warning(f"Failed to create interactive equity curve: {e}")
        
        # Interactive trade analysis
        if trades:
            try:
                self.plot_interactive_trade_analysis(
                    trades,
                    title="Interactive Trade Analysis",
                    save_path=os.path.join(dashboard_dir, "interactive_trade_analysis.html")
                )
            except Exception as e:
                logger.warning(f"Failed to create interactive trade analysis: {e}")
        
        # Create HTML dashboard
        self._generate_dashboard_html(
            dashboard_dir,
            title or "Backtest Dashboard",
            performance
        )
        
        logger.info(f"Dashboard generated at {dashboard_dir}")
    
    def _generate_dashboard_html(self, dashboard_dir: str, title: str, performance: Dict[str, Any]) -> None:
        """
        Generate HTML dashboard page.
        
        Args:
            dashboard_dir: Directory containing dashboard components
            title: Dashboard title
            performance: Performance metrics
        """
        # List image files in dashboard directory
        static_files = [f for f in os.listdir(dashboard_dir) if f.endswith('.png')]
        interactive_files = [f for f in os.listdir(dashboard_dir) if f.endswith('.html')]
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }}
                .dashboard-container {{ max-width: 1200px; margin: 0 auto; }}
                .dashboard-header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .metric-card {{ 
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
                    padding: 15px;
                    margin-bottom: 20px;
                    transition: all 0.3s;
                }}
                .metric-card:hover {{ box-shadow: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23); }}
                .metric-value {{ font-size: 2rem; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ font-size: 1rem; color: #6c757d; }}
                .chart-container {{ margin-bottom: 30px; }}
                .chart-title {{ font-size: 1.25rem; margin-bottom: 10px; }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .nav-tabs {{ margin-bottom: 20px; }}
                iframe {{ width: 100%; height: 600px; border: none; }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>{title}</h1>
                    <p class="text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="row">
        """
        
        # Add performance metrics
        metrics = [
            ('Total Return', f"{performance.get('total_return', 0):.2f}%", 'total_return'),
            ('Number of Trades', str(performance.get('num_trades', 0)), 'num_trades'),
            ('Win Rate', f"{performance.get('win_rate', 0):.2f}%", 'win_rate'),
            ('Profit Factor', f"{performance.get('profit_factor', 0):.2f}", 'profit_factor'),
            ('Max Drawdown', f"{performance.get('max_drawdown', 0):.2f}%", 'max_drawdown'),
            ('Sharpe Ratio', f"{performance.get('sharpe_ratio', 0):.2f}", 'sharpe_ratio'),
            ('Sortino Ratio', f"{performance.get('sortino_ratio', 0):.2f}", 'sortino_ratio'),
            ('Final Capital', f"${performance.get('final_capital', 0):,.2f}", 'final_capital')
        ]
        
        for label, value, key in metrics:
            # Determine if metric is positive or negative
            metric_class = ""
            if key in ['total_return', 'win_rate', 'profit_factor', 'sharpe_ratio', 'sortino_ratio', 'final_capital']:
                metric_value = performance.get(key, 0)
                if isinstance(metric_value, (int, float)):
                    metric_class = "positive" if metric_value > 0 else "negative" if metric_value < 0 else ""
            
            if key == 'max_drawdown':
                metric_class = "negative"  # Max drawdown is always negative
            
            html_content += f"""
                    <div class="col-md-3 col-sm-6">
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value {metric_class}">{value}</div>
                        </div>
                    </div>
            """
        
        # Add tabs for static and interactive charts
        html_content += """
                </div>
                
                <ul class="nav nav-tabs" id="dashboardTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="static-tab" data-bs-toggle="tab" data-bs-target="#static" type="button" role="tab" aria-controls="static" aria-selected="true">Static Charts</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="interactive-tab" data-bs-toggle="tab" data-bs-target="#interactive" type="button" role="tab" aria-controls="interactive" aria-selected="false">Interactive Charts</button>
                    </li>
                </ul>
                
                <div class="tab-content" id="dashboardTabsContent">
                    <div class="tab-pane fade show active" id="static" role="tabpanel" aria-labelledby="static-tab">
                        <div class="row">
        """
        
        # Add static charts
        for image_file in static_files:
            file_path = image_file
            title = image_file.replace('.png', '').replace('_', ' ').title()
            
            html_content += f"""
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <div class="chart-title">{title}</div>
                                    <img src="{file_path}" alt="{title}" class="img-fluid">
                                </div>
                            </div>
            """
        
        # Add interactive charts tab
        html_content += """
                        </div>
                    </div>
                    <div class="tab-pane fade" id="interactive" role="tabpanel" aria-labelledby="interactive-tab">
                        <div class="row">
        """
        
        # Add interactive charts
        for html_file in interactive_files:
            file_path = html_file
            title = html_file.replace('.html', '').replace('_', ' ').title()
            
            html_content += f"""
                            <div class="col-md-12">
                                <div class="chart-container">
                                    <div class="chart-title">{title}</div>
                                    <iframe src="{file_path}"></iframe>
                                </div>
                            </div>
            """
        
        # Close HTML
        html_content += """
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        # Write HTML file
        dashboard_path = os.path.join(dashboard_dir, "dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard HTML saved to {dashboard_path}")

if __name__ == "__main__":
    # Test the visualizer
    from datetime import datetime, timedelta
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    
    # Equity curve
    initial_capital = 10000
    returns = np.random.normal(0.001, 0.01, 100)
    equity = initial_capital * np.cumprod(1 + returns)
    equity_series = pd.Series(equity, index=dates)
    
    # Trades
    trades = []
    current_equity = initial_capital
    for i in range(20):
        entry_date = dates[i*5]
        exit_date = entry_date + timedelta(days=2)
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        entry_price = 100 + np.random.normal(0, 5)
        exit_price = entry_price * (1 + np.random.normal(0.01, 0.05))
        profit_amount = (exit_price - entry_price) if side == 'buy' else (entry_price - exit_price)
        profit_pct = (exit_price / entry_price - 1) * 100 if side == 'buy' else (entry_price / exit_price - 1) * 100
        
        current_equity += profit_amount
        
        exit_reason = np.random.choice(['TP', 'SL', 'SIGNAL', 'END_OF_DATA'])
        
        trades.append({
            'symbol': 'BTCUSDT',
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': 1,
            'entry_time': entry_date,
            'exit_time': exit_date,
            'profit_amount': profit_amount,
            'profit_percentage': profit_pct,
            'exit_reason': exit_reason
        })
    
    # Price data
    price_data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(102, 5, 100),
        'low': np.random.normal(98, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000, 100, 100)
    }, index=dates)
    
    # Training history
    history = {
        'loss': np.random.normal(0.1, 0.02, 50),
        'val_loss': np.random.normal(0.15, 0.02, 50),
        'mae': np.random.normal(0.08, 0.01, 50),
        'val_mae': np.random.normal(0.12, 0.01, 50)
    }
    
    # Feature importance
    feature_names = [f'feature_{i}' for i in range(20)]
    importances = np.random.normal(0.05, 0.1, 20)
    stds = np.random.normal(0.01, 0.005, 20)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'std': stds
    })
    
    # Create visualizer
    visualizer = Visualizer()
    
    # Generate all plots
    visualizer.plot_equity_curve(equity_series, initial_capital)
    visualizer.plot_drawdown(equity_series)
    visualizer.plot_monthly_returns(equity_series)
    visualizer.plot_trade_distribution(trades)
    visualizer.plot_learning_progress(history)
    visualizer.plot_feature_importance(importance_df)
    visualizer.plot_price_with_signals(price_data, trades)
    
    # Generate interactive plots
    try:
        visualizer.plot_interactive_equity_curve(equity_series, trades)
        visualizer.plot_interactive_trade_analysis(trades)
    except Exception as e:
        print(f"Failed to create interactive plots: {e}")
    
    # Generate dashboard
    backtest_results = {
        'performance': {
            'initial_capital': initial_capital,
            'final_capital': equity[-1],
            'total_return': (equity[-1] / initial_capital - 1) * 100,
            'num_trades': len(trades),
            'win_rate': sum(t['profit_percentage'] > 0 for t in trades) / len(trades) * 100,
            'profit_factor': 1.5,
            'max_drawdown': 10,
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.5
        },
        'equity_curve': equity,
        'trades': trades,
        'signals': price_data
    }
    
    visualizer.generate_backtest_dashboard(backtest_results, title="Sample Backtest Dashboard")
    
    print("Visualizations generated successfully!")