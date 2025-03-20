"""
Trading strategy module for the AI trading bot.
"""
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from multi_timeframe_utils import create_multi_timeframe_data, align_timeframes_to_base
import tqdm


import numpy as np
import pandas as pd

from config import (
    TRADING_FEE, SLIPPAGE, ADAPTIVE_SL_TP, TRAILING_STOP,
    FIXED_SL_PERCENTAGE, FIXED_TP_PERCENTAGE, ATR_SL_MULTIPLIER,
    MIN_RISK_REWARD_RATIO
)

logger = logging.getLogger(__name__)

class Position(Enum):
    """Possible trading positions."""
    LONG = 1
    SHORT = -1
    FLAT = 0

class Order:
    """Class representing a trading order."""
    
    def __init__(
        self,
        symbol: str,
        order_type: str,
        side: str,
        price: float,
        quantity: float,
        timestamp: pd.Timestamp,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        trailing_stop: bool = False,
        trailing_distance: Optional[float] = None,
        order_id: Optional[str] = None
    ):
        """
        Initialize an order.
        
        Args:
            symbol: Trading symbol
            order_type: Type of order (market, limit, etc.)
            side: Order side (buy, sell)
            price: Order price
            quantity: Order quantity
            timestamp: Order timestamp
            sl_price: Stop loss price
            tp_price: Take profit price
            trailing_stop: Whether to use trailing stop
            trailing_distance: Distance for trailing stop
            order_id: Unique order ID
        """
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.trailing_stop = trailing_stop
        self.trailing_distance = trailing_distance
        self.order_id = order_id or f"{symbol}_{side}_{timestamp.strftime('%Y%m%d%H%M%S')}_{id(self)}"
        
        # Execution details
        self.executed_price = None
        self.executed_quantity = None
        self.executed_timestamp = None
        self.status = "CREATED"
        
        # Profit details
        self.exit_price = None
        self.exit_timestamp = None
        self.profit_amount = None
        self.profit_percentage = None
        self.exit_reason = None
    
    def execute(self, price: float, timestamp: pd.Timestamp, slippage: float = SLIPPAGE):
        """
        Execute the order with slippage.
        
        Args:
            price: Execution price
            timestamp: Execution timestamp
            slippage: Slippage percentage
        """
        # Apply slippage
        if self.side == "buy":
            execution_price = price * (1 + slippage)
        else:  # sell
            execution_price = price * (1 - slippage)
        
        self.executed_price = execution_price
        self.executed_quantity = self.quantity
        self.executed_timestamp = timestamp
        self.status = "FILLED"
        
        return execution_price
    
    def update_trailing_stop(self, current_price: float) -> float:
        """
        Update trailing stop price based on current price.
        
        Args:
            current_price: Current market price
            
        Returns:
            Updated stop loss price
        """
        if not self.trailing_stop or self.trailing_distance is None:
            return self.sl_price
        
        if self.side == "buy":  # Long position
            # Trail stop loss higher as price increases
            new_sl = current_price - self.trailing_distance
            if self.sl_price is None or new_sl > self.sl_price:
                self.sl_price = new_sl
        else:  # Short position
            # Trail stop loss lower as price decreases
            new_sl = current_price + self.trailing_distance
            if self.sl_price is None or new_sl < self.sl_price:
                self.sl_price = new_sl
        
        return self.sl_price
    
    def close(self, price: float, timestamp: pd.Timestamp, reason: str, slippage: float = SLIPPAGE):
        """
        Close the order with slippage.
        
        Args:
            price: Close price
            timestamp: Close timestamp
            reason: Reason for closing (TP, SL, signal, etc.)
            slippage: Slippage percentage
        """
        # Apply slippage
        if self.side == "buy":  # For a long position, we sell to close
            exit_price = price * (1 - slippage)
        else:  # For a short position, we buy to close
            exit_price = price * (1 + slippage)
        
        self.exit_price = exit_price
        self.exit_timestamp = timestamp
        self.exit_reason = reason
        self.status = "CLOSED"
        
        # Calculate profit
        if self.side == "buy":
            self.profit_amount = (exit_price - self.executed_price) * self.executed_quantity
            self.profit_percentage = (exit_price / self.executed_price - 1) * 100
        else:
            self.profit_amount = (self.executed_price - exit_price) * self.executed_quantity
            self.profit_percentage = (self.executed_price / exit_price - 1) * 100
        
        return self.profit_amount, self.profit_percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "order_type": self.order_type,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "trailing_stop": self.trailing_stop,
            "trailing_distance": self.trailing_distance,
            "executed_price": self.executed_price,
            "executed_quantity": self.executed_quantity,
            "executed_timestamp": self.executed_timestamp.isoformat() if self.executed_timestamp else None,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "profit_amount": self.profit_amount,
            "profit_percentage": self.profit_percentage,
            "exit_reason": self.exit_reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create order from dictionary."""
        # Convert timestamp strings to datetime
        timestamp = pd.Timestamp(data["timestamp"])
        executed_timestamp = pd.Timestamp(data["executed_timestamp"]) if data["executed_timestamp"] else None
        exit_timestamp = pd.Timestamp(data["exit_timestamp"]) if data["exit_timestamp"] else None
        
        # Create order
        order = cls(
            symbol=data["symbol"],
            order_type=data["order_type"],
            side=data["side"],
            price=data["price"],
            quantity=data["quantity"],
            timestamp=timestamp,
            sl_price=data["sl_price"],
            tp_price=data["tp_price"],
            trailing_stop=data["trailing_stop"],
            trailing_distance=data["trailing_distance"],
            order_id=data["order_id"]
        )
        
        # Set execution details
        order.executed_price = data["executed_price"]
        order.executed_quantity = data["executed_quantity"]
        order.executed_timestamp = executed_timestamp
        order.status = data["status"]
        
        # Set profit details
        order.exit_price = data["exit_price"]
        order.exit_timestamp = exit_timestamp
        order.profit_amount = data["profit_amount"]
        order.profit_percentage = data["profit_percentage"]
        order.exit_reason = data["exit_reason"]
        
        return order

class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,
        trading_fee: float = TRADING_FEE,
        slippage: float = SLIPPAGE,
        adaptive_sl_tp: bool = ADAPTIVE_SL_TP,
        trailing_stop: bool = TRAILING_STOP,
        expected_feature_count : int = 30,
    ):
        """
        Initialize the trading strategy.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            initial_capital: Initial capital for backtesting
            position_size: Position size as fraction of capital
            trading_fee: Trading fee percentage
            slippage: Slippage percentage
            adaptive_sl_tp: Whether to use adaptive SL/TP
            trailing_stop: Whether to use trailing stop
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = position_size
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.adaptive_sl_tp = adaptive_sl_tp
        self.trailing_stop = trailing_stop
        
        # Position tracking
        self.position = Position.FLAT
        self.current_order = None
        self.orders = []
        self.trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.drawdowns = []
        self.returns = []
        
        # Market state
        self.data = None
        self.current_index = None
        self.current_timestamp = None
        self.current_price = None

        self.expected_feature_count = expected_feature_count # new

    
    def set_data(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Set the market data for the strategy.
        
        Args:
            data: Market data as pandas DataFrame
            **kwargs: Additional keyword arguments (for extensibility)
        """
        self.data = data.copy()
        
        # Reset position and performance tracking
        self.position = Position.FLAT
        self.current_order = None
        self.orders = []
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.drawdowns = [0.0]
        self.returns = [0.0]
        
        # Set initial market state
        self.current_index = 0
        self.current_timestamp = self.data.index[0]
        self.current_price = self.data['close'].iloc[0]

    def reset_backtest(self):
        """Reset all backtest state variables to initial values."""
        # Reset position tracking
        self.position = Position.FLAT
        self.current_order = None
        self.orders = []
        self.trades = []
        
        # Reset performance tracking
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.drawdowns = [0.0]
        self.returns = [0.0]
        
        # Reset market state
        if self.data is not None and len(self.data) > 0:
            self.current_index = 0
            self.current_timestamp = self.data.index[0]
            self.current_price = self.data['close'].iloc[0]
        
        logger.info("Backtest state reset")
    
    def update_state(self, index: int) -> None:
        """
        Update the current market state.
        
        Args:
            index: Current data index
        """
        if index >= len(self.data):
            raise IndexError(f"Index {index} out of bounds for data of length {len(self.data)}")
        
        self.current_index = index
        self.current_timestamp = self.data.index[index]
        self.current_price = self.data['close'].iloc[index]
        
        # Update trailing stop if active
        if self.current_order and self.current_order.status == "FILLED" and self.current_order.trailing_stop:
            self.current_order.update_trailing_stop(self.current_price)
        
        # Check for SL/TP hits
        self._check_exit_conditions()
        
        # Update equity curve
        self._update_equity()
    
    def _update_equity(self) -> None:
        """Update equity curve, drawdowns, and returns."""
        if self.position == Position.FLAT:
            current_equity = self.capital
        else:
            # Calculate unrealized P&L
            if self.current_order and self.current_order.status == "FILLED":
                if self.current_order.side == "buy":
                    pnl = (self.current_price - self.current_order.executed_price) * self.current_order.executed_quantity
                else:
                    pnl = (self.current_order.executed_price - self.current_price) * self.current_order.executed_quantity
                
                current_equity = self.capital + pnl
            else:
                current_equity = self.capital
        
        # Update equity curve
        self.equity_curve.append(current_equity)
        
        # Calculate return
        prev_equity = self.equity_curve[-2] if len(self.equity_curve) > 1 else self.initial_capital
        if prev_equity > 0:
            ret = (current_equity / prev_equity) - 1
        else:
            ret = 0
        self.returns.append(ret)
        
        # Calculate drawdown
        max_equity = max(self.equity_curve)
        drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0
        self.drawdowns.append(drawdown)
    
    def _check_exit_conditions(self) -> None:
        """Check for stop loss and take profit conditions."""
        if not self.current_order or self.current_order.status != "FILLED":
            return
        
        # Check stop loss
        if self.current_order.sl_price is not None:
            if (self.current_order.side == "buy" and self.current_price <= self.current_order.sl_price) or \
               (self.current_order.side == "sell" and self.current_price >= self.current_order.sl_price):
                self._close_position("SL")
                return
        
        # Check take profit
        if self.current_order.tp_price is not None:
            if (self.current_order.side == "buy" and self.current_price >= self.current_order.tp_price) or \
               (self.current_order.side == "sell" and self.current_price <= self.current_order.tp_price):
                self._close_position("TP")
                return
    
    def _calculate_sl_tp_levels(self, side: str, entry_price: float) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            side: Order side (buy, sell)
            entry_price: Entry price
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if not self.adaptive_sl_tp:
            # Fixed percentage
            if side == "buy":
                sl_price = entry_price * (1 - FIXED_SL_PERCENTAGE)
                tp_price = entry_price * (1 + FIXED_TP_PERCENTAGE)
            else:
                sl_price = entry_price * (1 + FIXED_SL_PERCENTAGE)
                tp_price = entry_price * (1 - FIXED_TP_PERCENTAGE)
            
            return sl_price, tp_price
        
        # Adaptive SL/TP based on market structure
        
        # Get ATR for volatility-based stops
        atr = self.data['atr'].iloc[self.current_index] if 'atr' in self.data.columns else None
        
        if atr is not None and atr > 0:
            # ATR-based stops
            atr_multiplier = ATR_SL_MULTIPLIER
            if side == "buy":
                sl_price = entry_price - atr * atr_multiplier
                tp_price = entry_price + atr * atr_multiplier * MIN_RISK_REWARD_RATIO
            else:
                sl_price = entry_price + atr * atr_multiplier
                tp_price = entry_price - atr * atr_multiplier * MIN_RISK_REWARD_RATIO
        else:
            # Fall back to fixed percentage
            if side == "buy":
                sl_price = entry_price * (1 - FIXED_SL_PERCENTAGE)
                tp_price = entry_price * (1 + FIXED_TP_PERCENTAGE)
            else:
                sl_price = entry_price * (1 + FIXED_SL_PERCENTAGE)
                tp_price = entry_price * (1 - FIXED_TP_PERCENTAGE)
        
        # Use swing points if available
        if 'swing_high' in self.data.columns and 'swing_low' in self.data.columns:
            # Look back for recent swing points
            lookback = 20
            start_idx = max(0, self.current_index - lookback)
            
            if side == "buy":
                # For long positions, use swing lows for SL and swing highs for TP
                recent_lows = self.data[self.data['swing_low'] == 1].iloc[start_idx:self.current_index]
                recent_highs = self.data[self.data['swing_high'] == 1].iloc[start_idx:self.current_index]
                
                if not recent_lows.empty:
                    # Use the nearest swing low below entry price
                    valid_lows = recent_lows[recent_lows['low'] < entry_price]
                    if not valid_lows.empty:
                        nearest_low = valid_lows['low'].max()
                        sl_price = nearest_low
                
                if not recent_highs.empty:
                    # Use the nearest swing high above entry price
                    valid_highs = recent_highs[recent_highs['high'] > entry_price]
                    if not valid_highs.empty:
                        nearest_high = valid_highs['high'].min()
                        tp_price = nearest_high
            else:
                # For short positions, use swing highs for SL and swing lows for TP
                recent_highs = self.data[self.data['swing_high'] == 1].iloc[start_idx:self.current_index]
                recent_lows = self.data[self.data['swing_low'] == 1].iloc[start_idx:self.current_index]
                
                if not recent_highs.empty:
                    # Use the nearest swing high above entry price
                    valid_highs = recent_highs[recent_highs['high'] > entry_price]
                    if not valid_highs.empty:
                        nearest_high = valid_highs['high'].min()
                        sl_price = nearest_high
                
                if not recent_lows.empty:
                    # Use the nearest swing low below entry price
                    valid_lows = recent_lows[recent_lows['low'] < entry_price]
                    if not valid_lows.empty:
                        nearest_low = valid_lows['low'].max()
                        tp_price = nearest_low
        
        # Ensure minimum risk-reward ratio
        if side == "buy":
            risk = entry_price - sl_price
            reward = tp_price - entry_price
            
            if risk > 0 and reward / risk < MIN_RISK_REWARD_RATIO:
                tp_price = entry_price + risk * MIN_RISK_REWARD_RATIO
        else:
            risk = sl_price - entry_price
            reward = entry_price - tp_price
            
            if risk > 0 and reward / risk < MIN_RISK_REWARD_RATIO:
                tp_price = entry_price - risk * MIN_RISK_REWARD_RATIO
        
        return sl_price, tp_price
    
    def _calculate_trailing_distance(self, side: str, entry_price: float, sl_price: float) -> float:
        """
        Calculate trailing stop distance.
        
        Args:
            side: Order side (buy, sell)
            entry_price: Entry price
            sl_price: Initial stop loss price
            
        Returns:
            Trailing stop distance
        """
        if side == "buy":
            return entry_price - sl_price
        else:
            return sl_price - entry_price
    
    def _open_position(self, position: Position, reason: str) -> Optional[Order]:
        """
        Open a new position.
        
        Args:
            position: Position to open (LONG or SHORT)
            reason: Reason for opening the position
            
        Returns:
            Newly created order
        """
        # Ensure we're not already in a position
        if self.position != Position.FLAT:
            logger.warning(f"Cannot open position, already in {self.position} position")
            return None
        
        # Set position
        self.position = position
        
        # Calculate order details
        side = "buy" if position == Position.LONG else "sell"
        price = self.current_price
        quantity = self.position_size * self.capital / price
        
        # Calculate stop loss and take profit levels
        sl_price, tp_price = self._calculate_sl_tp_levels(side, price)
        
        # Calculate trailing stop distance if enabled
        trailing_distance = None
        if self.trailing_stop:
            trailing_distance = self._calculate_trailing_distance(side, price, sl_price)
        
        # Create and execute order
        order = Order(
            symbol=self.symbol,
            order_type="market",
            side=side,
            price=price,
            quantity=quantity,
            timestamp=self.current_timestamp,
            sl_price=sl_price,
            tp_price=tp_price,
            trailing_stop=self.trailing_stop,
            trailing_distance=trailing_distance
        )
        
        # Execute order with slippage
        executed_price = order.execute(price, self.current_timestamp, self.slippage)
        
        # Apply trading fee
        fee = executed_price * quantity * self.trading_fee
        self.capital -= fee
        
        # Store order
        self.current_order = order
        self.orders.append(order)
        
        logger.info(f"Opened {side} position at {executed_price} with quantity {quantity}")
        logger.info(f"SL: {sl_price}, TP: {tp_price}, Fee: {fee}")
        
        return order
    
    def _close_position(self, reason: str) -> Optional[Tuple[float, float]]:
        """
        Close the current position.
        
        Args:
            reason: Reason for closing the position
            
        Returns:
            Tuple of (profit_amount, profit_percentage)
        """
        # Ensure we're in a position
        if self.position == Position.FLAT or not self.current_order:
            logger.warning("Cannot close position, no position open")
            return None
        
        # Close the order
        price = self.current_price
        profit_amount, profit_percentage = self.current_order.close(
            price, self.current_timestamp, reason, self.slippage
        )
        
        # Apply trading fee
        fee = price * self.current_order.executed_quantity * self.trading_fee
        
        # Update capital
        self.capital += profit_amount - fee
        
        # Store trade details
        trade = {
            "order_id": self.current_order.order_id,
            "symbol": self.symbol,
            "side": self.current_order.side,
            "entry_price": self.current_order.executed_price,
            "exit_price": self.current_order.exit_price,
            "quantity": self.current_order.executed_quantity,
            "entry_time": self.current_order.executed_timestamp,
            "exit_time": self.current_order.exit_timestamp,
            "profit_amount": profit_amount,
            "profit_percentage": profit_percentage,
            "exit_reason": reason,
            "fees": fee
        }
        self.trades.append(trade)
        
        logger.info(f"Closed position at {price} with {profit_percentage:.2f}% profit")
        logger.info(f"Reason: {reason}, Fee: {fee}")
        
        # Reset position
        self.position = Position.FLAT
        self.current_order = None
        
        return profit_amount, profit_percentage
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals with enhanced feature logic."""
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Add signal column
        signals = self.data.copy()
        signals['signal'] = 0
        
        # Get model prediction signal
        for i in range(self.lookback_window, len(signals)):
            # Create sequence for model prediction
            X = self._prepare_model_input(signals, i)
            
            # Get model prediction
            pred = self.model.predict(X)[0]
            predicted_return = (pred[-1] / signals['close'].iloc[i]) - 1
            
            # Base signal from model prediction
            base_signal = 0
            if predicted_return > self.threshold:
                base_signal = 1
            elif predicted_return < -self.threshold:
                base_signal = -1
            
            # Enhance signal with pattern recognition
            signal_strength = base_signal
            
            # Strengthen bullish signals with confirmation
            if base_signal > 0:
                # Strengthen if at support
                if signals['at_support'].iloc[i] > 0:
                    signal_strength += 0.5
                
                # Strengthen if bullish candle pattern
                if signals['bullish_engulfing'].iloc[i] > 0 or signals['hammer'].iloc[i] > 0 or signals['morning_star'].iloc[i] > 0:
                    signal_strength += 0.5
                
                # Reduce if against strong trend
                if signals['strong_downtrend'].iloc[i] > 0:
                    signal_strength -= 1
            
            # Strengthen bearish signals with confirmation
            elif base_signal < 0:
                # Strengthen if at resistance
                if signals['at_resistance'].iloc[i] > 0:
                    signal_strength -= 0.5
                
                # Strengthen if bearish candle pattern
                if signals['bearish_engulfing'].iloc[i] > 0 or signals['shooting_star'].iloc[i] > 0 or signals['evening_star'].iloc[i] > 0:
                    signal_strength -= 0.5
                
                # Reduce if against strong trend
                if signals['strong_uptrend'].iloc[i] > 0:
                    signal_strength += 1
            
            # Convert signal strength to discrete signal
            if signal_strength >= 1:
                signals.loc[signals.index[i], 'signal'] = 1
            elif signal_strength <= -1:
                signals.loc[signals.index[i], 'signal'] = -1
            else:
                signals.loc[signals.index[i], 'signal'] = 0
        
        return signals
    
    import tqdm

    def backtest(self, data, progress_bar=True):
        """Run a backtest with the given data and return performance metrics.
        
        Args:
            data (pd.DataFrame): The data to backtest on
            progress_bar (bool): Whether to show a progress bar
            
        Returns:
            dict: A dictionary of performance metrics
        """
        self.data = data
        self.reset_backtest()
        
        # Generate signals first
        signals = self.generate_signals()
        
        # Create progress bar if enabled
        if progress_bar:
            pbar = tqdm.tqdm(total=len(data), 
                            desc="Backtesting", 
                            bar_format="{l_bar}{bar:30}{r_bar}",
                            ncols=100)
        
        # Run backtest
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Update positions with current price
            self._update_positions(timestamp, row)
            
            # Process signals if any
            if i in signals:
                signal = signals[i]
                self._process_signal(timestamp, row, signal)
            
            # Update equity curve
            self._update_equity(timestamp, row)
            
            # Update progress bar
            if progress_bar:
                # Update progress bar with stats
                pbar.set_postfix(equity=f"${self.equity:.2f}", 
                            trades=len(self.trades),
                            active=len(self.active_positions))
                pbar.update(1)
        
        if progress_bar:
            pbar.close()
        
        # Calculate performance metrics
        return self._calculate_performance()
    
    def calculate_performance(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Basic metrics
        initial_capital = self.initial_capital
        final_capital = self.equity_curve[-1] if self.equity_curve else initial_capital
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Trade metrics
        num_trades = len(self.trades)
        if num_trades == 0:
            return {
                "initial_capital": initial_capital,
                "final_capital": final_capital,
                "total_return": total_return,
                "num_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0
            }
        
        # Win/loss metrics
        winning_trades = [t for t in self.trades if t['profit_amount'] > 0]
        losing_trades = [t for t in self.trades if t['profit_amount'] <= 0]
        
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        avg_profit = np.mean([t['profit_percentage'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_percentage'] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum(t['profit_amount'] for t in winning_trades)
        total_loss = abs(sum(t['profit_amount'] for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Drawdown
        max_drawdown = max(self.drawdowns) * 100 if self.drawdowns else 0
        
        # Risk-adjusted returns
        returns_series = pd.Series(self.returns)
        sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        
        # Sortino ratio (downside risk only)
        downside_returns = returns_series[returns_series < 0]
        sortino_ratio = returns_series.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        return {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio
        }

class MLTradingStrategy(TradingStrategy):
    """
    Machine learning-based trading strategy.
    Uses predictions from a deep learning model.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model,
        lookback_window: int,
        prediction_horizon: int,
        threshold: float = 0.005,
        use_multi_timeframe: bool = True,  # New parameter
        multi_tf_weight: float = 0.5,      # New parameter
        **kwargs
    ):
        """
        Initialize the ML trading strategy.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            model: Trained ML model
            lookback_window: Number of candles to consider for prediction
            prediction_horizon: Number of future candles to predict
            threshold: Price movement threshold for signals
            use_multi_timeframe: Whether to use multi-timeframe confirmation
            multi_tf_weight: Weight for multi-timeframe signals (0-1)
            **kwargs: Additional arguments for TradingStrategy
        """
        super().__init__(symbol, timeframe, **kwargs)
        
        self.model = model
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.threshold = threshold
        self.use_multi_timeframe = use_multi_timeframe
        self.multi_tf_weight = multi_tf_weight
        
        # Storage for multi-timeframe data
        self.multi_tf_data = {}
        self.multi_tf_signals = {}

    def set_data(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Set the market data with multi-timeframe support.
        """
        # Call parent implementation
        super().set_data(data, **kwargs)
        
        # Create multi-timeframe data if enabled
        if self.use_multi_timeframe:
            try:
                self.multi_tf_data = create_multi_timeframe_data(data, self.timeframe)
                logger.info(f"Created multi-timeframe data for {len(self.multi_tf_data)} timeframes")
            except Exception as e:
                logger.error(f"Error creating multi-timeframe data: {e}")
                self.multi_tf_data = {self.timeframe: data}

    def _transform_features_with_pca(self, pca_model, scaler_model):
        """
        Transform features using PCA and scaler models.
        
        Args:
            pca_model: PCA model for dimensionality reduction
            scaler_model: Scaler model for standardization
        """
        # Separate OHLCV
        ohlcv = self.data[['open', 'high', 'low', 'close', 'volume']]
        
        # Get feature columns (excluding OHLCV)
        feature_cols = [col for col in self.data.columns 
                    if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if not feature_cols:
            return  # No features to transform
        
        features = self.data[feature_cols].copy()
        
        # Handle extreme values
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(features.mean(), inplace=True)
        
        # Ensure enough features for the model
        expected_features = len(feature_cols)
        if expected_features < 30:
            # Add dummy features to match expected dimensions
            for i in range(expected_features, 30):
                features[f'dummy_{i}'] = 0.0


        # Standardize features
        try:
            features_scaled = scaler_model.transform(features)
            
            # Apply PCA
            components = pca_model.transform(features_scaled)
            
            # Create component dataframe
            component_df = pd.DataFrame(
                components,
                columns=[f'pc_{i+1}' for i in range(components.shape[1])],
                index=self.data.index
            )
            
            # Combine with OHLCV
            self.data = pd.concat([ohlcv, component_df], axis=1)
            logger.info(f"PCA transformation applied: {components.shape[1]} components")

        except Exception as e:
            logger.error(f"Error applying PCA transformation: {e}")
            # Continue with original data if transformation fails
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals with multi-timeframe confirmation.
        """
        # Generate signals for each timeframe
        if self.use_multi_timeframe and len(self.multi_tf_data) > 1:
            logger.info("Generating signals with multi-timeframe confirmation")
            
            # Process each timeframe
            all_tf_signals = {}
            for tf, tf_data in self.multi_tf_data.items():
                # Generate signals for this timeframe
                signals_tf = self._generate_signals_for_timeframe(tf_data)
                all_tf_signals[tf] = signals_tf
            
            # Align all signals to base timeframe
            aligned_signals = align_timeframes_to_base(all_tf_signals, self.timeframe)
            
            # Combine signals with weighted approach
            base_signals = all_tf_signals[self.timeframe]
            
            # Add columns for each timeframe's signals
            for tf, signals_series in aligned_signals.items():
                if tf != self.timeframe:
                    base_signals[f'signal_{tf}'] = signals_series
            
            # Create weighted signal using all timeframes
            base_weight = 1.0 - self.multi_tf_weight
            higher_weights = self.multi_tf_weight / (len(aligned_signals) - 1) if len(aligned_signals) > 1 else 0
            
            # Calculate combined signal
            base_signals['combined_signal'] = base_signals['signal'] * base_weight
            
            for tf, signals_series in aligned_signals.items():
                if tf != self.timeframe:
                    base_signals['combined_signal'] += signals_series * higher_weights
            
            # Discretize combined signal
            base_signals['signal'] = np.where(
                base_signals['combined_signal'] >= 0.5, 1,
                np.where(base_signals['combined_signal'] <= -0.5, -1, 0)
            )
            
            return base_signals
        
        else:
            # Fall back to single timeframe
            return self._generate_signals_for_timeframe(self.data)
        


    def _update_positions(self, current_time, current_price):
        """
        Update the status of all open positions based on current price.
        This method checks for stop-loss, take-profit, and trailing stop conditions.
        
        Args:
            current_time: Current timestamp
            current_price: Current price
        """
        if not self.positions:
            return
        
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            # Skip already closed positions (should not be in self.positions anyway)
            if position['status'] != 'open':
                continue
            
            # Calculate current P&L
            if position['side'] == 'long':
                profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            else:  # short
                profit_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
            
            position['current_price'] = current_price
            position['current_profit_pct'] = profit_pct
            position['current_profit'] = position['size'] * profit_pct / 100
            
            # Update trailing stop if enabled and in profit
            if self.trailing_stop and profit_pct > 0:
                # For longs, move stop loss up as price increases
                if position['side'] == 'long':
                    new_stop = current_price * (1 - self.trailing_stop_pct/100)
                    if new_stop > position['stop_loss'] and new_stop < current_price:
                        position['stop_loss'] = new_stop
                # For shorts, move stop loss down as price decreases
                else:
                    new_stop = current_price * (1 + self.trailing_stop_pct/100)
                    if new_stop < position['stop_loss'] and new_stop > current_price:
                        position['stop_loss'] = new_stop
            
            # Check if stop loss hit
            if (position['side'] == 'long' and current_price <= position['stop_loss']) or \
            (position['side'] == 'short' and current_price >= position['stop_loss']):
                position['exit_price'] = current_price
                position['exit_time'] = current_time
                position['status'] = 'closed'
                position['exit_reason'] = 'stop_loss'
                positions_to_close.append(position_id)
                continue
            
            # Check if take profit hit
            if (position['side'] == 'long' and current_price >= position['take_profit']) or \
            (position['side'] == 'short' and current_price <= position['take_profit']):
                position['exit_price'] = current_price
                position['exit_time'] = current_time
                position['status'] = 'closed'
                position['exit_reason'] = 'take_profit'
                positions_to_close.append(position_id)
                continue
        
        # Close positions and update portfolio
        for position_id in positions_to_close:
            position = self.positions[position_id]
            self._close_position(position)
            del self.positions[position_id]
    
    def _generate_signals_for_timeframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for a specific timeframe.
        
        Args:
            data: DataFrame with OHLCV data for the timeframe
            
        Returns:
            DataFrame with signals
        """
        if data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Add signal column
        signals = data.copy()
        signals['signal'] = 0
        
        # Check if we're using PCA components or raw features
        pca_cols = [col for col in signals.columns if col.startswith('pc_')]
        using_pca = len(pca_cols) > 0
        
        # Define feature columns based on what's available
        feature_columns = pca_cols if using_pca else [
            col for col in signals.columns 
            if col not in ['open', 'high', 'low', 'close', 'volume', 'signal']
        ]
        
        # Generate predictions for each candle
        for i in range(self.lookback_window, len(signals)):
            try:
                # Get lookback window for features
                window_data = signals[feature_columns].iloc[i-self.lookback_window:i].values
                
                # Normalize data
                mean = np.mean(window_data, axis=0)
                std = np.std(window_data, axis=0)
                std = np.where(std == 0, 1e-8, std)  # Avoid division by zero
                window_data_norm = (window_data - mean) / std
                
                # Reshape for model input (add batch dimension)
                X = np.expand_dims(window_data_norm, axis=0)
                
                # Get prediction
                pred = self.model.predict(X, verbose=0)[0]
                
                # Calculate predicted return
                current_price = signals['close'].iloc[i]
                predicted_price = pred[-1]  # Last value in prediction horizon
                predicted_return = (predicted_price / current_price) - 1
                
                # Base signal from model prediction with market context enhancement
                predicted_return_adj = predicted_return  # Start with model prediction

                # Add market context enhancement even when using PCA
                # 1. Simple trend following component
                price_sma20 = signals['close'].rolling(20).mean()
                price_sma50 = signals['close'].rolling(50).mean()

                if i >= 50:  # We need enough data for the SMAs
                    # Get current price and SMAs
                    curr_price = signals['close'].iloc[i]
                    curr_sma20 = price_sma20.iloc[i]
                    curr_sma50 = price_sma50.iloc[i]
                    
                    # Calculate trend bias factor (0.2 = moderate enhancement)
                    trend_bias = 0.2
                    
                    # Price above both MAs - enhance bullish predictions
                    if curr_price > curr_sma20 and curr_price > curr_sma50:
                        predicted_return_adj += trend_bias * abs(predicted_return)
                        
                    # Price below both MAs - enhance bearish predictions
                    elif curr_price < curr_sma20 and curr_price < curr_sma50:
                        predicted_return_adj -= trend_bias * abs(predicted_return)
                        
                    # MAs crossing - stronger signal
                    if (curr_sma20 > curr_sma50 and price_sma20.iloc[i-1] <= price_sma50.iloc[i-1]):
                        # Bullish crossover
                        predicted_return_adj += trend_bias * 2
                    elif (curr_sma20 < curr_sma50 and price_sma20.iloc[i-1] >= price_sma50.iloc[i-1]):
                        # Bearish crossover
                        predicted_return_adj -= trend_bias * 2

                # 2. Volatility context
                if 'atr' in signals.columns:
                    # Get ATR and normalize by price
                    atr_pct = signals['atr'].iloc[i] / signals['close'].iloc[i]
                    
                    # If in high volatility period, amplify signals
                    rolling_atr = signals['atr'].rolling(20).mean()
                    if i >= 20 and signals['atr'].iloc[i] > rolling_atr.iloc[i] * 1.5:
                        volatility_factor = 1.5  # Amplify during high volatility
                        predicted_return_adj *= volatility_factor

                # Use adjusted predicted return for signal generation
                if predicted_return_adj > self.threshold:
                    base_signal = 1
                elif predicted_return_adj < -self.threshold:
                    base_signal = -1
                else:
                    base_signal = 0
                
                # Store signals
                signals.loc[signals.index[i], 'signal'] = base_signal
                
            except Exception as e:
                logger.error(f"Error generating signal at index {i}: {e}")
                # Continue to next candle
        
        return signals



class EnsembleTradingStrategy(TradingStrategy):
    """
    Ensemble trading strategy.
    Combines multiple strategies or signal sources.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        strategies: List[TradingStrategy],
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize the ensemble trading strategy.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            strategies: List of strategies to combine
            weights: Weights for each strategy (normalized if not summing to 1)
            **kwargs: Additional arguments for TradingStrategy
        """
        super().__init__(symbol, timeframe, **kwargs)
        
        self.strategies = strategies
        
        # Normalize weights if provided
        if weights is not None:
            weights_sum = sum(weights)
            self.weights = [w / weights_sum for w in weights] if weights_sum > 0 else [1.0 / len(strategies)] * len(strategies)
        else:
            # Equal weights by default
            self.weights = [1.0 / len(strategies)] * len(strategies)
    
    def set_data(self, data: pd.DataFrame, pca_model=None, scaler_model=None) -> None:
        """
        Set the market data for the strategy.
        
        Args:
            data: Market data as pandas DataFrame
            pca_model: Optional PCA model for feature transformation
            scaler_model: Optional scaler model for feature standardization
        """
        # Store original data
        self.data = data.copy()
        
        # Reset position and performance tracking
        self.position = Position.FLAT
        self.current_order = None
        self.orders = []
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.drawdowns = [0.0]
        self.returns = [0.0]
        
        # Set initial market state
        self.current_index = 0
        self.current_timestamp = self.data.index[0]
        self.current_price = self.data['close'].iloc[0]
        
        # Set data for each strategy in the ensemble
        for strategy in self.strategies:
            try:
                strategy.set_data(self.data)
            except Exception as e:
                logger.error(f"Error setting data for strategy {type(strategy).__name__}: {e}")
        

    def _transform_features_with_pca(self, pca_model, scaler_model):
        """
        Transform features using PCA and scaler models.
        
        Args:
            pca_model: PCA model for dimensionality reduction
            scaler_model: Scaler model for standardization
        """
        # Separate OHLCV
        ohlcv = self.data[['open', 'high', 'low', 'close', 'volume']]
        
        # Get feature columns (excluding OHLCV)
        feature_cols = [col for col in self.data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if not feature_cols:
            return  # No features to transform
        
        features = self.data[feature_cols]
        
        # Handle extreme values
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(features.mean(), inplace=True)
        
        # Standardize features
        features_scaled = scaler_model.transform(features)
        
        # Apply PCA
        components = pca_model.transform(features_scaled)
        
        # Create component dataframe
        component_df = pd.DataFrame(
            components,
            columns=[f'pc_{i+1}' for i in range(components.shape[1])],
            index=self.data.index
        )
        
        # Combine with OHLCV
        self.data = pd.concat([ohlcv, component_df], axis=1)
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals by combining signals from all strategies.
        
        Returns:
            DataFrame with added signal column
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Add signal column
        signals = self.data.copy()
        signals['signal'] = 0
        
        # Generate signals for each strategy
        all_signals = []
        for i, strategy in enumerate(self.strategies):
            strategy_signals = strategy.generate_signals()
            all_signals.append(strategy_signals['signal'] * self.weights[i])
        
        # Combine signals
        combined_signal = sum(all_signals)
        
        # Discretize combined signal
        signals['signal'] = np.where(
            combined_signal > 0.2, 1,  # Strong buy signal
            np.where(
                combined_signal < -0.2, -1,  # Strong sell signal
                0  # Neutral signal
            )
        )
        
        return signals
    
    # Add this method to the EnsembleTradingStrategy class
    def set_data(self, data: pd.DataFrame, pca_model=None, scaler_model=None) -> None:
        """
        Set the market data for the strategy.
        
        Args:
            data: Market data as pandas DataFrame
            pca_model: Optional PCA model for feature transformation
            scaler_model: Optional scaler model for feature standardization
        """
        # Store original data
        self.data = data.copy()
        
        # Transform features with PCA if models are provided
        if pca_model is not None and scaler_model is not None:
            try:
                self._transform_features_with_pca(pca_model, scaler_model)
            except Exception as e:
                logger.error(f"Error transforming features with PCA: {e}")
                # Continue with original data if transformation fails
        
        # Reset position and performance tracking
        self.position = Position.FLAT
        self.current_order = None
        self.orders = []
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.drawdowns = [0.0]
        self.returns = [0.0]
        
        # Set initial market state
        self.current_index = 0
        self.current_timestamp = self.data.index[0]
        self.current_price = self.data['close'].iloc[0]
        
        # Set data for each strategy in the ensemble
        for strategy in self.strategies:
            try:
                strategy.set_data(self.data)
            except Exception as e:
                logger.error(f"Error setting data for strategy {type(strategy).__name__}: {e}")
    

if __name__ == "__main__":
    # Test the strategy
    from datetime import timedelta
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    data = pd.DataFrame({
        "open": np.random.normal(100, 1, 100),
        "high": np.random.normal(101, 1, 100),
        "low": np.random.normal(99, 1, 100),
        "close": np.random.normal(100, 1, 100),
        "volume": np.random.normal(1000, 100, 100)
    }, index=dates)
    
    # Create ATR for adaptive SL/TP
    data['atr'] = pd.Series(np.random.normal(1, 0.1, 100), index=dates)
    
    # Create swing points for testing
    data['swing_high'] = np.zeros(100)
    data['swing_low'] = np.zeros(100)
    data.loc[data.index[20], 'swing_high'] = 1
    data.loc[data.index[40], 'swing_low'] = 1
    data.loc[data.index[60], 'swing_high'] = 1
    data.loc[data.index[80], 'swing_low'] = 1
    
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
    
    # Run backtest
    results = strategy.backtest(data)
    
    # Print performance
    print("Performance:")
    for key, value in results['performance'].items():
        print(f"{key}: {value}")
    
    # Print trades
    print("\nTrades:")
    for trade in results['trades'][:3]:  # Show first 3 trades
        print(trade)