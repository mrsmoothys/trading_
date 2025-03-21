"""
Reinforcement Learning trading strategy for the AI trading bot.
"""
import os
import sys
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODELS_DIR, RESULTS_DIR, LOOKBACK_WINDOW, PREDICTION_HORIZON,
    TRADING_FEE, SLIPPAGE, POSITION_SIZE, INITIAL_CAPITAL
)
from strategy import TradingStrategy, Position, Order

class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = INITIAL_CAPITAL,
        lookback_window: int = LOOKBACK_WINDOW,
        commission: float = TRADING_FEE,
        slippage: float = SLIPPAGE,
        reward_scaling: float = 1e-4,
        position_size: float = POSITION_SIZE,
        max_position: int = 1,
        enable_short: bool = True
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data and features
            initial_balance: Initial account balance
            lookback_window: Number of candles to consider for state
            commission: Trading commission (percentage)
            slippage: Price slippage (percentage)
            reward_scaling: Scaling factor for rewards
            position_size: Size of each position as fraction of balance
            max_position: Maximum number of positions
            enable_short: Whether to allow short positions
        """
        super(TradingEnv, self).__init__()
        
        # Store parameters
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.commission = commission
        self.slippage = slippage
        self.reward_scaling = reward_scaling
        self.position_size = position_size
        self.max_position = max_position
        self.enable_short = enable_short
        
        # Extract feature columns
        self.feature_columns = [col for col in self.df.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'signal'
        ]]
        
        # Set up logging
        self.logger = logging.getLogger("rl_strategy")
        
        # State space: features + balance + position
        self.state_dim = len(self.feature_columns) + 2
        
        # Action space: -1 (sell), 0 (hold), 1 (buy)
        if enable_short:
            self.action_space = spaces.Discrete(3)
            self.actions = [-1, 0, 1]  # Maps action index to actual action
        else:
            self.action_space = spaces.Discrete(2)
            self.actions = [0, 1]  # Only hold or buy
        
        # Observation space
        # We use a combination of features and account information
        # Each feature is normalized to a reasonable range
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        # Reset account and position
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        
        # Track performance
        self.returns = []
        self.equity_curve = [self.initial_balance]
        
        # Set initial step
        self.current_step = self.lookback_window
        
        # Calculate initial state
        self._update_state()
        
        return self.state
    
    def _update_state(self):
        """Update the current state representation."""
        # Get current features
        features = self.df.iloc[self.current_step][self.feature_columns].values
        
        # Normalize balance and position
        norm_balance = self.balance / self.initial_balance - 1
        norm_position = self.position / self.max_position if self.max_position > 0 else 0
        
        # Combine features and account information
        self.state = np.concatenate([features, [norm_balance, norm_position]])
        
        return self.state
    
    def _take_action(self, action_idx: int):
        """
        Execute the specified action.
        
        Args:
            action_idx: Index of action in action space
            
        Returns:
            Reward from the action
        """
        # Map action index to actual action
        action = self.actions[action_idx]
        
            # Retrieve the 'close' price for the current step
        value = self.df.iloc[self.current_step]['close']

        if isinstance(value, (pd.Series, np.ndarray)):
            # If the value contains multiple elements, take the first one
            if value.size != 1:
                # For a pandas Series, use .iloc[0]; for a NumPy array, use [0]
                if isinstance(value, pd.Series):
                    current_price = float(value.iloc[0])
                else:
                    current_price = float(value[0])
            else:
                current_price = float(value.item())
        else:
            current_price = float(value)
    
        
        # Calculate action-dependent reward and execute trades
        reward = 0
        done = False
        
        # Buy action
        if action == 1 and self.position < self.max_position:
            # Calculate position size based on current balance
            size_to_buy = 1  # Buy one unit of max_position
            cost = current_price * size_to_buy * self.position_size * self.balance
            
            # Apply slippage and commission
            cost_with_fees = cost * (1 + self.slippage + self.commission)
            
            # Check if we have enough balance
            if cost_with_fees <= self.balance:
                # Execute buy
                self.balance -= cost_with_fees
                self.position += size_to_buy
                self.position_value = current_price * self.position * self.position_size * self.initial_balance
                self.logger.debug(f"Buy: {size_to_buy} units at {current_price:.2f}")
        
        # Sell action
        elif action == -1 and self.position > 0:
            # Sell all current position
            size_to_sell = self.position
            revenue = current_price * size_to_sell * self.position_size * self.initial_balance
            
            # Apply slippage and commission
            revenue_with_fees = revenue * (1 - self.slippage - self.commission)
            
            # Execute sell
            self.balance += revenue_with_fees
            self.position = 0
            self.position_value = 0
            self.logger.debug(f"Sell: {size_to_sell} units at {current_price:.2f}")
        
        # Open short position
        elif action == -1 and self.position == 0 and self.enable_short:
            # Calculate position size based on current balance
            size_to_short = 1  # Short one unit of max_position
            self.position = -size_to_short
            self.position_value = current_price * abs(self.position) * self.position_size * self.initial_balance
            self.logger.debug(f"Short: {size_to_short} units at {current_price:.2f}")
        
        # Close short position
        elif action == 1 and self.position < 0:
            # Buy back short position
            size_to_cover = abs(self.position)
            cost = current_price * size_to_cover * self.position_size * self.initial_balance
            
            # Apply slippage and commission
            cost_with_fees = cost * (1 + self.slippage + self.commission)
            
            # Calculate profit/loss from short
            entry_price = self.position_value / (abs(self.position) * self.position_size * self.initial_balance)
            profit = (entry_price - current_price) * size_to_cover * self.position_size * self.initial_balance
            profit_with_fees = profit - cost_with_fees * self.commission - cost * self.slippage
            
            # Execute cover
            self.balance += profit_with_fees
            self.position = 0
            self.position_value = 0
            self.logger.debug(f"Cover short: {size_to_cover} units at {current_price:.2f}, profit: {profit_with_fees:.2f}")
        
        # Calculate portfolio value
        portfolio_value = self.balance
        if self.position != 0:
            portfolio_value += self.position_value
        
        # Calculate returns
        self.equity_curve.append(portfolio_value)
        if len(self.equity_curve) >= 2:
            current_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
            self.returns.append(current_return)
        
        # Calculate reward: change in portfolio value
        reward = (portfolio_value / self.equity_curve[-2] - 1) if len(self.equity_curve) >= 2 else 0
        
        # Scale reward
        reward = reward * self.reward_scaling
        
        return reward, done
    
    def step(self, action_idx: int):
        """
        Take a step in the environment.
        
        Args:
            action_idx: Index of action in action space
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Take action
        reward, done = self._take_action(action_idx)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.df) - 1:
            done = True
        
        # Calculate next state
        if not done:
            self._update_state()
        
        # Calculate info
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.equity_curve[-1],
            'return': self.returns[-1] if self.returns else 0,
            'step': self.current_step
        }
        
        return self.state, reward, done, info

class DQNAgent:
    """
    Deep Q-Network agent for trading.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.1,
        batch_size: int = 64,
        memory_size: int = 10000,
        enable_double_dqn: bool = True,
        enable_dueling_dqn: bool = True,
        target_update_freq: int = 100
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum value of epsilon
            batch_size: Size of training batch
            memory_size: Size of replay memory
            enable_double_dqn: Whether to use Double DQN
            enable_dueling_dqn: Whether to use Dueling DQN
            target_update_freq: Frequency of target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_dqn = enable_dueling_dqn
        self.target_update_freq = target_update_freq
        
        # Initialize replay memory
        self.memory = []
        self.memory_counter = 0
        
        # Create networks
        self.model = self._build_model()
        
        if self.enable_double_dqn:
            # Create target network for Double DQN
            self.target_model = self._build_model()
            self.update_target_network()
        
        # Set up logging
        self.logger = logging.getLogger("dqn_agent")
        
        # Training state
        self.train_step_counter = 0
    
    def _build_model(self):
        """
        Build neural network model.
        
        Returns:
            Keras model
        """
        if self.enable_dueling_dqn:
            # Import at the top of the file or function
            from tensorflow.keras.layers import Lambda
            import tensorflow.keras.backend as K
            # Dueling DQN architecture
            inputs = Input(shape=(self.state_dim,))
            
            # Shared layers
            x = Dense(64, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Value stream
            value_stream = Dense(32, activation='relu')(x)
            value_stream = Dense(1)(value_stream)
            
            # Advantage stream
            advantage_stream = Dense(32, activation='relu')(x)
            advantage_stream = Dense(self.action_dim)(advantage_stream)
            
            # FIXED VERSION - Using Lambda layers:
            # Calculate mean of advantage stream using K.mean (Keras backend)
            advantage_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage_stream)
            
            # Combine streams using Lambda layer
            outputs = Lambda(lambda inputs: inputs[0] + inputs[1] - inputs[2])([
                value_stream, advantage_stream, advantage_mean
            ])
            
            model = Model(inputs=inputs, outputs=outputs)
        else:
            # Standard DQN architecture
            model = Sequential([
                Dense(64, activation='relu', input_shape=(self.state_dim,)),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(self.action_dim, activation='linear')
            ])
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_network(self):
        """Update target network weights from main network."""
        if self.enable_double_dqn:
            self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            # Replace old experiences
            idx = self.memory_counter % self.memory_size
            self.memory[idx] = experience
        
        self.memory_counter += 1
    
    def choose_action(self, state, training: bool = True):
        """
        Choose action based on epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether agent is training
            
        Returns:
            Selected action
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_dim)
        else:
            # Exploitation: best action from Q-network
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            return np.argmax(q_values)
    
    def replay(self):
        """
        Train the network using experience replay.
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from replay memory
        indices = np.random.choice(min(len(self.memory), self.memory_size), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Unpack batch
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        if self.enable_double_dqn:
            # Double DQN: use main network to select action, target network to evaluate
            next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
            next_q = self.target_model.predict(next_states, verbose=0)
            target_q = rewards + (1 - dones) * self.gamma * next_q[np.arange(self.batch_size), next_actions]
        else:
            # Standard DQN
            next_q = self.model.predict(next_states, verbose=0)
            target_q = rewards + (1 - dones) * self.gamma * np.max(next_q, axis=1)
        
        # Update target Q values for actions taken
        for i in range(self.batch_size):
            current_q[i, actions[i]] = target_q[i]
        
        # Train the network
        history = self.model.fit(states, current_q, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Update target network periodically
        self.train_step_counter += 1
        if self.enable_double_dqn and self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save(self, path: str):
        """
        Save the agent to disk.
        
        Args:
            path: Path to save directory
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save main model
        self.model.save(os.path.join(path, 'model.h5'))
        
        # Save target model if using Double DQN
        if self.enable_double_dqn:
            self.target_model.save(os.path.join(path, 'target_model.h5'))
        
        # Save agent parameters
        params = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'enable_double_dqn': self.enable_double_dqn,
            'enable_dueling_dqn': self.enable_dueling_dqn,
            'target_update_freq': self.target_update_freq,
            'train_step_counter': self.train_step_counter
        }
        
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)
    
    @classmethod
    def load(cls, path: str):
        """
        Load agent from disk.
        
        Args:
            path: Path to load directory
            
        Returns:
            Loaded agent
        """
        # Load parameters
        with open(os.path.join(path, 'params.pkl'), 'rb') as f:
            params = pickle.load(f)
        
        # Create agent
        agent = cls(**params)
        
        # Load models
        agent.model = tf.keras.models.load_model(os.path.join(path, 'model.h5'))
        
        if agent.enable_double_dqn:
            agent.target_model = tf.keras.models.load_model(os.path.join(path, 'target_model.h5'))
        
        return agent

class RLTradingStrategy(TradingStrategy):
    """
    Reinforcement Learning trading strategy.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        agent: Optional[DQNAgent] = None,
        train_mode: bool = False,
        lookback_window: int = LOOKBACK_WINDOW,
        initial_capital: float = INITIAL_CAPITAL,
        position_size: float = POSITION_SIZE,
        trading_fee: float = TRADING_FEE,
        slippage: float = SLIPPAGE,
        adaptive_sl_tp: bool = True,
        trailing_stop: bool = True
    ):
        """
        Initialize the RL trading strategy.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            agent: DQN agent (will be created if None)
            train_mode: Whether to train the agent
            lookback_window: Number of candles to consider for state
            initial_capital: Initial capital for backtesting
            position_size: Position size as fraction of capital
            trading_fee: Trading fee percentage
            slippage: Slippage percentage
            adaptive_sl_tp: Whether to use adaptive SL/TP
            trailing_stop: Whether to use trailing stop
        """
        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=initial_capital,
            position_size=position_size,
            trading_fee=trading_fee,
            slippage=slippage,
            adaptive_sl_tp=adaptive_sl_tp,
            trailing_stop=trailing_stop
        )
        
        # Initialize the logger
        self.logger = logging.getLogger("rl_strategy")

        self.lookback_window = lookback_window
        self.train_mode = train_mode
        
        # RL components
        self.env = None
        self.agent = agent
        
        # Training parameters
        self.episodes = 10
        self.train_batch_size = 32
        self.train_interval = 4
        
        # Performance tracking
        self.training_history = []
    
    def _create_env(self, data: pd.DataFrame) -> TradingEnv:
        """
        Create trading environment.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Trading environment
        """
        env = TradingEnv(
            df=data,
            initial_balance=self.initial_capital,
            lookback_window=self.lookback_window,
            commission=self.trading_fee,
            slippage=self.slippage,
            position_size=self.position_size,
            max_position=1,
            enable_short=True
        )
        
        return env
    
    def _create_agent(self, state_dim: int, action_dim: int) -> DQNAgent:
        """
        Create DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            
        Returns:
            DQN agent
        """
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.1,
            batch_size=64,
            memory_size=10000,
            enable_double_dqn=True,
            enable_dueling_dqn=True,
            target_update_freq=100
        )
        
        return agent
    
    def train(self, df, symbol, timeframe, feature_columns, epochs=10, batch_size=32, validation_split=0.2, apply_pca=True):
        """Train the model on data for a specific symbol and timeframe."""
        logger.info(f"Training universal model on {symbol} {timeframe}")
        
        # Prepare data
        X, symbol_ids, timeframe_values = self.prepare_data(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            feature_columns=feature_columns,
            apply_pca=apply_pca
        )
        
        # Get the target values (future price changes)
        y = df['close'].pct_change(periods=self.prediction_horizon).shift(-self.prediction_horizon).values
        y = y[self.lookback_window:].reshape(-1, 1)  # Start from lookback_window to match X
        
        # Remove NaN values
        valid_indices = ~np.isnan(y).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        symbol_ids = symbol_ids[valid_indices]
        timeframe_values = timeframe_values[valid_indices]
        
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        symbol_ids_train, symbol_ids_val = symbol_ids[:split_idx], symbol_ids[split_idx:]
        timeframe_values_train, timeframe_values_val = timeframe_values[:split_idx], timeframe_values[split_idx:]
        
        # Update training summary
        if symbol not in self.training_summary['trained_symbols']:
            self.training_summary['trained_symbols'].append(symbol)
        
        # Train model
        try:
            history = self.model.fit(
                [X_train, symbol_ids_train, timeframe_values_train],
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_val, symbol_ids_val, timeframe_values_val], y_val),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ],
                verbose=1
            )
            
            # Update training metrics
            key = f"{symbol}_{timeframe}"
            self.training_summary['training_history'][key] = {
                'final_loss': float(history.history['loss'][-1]) if 'loss' in history.history else None,
                'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
                'final_mae': float(history.history['mae'][-1]) if 'mae' in history.history else None,
                'epochs_trained': len(history.history['loss']) if 'loss' in history.history else 0
            }
            
            return history
            
        except Exception as e:
            logger.error(f"Error training on {symbol} {timeframe}: {str(e)}")
            return None
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals from the RL agent.
        
        Returns:
            DataFrame with added signal column
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Create a copy of the data
        signals = self.data.copy()
        
        # Add signal column
        signals['signal'] = 0
        
        # Create environment if not already created
        if self.env is None:
            self.env = self._create_env(signals)
        
        # Create agent if not provided
        if self.agent is None:
            self.agent = self._create_agent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_space.n
            )
        
        # Reset environment
        state = self.env.reset()
        done = False
        
        # Generate signals
        actions = []
        
        while not done:
            # Choose action
            action_idx = self.agent.choose_action(state, training=False)
            action = self.env.actions[action_idx]
            
            # Store action
            actions.append(action)
            
            # Take action
            next_state, _, done, _ = self.env.step(action_idx)
            
            # Update state
            state = next_state
        
        # Fill signal column
        for i, action in enumerate(actions):
            idx = i + self.lookback_window
            if idx < len(signals):
                signals.loc[signals.index[idx], 'signal'] = action
        
        return signals
    
    def save_agent(self, path: Optional[str] = None) -> str:
        """
        Save the RL agent to disk.
        
        Args:
            path: Path to save directory
            
        Returns:
            Path to saved agent
        """
        if self.agent is None:
            raise ValueError("Agent not initialized")
        
        # Create default path if not provided
        if path is None:
            path = os.path.join(MODELS_DIR, f"rl_{self.symbol}_{self.timeframe}")
        
        # Save agent
        self.agent.save(path)
        
        self.logger.info(f"Agent saved to {path}")
        return path
    
    def load_agent(self, path: str) -> None:
        """
        Load the RL agent from disk.
        
        Args:
            path: Path to load directory
        """
        # Load agent
        self.agent = DQNAgent.load(path)
        
        self.logger.info(f"Agent loaded from {path}")

def train_rl_strategy(
    symbol: str,
    timeframe: str,
    data_path: str,
    episodes: int = 50,
    save_path: Optional[str] = None
) -> RLTradingStrategy:
    """
    Train an RL trading strategy for a symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Trading timeframe
        data_path: Path to data file
        episodes: Number of training episodes
        save_path: Path to save trained agent
        
    Returns:
        Trained RL trading strategy
    """
    # Set up logging
    logger = logging.getLogger("rl_strategy")
    
    # Load data
    from data_processor import load_data
    from feature_engineering import generate_features
    
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    
    # Generate features
    logger.info("Generating features")
    df = generate_features(df)
    
    # Create strategy
    strategy = RLTradingStrategy(
        symbol=symbol,
        timeframe=timeframe,
        train_mode=True
    )
    
    # Train strategy
    logger.info(f"Training RL strategy for {symbol} {timeframe} with {episodes} episodes")
    history = strategy.train(df, episodes=episodes)
    
    # Save trained agent
    if save_path is not None:
        strategy.save_agent(save_path)
    
    return strategy

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train RL trading strategy.')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol')
    parser.add_argument('--timeframe', type=str, required=True, help='Trading timeframe')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data file')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--save-path', type=str, help='Path to save trained agent')
    
    args = parser.parse_args()
    
    # Train strategy
    strategy = train_rl_strategy(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_path=args.data_path,
        episodes=args.episodes,
        save_path=args.save_path
    )
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Episodes: {args.episodes}")
    
    if strategy.training_history:
        last_episode = strategy.training_history[-1]
        print(f"Final Return: {last_episode['return']:.2f}%")
        print(f"Final Sharpe Ratio: {last_episode['sharpe']:.2f}")
        print(f"Final Epsilon: {last_episode['epsilon']:.4f}")
    
    if args.save_path:
        print(f"Agent saved to: {args.save_path}")