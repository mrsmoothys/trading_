"""
Live trading module for the AI trading bot.
Connects to cryptocurrency exchanges and executes trades in real-time.
"""
import os
import sys
import time
import json
import logging
import hmac
import hashlib
import requests
import websocket
import threading
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from requests.exceptions import RequestException

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LOGS_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR,
    TRADING_FEE, SLIPPAGE, POSITION_SIZE, INITIAL_CAPITAL,
    LOOKBACK_WINDOW, PREDICTION_HORIZON
)
from data_processor import load_data, create_training_sequences
from feature_engineering import generate_features
from model import DeepLearningModel, ModelManager
from strategy import MLTradingStrategy, Position, Order
from utils import setup_logging, save_metadata, format_time, print_system_info

# Constants for API throttling
API_REQUEST_LIMIT = 1200  # Requests per minute (adjust based on exchange limits)
API_REQUEST_WINDOW = 60  # Window in seconds
API_RETRY_DELAY = 5  # Seconds to wait after rate limit hit
MAX_RETRIES = 3  # Maximum number of retries for API calls

class ExchangeConnection:
    """
    Base class for connecting to cryptocurrency exchanges.
    """
    
    def __init__(
        self,
        exchange_name: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize the exchange connection.
        
        Args:
            exchange_name: Name of the exchange
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet/sandbox mode
        """
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Request tracking for rate limiting
        self.request_timestamps = []
        
        # Logger
        self.logger = logging.getLogger(f"exchange.{exchange_name}")
    
    def _check_rate_limit(self) -> None:
        """
        Check if we're about to hit the rate limit and wait if necessary.
        """
        current_time = time.time()
        
        # Remove timestamps older than the window
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < API_REQUEST_WINDOW]
        
        # Check if we've hit the limit
        if len(self.request_timestamps) >= API_REQUEST_LIMIT:
            wait_time = self.request_timestamps[0] + API_REQUEST_WINDOW - current_time
            if wait_time > 0:
                self.logger.warning(f"Rate limit approached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Add current timestamp
        self.request_timestamps.append(current_time)
    
    def _handle_request_exception(self, e: Exception, endpoint: str, retry_count: int) -> bool:
        """
        Handle request exceptions with appropriate retry logic.
        
        Args:
            e: The exception that was raised
            endpoint: The API endpoint that was called
            retry_count: Current retry count
            
        Returns:
            True if should retry, False otherwise
        """
        if retry_count >= MAX_RETRIES:
            self.logger.error(f"Max retries exceeded for {endpoint}")
            return False
        
        if isinstance(e, RequestException):
            if hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
                
                # Rate limit exceeded
                if status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', API_RETRY_DELAY))
                    self.logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    return True
                
                # Server errors (5xx)
                elif 500 <= status_code < 600:
                    wait_time = (2 ** retry_count) * API_RETRY_DELAY
                    self.logger.warning(f"Server error {status_code}. Retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                    return True
        
        # General exception
        wait_time = (2 ** retry_count) * API_RETRY_DELAY
        self.logger.warning(f"Request failed: {e}. Retrying in {wait_time} seconds")
        time.sleep(wait_time)
        return True
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account information
        """
        raise NotImplementedError("Derived classes must implement get_account_info")
    
    def get_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with market data
        """
        raise NotImplementedError("Derived classes must implement get_market_data")
    
    def place_order(self, order: Order) -> Dict[str, Any]:
        """
        Place a trading order.
        
        Args:
            order: Order to place
            
        Returns:
            Dictionary with order information
        """
        raise NotImplementedError("Derived classes must implement place_order")
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Trading symbol
            
        Returns:
            Dictionary with cancellation information
        """
        raise NotImplementedError("Derived classes must implement cancel_order")
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order
            symbol: Trading symbol
            
        Returns:
            Dictionary with order status
        """
        raise NotImplementedError("Derived classes must implement get_order_status")
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of open orders
        """
        raise NotImplementedError("Derived classes must implement get_open_orders")
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with position information
        """
        raise NotImplementedError("Derived classes must implement get_position")
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with ticker information
        """
        raise NotImplementedError("Derived classes must implement get_ticker")

class BinanceConnection(ExchangeConnection):
    """
    Connection class for Binance exchange.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True
    ):
        """
        Initialize the Binance connection.
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet mode
        """
        super().__init__("binance", api_key, api_secret, testnet)
        
        # Set API base URL
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            self.wss_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com"
            self.wss_url = "wss://stream.binance.com:9443/ws"
        
        # Websocket connections
        self.ws_connections = {}
        
        # Test connection
        if api_key and api_secret:
            self.test_connection()
    
    def test_connection(self) -> bool:
        """
        Test API connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            url = f"{self.base_url}/api/v3/ping"
            response = requests.get(url)
            response.raise_for_status()
            
            # Test authenticated endpoint
            account_info = self.get_account_info()
            
            self.logger.info("Binance connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Binance connection test failed: {e}")
            return False
    
    def _get_signature(self, query_string: str) -> str:
        """
        Generate HMAC-SHA256 signature for API request.
        
        Args:
            query_string: Query string to sign
            
        Returns:
            Signature string
        """
        if not self.api_secret:
            raise ValueError("API secret not set")
        
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get request headers with API key.
        
        Returns:
            Dictionary of headers
        """
        if not self.api_key:
            raise ValueError("API key not set")
        
        return {
            'X-MBX-APIKEY': self.api_key
        }
    
    def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, signed: bool = False) -> Any:
        """
        Make API request with rate limiting and authentication.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether the request needs signature
            
        Returns:
            JSON response data
        """
        # Check rate limits
        self._check_rate_limit()
        
        # Prepare params
        params = params or {}
        
        if signed:
            # Add timestamp for signed requests
            params['timestamp'] = int(time.time() * 1000)
            
            # Convert params to query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            
            # Generate signature
            signature = self._get_signature(query_string)
            
            # Add signature to params
            params['signature'] = signature
        
        # Prepare URL
        url = f"{self.base_url}{endpoint}"
        
        # Make request with retries
        retry_count = 0
        while retry_count <= MAX_RETRIES:
            try:
                if method == 'GET':
                    response = requests.get(url, params=params, headers=self._get_headers() if self.api_key else None)
                elif method == 'POST':
                    response = requests.post(url, params=params, headers=self._get_headers())
                elif method == 'DELETE':
                    response = requests.delete(url, params=params, headers=self._get_headers())
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response.raise_for_status()
                return response.json()
            
            except Exception as e:
                if not self._handle_request_exception(e, endpoint, retry_count):
                    raise
                
                retry_count += 1
        
        raise Exception(f"Failed to make request to {endpoint} after {MAX_RETRIES} retries")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account information
        """
        return self._make_request('GET', '/api/v3/account', signed=True)
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            Dictionary with exchange information
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._make_request('GET', '/api/v3/exchangeInfo', params=params)
    
    def get_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with market data
        """
        # Convert timeframe to binance interval format
        interval_map = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
        
        interval = interval_map.get(timeframe)
        if not interval:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Prepare parameters
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        # Make request
        response = self._make_request('GET', '/api/v3/klines', params=params)
        
        # Convert to DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ]
        
        df = pd.DataFrame(response, columns=columns)
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                          'trades', 'taker_buy_base', 'taker_buy_quote']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def place_order(self, order: Order) -> Dict[str, Any]:
        """
        Place a trading order.
        
        Args:
            order: Order to place
            
        Returns:
            Dictionary with order information
        """
        # Prepare parameters
        params = {
            'symbol': order.symbol,
            'side': order.side.upper(),
            'quantity': order.quantity
        }
        
        # Set order type
        if order.order_type.lower() == 'market':
            params['type'] = 'MARKET'
        elif order.order_type.lower() == 'limit':
            params['type'] = 'LIMIT'
            params['timeInForce'] = 'GTC'
            params['price'] = order.price
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")
        
        # Add stop loss and take profit if provided
        if order.sl_price is not None:
            params['stopPrice'] = order.sl_price
            params['stopLimitPrice'] = order.sl_price
        
        if order.tp_price is not None:
            # Binance doesn't support TP in the same order, would need a separate order
            pass
        
        # Place order
        response = self._make_request('POST', '/api/v3/order', params=params, signed=True)
        
        return response
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Trading symbol
            
        Returns:
            Dictionary with cancellation information
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return self._make_request('DELETE', '/api/v3/order', params=params, signed=True)
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order
            symbol: Trading symbol
            
        Returns:
            Dictionary with order status
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return self._make_request('GET', '/api/v3/order', params=params, signed=True)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._make_request('GET', '/api/v3/openOrders', params=params, signed=True)
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with position information
        """
        # Get account info
        account_info = self.get_account_info()
        
        # Find asset balance
        base_asset = symbol[:-4]  # Assuming USDT pairs like BTCUSDT
        quote_asset = symbol[-4:]  # USDT
        
        base_balance = next((asset for asset in account_info['balances'] 
                            if asset['asset'] == base_asset), {'free': '0', 'locked': '0'})
        
        quote_balance = next((asset for asset in account_info['balances'] 
                             if asset['asset'] == quote_asset), {'free': '0', 'locked': '0'})
        
        # Get ticker for current price
        ticker = self.get_ticker(symbol)
        current_price = float(ticker['price'])
        
        # Calculate position value
        base_free = float(base_balance['free'])
        base_locked = float(base_balance['locked'])
        total_base = base_free + base_locked
        position_value = total_base * current_price
        
        return {
            'symbol': symbol,
            'base_asset': base_asset,
            'quote_asset': quote_asset,
            'base_free': base_free,
            'base_locked': base_locked,
            'quote_free': float(quote_balance['free']),
            'quote_locked': float(quote_balance['locked']),
            'total_base': total_base,
            'position_value': position_value,
            'current_price': current_price
        }
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with ticker information
        """
        params = {
            'symbol': symbol
        }
        
        return self._make_request('GET', '/api/v3/ticker/price', params=params)
    
    def subscribe_to_klines(self, symbol: str, interval: str, callback: Callable) -> None:
        """
        Subscribe to kline (candlestick) updates.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe interval
            callback: Callback function for updates
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        
        # Create WebSocket connection
        def on_message(ws, message):
            data = json.loads(message)
            callback(data)
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info(f"WebSocket closed: {close_msg}")
        
        def on_open(ws):
            self.logger.info(f"WebSocket connected: {stream_name}")
        
        # Create and start WebSocket
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(
            f"{self.wss_url}/{stream_name}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Store connection
        self.ws_connections[stream_name] = ws
        
        # Start in a thread
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def close_all_connections(self) -> None:
        """Close all WebSocket connections."""
        for name, ws in self.ws_connections.items():
            self.logger.info(f"Closing WebSocket: {name}")
            ws.close()
        
        self.ws_connections = {}

class LiveTrader:
    """
    Enhanced class for live trading using ML models and strategies.
    Includes improved model loading and PCA support.
    """
    
    def __init__(
        self,
        exchange_connection: ExchangeConnection,
        model_manager: ModelManager,
        symbols: List[str],
        timeframes: List[str],
        config_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the live trader.
        
        Args:
            exchange_connection: Connection to trading exchange
            model_manager: Model manager for loading models
            symbols: List of symbols to trade
            timeframes: List of timeframes to trade
            config_path: Path to configuration file
            data_dir: Directory for data files
            results_dir: Directory for results
            log_level: Logging level
        """
        # Setup logging
        setup_logging(log_level)
        self.logger = logging.getLogger("live_trader")
        
        # Store parameters
        self.exchange = exchange_connection
        self.model_manager = model_manager
        self.symbols = symbols
        self.timeframes = timeframes
        self.data_dir = data_dir or DATA_DIR
        self.results_dir = results_dir or RESULTS_DIR
        
        # Ensure result directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize trading state
        self.strategies = {}
        self.data_cache = {}
        self.last_update_time = {}
        self.active_orders = {}
        self.positions = {}
        self.feature_models = {}  # Store PCA and scaler models
        
        # Control flags
        self.running = False
        self.paused = False
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
        # Event handlers
        self.on_trade_executed = None
        self.on_position_changed = None
        self.on_error = None
        
        self.logger.info("LiveTrader initialized")
    
    def load_trading_models(self, symbols: List[str], timeframes: List[str]) -> bool:
        """
        Enhanced model loading function for live trading.
        Loads ML models and associated PCA/scaler models for all symbols and timeframes.
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes to trade
            
        Returns:
            True if all necessary models were loaded successfully, False otherwise
        """
        self.logger.info(f"Loading models for {symbols} on timeframes {timeframes}")
        success = True
        loaded_models = 0
        
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        for symbol in symbols:
            for timeframe in timeframes:
                model_id = f"{symbol}_{timeframe}"
                model_dir = os.path.join(MODELS_DIR, model_id)
                
                # Skip if model directory doesn't exist
                if not os.path.exists(model_dir):
                    self.logger.warning(f"No model directory found for {model_id}, skipping")
                    success = False
                    continue
                
                # Find latest model file
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5') or f.endswith('.keras')]
                if not model_files:
                    self.logger.warning(f"No model files found for {model_id}, skipping")
                    success = False
                    continue
                
                # Get latest model file by creation time
                latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
                model_path = os.path.join(model_dir, latest_model)
                
                try:
                    # Load model
                    self.logger.info(f"Loading model from {model_path}")
                    
                    # Create model with lookback_window and feature count placeholder that will be replaced
                    model = DeepLearningModel(
                        input_shape=(LOOKBACK_WINDOW, 36),  # Default to 36 features
                        output_dim=PREDICTION_HORIZON,
                        model_path=model_path
                    )
                    
                    # Try to load PCA and scaler models if they exist
                    pca_model = None
                    scaler_model = None
                    
                    # Find PCA and scaler files with matching timestamp to the model
                    model_timestamp = re.search(r'(\d{8}_\d{6})', latest_model)
                    if model_timestamp:
                        timestamp = model_timestamp.group(1)
                        pca_files = [f for f in os.listdir(model_dir) if f.startswith('pca_') and timestamp in f]
                        scaler_files = [f for f in os.listdir(model_dir) if f.startswith('scaler_') and timestamp in f]
                        
                        if pca_files and scaler_files:
                            try:
                                import pickle
                                pca_path = os.path.join(model_dir, pca_files[0])
                                scaler_path = os.path.join(model_dir, scaler_files[0])
                                
                                with open(pca_path, 'rb') as f:
                                    pca_model = pickle.load(f)
                                
                                with open(scaler_path, 'rb') as f:
                                    scaler_model = pickle.load(f)
                                
                                self.logger.info(f"Loaded PCA and scaler models for {model_id}")
                            except Exception as e:
                                self.logger.warning(f"Error loading PCA/scaler models for {model_id}: {e}")
                    
                    # Store in model manager and strategy params
                    self.model_manager.models[model_id] = model
                    
                    # Store PCA and scaler models for this symbol/timeframe
                    if pca_model is not None and scaler_model is not None:
                        if not hasattr(self, 'feature_models'):
                            self.feature_models = {}
                        self.feature_models[model_id] = {
                            'pca': pca_model,
                            'scaler': scaler_model
                        }
                    
                    loaded_models += 1
                    self.logger.info(f"Model loaded successfully for {model_id}")
                
                except Exception as e:
                    self.logger.error(f"Error loading model for {model_id}: {e}")
                    success = False
        
        self.logger.info(f"Loaded {loaded_models} models out of {len(symbols) * len(timeframes)} required")
        return success and loaded_models == len(symbols) * len(timeframes)
    
    def initialize_strategies(self) -> bool:
        """
        Initialize trading strategies for all symbols and timeframes with PCA support.
        
        Returns:
            True if all strategies initialized successfully, False otherwise
        """
        self.logger.info("Initializing strategies with PCA support")
        success = True
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                model_id = f"{symbol}_{timeframe}"
                
                try:
                    # Get model
                    model = self.model_manager.get_model(symbol, timeframe)
                    
                    if model is None:
                        self.logger.warning(f"Model not available for {model_id}, skipping")
                        success = False
                        continue
                    
                    # Get strategy parameters from config
                    strategy_params = self.config.get('strategies', {}).get(model_id, {})
                    
                    # Get defaults from config
                    lookback_window = strategy_params.get('lookback_window', LOOKBACK_WINDOW)
                    threshold = strategy_params.get('threshold', 0.005)
                    position_size = strategy_params.get('position_size', POSITION_SIZE)
                    
                    # Create strategy
                    strategy = MLTradingStrategy(
                        symbol=symbol,
                        timeframe=timeframe,
                        model=model,
                        lookback_window=lookback_window,
                        prediction_horizon=PREDICTION_HORIZON,
                        threshold=threshold,
                        position_size=position_size,
                        initial_capital=INITIAL_CAPITAL,
                        trading_fee=TRADING_FEE,
                        slippage=SLIPPAGE,
                        adaptive_sl_tp=True,
                        trailing_stop=True
                    )
                    
                    # Store strategy
                    self.strategies[model_id] = strategy
                    
                    self.logger.info(f"Strategy initialized for {model_id}")
                
                except Exception as e:
                    self.logger.error(f"Error initializing strategy for {model_id}: {e}")
                    success = False
            
        return success
    
    def _update_data(self, symbol: str, timeframe: str, force: bool = False) -> pd.DataFrame:
        """
        Update market data for a symbol and timeframe with PCA transformation support.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            force: Force update even if recently updated
            
        Returns:
            Updated DataFrame with market data and PCA-transformed features
        """
        model_id = f"{symbol}_{timeframe}"
        current_time = time.time()
        
        # Check if data was recently updated
        last_update = self.last_update_time.get(model_id, 0)
        
        # Determine update interval based on timeframe
        if timeframe.endswith('m'):
            update_interval = int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            update_interval = int(timeframe[:-1]) * 3600
        elif timeframe.endswith('d'):
            update_interval = int(timeframe[:-1]) * 86400
        else:
            update_interval = 3600  # Default to 1 hour
        
        # Check if update is needed
        if not force and current_time - last_update < update_interval / 2:
            return self.data_cache.get(model_id, pd.DataFrame())
        
        try:
            # Get market data
            lookback = self.strategies[model_id].lookback_window if model_id in self.strategies else LOOKBACK_WINDOW
            limit = max(lookback * 3, 100)  # Get more data than needed for feature generation
            
            data = self.exchange.get_market_data(symbol, timeframe, limit=limit)
            
            # Generate regular features
            from feature_engineering import generate_features
            data = generate_features(data)
            
            # Apply PCA transformation if available
            if hasattr(self, 'feature_models') and model_id in self.feature_models:
                try:
                    # Extract OHLCV
                    ohlcv = data[['open', 'high', 'low', 'close', 'volume']]
                    
                    # Get feature columns (exclude OHLCV)
                    feature_cols = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                    features = data[feature_cols]
                    
                    # Handle extreme values
                    features.replace([np.inf, -np.inf], np.nan, inplace=True)
                    features.fillna(features.mean(), inplace=True)
                    
                    # Apply scaler
                    scaler_model = self.feature_models[model_id]['scaler']
                    features_scaled = scaler_model.transform(features)
                    
                    # Apply PCA
                    pca_model = self.feature_models[model_id]['pca']
                    components = pca_model.transform(features_scaled)
                    
                    # Create component dataframe
                    component_df = pd.DataFrame(
                        components,
                        columns=[f'pc_{i+1}' for i in range(components.shape[1])],
                        index=data.index
                    )
                    
                    # Combine with OHLCV
                    data = pd.concat([ohlcv, component_df], axis=1)
                    self.logger.debug(f"Applied PCA transformation for {model_id}")
                
                except Exception as e:
                    self.logger.error(f"Error applying PCA transformation for {model_id}: {e}")
                    self.logger.info("Using regular features without PCA")
            
            # Update cache
            self.data_cache[model_id] = data
            self.last_update_time[model_id] = current_time
            
            self.logger.debug(f"Data updated for {model_id}, {len(data)} candles")
            
            return data
        
        except Exception as e:
            self.logger.error(f"Error updating data for {model_id}: {e}")
            
            # Return cached data if available
            if model_id in self.data_cache:
                return self.data_cache[model_id]
            
            return pd.DataFrame()
    
    def _get_current_signals(self) -> Dict[str, int]:
        """
        Get current trading signals for all symbols and timeframes with PCA support.
        
        Returns:
            Dictionary of signals by model_id
        """
        signals = {}
        
        for model_id, strategy in self.strategies.items():
            symbol, timeframe = model_id.split('_')
            
            try:
                # Update data
                data = self._update_data(symbol, timeframe)
                
                if data.empty:
                    self.logger.warning(f"No data available for {model_id}, skipping signal")
                    continue
                
                # Check if we have PCA models for this symbol/timeframe
                pca_model = None
                scaler_model = None
                if hasattr(self, 'feature_models') and model_id in self.feature_models:
                    pca_model = self.feature_models[model_id]['pca']
                    scaler_model = self.feature_models[model_id]['scaler']
                
                # Generate signals with optional PCA models
                strategy.set_data(data, pca_model=pca_model, scaler_model=scaler_model)
                signal_df = strategy.generate_signals()
                
                # Get latest signal
                latest_signal = signal_df['signal'].iloc[-1]
                
                # Store signal
                signals[model_id] = latest_signal
                
                self.logger.info(f"Signal for {model_id}: {latest_signal}")
            
            except Exception as e:
                self.logger.error(f"Error generating signal for {model_id}: {e}")
        
        return signals
    
    def _execute_signals(self, signals: Dict[str, int]) -> None:
        """
        Execute trading signals by placing orders with improved error handling.
        
        Args:
            signals: Dictionary of signals by model_id
        """
        # Group signals by symbol
        symbol_signals = {}
        for model_id, signal in signals.items():
            symbol, _ = model_id.split('_')
            
            if symbol not in symbol_signals:
                symbol_signals[symbol] = []
            
            symbol_signals[symbol].append(signal)
        
        # Process each symbol
        for symbol, signal_list in symbol_signals.items():
            try:
                # Get current position
                position = self.positions.get(symbol, {'total_base': 0})
                current_position = position['total_base']
                
                # Determine aggregate signal
                # Simple majority vote
                long_signals = sum(1 for s in signal_list if s > 0)
                short_signals = sum(1 for s in signal_list if s < 0)
                neutral_signals = sum(1 for s in signal_list if s == 0)
                
                # Determine overall signal
                if long_signals > short_signals and long_signals > neutral_signals:
                    overall_signal = 1  # Long
                elif short_signals > long_signals and short_signals > neutral_signals:
                    overall_signal = -1  # Short
                else:
                    overall_signal = 0  # Neutral
                
                # Get current price
                ticker = self.exchange.get_ticker(symbol)
                current_price = float(ticker['price'])
                
                # Log the decision
                self.logger.info(f"{symbol} signals: {long_signals} long, {short_signals} short, {neutral_signals} neutral -> Overall: {overall_signal}")
                
                # Determine action based on current position and signal
                if current_position > 0 and overall_signal <= 0:
                    # Close long position
                    self._close_position(symbol, current_position, current_price)
                    
                    # Open short position if signal is short
                    if overall_signal < 0:
                        self._open_position(symbol, 'sell', current_price)
                
                elif current_position < 0 and overall_signal >= 0:
                    # Close short position
                    self._close_position(symbol, abs(current_position), current_price)
                    
                    # Open long position if signal is long
                    if overall_signal > 0:
                        self._open_position(symbol, 'buy', current_price)
                
                elif current_position == 0 and overall_signal != 0:
                    # Open position according to signal
                    side = 'buy' if overall_signal > 0 else 'sell'
                    self._open_position(symbol, side, current_price)
                
                else:
                    self.logger.info(f"No action needed for {symbol}: position={current_position}, signal={overall_signal}")
            
            except Exception as e:
                self.logger.error(f"Error executing signals for {symbol}: {e}")
                
                # Call error handler if set
                if self.on_error:
                    self.on_error(symbol, str(e))
    
    def start(self) -> None:
        """Start the live trading loop with improved monitoring and safeguards."""
        if self.running:
            self.logger.warning("Live trader already running")
            return
        
        self.logger.info("Starting live trader")
        
        # Load models if not already loaded
        model_count = sum(1 for k in self.model_manager.models.keys() if any(symbol in k for symbol in self.symbols))
        if model_count < len(self.symbols) * len(self.timeframes):
            self.logger.info("Models not fully loaded, loading now...")
            if not self.load_trading_models(self.symbols, self.timeframes):
                self.logger.error("Failed to load all models, aborting")
                return
        
        # Initialize strategies if not already initialized
        if len(self.strategies) < len(self.symbols) * len(self.timeframes):
            self.logger.info("Strategies not fully initialized, initializing now...")
            if not self.initialize_strategies():
                self.logger.error("Failed to initialize all strategies, aborting")
                return
        
        # Update positions to get initial state
        self._update_positions()
        
        # Start trading loop
        self.running = True
        self.paused = False
        
        # Print initial position summary
        self.logger.info("Initial positions:")
        for symbol, position in self.positions.items():
            self.logger.info(f"  {symbol}: {position['total_base']} {position.get('base_asset', '')}")
        
        try:
            # Main trading loop
            iteration = 0
            while self.running:
                if not self.paused:
                    iteration += 1
                    self.logger.info(f"Trading iteration {iteration}")
                    
                    # Get signals
                    signals = self._get_current_signals()
                    
                    # Log all signals
                    if signals:
                        self.logger.info("Current signals:")
                        for model_id, signal in signals.items():
                            signal_text = "BUY" if signal > 0 else "SELL" if signal < 0 else "NEUTRAL"
                            self.logger.info(f"  {model_id}: {signal_text} ({signal})")
                    else:
                        self.logger.warning("No signals generated")
                    
                    # Execute signals
                    self._execute_signals(signals)
                    
                    # Check active orders
                    self._check_active_orders()
                    
                    # Update positions
                    self._update_positions()
                    
                    # Calculate performance metrics
                    self._calculate_performance_metrics()
                    
                    # Log current performance
                    if self.performance_metrics:
                        self.logger.info("Current performance:")
                        for symbol, metrics in self.performance_metrics.items():
                            self.logger.info(f"  {symbol}: PnL=${metrics['pnl']:.2f}, Trades={metrics['num_trades']}, Win={metrics['win_rate']*100:.1f}%")
                
                # Determine sleep time based on shortest timeframe
                min_interval = min(self._get_timeframe_seconds(tf) for tf in self.timeframes)
                sleep_time = min(min_interval / 4, 60)  # Max 60 seconds, min 25% of smallest timeframe
                
                self.logger.info(f"Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.info("Live trader stopped by user")
        
        except Exception as e:
            self.logger.error(f"Error in live trading loop: {e}", exc_info=True)
            
            # Call error handler if set
            if self.on_error:
                self.on_error("system", str(e))
        
        finally:
            self.running = False
            self.logger.info("Live trader stopped")
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe to seconds.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '15m', '1d')
            
        Returns:
            Number of seconds
        """
        if timeframe.endswith('m'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 3600
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 86400
        else:
            return 3600  # Default to 1 hour

def run_live_trading(args: argparse.Namespace) -> int:
    """
    Run live trading with command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    logger = logging.getLogger("live_trading")
    
    try:
        # Print system information
        print_system_info()
        
        # Create exchange connection
        if args.exchange.lower() == 'binance':
            exchange = BinanceConnection(
                api_key=args.api_key,
                api_secret=args.api_secret,
                testnet=args.testnet
            )
        else:
            logger.error(f"Unsupported exchange: {args.exchange}")
            return 1
        
        # Create model manager
        model_manager = ModelManager()
        
        # Create live trader
        trader = LiveTrader(
            exchange_connection=exchange,
            model_manager=model_manager,
            symbols=args.symbols,
            timeframes=args.timeframes,
            config_path=args.config,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            log_level=args.log_level
        )
        
        # Define event handlers
        def on_trade_executed(symbol, side, quantity, price):
            logger.info(f"Trade executed: {symbol} {side} {quantity} @ {price}")
        
        def on_position_changed(symbol):
            position = trader.positions.get(symbol, {'total_base': 0})
            logger.info(f"Position changed: {symbol} {position['total_base']} {position.get('base_asset', '')}")
        
        def on_error(source, error_msg):
            logger.error(f"Error in {source}: {error_msg}")
        
        # Set event handlers
        trader.on_trade_executed = on_trade_executed
        trader.on_position_changed = on_position_changed
        trader.on_error = on_error
        
        # Start trading
        trader.start()
        
        # Save report when done
        trader.save_report()
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in live trading: {e}")
        return 1

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run live trading with the AI trading bot.')
    
    # Exchange settings
    parser.add_argument('--exchange', type=str, default='binance',
                        help='Exchange to use (default: binance)')
    parser.add_argument('--api-key', type=str, required=True,
                        help='API key for exchange')
    parser.add_argument('--api-secret', type=str, required=True,
                        help='API secret for exchange')
    parser.add_argument('--testnet', action='store_true',
                        help='Use testnet/sandbox mode')
    
    # Trading settings
    parser.add_argument('--symbols', type=str, nargs='+', required=True,
                        help='Symbols to trade')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1h'],
                        help='Timeframes to trade (default: 1h)')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    
    # Output options
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Directory for data files')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_live_trading(args))