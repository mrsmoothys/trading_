#!/usr/bin/env python
"""
Enhanced live trading script with improved model loading and PCA support.
"""
import os
import sys
import logging
import argparse
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LOGS_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR,
    TRADING_FEE, SLIPPAGE, POSITION_SIZE, INITIAL_CAPITAL
)
from model import ModelManager
from live_trading import BinanceConnection, LiveTrader
from utils import setup_logging, print_system_info

def parse_args():
    """Parse command line arguments."""
    # Import from config to use as defaults
    from config import DATA_DIR, MODELS_DIR, RESULTS_DIR
    
    parser = argparse.ArgumentParser(description='Run live trading with enhanced model support.')
    
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
    
    # Output options - use the actual paths from config as defaults
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help=f'Directory for data files (default: {DATA_DIR})')
    parser.add_argument('--models-dir', type=str, default=str(MODELS_DIR),
                        help=f'Directory for model files (default: {MODELS_DIR})')
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR),
                        help=f'Directory for results (default: {RESULTS_DIR})')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    
    return parser.parse_args()

def run_live_trading(args):
    """Run live trading with enhanced model support."""
    logger = logging.getLogger("live_trading")
    
    try:
        # Print system information
        print_system_info()
        
        # Get directory paths from args or config
        models_dir = args.models_dir
        data_dir = args.data_dir
        results_dir = args.results_dir
        
        # Validate directories
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            return 1
        
        # Check if there are any model directories for the requested symbols
        model_dirs_found = False
        for symbol in args.symbols:
            for timeframe in args.timeframes:
                model_dir = os.path.join(models_dir, f"{symbol}_{timeframe}")
                if os.path.exists(model_dir):
                    model_dirs_found = True
                    break
            if model_dirs_found:
                break
        
        if not model_dirs_found:
            logger.error(f"No model directories found for any of the requested symbols/timeframes")
            logger.info(f"Available directories in {models_dir}:")
            for item in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, item)):
                    logger.info(f"  - {item}")
            return 1
        
        # Create exchange connection
        logger.info(f"Connecting to {args.exchange} {'testnet' if args.testnet else 'live'}")
        exchange = BinanceConnection(
            api_key=args.api_key,
            api_secret=args.api_secret,
            testnet=args.testnet
        )
        
        # Create model manager
        model_manager = ModelManager(base_dir=models_dir)
        
        # Create live trader
        trader = LiveTrader(
            exchange_connection=exchange,
            model_manager=model_manager,
            symbols=args.symbols,
            timeframes=args.timeframes,
            config_path=args.config,
            data_dir=data_dir,
            results_dir=results_dir,
            log_level=args.log_level
        )
        
        # Use enhanced model loading function
        if not trader.load_trading_models(args.symbols, args.timeframes):
            logger.error("Failed to load all required models")
            return 1
        
        # Initialize strategies
        if not trader.initialize_strategies():
            logger.error("Failed to initialize all strategies")
            return 1
        
        # Define event handlers
        def on_trade_executed(symbol, side, quantity, price):
            logger.info(f"Trade executed: {symbol} {side} {quantity} @ {price:.8f}")
        
        def on_position_changed(symbol):
            position = trader.positions.get(symbol, {'total_base': 0, 'base_asset': ''})
            base_asset = position.get('base_asset', '')
            logger.info(f"Position changed: {symbol} {position['total_base']:.8f} {base_asset}")
        
        def on_error(source, error_msg):
            logger.error(f"Error in {source}: {error_msg}")
        
        # Set event handlers
        trader.on_trade_executed = on_trade_executed
        trader.on_position_changed = on_position_changed
        trader.on_error = on_error
        
        # Start trading
        logger.info("Starting live trading")
        trader.start()
        
        # Save report when done
        trader.save_report()
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Live trading interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error in live trading: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    sys.exit(run_live_trading(args))