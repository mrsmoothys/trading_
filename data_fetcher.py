"""
Data fetcher module for downloading historical cryptocurrency data from exchanges.
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import concurrent.futures

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR
from utils import setup_logging

class DataFetcher:
    """
    Base class for downloading historical cryptocurrency data.
    """
    
    def __init__(
        self,
        exchange: str,
        data_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Initialize the data fetcher.
        
        Args:
            exchange: Exchange name
            data_dir: Directory to save data files
            api_key: API key for authentication
            api_secret: API secret for authentication
        """
        self.exchange = exchange
        self.data_dir = data_dir or DATA_DIR
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"data_fetcher.{exchange}")
    
    def fetch_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        save_to_file: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_to_file: Whether to save the data to a file
            
        Returns:
            DataFrame with historical data
        """
        raise NotImplementedError("Derived classes must implement fetch_data")
    
    def fetch_symbols(self) -> List[str]:
        """
        Fetch available trading symbols.
        
        Returns:
            List of available symbols
        """
        raise NotImplementedError("Derived classes must implement fetch_symbols")
    
    def fetch_timeframes(self) -> List[str]:
        """
        Fetch available timeframes.
        
        Returns:
            List of available timeframes
        """
        raise NotImplementedError("Derived classes must implement fetch_timeframes")
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """
        Save data to CSV file.
        
        Args:
            df: DataFrame with historical data
            symbol: Trading symbol
            timeframe: Trading timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            Path to saved file
        """
        # Extract just the years for the filename
        start_year = datetime.strptime(start_date, '%Y-%m-%d').year
        end_year = datetime.strptime(end_date, '%Y-%m-%d').year
        
        # Create filename
        filename = f"{symbol}_{timeframe}_data_{start_year}_to_{end_year}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=True)
        
        self.logger.info(f"Data saved to {filepath}")
        return filepath

class BinanceDataFetcher(DataFetcher):
    """
    Data fetcher for Binance exchange.
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False
    ):
        """
        Initialize the Binance data fetcher.
        
        Args:
            data_dir: Directory to save data files
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet
        """
        super().__init__("binance", data_dir, api_key, api_secret)
        
        self.testnet = testnet
        
        # Set base URL
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api"
        
        # Rate limiting parameters
        self.rate_limit = 1200  # requests per minute
        self.request_weight = 1  # weight per request
        self.request_timestamps = []
    
    def _check_rate_limit(self) -> None:
        """
        Check rate limit and wait if necessary.
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
        
        # Check if rate limit is reached
        if len(self.request_timestamps) >= self.rate_limit / self.request_weight:
            # Wait until oldest timestamp is more than 1 minute old
            wait_time = 60 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                self.logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Add current timestamp
        self.request_timestamps.append(current_time)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """
        Make API request with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Response data
        """
        # Check rate limit
        self._check_rate_limit()
        
        # Prepare URL
        url = f"{self.base_url}{endpoint}"
        
        # Make request
        response = requests.get(url, params=params)
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"API request failed: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        return response.json()
    
    def fetch_symbols(self) -> List[str]:
        """
        Fetch available trading symbols.
        
        Returns:
            List of available USDT trading pairs
        """
        # Get exchange info
        exchange_info = self._make_request("/v3/exchangeInfo")
        
        # Filter USDT trading pairs
        symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] 
                  if symbol['quoteAsset'] == 'USDT' and symbol['status'] == 'TRADING']
        
        self.logger.info(f"Found {len(symbols)} USDT trading pairs")
        return symbols
    
    def fetch_timeframes(self) -> List[str]:
        """
        Fetch available timeframes.
        
        Returns:
            List of available timeframes
        """
        # Binance supports these intervals
        timeframes = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]
        
        return timeframes
    
    def fetch_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        save_to_file: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save_to_file: Whether to save the data to a file
            
        Returns:
            DataFrame with historical data
        """
        # Validate timeframe
        valid_timeframes = self.fetch_timeframes()
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Valid timeframes: {valid_timeframes}")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Initialize data list
        all_candles = []
        
        # Calculate number of days to fetch
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_to_fetch = (end_dt - start_dt).days + 1
        
        # Set up progress bar
        self.logger.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}")
        pbar = tqdm(total=days_to_fetch, desc=f"{symbol} {timeframe}")
        
        # Fetch data in chunks
        chunk_start_ts = start_ts
        while chunk_start_ts < end_ts:
            # Binance limits 1000 candles per request
            # Calculate maximum chunk duration based on timeframe
            if timeframe.endswith('m'):
                minutes = int(timeframe[:-1])
                chunk_size = 1000 * minutes * 60 * 1000  # in milliseconds
            elif timeframe.endswith('h'):
                hours = int(timeframe[:-1])
                chunk_size = 1000 * hours * 60 * 60 * 1000
            elif timeframe.endswith('d'):
                days = int(timeframe[:-1])
                chunk_size = 1000 * days * 24 * 60 * 60 * 1000
            elif timeframe == '1w':
                chunk_size = 1000 * 7 * 24 * 60 * 60 * 1000
            elif timeframe == '1M':
                chunk_size = 1000 * 30 * 24 * 60 * 60 * 1000
            else:
                raise ValueError(f"Invalid timeframe format: {timeframe}")
            
            # Calculate chunk end
            chunk_end_ts = min(chunk_start_ts + chunk_size, end_ts)
            
            # Prepare parameters
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'startTime': chunk_start_ts,
                'endTime': chunk_end_ts,
                'limit': 1000
            }
            
            try:
                # Make request
                candles = self._make_request("/v3/klines", params)
                
                # Add to list
                all_candles.extend(candles)
                
                # Update progress
                days_fetched = (datetime.fromtimestamp(chunk_end_ts / 1000) - start_dt).days + 1
                pbar.update(min(days_fetched, days_to_fetch) - pbar.n)
                
                # Update chunk start
                if len(candles) > 0:
                    # Use last candle timestamp + 1 ms as next start
                    chunk_start_ts = candles[-1][0] + 1
                else:
                    # If no candles returned, advance by chunk size
                    chunk_start_ts = chunk_end_ts + 1
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Error fetching {symbol} {timeframe} data: {e}")
                # Wait longer on error
                time.sleep(2)
                # Try to continue from where we left off
                chunk_start_ts = chunk_end_ts + 1
        
        pbar.close()
        
        # Check if we got any data
        if not all_candles:
            self.logger.warning(f"No data fetched for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        
        df = pd.DataFrame(all_candles, columns=columns)
        
        # Convert timestamp columns to datetime
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        
        # Convert numeric columns
        numeric_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume'
        ]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Set Open time as index
        df.set_index('Open time', inplace=True)
        
        # Sort by index
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        self.logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        
        # Save to file if requested
        if save_to_file:
            self.save_to_csv(df, symbol, timeframe, start_date, end_date)
        
        return df

def fetch_multi_symbols(
    symbols: List[str],
    timeframes: List[str],
    start_date: str,
    end_date: str,
    data_dir: Optional[str] = None,
    max_workers: int = 4
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch data for multiple symbols and timeframes in parallel.
    
    Args:
        symbols: List of symbols to fetch
        timeframes: List of timeframes to fetch
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_dir: Directory to save data files
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary of DataFrames by symbol and timeframe
    """
    # Set up logger
    logger = logging.getLogger("data_fetcher")
    
    # Create data fetcher
    fetcher = BinanceDataFetcher(data_dir=data_dir)
    
    # Initialize results dictionary
    results = {}
    
    # Create a list of tasks
    tasks = []
    for symbol in symbols:
        results[symbol] = {}
        for timeframe in timeframes:
            tasks.append((symbol, timeframe))
    
    # Define worker function
    def fetch_task(args):
        symbol, timeframe = args
        try:
            df = fetcher.fetch_data(symbol, timeframe, start_date, end_date)
            return symbol, timeframe, df
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return symbol, timeframe, None
    
    # Execute tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_task, task) for task in tasks]
        
        # Create progress bar
        pbar = tqdm(total=len(tasks), desc="Fetching data")
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            symbol, timeframe, df = future.result()
            if df is not None:
                results[symbol][timeframe] = df
            pbar.update(1)
        
        pbar.close()
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fetch historical cryptocurrency data.')
    
    # Data parameters
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Symbols to fetch (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1h'],
                        help='Timeframes to fetch (default: 1h)')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help=f'End date (YYYY-MM-DD, default: {datetime.now().strftime("%Y-%m-%d")})')
    
    # Output options
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Directory to save data files')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    
    # Fetch options
    parser.add_argument('--workers', type=int, default=4,
                        help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--list-symbols', action='store_true',
                        help='List available symbols and exit')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("data_fetcher")
    
    try:
        # Create data fetcher
        fetcher = BinanceDataFetcher(data_dir=args.data_dir)
        
        # List symbols if requested
        if args.list_symbols:
            symbols = fetcher.fetch_symbols()
            print(f"Available symbols ({len(symbols)}):")
            for i, symbol in enumerate(sorted(symbols)):
                print(f"{symbol:<10}", end="\t")
                if (i + 1) % 8 == 0:
                    print()
            print()
            return 0
        
        # Validate symbols
        if not args.symbols:
            logger.error("No symbols specified")
            return 1
        
        # Validate timeframes
        valid_timeframes = fetcher.fetch_timeframes()
        for timeframe in args.timeframes:
            if timeframe not in valid_timeframes:
                logger.error(f"Invalid timeframe: {timeframe}")
                return 1
        
        # Validate dates
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            
            if start_date > end_date:
                logger.error("Start date cannot be after end date")
                return 1
            
            if end_date > datetime.now():
                logger.warning("End date is in the future, using current date instead")
                end_date = datetime.now()
                args.end_date = end_date.strftime('%Y-%m-%d')
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            return 1
        
        # Fetch data
        fetch_multi_symbols(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            data_dir=args.data_dir,
            max_workers=args.workers
        )
        
        logger.info("Data fetching completed")
        return 0
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())