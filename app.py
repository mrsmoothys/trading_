"""
Web dashboard for the AI trading bot.
Provides a user interface for monitoring and controlling the trading system.
"""
import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LOGS_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR
)
from data_processor import load_data
from model import ModelManager
from live_trading import BinanceConnection, LiveTrader
from utils import setup_logging, find_all_datasets, find_common_timeframes

# Initialize Flask app
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(__file__), 'static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Setup logging
setup_logging()
logger = logging.getLogger("web_dashboard")

# Global state
trader = None
exchange = None
model_manager = None
datasets = None
backtest_results = {}

@app.route('/')
def index():
    """Render dashboard homepage."""
    return render_template('index.html', 
                          title='AI Trading Bot Dashboard',
                          trader_status=get_trader_status())

@app.route('/status')
def status():
    """Get current trader status."""
    return jsonify(get_trader_status())

def get_trader_status():
    """Get current trader status as a dictionary."""
    if trader is None:
        return {
            'initialized': False,
            'running': False,
            'paused': False,
            'positions': {},
            'active_orders': {},
            'performance': {
                'total_pnl': 0,
                'total_trades': 0,
                'overall_win_rate': 0
            }
        }
    
    # Get status from trader
    status = trader.get_status()
    
    # Get performance report
    performance = trader.get_performance_report()
    
    # Return combined status
    return {
        'initialized': True,
        'running': status['running'],
        'paused': status['paused'],
        'positions': status['positions'],
        'active_orders': status['active_orders'],
        'performance': {
            'total_pnl': performance['total_pnl'],
            'total_trades': performance['total_trades'],
            'overall_win_rate': performance['overall_win_rate']
        }
    }

@app.route('/initialize', methods=['POST'])
def initialize_trader():
    """Initialize the live trader."""
    global trader, exchange, model_manager
    
    try:
        # Get parameters from request
        data = request.json
        api_key = data.get('api_key')
        api_secret = data.get('api_secret')
        testnet = data.get('testnet', True)
        exchange_name = data.get('exchange', 'binance')
        symbols = data.get('symbols', [])
        timeframes = data.get('timeframes', ['1h'])
        
        # Validate required parameters
        if not api_key or not api_secret or not symbols:
            return jsonify({'success': False, 'error': 'Missing required parameters'})
        
        # Create exchange connection
        if exchange_name.lower() == 'binance':
            exchange = BinanceConnection(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
        else:
            return jsonify({'success': False, 'error': f'Unsupported exchange: {exchange_name}'})
        
        # Create model manager
        model_manager = ModelManager()
        
        # Create live trader
        trader = LiveTrader(
            exchange_connection=exchange,
            model_manager=model_manager,
            symbols=symbols,
            timeframes=timeframes,
            log_level='INFO'
        )
        
        # Load models
        success = trader.load_models()
        if not success:
            return jsonify({'success': False, 'error': 'Failed to load all models'})
        
        # Initialize strategies
        success = trader.initialize_strategies()
        if not success:
            return jsonify({'success': False, 'error': 'Failed to initialize all strategies'})
        
        # Set up event handlers
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
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Error initializing trader: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start', methods=['POST'])
def start_trader():
    """Start the live trader."""
    if trader is None:
        return jsonify({'success': False, 'error': 'Trader not initialized'})
    
    if trader.running:
        return jsonify({'success': False, 'error': 'Trader already running'})
    
    # Start trader in a thread
    def trader_thread():
        trader.start()
    
    thread = threading.Thread(target=trader_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True})

@app.route('/stop', methods=['POST'])
def stop_trader():
    """Stop the live trader."""
    if trader is None:
        return jsonify({'success': False, 'error': 'Trader not initialized'})
    
    if not trader.running:
        return jsonify({'success': False, 'error': 'Trader not running'})
    
    # Stop trader
    trader.stop()
    
    return jsonify({'success': True})

@app.route('/pause', methods=['POST'])
def pause_trader():
    """Pause the live trader."""
    if trader is None:
        return jsonify({'success': False, 'error': 'Trader not initialized'})
    
    if not trader.running:
        return jsonify({'success': False, 'error': 'Trader not running'})
    
    if trader.paused:
        return jsonify({'success': False, 'error': 'Trader already paused'})
    
    # Pause trader
    trader.pause()
    
    return jsonify({'success': True})

@app.route('/resume', methods=['POST'])
def resume_trader():
    """Resume the live trader."""
    if trader is None:
        return jsonify({'success': False, 'error': 'Trader not initialized'})
    
    if not trader.running:
        return jsonify({'success': False, 'error': 'Trader not running'})
    
    if not trader.paused:
        return jsonify({'success': False, 'error': 'Trader not paused'})
    
    # Resume trader
    trader.resume()
    
    return jsonify({'success': True})

@app.route('/performance')
def performance():
    """Render performance page."""
    return render_template('performance.html', 
                          title='Performance Dashboard',
                          trader_status=get_trader_status())

@app.route('/performance/data')
def performance_data():
    """Get performance data for charts."""
    if trader is None:
        return jsonify({
            'performance': {
                'total_pnl': 0,
                'total_trades': 0,
                'overall_win_rate': 0,
                'by_symbol': {},
                'trade_history': []
            }
        })
    
    # Get performance report
    performance = trader.get_performance_report()
    
    return jsonify({'performance': performance})

@app.route('/performance/charts')
def performance_charts():
    """Generate performance charts."""
    if trader is None:
        return jsonify({'charts': {}})
    
    performance = trader.get_performance_report()
    charts = {}
    
    # Only create charts if we have trades
    if performance['total_trades'] > 0:
        # Create PnL chart
        if performance['by_symbol']:
            symbol_pnl = [{'symbol': symbol, 'pnl': metrics['pnl']} 
                         for symbol, metrics in performance['by_symbol'].items()]
            
            fig = px.bar(symbol_pnl, x='symbol', y='pnl', title='Profit & Loss by Symbol')
            charts['pnl_by_symbol'] = fig.to_json()
        
        # Create win rate chart
        if performance['by_symbol']:
            symbol_win_rate = [{'symbol': symbol, 'win_rate': metrics['win_rate'] * 100} 
                              for symbol, metrics in performance['by_symbol'].items()]
            
            fig = px.bar(symbol_win_rate, x='symbol', y='win_rate', title='Win Rate by Symbol (%)')
            charts['win_rate_by_symbol'] = fig.to_json()
        
        # Create trade history chart
        if performance['trade_history']:
            trade_df = pd.DataFrame(performance['trade_history'])
            
            # Convert timestamp to datetime if it's a string
            if isinstance(trade_df['timestamp'].iloc[0], str):
                trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
            
            # Group by date and side
            trade_df['date'] = trade_df['timestamp'].dt.date
            trade_count = trade_df.groupby(['date', 'side']).size().reset_index(name='count')
            
            fig = px.bar(trade_count, x='date', y='count', color='side', 
                        title='Trades by Day', barmode='group')
            charts['trades_by_day'] = fig.to_json()
    
    return jsonify({'charts': charts})

@app.route('/backtest')
def backtest():
    """Render backtest page."""
    global datasets
    
    # Find available datasets if not already loaded
    if datasets is None:
        datasets = find_all_datasets()
    
    # Find common timeframes
    timeframes = find_common_timeframes(datasets)
    
    return render_template('backtest.html', 
                          title='Backtest Dashboard',
                          datasets=datasets,
                          timeframes=timeframes)

@app.route('/backtest/run', methods=['POST'])
def run_backtest():
    """Run backtest with selected parameters."""
    global backtest_results
    
    try:
        # Get parameters from request
        data = request.json
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Validate required parameters
        if not symbol or not timeframe:
            return jsonify({'success': False, 'error': 'Missing required parameters'})
        
        # Construct data path
        data_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}_data_2018_to_2025.csv")
        
        if not os.path.exists(data_path):
            return jsonify({'success': False, 'error': f'Data file not found: {data_path}'})
        
        # Run backtest script in a thread
        def backtest_thread():
            import subprocess
            
            cmd = [
                'python', 'run_single.py',
                '--data-path', data_path,
                '--visualize'
            ]
            
            if start_date:
                cmd.extend(['--start-date', start_date])
            
            if end_date:
                cmd.extend(['--end-date', end_date])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results
            if result.returncode == 0:
                # Find results file
                results_dir = os.path.join(RESULTS_DIR)
                
                # Get most recent results file
                results_files = [f for f in os.listdir(results_dir) if f.startswith('results_')]
                
                if results_files:
                    latest_file = max(results_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
                    
                    with open(os.path.join(results_dir, latest_file), 'r') as f:
                        backtest_results[f"{symbol}_{timeframe}"] = json.load(f)
            
            logger.info(f"Backtest finished with return code {result.returncode}")
        
        # Start backtest thread
        thread = threading.Thread(target=backtest_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Backtest started'})
    
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/backtest/results')
def backtest_results_endpoint():
    """Get backtest results."""
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe')
    
    if not symbol or not timeframe:
        return jsonify({'results': backtest_results})
    
    # Get specific results
    key = f"{symbol}_{timeframe}"
    
    if key in backtest_results:
        return jsonify({'results': {key: backtest_results[key]}})
    
    return jsonify({'results': {}})

@app.route('/backtest/results/charts')
def backtest_charts():
    """Generate backtest charts."""
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe')
    
    if not symbol or not timeframe:
        return jsonify({'charts': {}})
    
    # Get specific results
    key = f"{symbol}_{timeframe}"
    
    if key not in backtest_results:
        return jsonify({'charts': {}})
    
    # Generate charts
    results = backtest_results[key]
    charts = {}
    
    # Create performance chart
    if 'performance' in results:
        perf = results['performance']
        
        # Create gauge chart for total return
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = perf['total_return'],
            title = {'text': "Total Return (%)"},
            gauge = {
                'axis': {'range': [-50, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-50, 0], 'color': "lightcoral"},
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "darkgreen"}
                ]
            }
        ))
        
        charts['total_return'] = fig.to_json()
        
        # Create metrics grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Win Rate (%)", "Profit Factor", "Sharpe Ratio", "Max Drawdown (%)"),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=perf['win_rate'],
                number={'suffix': "%"}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=perf['profit_factor']
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=perf['sharpe_ratio']
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=perf['max_drawdown'],
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=400)
        charts['metrics_grid'] = fig.to_json()
    
    return jsonify({'charts': charts})

@app.route('/models')
def models_page():
    """Render models page."""
    return render_template('models.html', 
                          title='Model Management')

@app.route('/models/list')
def list_models():
    """List available models."""
    # List files in models directory
    if not os.path.exists(MODELS_DIR):
        return jsonify({'models': []})
    
    models = []
    
    for root, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith('.h5'):
                # Get relative path
                rel_path = os.path.relpath(os.path.join(root, file), MODELS_DIR)
                
                # Extract model details
                parts = rel_path.split(os.path.sep)
                
                if len(parts) >= 2:
                    symbol, timeframe = parts[0].split('_')
                    
                    # Get file size
                    file_path = os.path.join(root, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    
                    # Get file modification time
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    models.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'file': rel_path,
                        'size_mb': round(size_mb, 2),
                        'last_modified': mtime.strftime('%Y-%m-%d %H:%M:%S')
                    })
    
    return jsonify({'models': models})

@app.route('/logs')
def logs_page():
    """Render logs page."""
    return render_template('logs.html', 
                          title='Log Viewer')

@app.route('/logs/list')
def list_logs():
    """List available log files."""
    if not os.path.exists(LOGS_DIR):
        return jsonify({'logs': []})
    
    logs = []
    
    for file in os.listdir(LOGS_DIR):
        if file.endswith('.log'):
            file_path = os.path.join(LOGS_DIR, file)
            
            # Get file size
            size_kb = os.path.getsize(file_path) / 1024
            
            # Get file modification time
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            logs.append({
                'file': file,
                'size_kb': round(size_kb, 2),
                'last_modified': mtime.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return jsonify({'logs': sorted(logs, key=lambda x: x['last_modified'], reverse=True)})

@app.route('/logs/view')
def view_log():
    """View log file contents."""
    file = request.args.get('file')
    
    if not file:
        return jsonify({'content': ''})
    
    # Ensure file is within logs directory
    log_path = os.path.join(LOGS_DIR, file)
    
    if not os.path.exists(log_path) or not log_path.startswith(LOGS_DIR):
        return jsonify({'content': 'Log file not found'})
    
    # Read last 1000 lines (for performance)
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        # Get last 1000 lines
        content = ''.join(lines[-1000:])
        
        return jsonify({'content': content})
    
    except Exception as e:
        return jsonify({'content': f'Error reading log file: {str(e)}'})

@app.route('/settings')
def settings_page():
    """Render settings page."""
    return render_template('settings.html', 
                          title='Settings')

@app.route('/settings/config')
def get_config():
    """Get current configuration."""
    # Read config from file
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
    
    config = {}
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                # Parse variable assignment
                if '=' in line:
                    parts = line.split('=', 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Skip imports and other non-config lines
                    if key.startswith('import') or key.startswith('from'):
                        continue
                    
                    # Clean up value
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    else:
                        try:
                            # Try to evaluate as Python literal
                            value = eval(value)
                        except:
                            pass
                    
                    config[key] = value
    
    return jsonify({'config': config})

if __name__ == '__main__':
    # Create required directories
    for directory in [LOGS_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Create static and templates directories
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create basic HTML templates if they don't exist
    index_template = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_template):
        with open(index_template, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.1/plotly.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">AI Trading Bot</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/performance">Performance</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/backtest">Backtest</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/models">Models</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/logs">Logs</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/settings">Settings</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <h1>Trading Dashboard</h1>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        Initialize Trader
                    </div>
                    <div class="card-body">
                        <form id="initialize-form">
                            <div class="mb-3">
                                <label for="exchange" class="form-label">Exchange</label>
                                <select class="form-select" id="exchange">
                                    <option value="binance">Binance</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label for="api-key" class="form-label">API Key</label>
                                <input type="password" class="form-control" id="api-key" required>
                            </div>
                            <div class="mb-3">
                                <label for="api-secret" class="form-label">API Secret</label>
                                <input type="password" class="form-control" id="api-secret" required>
                            </div>
                            <div class="mb-3 form-check">
                                <input type="checkbox" class="form-check-input" id="testnet" checked>
                                <label class="form-check-label" for="testnet">Use Testnet/Sandbox</label>
                            </div>
                            <div class="mb-3">
                                <label for="symbols" class="form-label">Symbols (comma-separated)</label>
                                <input type="text" class="form-control" id="symbols" value="BTCUSDT,ETHUSDT" required>
                            </div>
                            <div class="mb-3">
                                <label for="timeframes" class="form-label">Timeframes (comma-separated)</label>
                                <input type="text" class="form-control" id="timeframes" value="1h,4h" required>
                            </div>
                            <button type="submit" class="btn btn-primary">Initialize</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        Trader Status
                    </div>
                    <div class="card-body">
                        <div id="status-display">
                            <p><strong>Status:</strong> <span id="trader-status">Not initialized</span></p>
                            <p><strong>Running:</strong> <span id="trader-running">No</span></p>
                            <p><strong>Paused:</strong> <span id="trader-paused">No</span></p>
                            <p><strong>Total PnL:</strong> <span id="trader-pnl">$0</span></p>
                            <p><strong>Total Trades:</strong> <span id="trader-trades">0</span></p>
                            <p><strong>Win Rate:</strong> <span id="trader-win-rate">0%</span></p>
                        </div>
                        
                        <div class="btn-group mt-3">
                            <button id="start-btn" class="btn btn-success" disabled>Start</button>
                            <button id="stop-btn" class="btn btn-danger" disabled>Stop</button>
                            <button id="pause-btn" class="btn btn-warning" disabled>Pause</button>
                            <button id="resume-btn" class="btn btn-info" disabled>Resume</button>
                        </div>
                    </div>
                </div>
                
                <div class="card mt-4">
                    <div class="card-header">
                        Open Positions
                    </div>
                    <div class="card-body">
                        <div id="positions-table">
                            <p>No open positions</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Update status every 5 seconds
        function updateStatus() {
            $.getJSON('/status', function(data) {
                $('#trader-status').text(data.initialized ? 'Initialized' : 'Not initialized');
                $('#trader-running').text(data.running ? 'Yes' : 'No');
                $('#trader-paused').text(data.paused ? 'Yes' : 'No');
                $('#trader-pnl').text('$' + data.performance.total_pnl.toFixed(2));
                $('#trader-trades').text(data.performance.total_trades);
                $('#trader-win-rate').text((data.performance.overall_win_rate * 100).toFixed(2) + '%');
                
                // Update buttons
                $('#start-btn').prop('disabled', !data.initialized || data.running);
                $('#stop-btn').prop('disabled', !data.initialized || !data.running);
                $('#pause-btn').prop('disabled', !data.initialized || !data.running || data.paused);
                $('#resume-btn').prop('disabled', !data.initialized || !data.running || !data.paused);
                
                // Update positions
                let positionsHtml = '';
                if (Object.keys(data.positions).length > 0) {
                    positionsHtml = '<table class="table table-sm"><thead><tr><th>Symbol</th><th>Size</th><th>Value</th></tr></thead><tbody>';
                    for (const symbol in data.positions) {
                        const position = data.positions[symbol];
                        positionsHtml += `<tr>
                            <td>${symbol}</td>
                            <td>${position.total_base.toFixed(8)}</td>
                            <td>$${position.position_value.toFixed(2)}</td>
                        </tr>`;
                    }
                    positionsHtml += '</tbody></table>';
                } else {
                    positionsHtml = '<p>No open positions</p>';
                }
                $('#positions-table').html(positionsHtml);
            });
        }
        
        // Update status initially and then every 5 seconds
        updateStatus();
        setInterval(updateStatus, 5000);
        
        // Initialize form submission
        $('#initialize-form').submit(function(e) {
            e.preventDefault();
            
            const apiKey = $('#api-key').val();
            const apiSecret = $('#api-secret').val();
            const testnet = $('#testnet').is(':checked');
            const exchange = $('#exchange').val();
            const symbols = $('#symbols').val().split(',').map(s => s.trim());
            const timeframes = $('#timeframes').val().split(',').map(s => s.trim());
            
            $.ajax({
                url: '/initialize',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    api_key: apiKey,
                    api_secret: apiSecret,
                    testnet: testnet,
                    exchange: exchange,
                    symbols: symbols,
                    timeframes: timeframes
                }),
                success: function(response) {
                    if (response.success) {
                        alert('Trader initialized successfully');
                        updateStatus();
                    } else {
                        alert('Error initializing trader: ' + response.error);
                    }
                },
                error: function() {
                    alert('Error initializing trader');
                }
            });
        });
        
        // Button handlers
        $('#start-btn').click(function() {
            $.post('/start', function(response) {
                if (response.success) {
                    alert('Trader started successfully');
                    updateStatus();
                } else {
                    alert('Error starting trader: ' + response.error);
                }
            });
        });
        
        $('#stop-btn').click(function() {
            $.post('/stop', function(response) {
                if (response.success) {
                    alert('Trader stopped successfully');
                    updateStatus();
                } else {
                    alert('Error stopping trader: ' + response.error);
                }
            });
        });
        
        $('#pause-btn').click(function() {
            $.post('/pause', function(response) {
                if (response.success) {
                    alert('Trader paused successfully');
                    updateStatus();
                } else {
                    alert('Error pausing trader: ' + response.error);
                }
            });
        });
        
        $('#resume-btn').click(function() {
            $.post('/resume', function(response) {
                if (response.success) {
                    alert('Trader resumed successfully');
                    updateStatus();
                } else {
                    alert('Error resuming trader: ' + response.error);
                }
            });
        });
    </script>
</body>
</html>
            """)
    
    # Create performance template
    performance_template = os.path.join(templates_dir, 'performance.html')
    if not os.path.exists(performance_template):
        with open(performance_template, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.1/plotly.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">AI Trading Bot</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/performance">Performance</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/backtest">Backtest</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/models">Models</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/logs">Logs</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/settings">Settings</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <h1>Performance Dashboard</h1>
        
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        Performance Summary
                    </div>
                    <div class="card-body">
                        <div id="performance-summary">
                            <p><strong>Total PnL:</strong> <span id="total-pnl">$0</span></p>
                            <p><strong>Total Trades:</strong> <span id="total-trades">0</span></p>
                            <p><strong>Win Rate:</strong> <span id="win-rate">0%</span></p>
                        </div>
                    </div>
                </div>
                
                <div class="card mt-4">
                    <div class="card-header">
                        Trade History
                    </div>
                    <div class="card-body">
                        <div id="trade-history">
                            <p>No trades yet</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        Performance Charts
                    </div>
                    <div class="card-body">
                        <div id="pnl-chart" style="height: 300px;"></div>
                        <div id="win-rate-chart" style="height: 300px; margin-top: 20px;"></div>
                        <div id="trades-chart" style="height: 300px; margin-top: 20px;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Update performance data
        function updatePerformance() {
            $.getJSON('/performance/data', function(data) {
                const performance = data.performance;
                
                // Update summary
                $('#total-pnl').text('$' + performance.total_pnl.toFixed(2));
                $('#total-trades').text(performance.total_trades);
                $('#win-rate').text((performance.overall_win_rate * 100).toFixed(2) + '%');
                
                // Update trade history
                let tradeHtml = '';
                if (performance.trade_history && performance.trade_history.length > 0) {
                    tradeHtml = '<table class="table table-sm"><thead><tr><th>Symbol</th><th>Side</th><th>Price</th><th>Time</th></tr></thead><tbody>';
                    
                    // Show last 10 trades
                    const trades = performance.trade_history.slice(-10).reverse();
                    
                    for (const trade of trades) {
                        tradeHtml += `<tr>
                            <td>${trade.symbol}</td>
                            <td>${trade.side}</td>
                            <td>${parseFloat(trade.price).toFixed(4)}</td>
                            <td>${new Date(trade.timestamp).toLocaleString()}</td>
                        </tr>`;
                    }
                    
                    tradeHtml += '</tbody></table>';
                } else {
                    tradeHtml = '<p>No trades yet</p>';
                }
                
                $('#trade-history').html(tradeHtml);
                
                // Update charts
                $.getJSON('/performance/charts', function(chartData) {
                    if (chartData.charts.pnl_by_symbol) {
                        Plotly.newPlot('pnl-chart', JSON.parse(chartData.charts.pnl_by_symbol).data, JSON.parse(chartData.charts.pnl_by_symbol).layout);
                    }
                    
                    if (chartData.charts.win_rate_by_symbol) {
                        Plotly.newPlot('win-rate-chart', JSON.parse(chartData.charts.win_rate_by_symbol).data, JSON.parse(chartData.charts.win_rate_by_symbol).layout);
                    }
                    
                    if (chartData.charts.trades_by_day) {
                        Plotly.newPlot('trades-chart', JSON.parse(chartData.charts.trades_by_day).data, JSON.parse(chartData.charts.trades_by_day).layout);
                    }
                });
            });
        }
        
        // Update performance initially and then every 10 seconds
        updatePerformance();
        setInterval(updatePerformance, 10000);
    </script>
</body>
</html>
            """)
    
    # Create backtest template
    backtest_template = os.path.join(templates_dir, 'backtest.html')
    if not os.path.exists(backtest_template):
        with open(backtest_template, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.1/plotly.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">AI Trading Bot</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/performance">Performance</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="/backtest">Backtest</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/models">Models</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/logs">Logs</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/settings">Settings</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <h1>Backtest Dashboard</h1>
        
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        Run Backtest
                    </div>
                    <div class="card-body">
                        <form id="backtest-form">
                            <div class="mb-3">
                                <label for="symbol" class="form-label">Symbol</label>
                                <select class="form-select" id="symbol" required>
                                    <option value="">Select Symbol</option>
                                    {% for symbol in datasets %}
                                        <option value="{{ symbol }}">{{ symbol }}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <div class="mb-3">
                                <label for="timeframe" class="form-label">Timeframe</label>
                                <select class="form-select" id="timeframe" required>
                                    <option value="">Select Timeframe</option>
                                    {% for timeframe in timeframes %}
                                        <option value="{{ timeframe }}">{{ timeframe }}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <div class="mb-3">
                                <label for="start-date" class="form-label">Start Date</label>
                                <input type="date" class="form-control" id="start-date">
                            </div>
                            <div class="mb-3">
                                <label for="end-date" class="form-label">End Date</label>
                                <input type="date" class="form-control" id="end-date">
                            </div>
                            <button type="submit" class="btn btn-primary">Run Backtest</button>
                        </form>
                    </div>
                </div>
                
                <div class="card mt-4">
                    <div class="card-header">
                        Backtest Results
                    </div>
                    <div class="card-body">
                        <div id="backtest-summary">
                            <p>No backtest results yet. Run a backtest to see results.</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        Backtest Charts
                    </div>
                    <div class="card-body">
                        <div id="backtest-chart-container">
                            <div id="total-return-chart" style="height: 300px;"></div>
                            <div id="metrics-chart" style="height: 400px; margin-top: 20px;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Update symbol dropdown based on selected timeframe
        $('#timeframe').change(function() {
            const timeframe = $(this).val();
            
            $.getJSON('/backtest', function(data) {
                const symbols = data.datasets;
                
                // Filter symbols that have the selected timeframe
                const filteredSymbols = Object.keys(symbols).filter(symbol => 
                    symbols[symbol].includes(timeframe)
                );
                
                // Update symbol dropdown
                const symbolDropdown = $('#symbol');
                symbolDropdown.empty();
                symbolDropdown.append('<option value="">Select Symbol</option>');
                
                filteredSymbols.forEach(symbol => {
                    symbolDropdown.append(`<option value="${symbol}">${symbol}</option>`);
                });
            });
        });
        
        // Run backtest form submission
        $('#backtest-form').submit(function(e) {
            e.preventDefault();
            
            const symbol = $('#symbol').val();
            const timeframe = $('#timeframe').val();
            const startDate = $('#start-date').val();
            const endDate = $('#end-date').val();
            
            // Show loading message
            $('#backtest-summary').html('<p>Running backtest, please wait...</p>');
            
            $.ajax({
                url: '/backtest/run',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    symbol: symbol,
                    timeframe: timeframe,
                    start_date: startDate,
                    end_date: endDate
                }),
                success: function(response) {
                    if (response.success) {
                        $('#backtest-summary').html('<p>Backtest started. Results will appear when complete.</p>');
                        
                        // Check for results every 5 seconds
                        checkBacktestResults(symbol, timeframe);
                    } else {
                        $('#backtest-summary').html(`<p class="text-danger">Error: ${response.error}</p>`);
                    }
                },
                error: function() {
                    $('#backtest-summary').html('<p class="text-danger">Error running backtest</p>');
                }
            });
        });
        
        // Check backtest results
        function checkBacktestResults(symbol, timeframe) {
            const checkInterval = setInterval(function() {
                $.getJSON(`/backtest/results?symbol=${symbol}&timeframe=${timeframe}`, function(data) {
                    const key = `${symbol}_${timeframe}`;
                    
                    if (data.results[key]) {
                        clearInterval(checkInterval);
                        updateBacktestResults(data.results[key], symbol, timeframe);
                    }
                });
            }, 5000);
        }
        
        // Update backtest results
        function updateBacktestResults(results, symbol, timeframe) {
            // Update summary
            let summaryHtml = '<dl class="row">';
            
            if (results.performance) {
                const perf = results.performance;
                
                summaryHtml += `
                    <dt class="col-sm-6">Symbol</dt>
                    <dd class="col-sm-6">${symbol}</dd>
                    
                    <dt class="col-sm-6">Timeframe</dt>
                    <dd class="col-sm-6">${timeframe}</dd>
                    
                    <dt class="col-sm-6">Total Return</dt>
                    <dd class="col-sm-6">${perf.total_return.toFixed(2)}%</dd>
                    
                    <dt class="col-sm-6">Number of Trades</dt>
                    <dd class="col-sm-6">${perf.num_trades}</dd>
                    
                    <dt class="col-sm-6">Win Rate</dt>
                    <dd class="col-sm-6">${perf.win_rate.toFixed(2)}%</dd>
                    
                    <dt class="col-sm-6">Profit Factor</dt>
                    <dd class="col-sm-6">${perf.profit_factor.toFixed(2)}</dd>
                    
                    <dt class="col-sm-6">Sharpe Ratio</dt>
                    <dd class="col-sm-6">${perf.sharpe_ratio.toFixed(2)}</dd>
                    
                    <dt class="col-sm-6">Max Drawdown</dt>
                    <dd class="col-sm-6">${perf.max_drawdown.toFixed(2)}%</dd>
                `;
            }
            
            summaryHtml += '</dl>';
            $('#backtest-summary').html(summaryHtml);
            
            // Update charts
            $.getJSON(`/backtest/results/charts?symbol=${symbol}&timeframe=${timeframe}`, function(chartData) {
                if (chartData.charts.total_return) {
                    Plotly.newPlot('total-return-chart', JSON.parse(chartData.charts.total_return).data, JSON.parse(chartData.charts.total_return).layout);
                }
                
                if (chartData.charts.metrics_grid) {
                    Plotly.newPlot('metrics-chart', JSON.parse(chartData.charts.metrics_grid).data, JSON.parse(chartData.charts.metrics_grid).layout);
                }
            });
        }
    </script>
</body>
</html>
            """)
    
    # Create CSS directory and style.css
    css_dir = os.path.join(static_dir, 'css')
    os.makedirs(css_dir, exist_ok=True)
    
    style_css = os.path.join(css_dir, 'style.css')
    if not os.path.exists(style_css):
        with open(style_css, 'w') as f:
            f.write("""
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}

.navbar {
    margin-bottom: 20px;
}

.card {
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.card-header {
    background-color: #f8f9fa;
    font-weight: bold;
}

.btn-group .btn {
    margin-right: 5px;
}

#positions-table table {
    font-size: 0.9rem;
}
            """)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)