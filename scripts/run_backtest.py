#!/usr/bin/env python3
"""
Main execution script for running backtests.

Usage:
    python scripts/run_backtest.py --symbol AAPL --start 2023-01-01 --end 2023-12-31
"""

# --- SANITY CHECK ---
print("DEBUG: Script is starting...") 

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd  # <--- NEW: Required for the fix

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

try:
    from src.core.config import Config, set_config
    from src.data.data_loader import DataLoader
    from src.strategies.golden_cross import GoldenCrossStrategy, EnhancedGoldenCross
    from src.backtesting.backtester import Backtester
    from src.analytics.analyzers import PerformanceAnalyzer
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import project modules. {e}")
    sys.exit(1)


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level=log_level
    )
    logger.add(
        "logs/backtest_{time}.log",
        rotation="1 day",
        retention="30 days",
        level=log_level
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run quantitative backtesting engine")
    
    # Data parameters
    parser.add_argument("--symbol", type=str, default="AAPL", help="Trading symbol (default: AAPL)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1d", choices=["1d", "1h", "4h", "1w"], help="Data timeframe")
    
    # Strategy parameters
    parser.add_argument("--strategy", type=str, default="golden_cross", choices=["golden_cross", "enhanced_golden_cross"], help="Strategy to backtest")
    parser.add_argument("--fast-period", type=int, default=50, help="Fast moving average period")
    parser.add_argument("--slow-period", type=int, default=200, help="Slow moving average period")
    
    # Trading parameters
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate (0.001 = 0.1%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage rate")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--save-plots", action="store_true", help="Generate and save plots")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    return parser.parse_args()


def initialize_strategy(strategy_name: str, parameters: dict):
    """Initialize strategy based on name."""
    strategies = {
        "golden_cross": GoldenCrossStrategy,
        "enhanced_golden_cross": EnhancedGoldenCross
    }
    
    strategy_class = strategies.get(strategy_name)
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_class(parameters=parameters)


def main():
    """Main execution function."""
    print("DEBUG: Entering main function...")
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("=" * 80)
    logger.info("QUANTITATIVE BACKTESTING ENGINE")
    logger.info("=" * 80)
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
        config.trading.initial_capital = args.capital
        config.trading.commission = args.commission
        config.trading.slippage = args.slippage
    
    set_config(config)
    
    # -------------------------------------------------------------------------
    # Warm-up Calculation
    # -------------------------------------------------------------------------
    logger.info("Loading market data with warm-up buffer...")
    
    warmup_days = 365 
    
    try:
        original_start_date = datetime.strptime(args.start, "%Y-%m-%d")
        buffer_start_date = original_start_date - timedelta(days=warmup_days)
        buffer_start_str = buffer_start_date.strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Date format error: {e}")
        return 1

    logger.info(f"Requested Start: {args.start}")
    logger.info(f"Fetch Start (with buffer): {buffer_start_str}")

    data_loader = DataLoader()
    
    try:
        full_data = data_loader.fetch_data(
            symbol=args.symbol,
            start_date=buffer_start_str,
            end_date=args.end,
            timeframe=args.timeframe
        )
        logger.info(f"Loaded {len(full_data)} bars (including buffer)")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Initialize strategy
    strategy_params = {
        'fast_period': args.fast_period,
        'slow_period': args.slow_period
    }
    
    logger.info(f"Initializing {args.strategy} strategy...")
    strategy = initialize_strategy(args.strategy, strategy_params)
    
    # -------------------------------------------------------------------------
    # Strategy Execution
    # -------------------------------------------------------------------------
    logger.info("Generating signals on full history...")
    data_with_signals = strategy.run(full_data)
    
    # -------------------------------------------------------------------------
    # FIX: Robust Slicing Logic (Prevents crash if date is missing)
    # -------------------------------------------------------------------------
    try:
        # Ensure index is a proper DatetimeIndex for comparison
        if not isinstance(data_with_signals.index, pd.DatetimeIndex):
            # Handle timezone-aware strings by using utc=True then converting
            data_with_signals.index = pd.to_datetime(data_with_signals.index, utc=True)
        
        # Convert args to Timestamps with UTC timezone to match the data
        start_dt = pd.Timestamp(args.start, tz='UTC')
        end_dt = pd.Timestamp(args.end, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include full end day
        
        # Use Boolean Mask (This is the crash fix!)
        mask = (data_with_signals.index >= start_dt) & (data_with_signals.index <= end_dt)
        backtest_data = data_with_signals.loc[mask].copy()
        
    except Exception as e:
        logger.error(f"Slicing error: {e}")
        return 1
    
    if len(backtest_data) == 0:
        logger.error("Error: No data remaining after slicing! Check your dates.")
        return 1
        
    logger.info(f"Backtesting simulation period: {args.start} to {args.end}")
    logger.info(f"Trading bars: {len(backtest_data)}")

    # -------------------------------------------------------------------------
    # Simulation Loop
    # -------------------------------------------------------------------------
    backtester = Backtester(
        strategy=strategy, 
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage
    )
    
    logger.info("Running simulation loop...")

    # Initialize tracking columns
    backtest_data['cash'] = float(backtester.initial_capital)
    backtest_data['position_size'] = 0.0
    backtest_data['portfolio_value'] = float(backtester.initial_capital)
    backtest_data['returns'] = 0.0
    backtest_data['cumulative_returns'] = 0.0
    
    # Run simulation
    results = backtester._simulate_trading(backtest_data)
    metrics = backtester._calculate_metrics(results)
    
    # Print summary
    backtester._print_summary(metrics)
    
    # Analytics
    logger.info("Calculating advanced metrics...")
    analyzer = PerformanceAnalyzer(results)
    
    trades_df = backtester.get_trades_dataframe()
    if len(trades_df) > 0:
        kelly = analyzer.calculate_kelly_criterion(trades_df)
        metrics['kelly_criterion'] = kelly
        logger.info(f"Kelly Criterion: {kelly:.2%}")
    
    # Risk assessment
    var_95 = analyzer.calculate_value_at_risk(0.95)
    cvar_95 = analyzer.calculate_conditional_var(0.95)
    logger.info(f"VaR (95%): {var_95:.2%}")
    logger.info(f"CVaR (95%): {cvar_95:.2%}")
    
    # Export results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = output_path / f"backtest_{args.symbol}_{timestamp}.csv"
    results.to_csv(results_file)
    logger.info(f"Saved results to {results_file}")
    
    if len(trades_df) > 0:
        trades_file = output_path / f"trades_{args.symbol}_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved trades to {trades_file}")
    
    metrics_file = output_path / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Final logs
    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)