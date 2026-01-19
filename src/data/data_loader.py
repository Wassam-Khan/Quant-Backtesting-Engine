"""
Data Loading and Caching Module

Handles fetching historical market data from various sources (yfinance, CCXT)
with intelligent caching to minimize API calls and improve performance.
"""

from typing import Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from loguru import logger

from src.core.config import get_config


class DataLoader:
    """Handles data fetching, caching, and validation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.config = get_config()
        self.cache_dir = cache_dir or self.config.data.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = "1d",
        source: str = "yfinance",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch historical market data.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USDT')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Data timeframe ('1d', '1h', '4h', etc.)
            source: Data source ('yfinance' or 'ccxt')
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data and standard column names
            
        Example:
            >>> loader = DataLoader()
            >>> df = loader.fetch_data('AAPL', '2020-01-01', '2023-12-31')
            >>> print(df.head())
        """
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(symbol, start_date, end_date, timeframe)
            if cached_data is not None:
                logger.info(f"Loaded {symbol} from cache")
                return cached_data
        
        # Fetch from source
        logger.info(f"Fetching {symbol} from {source}")
        if source == "yfinance":
            df = self._fetch_yfinance(symbol, start_date, end_date, timeframe)
        elif source == "ccxt":
            df = self._fetch_ccxt(symbol, start_date, end_date, timeframe)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Validate and clean
        df = self._validate_and_clean(df)
        
        # Cache the data
        if use_cache:
            self._save_to_cache(df, symbol, start_date, end_date, timeframe)
        
        return df
    
    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            timeframe: Time interval
            
        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(symbol)
        
        # Map timeframe to yfinance interval
        interval_map = {
            '1d': '1d',
            '1h': '1h',
            '4h': '4h',
            '1w': '1wk',
            '1m': '1mo'
        }
        interval = interval_map.get(timeframe, '1d')
        
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True  # Adjust for splits and dividends
        )
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def _fetch_ccxt(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch cryptocurrency data from exchanges via CCXT.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Start date
            end_date: End date
            timeframe: Timeframe ('1m', '5m', '1h', '1d', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Initialize exchange (Binance as default)
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        # Convert dates to timestamps
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        # Fetch OHLCV data
        all_ohlcv = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_ts,
                limit=1000
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data.
        
        Args:
            df: Raw market data
            
        Returns:
            Cleaned DataFrame
        """
        # Check minimum data points
        if len(df) < self.config.data.min_data_points:
            logger.warning(
                f"Data contains only {len(df)} points, "
                f"minimum recommended: {self.config.data.min_data_points}"
            )
        
        # Remove NaN values
        initial_rows = len(df)
        df = df.dropna()
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} rows with NaN values")
        
        # Check for data quality issues
        if (df['high'] < df['low']).any():
            logger.error("Data quality issue: High < Low detected")
        
        if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
            logger.error("Data quality issue: Close outside High-Low range")
        
        # Remove zero or negative prices
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    def _load_from_cache(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            Cached DataFrame or None if not available
        """
        cache_file = self._get_cache_path(symbol, start_date, end_date, timeframe)
        
        if not cache_file.exists():
            return None
        
        # Check if cache is fresh (less than 1 day old for daily data)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age > timedelta(days=1) and timeframe == "1d":
            logger.info(f"Cache expired for {symbol}")
            return None
        
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str
    ) -> None:
        """Save data to cache.
        
        Args:
            df: DataFrame to cache
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
        """
        cache_file = self._get_cache_path(symbol, start_date, end_date, timeframe)
        df.to_csv(cache_file)
        logger.info(f"Cached data to {cache_file}")
    
    def _get_cache_path(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str
    ) -> Path:
        """Generate cache file path.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            Path to cache file
        """
        # Sanitize symbol for filename
        safe_symbol = symbol.replace('/', '_').replace('\\', '_')
        filename = f"{safe_symbol}_{start_date}_{end_date}_{timeframe}.csv"
        return self.cache_dir / filename
    
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = "1d"
    ) -> dict:
        """Fetch data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, start_date, end_date, timeframe)
                data[symbol] = df
                logger.info(f"Successfully fetched {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        return data