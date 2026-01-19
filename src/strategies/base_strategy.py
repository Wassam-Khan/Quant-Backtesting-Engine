"""
Abstract Base Strategy Class

All trading strategies must inherit from this class and implement
the required methods. This ensures consistency and enforces best practices.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.
    
    All strategies must implement:
    - generate_signals(): Core signal generation logic
    - calculate_indicators(): Technical indicator computation
    - get_parameters(): Strategy parameter definitions
    
    Attributes:
        name: Strategy name for identification
        parameters: Dictionary of strategy parameters
        data: DataFrame containing market data and indicators
    """
    
    def __init__(self, name: str, parameters: Optional[Dict] = None):
        """Initialize base strategy.
        
        Args:
            name: Descriptive name for the strategy
            parameters: Dictionary of strategy-specific parameters
        """
        self.name = name
        self.parameters = parameters or self.get_default_parameters()
        self.data: Optional[pd.DataFrame] = None
        self._validate_parameters()
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on market data.
        
        This is the core method that must be implemented by all strategies.
        It should add a 'signal' column to the data where:
        - 1.0 = Long (Buy)
        - -1.0 = Short (Sell)
        - 0.0 = No position (Neutral/Cash)
        
        Args:
            data: DataFrame with OHLCV data and calculated indicators
            
        Returns:
            DataFrame with added 'signal' column
            
        Example:
            >>> data['signal'] = 0.0
            >>> data.loc[buy_condition, 'signal'] = 1.0
            >>> data.loc[sell_condition, 'signal'] = -1.0
            >>> return data
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators required by the strategy.
        
        This method should add all necessary technical indicators as new
        columns to the input DataFrame.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
            
        Example:
            >>> data['sma_20'] = data['close'].rolling(20).mean()
            >>> data['sma_50'] = data['close'].rolling(50).mean()
            >>> return data
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict:
        """Return default strategy parameters.
        
        Returns:
            Dictionary of parameter names and default values
            
        Example:
            >>> return {
            ...     'fast_period': 20,
            ...     'slow_period': 50,
            ...     'signal_threshold': 0.02
            ... }
        """
        pass
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the complete strategy pipeline.
        
        This is the main entry point that orchestrates:
        1. Indicator calculation
        2. Signal generation
        3. Position management
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators, signals, and positions
        """
        logger.info(f"Running strategy: {self.name}")
        
        # Store data
        self.data = data.copy()
        
        # Calculate indicators
        self.data = self.calculate_indicators(self.data)
        
        # Generate signals
        self.data = self.generate_signals(self.data)
        
        # Convert signals to positions (forward-fill)
        self.data['position'] = self.data['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Calculate position changes (entries/exits)
        self.data['position_change'] = self.data['position'].diff()
        
        logger.info(f"Strategy execution complete. Generated {self._count_trades()} trades")
        
        return self.data
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(self.parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Check for negative periods
        for key, value in self.parameters.items():
            if 'period' in key.lower() and isinstance(value, (int, float)):
                if value <= 0:
                    raise ValueError(f"Parameter {key} must be positive, got {value}")
    
    def _count_trades(self) -> int:
        """Count the number of trades generated.
        
        Returns:
            Number of position changes (entry/exit signals)
        """
        if self.data is None or 'position_change' not in self.data.columns:
            return 0
        return int((self.data['position_change'] != 0).sum())
    
    def get_parameter_ranges(self) -> Dict:
        """Get parameter ranges for optimization.
        
        Returns:
            Dictionary mapping parameter names to (min, max, step) tuples
            
        Example:
            >>> return {
            ...     'fast_period': (10, 30, 5),
            ...     'slow_period': (40, 100, 10)
            ... }
        """
        return {}
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            raise ValueError("Data is empty")
        
        return True
    
    def apply_risk_management(
        self,
        data: pd.DataFrame,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.06
    ) -> pd.DataFrame:
        """Apply stop-loss and take-profit rules.
        
        Args:
            data: DataFrame with positions
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
            take_profit_pct: Take profit percentage (e.g., 0.06 for 6%)
            
        Returns:
            DataFrame with risk-adjusted positions
        """
        data = data.copy()
        
        # Track entry prices
        data['entry_price'] = np.nan
        position_changes = data['position'].diff() != 0
        data.loc[position_changes, 'entry_price'] = data.loc[position_changes, 'close']
        data['entry_price'] = data['entry_price'].ffill()
        
        # Calculate returns from entry
        data['return_from_entry'] = (data['close'] - data['entry_price']) / data['entry_price']
        
        # Apply stop-loss for long positions
        stop_loss_trigger = (
            (data['position'] == 1) &
            (data['return_from_entry'] <= -stop_loss_pct)
        )
        
        # Apply take-profit for long positions
        take_profit_trigger = (
            (data['position'] == 1) &
            (data['return_from_entry'] >= take_profit_pct)
        )
        
        # Exit position on stop-loss or take-profit
        exit_triggers = stop_loss_trigger | take_profit_trigger
        data.loc[exit_triggers, 'position'] = 0
        data.loc[exit_triggers, 'signal'] = 0
        
        # Forward-fill positions after exits
        data['position'] = data['position'].replace(0, np.nan).ffill().fillna(0)
        
        return data
    
    def __repr__(self) -> str:
        """String representation of the strategy."""
        param_str = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.name}({param_str})"