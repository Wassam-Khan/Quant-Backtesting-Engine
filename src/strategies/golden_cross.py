"""
Golden Cross / Death Cross Strategy

A classic trend-following strategy based on moving average crossovers:
- Golden Cross: Fast MA crosses above Slow MA → Buy Signal
- Death Cross: Fast MA crosses below Slow MA → Sell Signal

Theory:
    The Golden Cross strategy identifies trend changes using two moving averages:
    - Fast MA (shorter period): Reacts quickly to price changes
    - Slow MA (longer period): Represents the long-term trend
    
    When the fast MA crosses above the slow MA, it suggests an uptrend is beginning.
    When it crosses below, it suggests a downtrend.
    
    Mathematical Definition:
        SMA(n) = (P1 + P2 + ... + Pn) / n
        
        Buy Signal: SMA_fast(t) > SMA_slow(t) AND SMA_fast(t-1) <= SMA_slow(t-1)
        Sell Signal: SMA_fast(t) < SMA_slow(t) AND SMA_fast(t-1) >= SMA_slow(t-1)
"""

from typing import Dict
import pandas as pd
import numpy as np
from loguru import logger

from src.strategies.base_strategy import BaseStrategy


class GoldenCrossStrategy(BaseStrategy):
    """Golden Cross trend-following strategy.
    
    Attributes:
        fast_period: Period for fast moving average (default: 50)
        slow_period: Period for slow moving average (default: 200)
        use_ema: Whether to use EMA instead of SMA (default: False)
    """
    
    def __init__(self, parameters: Dict = None):
        """Initialize Golden Cross strategy.
        
        Args:
            parameters: Strategy parameters including:
                - fast_period: Fast MA period (default: 50)
                - slow_period: Slow MA period (default: 200)
                - use_ema: Use exponential MA (default: False)
        """
        super().__init__(name="GoldenCross", parameters=parameters)
        
    def get_default_parameters(self) -> Dict:
        """Get default parameters for Golden Cross.
        
        Returns:
            Dictionary with default parameter values
        """
        return {
            'fast_period': 50,
            'slow_period': 200,
            'use_ema': False,  # Simple MA by default
            'confirmation_period': 1  # Days to confirm signal
        }
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages and trend indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added MA columns
        """
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        use_ema = self.parameters.get('use_ema', False)
        
        # Calculate moving averages
        if use_ema:
            data['ma_fast'] = data['close'].ewm(span=fast_period, adjust=False).mean()
            data['ma_slow'] = data['close'].ewm(span=slow_period, adjust=False).mean()
            logger.info(f"Calculated EMA({fast_period}) and EMA({slow_period})")
        else:
            data['ma_fast'] = data['close'].rolling(window=fast_period).mean()
            data['ma_slow'] = data['close'].rolling(window=slow_period).mean()
            logger.info(f"Calculated SMA({fast_period}) and SMA({slow_period})")
        
        # Calculate crossover indicators
        data['ma_diff'] = data['ma_fast'] - data['ma_slow']
        data['ma_diff_prev'] = data['ma_diff'].shift(1)
        
        # Trend strength (distance between MAs as % of price)
        data['trend_strength'] = (data['ma_diff'] / data['close']) * 100
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Golden Cross / Death Cross signals.
        
        Args:
            data: DataFrame with calculated indicators
            
        Returns:
            DataFrame with 'signal' column added
        """
        confirmation_period = self.parameters.get('confirmation_period', 1)
        
        # Initialize signal column
        data['signal'] = 0.0
        
        # Golden Cross: Fast MA crosses above Slow MA
        golden_cross = (
            (data['ma_fast'] > data['ma_slow']) &
            (data['ma_fast'].shift(1) <= data['ma_slow'].shift(1))
        )
        
        # Death Cross: Fast MA crosses below Slow MA
        death_cross = (
            (data['ma_fast'] < data['ma_slow']) &
            (data['ma_fast'].shift(1) >= data['ma_slow'].shift(1))
        )
        
        # Apply confirmation period (wait N days after crossover)
        if confirmation_period > 1:
            golden_cross = golden_cross.rolling(window=confirmation_period).sum() > 0
            death_cross = death_cross.rolling(window=confirmation_period).sum() > 0
        
        # Generate signals
        data.loc[golden_cross, 'signal'] = 1.0  # Buy
        data.loc[death_cross, 'signal'] = -1.0  # Sell
        
        # Count signals
        n_golden = golden_cross.sum()
        n_death = death_cross.sum()
        logger.info(f"Generated {n_golden} Golden Cross and {n_death} Death Cross signals")
        
        return data
    
    def get_parameter_ranges(self) -> Dict:
        """Get parameter ranges for optimization.
        
        Returns:
            Dictionary with parameter ranges for grid search
        """
        return {
            'fast_period': (20, 100, 10),   # (min, max, step)
            'slow_period': (100, 300, 20),
            'use_ema': [True, False],        # Discrete choices
            'confirmation_period': (1, 5, 1)
        }


class EnhancedGoldenCross(GoldenCrossStrategy):
    """Enhanced Golden Cross with volume and momentum filters.
    
    Adds additional filters to reduce false signals:
    - Volume confirmation: Requires above-average volume
    - Momentum filter: Checks RSI to avoid overbought/oversold
    - Trend filter: Ensures price is above/below long-term MA
    """
    
    def __init__(self, parameters: Dict = None):
        """Initialize Enhanced Golden Cross strategy."""
        super().__init__(parameters=parameters)
        self.name = "EnhancedGoldenCross"
    
    def get_default_parameters(self) -> Dict:
        """Get default parameters."""
        params = super().get_default_parameters()
        params.update({
            'volume_ma_period': 20,
            'volume_multiplier': 1.5,  # Require 1.5x average volume
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30
        })
        return params
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators including volume and RSI."""
        # Base indicators
        data = super().calculate_indicators(data)
        
        # Volume indicators
        volume_period = self.parameters['volume_ma_period']
        data['volume_ma'] = data['volume'].rolling(window=volume_period).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # RSI calculation
        rsi_period = self.parameters['rsi_period']
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate filtered signals."""
        # Base signals
        data = super().generate_signals(data)
        
        # Apply filters
        volume_multiplier = self.parameters['volume_multiplier']
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_oversold = self.parameters['rsi_oversold']
        
        # Volume filter
        volume_ok = data['volume_ratio'] >= volume_multiplier
        
        # RSI filter (avoid extremes)
        rsi_ok_long = data['rsi'] < rsi_overbought
        rsi_ok_short = data['rsi'] > rsi_oversold
        
        # Apply filters to signals
        data.loc[(data['signal'] == 1.0) & (~volume_ok | ~rsi_ok_long), 'signal'] = 0.0
        data.loc[(data['signal'] == -1.0) & (~volume_ok | ~rsi_ok_short), 'signal'] = 0.0
        
        return data