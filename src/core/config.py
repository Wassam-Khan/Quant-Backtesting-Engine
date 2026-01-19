"""
Central Configuration Management Module

This module handles all configuration parameters for the backtesting engine,
including trading fees, slippage, risk parameters, and data sources.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml
from loguru import logger


@dataclass
class TradingConfig:
    """Trading execution parameters."""
    
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    position_size: float = 0.95  # Use 95% of available capital
    
    # Order execution
    order_type: str = "market"  # market, limit, stop
    execution_delay: int = 1  # bars delay for realistic execution


@dataclass
class RiskConfig:
    """Risk management parameters."""
    
    max_position_size: float = 0.2  # Max 20% per position
    max_drawdown_threshold: float = 0.15  # Stop trading at 15% drawdown
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.06  # 6% take profit (3:1 R:R)
    
    # Portfolio risk
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_open_positions: int = 5
    
    # Leverage (for crypto/forex)
    max_leverage: float = 1.0  # No leverage by default
    
    # Kelly Criterion parameters
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25  # Quarter Kelly for safety


@dataclass
class DataConfig:
    """Data source and storage configuration."""
    
    data_source: str = "yfinance"  # yfinance, ccxt, csv
    cache_enabled: bool = True
    cache_dir: Path = Path("data/cache")
    
    # Data quality
    min_data_points: int = 252  # Minimum 1 year of data
    fill_method: str = "ffill"  # Forward fill missing data
    
    # Time periods
    default_timeframe: str = "1d"  # 1d, 1h, 4h, etc.
    lookback_period: int = 365  # Days to look back


@dataclass
class BacktestConfig:
    """Backtesting engine parameters."""
    
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    
    # Optimization
    optimize: bool = False
    optimization_metric: str = "sharpe_ratio"  # sharpe_ratio, sortino, calmar
    
    # Monte Carlo simulation
    monte_carlo_runs: int = 1000
    confidence_level: float = 0.95


@dataclass
class Config:
    """Master configuration container."""
    
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = Path("logs/backtest.log")
    
    # Output
    results_dir: Path = Path("results")
    save_trades: bool = True
    generate_plots: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance populated from YAML
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            trading=TradingConfig(**config_dict.get('trading', {})),
            risk=RiskConfig(**config_dict.get('risk', {})),
            data=DataConfig(**config_dict.get('data', {})),
            backtest=BacktestConfig(**config_dict.get('backtest', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['trading', 'risk', 'data', 'backtest']}
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'trading': self.trading.__dict__,
            'risk': self.risk.__dict__,
            'data': self.data.__dict__,
            'backtest': self.backtest.__dict__,
            'log_level': self.log_level,
            'log_file': str(self.log_file),
            'results_dir': str(self.results_dir),
            'save_trades': self.save_trades,
            'generate_plots': self.generate_plots
        }
    
    def validate(self) -> bool:
        """Validate configuration parameters.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.trading.commission < 0 or self.trading.commission > 0.1:
            raise ValueError("Commission must be between 0 and 10%")
        
        if self.risk.max_position_size > 1.0:
            raise ValueError("Max position size cannot exceed 100%")
        
        if self.risk.kelly_fraction > 1.0:
            logger.warning("Kelly fraction > 1.0 increases risk significantly")
        
        if self.trading.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        logger.info("Configuration validation passed")
        return True


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    config.validate()
    _config = config