"""Custom Exceptions"""


class BacktestError(Exception):
    """Base exception for backtesting errors."""
    pass


class DataError(BacktestError):
    """Data loading or validation errors."""
    pass


class StrategyError(BacktestError):
    """Strategy configuration or execution errors."""
    pass


class InsufficientDataError(DataError):
    """Raised when data has too few points."""
    pass


class InvalidParameterError(StrategyError):
    """Raised when strategy parameters are invalid."""
    pass


class ExecutionError(BacktestError):
    """Trade execution errors."""
    pass