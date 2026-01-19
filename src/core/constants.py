"""Constants and Enumerations"""

from enum import Enum


class OrderType(Enum):
    """Order execution types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class PositionType(Enum):
    """Position types."""
    LONG = 1
    SHORT = -1
    FLAT = 0


class TimeFrame(Enum):
    """Supported timeframes."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


# Trading constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
MIN_DATA_POINTS = 100