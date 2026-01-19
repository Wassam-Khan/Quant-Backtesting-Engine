"""
Advanced Performance Analytics and Risk Metrics

Provides comprehensive analysis tools for backtesting results including
risk-adjusted returns, drawdown analysis, and statistical tests.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


class PerformanceAnalyzer:
    """Advanced performance analysis for trading strategies."""
    
    def __init__(self, results: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None):
        """Initialize performance analyzer.
        
        Args:
            results: DataFrame with portfolio values and returns
            benchmark: Optional benchmark returns (e.g., S&P 500)
        """
        self.results = results
        self.benchmark = benchmark
        self.returns = results['returns'].dropna()
    
    def calculate_sharpe_ratio(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sharpe Ratio.
        
        The Sharpe Ratio measures risk-adjusted return:
            SR = (E[R] - Rf) / σ(R)
        
        Where:
            E[R] = Expected portfolio return
            Rf = Risk-free rate
            σ(R) = Standard deviation of returns
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            periods_per_year: Trading periods per year
            
        Returns:
            Annualized Sharpe Ratio
        """
        excess_returns = self.returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sortino Ratio (downside deviation).
        
        Similar to Sharpe but only penalizes downside volatility:
            Sortino = (E[R] - Rf) / σ_downside(R)
        
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Annualized Sortino Ratio
        """
        excess_returns = self.returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    def calculate_max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown and its duration.
        
        Drawdown is the peak-to-trough decline:
            DD(t) = (V(t) - max(V[0:t])) / max(V[0:t])
        
        Returns:
            Tuple of (max_drawdown, start_date, end_date)
        """
        portfolio_value = self.results['portfolio_value']
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max
        
        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        
        # Find the peak before the maximum drawdown
        start_date = portfolio_value.loc[:end_date].idxmax()
        
        return max_dd, start_date, end_date
    
    def calculate_calmar_ratio(self, periods_per_year: int = 252) -> float:
        """Calculate Calmar Ratio (return / max drawdown).
        
        Calmar Ratio = Annualized Return / |Max Drawdown|
        
        Args:
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar Ratio
        """
        annualized_return = self.returns.mean() * periods_per_year
        max_dd, _, _ = self.calculate_max_drawdown()
        
        return annualized_return / abs(max_dd) if max_dd != 0 else 0
    
    def calculate_information_ratio(self) -> float:
        """Calculate Information Ratio vs benchmark.
        
        IR = (E[R_p] - E[R_b]) / σ(R_p - R_b)
        
        Returns:
            Information Ratio (or 0 if no benchmark)
        """
        if self.benchmark is None:
            logger.warning("No benchmark provided for Information Ratio")
            return 0.0
        
        benchmark_returns = self.benchmark['returns'].dropna()
        excess_returns = self.returns - benchmark_returns
        
        return excess_returns.mean() / excess_returns.std()
    
    def calculate_value_at_risk(
        self,
        confidence_level: float = 0.95,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Value at Risk (VaR).
        
        VaR is the maximum expected loss at a given confidence level.
        
        Args:
            confidence_level: Confidence level (default: 95%)
            periods_per_year: Trading periods per year
            
        Returns:
            Daily VaR as a percentage
        """
        return np.percentile(self.returns, (1 - confidence_level) * 100)
    
    def calculate_conditional_var(
        self,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        CVaR is the expected loss given that the loss exceeds VaR.
        
        Args:
            confidence_level: Confidence level
            
        Returns:
            CVaR as a percentage
        """
        var = self.calculate_value_at_risk(confidence_level)
        return self.returns[self.returns <= var].mean()
    
    def calculate_omega_ratio(
        self,
        threshold: float = 0.0
    ) -> float:
        """Calculate Omega Ratio.
        
        Omega = Probability-weighted gains / Probability-weighted losses
        
        Args:
            threshold: Return threshold (default: 0%)
            
        Returns:
            Omega Ratio
        """
        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns < threshold]
        
        return gains.sum() / losses.sum() if losses.sum() != 0 else np.inf
    
    def calculate_kelly_criterion(
        self,
        trades: pd.DataFrame
    ) -> float:
        """Calculate optimal position size using Kelly Criterion.
        
        Kelly Formula: f* = p - (1-p)/b
        
        Where:
            p = probability of winning
            b = win/loss ratio
            f* = fraction of capital to bet
        
        Args:
            trades: DataFrame with trade results
            
        Returns:
            Kelly fraction (0 to 1)
        """
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        if len(trades) == 0:
            return 0.0
        
        win_rate = len(winning_trades) / len(trades)
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Bound Kelly between 0 and 1 for safety
        return max(0, min(kelly, 1))
    
    def rolling_sharpe(
        self,
        window: int = 252,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """Calculate rolling Sharpe Ratio.
        
        Args:
            window: Rolling window size
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Series of rolling Sharpe Ratios
        """
        excess_returns = self.returns - (risk_free_rate / 252)
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = excess_returns.rolling(window).std()
        
        return np.sqrt(252) * rolling_mean / rolling_std
    
    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate win rate from trades.
        
        Args:
            trades: DataFrame with trade results
            
        Returns:
            Win rate as a fraction (0 to 1)
        """
        if len(trades) == 0:
            return 0.0
        
        winning_trades = trades[trades['pnl'] > 0]
        return len(winning_trades) / len(trades)
    
    def calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss).
        
        Args:
            trades: DataFrame with trade results
            
        Returns:
            Profit factor
        """
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        
        return gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    def monte_carlo_simulation(
        self,
        n_simulations: int = 1000,
        n_days: Optional[int] = None
    ) -> pd.DataFrame:
        """Run Monte Carlo simulation on returns.
        
        Args:
            n_simulations: Number of simulations to run
            n_days: Number of days to simulate (defaults to actual length)
            
        Returns:
            DataFrame with simulated portfolio paths
        """
        if n_days is None:
            n_days = len(self.returns)
        
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        
        simulations = np.zeros((n_days, n_simulations))
        simulations[0] = 1.0
        
        for i in range(1, n_days):
            daily_returns = np.random.normal(mean_return, std_return, n_simulations)
            simulations[i] = simulations[i-1] * (1 + daily_returns)
        
        return pd.DataFrame(simulations)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report.
        
        Returns:
            Dictionary with all performance metrics
        """
        max_dd, dd_start, dd_end = self.calculate_max_drawdown()
        
        report = {
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'max_drawdown': max_dd,
            'max_drawdown_start': dd_start,
            'max_drawdown_end': dd_end,
            'value_at_risk_95': self.calculate_value_at_risk(0.95),
            'conditional_var_95': self.calculate_conditional_var(0.95),
            'omega_ratio': self.calculate_omega_ratio(),
            'total_return': self.results['cumulative_returns'].iloc[-1],
            'annualized_return': self.returns.mean() * 252,
            'annualized_volatility': self.returns.std() * np.sqrt(252)
        }
        
        return report