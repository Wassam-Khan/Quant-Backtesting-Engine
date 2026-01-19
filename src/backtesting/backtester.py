"""
Core Backtesting Engine

Simulates trading strategies on historical data with realistic
execution modeling including commissions, slippage, and position sizing.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from src.core.config import get_config
from src.strategies.base_strategy import BaseStrategy
from src.data.data_loader import DataLoader


class Backtester:
    """Core backtesting engine with realistic execution simulation."""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: Optional[float] = None,
        commission: Optional[float] = None,
        slippage: Optional[float] = None
    ):
        self.strategy = strategy
        self.config = get_config()
        
        # Trading parameters
        self.initial_capital = initial_capital or self.config.trading.initial_capital
        self.commission = commission or self.config.trading.commission
        self.slippage = slippage or self.config.trading.slippage
        self.position_size = self.config.trading.position_size
        
        # State tracking
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.position = 0.0
        self.trades: List[Dict] = []
        
        logger.info(f"Initialized backtester with ${self.initial_capital:,.2f} capital")
    
    def run(self, data: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Execute backtest on historical data."""
        logger.info(f"Starting backtest: {self.strategy.name}")
        
        # Generate strategy signals
        data = self.strategy.run(data)
        
        # Initialize columns
        data['cash'] = float(self.initial_capital)
        data['position_size'] = 0.0
        data['portfolio_value'] = float(self.initial_capital)
        data['returns'] = 0.0
        data['cumulative_returns'] = 0.0
        
        # Simulate trading
        data = self._simulate_trading(data)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(data)
        
        if verbose:
            self._print_summary(metrics)
        
        return data, metrics
    
    def _simulate_trading(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simulate trade execution with realistic costs."""
        cash = self.initial_capital
        position_shares = 0.0
        last_trade_price = 0.0
        
        # Iterate through bars
        for i in range(len(data)):
            current_signal = data['signal'].iloc[i]
            current_price = data['close'].iloc[i]
            date = data.index[i]
            
            # Apply slippage
            if current_signal != 0:
                execution_price = current_price * (1 + self.slippage * np.sign(current_signal))
            else:
                execution_price = current_price
            
            # ---------------------------------------------------------
            # LOGIC: Buy / Sell
            # ---------------------------------------------------------
            if current_signal == 1.0 and position_shares == 0:
                # Open Long
                shares_to_buy = (cash * self.position_size) / execution_price
                cost = shares_to_buy * execution_price
                commission_cost = cost * self.commission
                
                if cash >= (cost + commission_cost):
                    position_shares = shares_to_buy
                    cash -= (cost + commission_cost)
                    last_trade_price = execution_price
                    
                    self._record_trade(date, 'BUY', execution_price, shares_to_buy, commission_cost)
            
            elif current_signal == -1.0 and position_shares > 0:
                # Close Long
                proceeds = position_shares * execution_price
                commission_cost = proceeds * self.commission
                
                cash += (proceeds - commission_cost)
                trade_pnl = (execution_price - last_trade_price) * position_shares - commission_cost
                
                self._record_trade(date, 'SELL', execution_price, position_shares, commission_cost, trade_pnl)
                position_shares = 0.0
            
            # Update step values
            position_value = position_shares * current_price
            portfolio_value = cash + position_value
            
            data.iat[i, data.columns.get_loc('cash')] = cash
            data.iat[i, data.columns.get_loc('position_size')] = position_value
            data.iat[i, data.columns.get_loc('portfolio_value')] = portfolio_value

        # ---------------------------------------------------------
        # FIX: Force Close at End of Backtest (This fixes the "0 Trades" issue)
        # ---------------------------------------------------------
        if position_shares > 0:
            last_price = data['close'].iloc[-1]
            last_date = data.index[-1]
            
            proceeds = position_shares * last_price
            commission_cost = proceeds * self.commission
            
            cash += (proceeds - commission_cost)
            trade_pnl = (last_price - last_trade_price) * position_shares - commission_cost
            
            logger.info("Forcing close of open position at end of backtest.")
            self._record_trade(last_date, 'SELL (Force)', last_price, position_shares, commission_cost, trade_pnl)
            
            # Update final row
            data.iat[-1, data.columns.get_loc('cash')] = cash
            data.iat[-1, data.columns.get_loc('position_size')] = 0.0
            data.iat[-1, data.columns.get_loc('portfolio_value')] = cash

        # Calculate returns
        data['returns'] = data['portfolio_value'].pct_change()
        data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
        
        return data
    
    def _record_trade(self, date, action, price, shares, commission, pnl=0.0):
        """Record trade details."""
        trade = {
            'date': date,
            'action': action,
            'price': price,
            'shares': shares,
            'value': price * shares,
            'commission': commission,
            'pnl': pnl
        }
        self.trades.append(trade)
        logger.debug(f"{action} {shares:.2f} shares @ ${price:.2f} (P&L: ${pnl:.2f})")

    def _calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = data['returns'].dropna()
        
        total_return = data['cumulative_returns'].iloc[-1]
        n_years = len(data) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Sortino
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - 0.02) / downside_std if downside_std > 0 else 0
        
        # Max Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trades Analysis
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            # Filter for selling trades to calculate Win Rate
            closing_trades = trades_df[trades_df['action'].str.contains('SELL')]
            
            n_trades = len(closing_trades)
            if n_trades > 0:
                winning_trades = closing_trades[closing_trades['pnl'] > 0]
                win_rate = len(winning_trades) / n_trades
                
                avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
                losing_trades = closing_trades[closing_trades['pnl'] <= 0]
                avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0
                
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
        else:
            n_trades = win_rate = avg_win = avg_loss = profit_factor = 0
            
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'kelly_criterion': 0.0,
            'final_portfolio_value': data['portfolio_value'].iloc[-1],
            'total_commission_paid': trades_df['commission'].sum() if not trades_df.empty else 0
        }

    def _print_summary(self, metrics: Dict) -> None:
        """Print summary."""
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Strategy: {self.strategy.name}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
        logger.info("-" * 60)
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info("-" * 60)
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
        logger.info("=" * 60)

    def get_trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
    
    def export_results(self, results, metrics, output_dir):
         # Dummy implementation to satisfy potential external calls, though run_backtest handles export
         pass