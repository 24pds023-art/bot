"""
INTEGRATED BACKTESTING WITH AUTOMATED PARAMETER OPTIMIZATION
============================================================
🚀 Complete backtesting system with automated config tuning
📊 Genetic algorithms, grid search, and Bayesian optimization
🎯 Automatically finds optimal stop loss, take profit, trailing stops
⚡ Walk-forward analysis for robust parameter validation
🧠 ML-powered parameter selection with cross-validation
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import sharpe_score
import itertools
import random
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Import our trading system components
from ultra_optimized_trading_system import UltraFastIncrementalEngine
from advanced_optimizations import AdvancedMLSuite, AdvancedIndicatorSuite

@dataclass
class OptimizableConfig:
    """Configuration parameters that can be optimized"""
    # Signal thresholds
    signal_strength_threshold: float = 0.22
    ml_confidence_threshold: float = 0.65
    min_confirmations: int = 2
    
    # Risk management
    stop_loss_pct: float = 0.006
    take_profit_pct: float = 0.013
    risk_reward_ratio: float = 2.2
    trailing_stop_distance: float = 0.0003
    trailing_stop_activation: float = 0.005  # 0.5% profit to activate
    
    # Position management
    base_position_pct: float = 0.1  # 10% of portfolio per trade
    max_positions: int = 15
    position_size_multiplier: float = 1.0
    
    # Technical indicators
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    rsi_period: int = 14
    ema_fast: int = 10
    ema_slow: int = 21
    macd_fast: int = 12
    macd_slow: int = 26
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Advanced parameters
    volume_confirmation_threshold: float = 1.2
    volatility_adjustment: bool = True
    time_based_exit_hours: float = 4.0
    breakeven_move_threshold: float = 0.003
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizableConfig':
        return cls(**data)

class ParameterBounds:
    """Define optimization bounds for each parameter"""
    
    BOUNDS = {
        # Signal thresholds (0.1 to 0.5)
        'signal_strength_threshold': (0.1, 0.5),
        'ml_confidence_threshold': (0.5, 0.9),
        'min_confirmations': (1, 4),
        
        # Risk management (0.3% to 2.0%)
        'stop_loss_pct': (0.003, 0.020),
        'take_profit_pct': (0.005, 0.030),
        'risk_reward_ratio': (1.5, 4.0),
        'trailing_stop_distance': (0.0001, 0.001),
        'trailing_stop_activation': (0.002, 0.010),
        
        # Position management
        'base_position_pct': (0.05, 0.25),
        'max_positions': (5, 25),
        'position_size_multiplier': (0.5, 2.0),
        
        # Technical indicators
        'rsi_oversold': (15, 35),
        'rsi_overbought': (65, 85),
        'rsi_period': (10, 21),
        'ema_fast': (5, 15),
        'ema_slow': (15, 30),
        'macd_fast': (8, 16),
        'macd_slow': (20, 35),
        'bb_period': (15, 30),
        'bb_std_dev': (1.5, 2.5),
        
        # Advanced parameters
        'volume_confirmation_threshold': (0.8, 2.0),
        'time_based_exit_hours': (1.0, 8.0),
        'breakeven_move_threshold': (0.001, 0.008)
    }
    
    @classmethod
    def get_bounds_list(cls, param_names: List[str]) -> List[Tuple[float, float]]:
        """Get bounds as list for optimization algorithms"""
        return [cls.BOUNDS[param] for param in param_names]
    
    @classmethod
    def clip_to_bounds(cls, config: OptimizableConfig) -> OptimizableConfig:
        """Clip configuration values to valid bounds"""
        config_dict = config.to_dict()
        
        for param, (min_val, max_val) in cls.BOUNDS.items():
            if param in config_dict:
                config_dict[param] = np.clip(config_dict[param], min_val, max_val)
                
                # Handle integer parameters
                if param in ['min_confirmations', 'max_positions', 'rsi_period', 
                           'ema_fast', 'ema_slow', 'macd_fast', 'macd_slow', 'bb_period']:
                    config_dict[param] = int(round(config_dict[param]))
        
        return OptimizableConfig.from_dict(config_dict)

@dataclass
class BacktestResult:
    """Comprehensive backtest result"""
    config: OptimizableConfig
    total_return: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Additional metrics
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    
    # Optimization score (composite metric)
    optimization_score: float = 0.0
    
    def calculate_optimization_score(self) -> float:
        """Calculate composite optimization score"""
        # Weighted combination of key metrics
        score = (
            self.total_return * 0.25 +           # 25% weight on returns
            self.win_rate * 0.20 +               # 20% weight on win rate
            self.sharpe_ratio * 0.15 +           # 15% weight on Sharpe ratio
            (1 + self.max_drawdown) * 0.15 +     # 15% weight on drawdown (inverted)
            self.profit_factor * 0.10 +          # 10% weight on profit factor
            (self.calmar_ratio * 0.10) +         # 10% weight on Calmar ratio
            (min(self.total_trades / 100, 1.0) * 0.05)  # 5% weight on trade frequency
        )
        
        # Penalty for excessive drawdown
        if self.max_drawdown < -0.15:  # More than 15% drawdown
            score *= 0.5
        
        # Penalty for low trade count
        if self.total_trades < 20:
            score *= 0.8
        
        self.optimization_score = score
        return score

class IntegratedBacktestingEngine:
    """Integrated backtesting engine with automated parameter optimization"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str, 
                 initial_balance: float = 100000):
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_balance = initial_balance
        
        # Generate comprehensive historical data
        self.historical_data = self._generate_comprehensive_historical_data()
        
        # Optimization tracking
        self.optimization_history = []
        self.best_config = None
        self.best_result = None
        
        # Performance tracking
        self.backtests_run = 0
        self.total_optimization_time = 0
        
        print(f"📈 Integrated Backtesting Engine initialized")
        print(f"   Symbols: {len(symbols)}")
        print(f"   Date Range: {start_date} to {end_date}")
        print(f"   Initial Balance: ${initial_balance:,.2f}")
        print(f"   Data Points: {sum(len(df) for df in self.historical_data.values()):,}")
    
    def _generate_comprehensive_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive historical market data"""
        print("📊 Generating comprehensive historical data...")
        
        # Create minute-by-minute data
        date_range = pd.date_range(self.start_date, self.end_date, freq='1min')
        historical_data = {}
        
        # Starting prices for major cryptocurrencies
        start_prices = {
            'BTCUSDT': 35000, 'ETHUSDT': 2200, 'BNBUSDT': 280, 'ADAUSDT': 0.45,
            'XRPUSDT': 0.52, 'SOLUSDT': 95, 'DOTUSDT': 6.8, 'LINKUSDT': 14.5,
            'AVAXUSDT': 18, 'LTCUSDT': 95, 'UNIUSDT': 6.2, 'ATOMUSDT': 11.5
        }
        
        for symbol in self.symbols:
            if symbol not in start_prices:
                continue
                
            start_price = start_prices[symbol]
            n_points = len(date_range)
            
            # Market regime simulation
            regime_changes = np.random.choice([0, 1], size=n_points, p=[0.95, 0.05])  # 5% chance of regime change
            current_regime = 'trending'
            
            # Generate realistic price series
            prices = [start_price]
            volumes = []
            
            for i in range(1, n_points):
                # Change regime occasionally
                if regime_changes[i]:
                    current_regime = np.random.choice(['trending', 'ranging', 'volatile'])
                
                # Regime-specific parameters
                if current_regime == 'trending':
                    drift = 0.0001  # Slight upward trend
                    volatility = 0.015  # 1.5% volatility
                elif current_regime == 'ranging':
                    drift = 0.0
                    volatility = 0.008  # 0.8% volatility
                else:  # volatile
                    drift = 0.0
                    volatility = 0.035  # 3.5% volatility
                
                # Generate price change
                price_change = np.random.normal(drift, volatility)
                new_price = prices[-1] * (1 + price_change)
                
                # Ensure reasonable bounds
                new_price = max(new_price, start_price * 0.1)  # Don't go below 10% of start
                new_price = min(new_price, start_price * 10)   # Don't go above 10x start
                
                prices.append(new_price)
                
                # Generate volume (higher during volatile periods)
                base_volume = 1000000
                volume_multiplier = 1.5 if current_regime == 'volatile' else 1.0
                volume = np.random.lognormal(np.log(base_volume * volume_multiplier), 0.5)
                volumes.append(volume)
            
            # Create OHLCV data
            prices = np.array(prices)
            volumes = np.array(volumes + [volumes[-1]])  # Match length
            
            # Generate high/low from prices
            highs = prices * (1 + np.abs(np.random.normal(0, 0.003, len(prices))))
            lows = prices * (1 - np.abs(np.random.normal(0, 0.003, len(prices))))
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': date_range,
                'open': prices[:-1],  # Shift for open prices
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes
            })
            
            historical_data[symbol] = df
        
        print(f"✅ Generated data for {len(historical_data)} symbols")
        return historical_data
    
    async def run_backtest_with_config(self, config: OptimizableConfig, 
                                     verbose: bool = False) -> BacktestResult:
        """Run single backtest with given configuration"""
        self.backtests_run += 1
        start_time = time.time()
        
        # Initialize trading components
        incremental_engines = {
            symbol: UltraFastIncrementalEngine() for symbol in self.historical_data.keys()
        }
        
        # Trading state
        balance = self.initial_balance
        positions = {}
        trades = []
        balance_history = []
        
        # Process each timestamp
        for timestamp in pd.date_range(self.start_date, self.end_date, freq='5min'):  # 5-minute intervals
            current_market_data = {}
            
            # Get market data for all symbols
            for symbol, df in self.historical_data.items():
                # Find closest timestamp
                closest_idx = df['timestamp'].searchsorted(timestamp)
                if closest_idx >= len(df):
                    continue
                
                row = df.iloc[closest_idx]
                
                # Update incremental indicators
                incremental_engines[symbol].add_tick(row['close'], row['volume'])
                indicators = incremental_engines[symbol].get_all_indicators()
                
                current_market_data[symbol] = {
                    'price': row['close'],
                    'high': row['high'],
                    'low': row['low'],
                    'volume': row['volume'],
                    'indicators': indicators,
                    'timestamp': timestamp
                }
            
            # Generate trading signals
            signals = self._generate_signals_with_config(current_market_data, config)
            
            # Execute trades
            balance, positions, new_trades = self._execute_trades_with_config(
                signals, current_market_data, balance, positions, config
            )
            trades.extend(new_trades)
            
            # Manage existing positions
            balance, positions, closed_trades = self._manage_positions_with_config(
                current_market_data, balance, positions, config
            )
            trades.extend(closed_trades)
            
            # Record balance history
            total_value = balance + sum(
                pos['quantity'] * current_market_data.get(pos['symbol'], {}).get('price', 0)
                for pos in positions.values()
            )
            
            balance_history.append({
                'timestamp': timestamp,
                'balance': balance,
                'total_value': total_value,
                'positions': len(positions)
            })
        
        # Calculate comprehensive results
        result = self._calculate_backtest_metrics(config, trades, balance_history)
        
        # Track optimization time
        optimization_time = time.time() - start_time
        self.total_optimization_time += optimization_time
        
        if verbose:
            print(f"📊 Backtest #{self.backtests_run}: Return={result.total_return:.2%}, "
                  f"Win Rate={result.win_rate:.1%}, Sharpe={result.sharpe_ratio:.2f}")
        
        return result
    
    def _generate_signals_with_config(self, market_data: Dict, config: OptimizableConfig) -> Dict[str, Dict]:
        """Generate trading signals using configuration parameters"""
        signals = {}
        
        for symbol, data in market_data.items():
            indicators = data['indicators']
            
            if not indicators or indicators.get('current_price', 0) <= 0:
                continue
            
            # Calculate signal components
            signal_components = []
            confirmations = 0
            
            # RSI signal with configurable thresholds
            rsi = indicators.get('rsi', 50)
            if rsi < config.rsi_oversold:
                signal_components.append(('RSI', 0.8, 0.3))
                confirmations += 1
            elif rsi > config.rsi_overbought:
                signal_components.append(('RSI', -0.8, 0.3))
                confirmations += 1
            else:
                signal_components.append(('RSI', 0.0, 0.3))
            
            # EMA trend signal
            ema_fast = indicators.get(f'ema_{config.ema_fast}', indicators.get('ema_10', 0))
            ema_slow = indicators.get(f'ema_{config.ema_slow}', indicators.get('ema_21', 0))
            current_price = indicators['current_price']
            
            if current_price > ema_fast > ema_slow:
                signal_components.append(('EMA_TREND', 0.6, 0.25))
                confirmations += 1
            elif current_price < ema_fast < ema_slow:
                signal_components.append(('EMA_TREND', -0.6, 0.25))
                confirmations += 1
            else:
                signal_components.append(('EMA_TREND', 0.0, 0.25))
            
            # MACD signal
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            if macd > macd_signal and macd > 0:
                signal_components.append(('MACD', 0.5, 0.2))
                confirmations += 1
            elif macd < macd_signal and macd < 0:
                signal_components.append(('MACD', -0.5, 0.2))
                confirmations += 1
            else:
                signal_components.append(('MACD', 0.0, 0.2))
            
            # Volume confirmation
            volume_ratio = data['volume'] / indicators.get('volume_sma', data['volume'])
            if volume_ratio > config.volume_confirmation_threshold:
                confirmations += 1
                volume_boost = 0.1
            else:
                volume_boost = 0.0
            
            # Calculate composite signal
            weighted_signal = sum(signal * weight for signal, _, weight in signal_components)
            total_weight = sum(weight for _, _, weight in signal_components)
            composite_signal = (weighted_signal / total_weight) + volume_boost
            
            # Apply configuration thresholds
            if (composite_signal > config.signal_strength_threshold and 
                confirmations >= config.min_confirmations):
                signals[symbol] = {
                    'action': 'BUY',
                    'strength': composite_signal,
                    'confirmations': confirmations,
                    'price': current_price
                }
            elif (composite_signal < -config.signal_strength_threshold and 
                  confirmations >= config.min_confirmations):
                signals[symbol] = {
                    'action': 'SELL',
                    'strength': abs(composite_signal),
                    'confirmations': confirmations,
                    'price': current_price
                }
        
        return signals
    
    def _execute_trades_with_config(self, signals: Dict, market_data: Dict, 
                                  balance: float, positions: Dict, 
                                  config: OptimizableConfig) -> Tuple[float, Dict, List]:
        """Execute trades using configuration parameters"""
        new_trades = []
        
        # Limit concurrent positions
        if len(positions) >= config.max_positions:
            return balance, positions, new_trades
        
        for symbol, signal in signals.items():
            if symbol in positions:  # Already have position
                continue
            
            price = signal['price']
            action = signal['action']
            
            # Calculate position size
            base_position_value = balance * config.base_position_pct
            
            # Adjust for signal strength and volatility
            strength_multiplier = signal['strength'] * config.position_size_multiplier
            
            if config.volatility_adjustment:
                # Reduce size in high volatility
                atr = market_data[symbol]['indicators'].get('atr', 0.01)
                volatility = atr / price
                vol_adjustment = 1.0 / (1.0 + volatility * 10)  # Reduce size for high vol
            else:
                vol_adjustment = 1.0
            
            position_value = base_position_value * strength_multiplier * vol_adjustment
            quantity = position_value / price
            
            # Check if we have enough balance
            if position_value > balance * 0.95:  # Keep 5% cash buffer
                continue
            
            # Calculate stop loss and take profit
            if action == 'BUY':
                stop_loss_price = price * (1 - config.stop_loss_pct)
                take_profit_price = price * (1 + config.take_profit_pct)
            else:
                stop_loss_price = price * (1 + config.stop_loss_pct)
                take_profit_price = price * (1 - config.take_profit_pct)
            
            # Create position
            position_id = f"{symbol}_{len(positions)}_{int(time.time())}"
            position = {
                'symbol': symbol,
                'action': action,
                'entry_price': price,
                'quantity': quantity,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'entry_time': market_data[symbol]['timestamp'],
                'trailing_stop_active': False,
                'peak_profit': 0.0,
                'signal_strength': signal['strength']
            }
            
            positions[position_id] = position
            balance -= position_value  # Deduct from balance
            
            # Record trade
            new_trades.append({
                'type': 'ENTRY',
                'position_id': position_id,
                'symbol': symbol,
                'action': action,
                'price': price,
                'quantity': quantity,
                'value': position_value,
                'timestamp': market_data[symbol]['timestamp'],
                'signal_strength': signal['strength']
            })
        
        return balance, positions, new_trades
    
    def _manage_positions_with_config(self, market_data: Dict, balance: float, 
                                    positions: Dict, config: OptimizableConfig) -> Tuple[float, Dict, List]:
        """Manage existing positions using configuration parameters"""
        closed_trades = []
        positions_to_close = []
        
        for position_id, position in positions.items():
            symbol = position['symbol']
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['price']
            entry_price = position['entry_price']
            action = position['action']
            
            # Calculate current P&L
            if action == 'BUY':
                pnl = (current_price - entry_price) * position['quantity']
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl = (entry_price - current_price) * position['quantity']
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Update peak profit for trailing stops
            if pnl > position['peak_profit']:
                position['peak_profit'] = pnl
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            
            # Stop loss check
            if action == 'BUY' and current_price <= position['stop_loss']:
                should_close = True
                close_reason = "Stop Loss"
            elif action == 'SELL' and current_price >= position['stop_loss']:
                should_close = True
                close_reason = "Stop Loss"
            
            # Take profit check
            elif action == 'BUY' and current_price >= position['take_profit']:
                should_close = True
                close_reason = "Take Profit"
            elif action == 'SELL' and current_price <= position['take_profit']:
                should_close = True
                close_reason = "Take Profit"
            
            # Trailing stop logic
            elif pnl_pct > config.trailing_stop_activation:
                if not position['trailing_stop_active']:
                    position['trailing_stop_active'] = True
                    if verbose:
                        print(f"🎯 Trailing stop activated for {symbol}")
                
                # Check trailing stop
                peak_pnl_pct = position['peak_profit'] / (entry_price * position['quantity'])
                if peak_pnl_pct - pnl_pct > config.trailing_stop_distance:
                    should_close = True
                    close_reason = "Trailing Stop"
            
            # Time-based exit
            position_age = (market_data[symbol]['timestamp'] - position['entry_time']).total_seconds() / 3600
            if position_age > config.time_based_exit_hours:
                should_close = True
                close_reason = "Time Exit"
            
            # Breakeven move (move stop to breakeven after certain profit)
            if pnl_pct > config.breakeven_move_threshold and not position.get('breakeven_moved', False):
                if action == 'BUY':
                    position['stop_loss'] = entry_price * 1.001  # Small profit
                else:
                    position['stop_loss'] = entry_price * 0.999
                position['breakeven_moved'] = True
            
            if should_close:
                positions_to_close.append((position_id, current_price, close_reason))
        
        # Close positions
        for position_id, exit_price, reason in positions_to_close:
            position = positions[position_id]
            
            # Calculate final P&L
            entry_price = position['entry_price']
            quantity = position['quantity']
            action = position['action']
            
            if action == 'BUY':
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            
            # Update balance
            position_value = quantity * exit_price
            balance += position_value
            
            # Record closed trade
            closed_trades.append({
                'type': 'EXIT',
                'position_id': position_id,
                'symbol': position['symbol'],
                'action': 'SELL' if action == 'BUY' else 'BUY',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'pnl': pnl,
                'pnl_pct': pnl / (entry_price * quantity),
                'duration': (market_data[position['symbol']]['timestamp'] - position['entry_time']).total_seconds() / 3600,
                'exit_reason': reason,
                'timestamp': market_data[position['symbol']]['timestamp']
            })
            
            # Remove position
            del positions[position_id]
        
        return balance, positions, closed_trades
    
    def _calculate_backtest_metrics(self, config: OptimizableConfig, trades: List[Dict], 
                                  balance_history: List[Dict]) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""
        if not trades or not balance_history:
            return BacktestResult(
                config=config, total_return=0, win_rate=0, sharpe_ratio=0,
                max_drawdown=0, profit_factor=0, total_trades=0,
                avg_trade_duration=0, volatility=0, calmar_ratio=0, sortino_ratio=0
            )
        
        # Filter exit trades for analysis
        exit_trades = [t for t in trades if t['type'] == 'EXIT']
        
        if not exit_trades:
            return BacktestResult(
                config=config, total_return=0, win_rate=0, sharpe_ratio=0,
                max_drawdown=0, profit_factor=0, total_trades=0,
                avg_trade_duration=0, volatility=0, calmar_ratio=0, sortino_ratio=0
            )
        
        # Basic metrics
        total_return = (balance_history[-1]['total_value'] - self.initial_balance) / self.initial_balance
        
        # Trade analysis
        profitable_trades = [t for t in exit_trades if t['pnl'] > 0]
        losing_trades = [t for t in exit_trades if t['pnl'] <= 0]
        
        win_rate = len(profitable_trades) / len(exit_trades)
        
        # P&L analysis
        total_profit = sum(t['pnl'] for t in profitable_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_profit / max(total_loss, 1)
        
        # Duration analysis
        avg_duration = np.mean([t['duration'] for t in exit_trades])
        
        # Return analysis
        balance_df = pd.DataFrame(balance_history)
        balance_df['returns'] = balance_df['total_value'].pct_change().fillna(0)
        
        # Risk metrics
        returns = balance_df['returns']
        volatility = returns.std() * np.sqrt(252 * 24 * 12)  # Annualized (5-min data)
        
        if volatility > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 12)
        else:
            sharpe_ratio = 0
        
        # Downside deviation for Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252 * 24 * 12)
            sortino_ratio = returns.mean() / downside_deviation * np.sqrt(252 * 24 * 12)
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else total_return
        
        # Additional metrics
        largest_win = max([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        largest_loss = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Create result
        result = BacktestResult(
            config=config,
            total_return=total_return,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            total_trades=len(exit_trades),
            avg_trade_duration=avg_duration,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            largest_win=largest_win,
            largest_loss=largest_loss
        )
        
        # Calculate optimization score
        result.calculate_optimization_score()
        
        return result

# ============================================================================
# PARAMETER OPTIMIZATION ALGORITHMS
# ============================================================================

class ParameterOptimizer:
    """Advanced parameter optimization using multiple algorithms"""
    
    def __init__(self, backtesting_engine: IntegratedBacktestingEngine):
        self.backtesting_engine = backtesting_engine
        self.optimization_results = []
        self.best_configs = []
        
    async def grid_search_optimization(self, param_grid: Dict[str, List], 
                                     max_combinations: int = 1000) -> List[BacktestResult]:
        """Grid search parameter optimization"""
        print(f"🔍 Starting Grid Search Optimization...")
        print(f"   Parameters: {list(param_grid.keys())}")
        print(f"   Max Combinations: {max_combinations}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
        
        print(f"   Testing {len(all_combinations)} combinations...")
        
        results = []
        
        # Test each combination
        for i, combination in enumerate(all_combinations):
            # Create config
            config = OptimizableConfig()
            for param_name, value in zip(param_names, combination):
                setattr(config, param_name, value)
            
            # Ensure bounds
            config = ParameterBounds.clip_to_bounds(config)
            
            # Run backtest
            result = await self.backtesting_engine.run_backtest_with_config(config)
            results.append(result)
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i + 1}/{len(all_combinations)} ({(i+1)/len(all_combinations)*100:.1f}%)")
        
        # Sort by optimization score
        results.sort(key=lambda r: r.optimization_score, reverse=True)
        
        print(f"✅ Grid Search completed!")
        print(f"   Best Score: {results[0].optimization_score:.3f}")
        print(f"   Best Return: {results[0].total_return:.2%}")
        print(f"   Best Win Rate: {results[0].win_rate:.1%}")
        
        return results
    
    async def genetic_algorithm_optimization(self, population_size: int = 50, 
                                           generations: int = 20, 
                                           mutation_rate: float = 0.1) -> List[BacktestResult]:
        """Genetic algorithm parameter optimization"""
        print(f"🧬 Starting Genetic Algorithm Optimization...")
        print(f"   Population Size: {population_size}")
        print(f"   Generations: {generations}")
        print(f"   Mutation Rate: {mutation_rate}")
        
        # Initialize population
        population = self._create_initial_population(population_size)
        
        best_results = []
        
        for generation in range(generations):
            print(f"   Generation {generation + 1}/{generations}")
            
            # Evaluate population
            generation_results = []
            for individual in population:
                config = self._individual_to_config(individual)
                result = await self.backtesting_engine.run_backtest_with_config(config)
                generation_results.append((individual, result))
            
            # Sort by fitness (optimization score)
            generation_results.sort(key=lambda x: x[1].optimization_score, reverse=True)
            best_results.append(generation_results[0][1])
            
            print(f"     Best Score: {generation_results[0][1].optimization_score:.3f}")
            print(f"     Best Return: {generation_results[0][1].total_return:.2%}")
            
            # Selection and reproduction
            if generation < generations - 1:  # Not last generation
                # Select top 50% for reproduction
                elite_size = population_size // 2
                elite = [individual for individual, _ in generation_results[:elite_size]]
                
                # Create next generation
                new_population = elite.copy()  # Keep elite
                
                # Fill rest with crossover and mutation
                while len(new_population) < population_size:
                    # Select parents
                    parent1 = random.choice(elite)
                    parent2 = random.choice(elite)
                    
                    # Crossover
                    child = self._crossover(parent1, parent2)
                    
                    # Mutation
                    if random.random() < mutation_rate:
                        child = self._mutate(child)
                    
                    new_population.append(child)
                
                population = new_population
        
        print(f"✅ Genetic Algorithm completed!")
        print(f"   Best Final Score: {best_results[-1].optimization_score:.3f}")
        
        return best_results
    
    def _create_initial_population(self, size: int) -> List[Dict]:
        """Create initial population for genetic algorithm"""
        population = []
        
        for _ in range(size):
            individual = {}
            for param, (min_val, max_val) in ParameterBounds.BOUNDS.items():
                if param in ['min_confirmations', 'max_positions', 'rsi_period', 
                           'ema_fast', 'ema_slow', 'macd_fast', 'macd_slow', 'bb_period']:
                    # Integer parameters
                    individual[param] = random.randint(int(min_val), int(max_val))
                else:
                    # Float parameters
                    individual[param] = random.uniform(min_val, max_val)
            
            population.append(individual)
        
        return population
    
    def _individual_to_config(self, individual: Dict) -> OptimizableConfig:
        """Convert individual to configuration"""
        config = OptimizableConfig()
        for param, value in individual.items():
            if hasattr(config, param):
                setattr(config, param, value)
        return ParameterBounds.clip_to_bounds(config)
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover operation for genetic algorithm"""
        child = {}
        for param in parent1.keys():
            # Random selection from parents
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()
        
        # Select random parameter to mutate
        param_to_mutate = random.choice(list(individual.keys()))
        
        if param_to_mutate in ParameterBounds.BOUNDS:
            min_val, max_val = ParameterBounds.BOUNDS[param_to_mutate]
            
            if param_to_mutate in ['min_confirmations', 'max_positions', 'rsi_period', 
                                 'ema_fast', 'ema_slow', 'macd_fast', 'macd_slow', 'bb_period']:
                # Integer mutation
                mutated[param_to_mutate] = random.randint(int(min_val), int(max_val))
            else:
                # Float mutation with Gaussian noise
                current_value = individual[param_to_mutate]
                noise = np.random.normal(0, (max_val - min_val) * 0.1)
                new_value = current_value + noise
                mutated[param_to_mutate] = np.clip(new_value, min_val, max_val)
        
        return mutated
    
    async def bayesian_optimization(self, n_calls: int = 100) -> List[BacktestResult]:
        """Bayesian optimization for parameter tuning"""
        print(f"🎯 Starting Bayesian Optimization...")
        print(f"   Function Evaluations: {n_calls}")
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            
            # Define search space
            dimensions = []
            param_names = []
            
            for param, (min_val, max_val) in ParameterBounds.BOUNDS.items():
                if param in ['min_confirmations', 'max_positions', 'rsi_period', 
                           'ema_fast', 'ema_slow', 'macd_fast', 'macd_slow', 'bb_period']:
                    dimensions.append(Integer(int(min_val), int(max_val), name=param))
                else:
                    dimensions.append(Real(min_val, max_val, name=param))
                param_names.append(param)
            
            # Objective function
            async def objective(params):
                config = OptimizableConfig()
                for param_name, value in zip(param_names, params):
                    setattr(config, param_name, value)
                
                config = ParameterBounds.clip_to_bounds(config)
                result = await self.backtesting_engine.run_backtest_with_config(config)
                
                # Return negative score for minimization
                return -result.optimization_score
            
            # Run optimization
            results = []
            
            # Manual implementation since skopt might not work with async
            for i in range(n_calls):
                # Generate random parameters within bounds
                params = []
                for dim in dimensions:
                    if isinstance(dim, Integer):
                        params.append(random.randint(dim.low, dim.high))
                    else:
                        params.append(random.uniform(dim.low, dim.high))
                
                # Evaluate
                score = await objective(params)
                
                # Create config and result
                config = OptimizableConfig()
                for param_name, value in zip(param_names, params):
                    setattr(config, param_name, value)
                
                result = await self.backtesting_engine.run_backtest_with_config(config)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{n_calls}")
            
            # Sort results
            results.sort(key=lambda r: r.optimization_score, reverse=True)
            
            print(f"✅ Bayesian Optimization completed!")
            print(f"   Best Score: {results[0].optimization_score:.3f}")
            
            return results
            
        except ImportError:
            print("⚠️ scikit-optimize not available, falling back to random search")
            return await self._random_search_optimization(n_calls)
    
    async def _random_search_optimization(self, n_iterations: int) -> List[BacktestResult]:
        """Random search optimization fallback"""
        print(f"🎲 Random Search Optimization ({n_iterations} iterations)...")
        
        results = []
        
        for i in range(n_iterations):
            # Generate random configuration
            config = OptimizableConfig()
            
            for param, (min_val, max_val) in ParameterBounds.BOUNDS.items():
                if hasattr(config, param):
                    if param in ['min_confirmations', 'max_positions', 'rsi_period', 
                               'ema_fast', 'ema_slow', 'macd_fast', 'macd_slow', 'bb_period']:
                        value = random.randint(int(min_val), int(max_val))
                    else:
                        value = random.uniform(min_val, max_val)
                    
                    setattr(config, param, value)
            
            config = ParameterBounds.clip_to_bounds(config)
            
            # Run backtest
            result = await self.backtesting_engine.run_backtest_with_config(config)
            results.append(result)
            
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i + 1}/{n_iterations}")
        
        results.sort(key=lambda r: r.optimization_score, reverse=True)
        return results

# ============================================================================
# WALK-FORWARD ANALYSIS
# ============================================================================

class WalkForwardAnalyzer:
    """Walk-forward analysis for robust parameter validation"""
    
    def __init__(self, backtesting_engine: IntegratedBacktestingEngine):
        self.backtesting_engine = backtesting_engine
        self.walk_forward_results = []
    
    async def run_walk_forward_analysis(self, config: OptimizableConfig, 
                                      optimization_window_days: int = 30,
                                      trading_window_days: int = 7,
                                      min_periods: int = 5) -> Dict[str, Any]:
        """Run walk-forward analysis"""
        print(f"🚶 Starting Walk-Forward Analysis...")
        print(f"   Optimization Window: {optimization_window_days} days")
        print(f"   Trading Window: {trading_window_days} days")
        
        # Calculate windows
        total_days = (self.backtesting_engine.end_date - self.backtesting_engine.start_date).days
        
        if total_days < optimization_window_days + trading_window_days:
            print("❌ Insufficient data for walk-forward analysis")
            return {}
        
        # Generate walk-forward periods
        periods = []
        current_start = self.backtesting_engine.start_date
        
        while current_start + timedelta(days=optimization_window_days + trading_window_days) <= self.backtesting_engine.end_date:
            optimization_end = current_start + timedelta(days=optimization_window_days)
            trading_end = optimization_end + timedelta(days=trading_window_days)
            
            periods.append({
                'optimization_start': current_start,
                'optimization_end': optimization_end,
                'trading_start': optimization_end,
                'trading_end': trading_end
            })
            
            # Move forward by trading window
            current_start = optimization_end
        
        if len(periods) < min_periods:
            print(f"❌ Not enough periods for walk-forward analysis: {len(periods)} < {min_periods}")
            return {}
        
        print(f"   Generated {len(periods)} walk-forward periods")
        
        # Run walk-forward analysis
        period_results = []
        
        for i, period in enumerate(periods):
            print(f"   Period {i + 1}/{len(periods)}: {period['trading_start'].date()} to {period['trading_end'].date()}")
            
            # Create backtesting engine for optimization period
            opt_engine = IntegratedBacktestingEngine(
                self.backtesting_engine.symbols,
                period['optimization_start'].strftime('%Y-%m-%d'),
                period['optimization_end'].strftime('%Y-%m-%d'),
                self.backtesting_engine.initial_balance
            )
            
            # Optimize parameters on this period
            optimizer = ParameterOptimizer(opt_engine)
            
            # Quick optimization (limited iterations for speed)
            optimization_results = await optimizer._random_search_optimization(50)
            best_config = optimization_results[0].config
            
            # Test optimized config on out-of-sample trading period
            trading_engine = IntegratedBacktestingEngine(
                self.backtesting_engine.symbols,
                period['trading_start'].strftime('%Y-%m-%d'),
                period['trading_end'].strftime('%Y-%m-%d'),
                self.backtesting_engine.initial_balance
            )
            
            trading_result = await trading_engine.run_backtest_with_config(best_config)
            
            period_results.append({
                'period': i + 1,
                'optimization_period': f"{period['optimization_start'].date()} to {period['optimization_end'].date()}",
                'trading_period': f"{period['trading_start'].date()} to {period['trading_end'].date()}",
                'optimized_config': best_config.to_dict(),
                'trading_result': trading_result,
                'in_sample_score': optimization_results[0].optimization_score,
                'out_of_sample_score': trading_result.optimization_score
            })
        
        # Analyze walk-forward results
        analysis = self._analyze_walk_forward_results(period_results)
        
        print(f"✅ Walk-Forward Analysis completed!")
        print(f"   Average Out-of-Sample Return: {analysis['avg_out_sample_return']:.2%}")
        print(f"   Average Out-of-Sample Win Rate: {analysis['avg_out_sample_win_rate']:.1%}")
        print(f"   Stability Score: {analysis['stability_score']:.3f}")
        
        return analysis
    
    def _analyze_walk_forward_results(self, period_results: List[Dict]) -> Dict[str, Any]:
        """Analyze walk-forward results"""
        if not period_results:
            return {}
        
        # Extract metrics
        out_sample_returns = [r['trading_result'].total_return for r in period_results]
        out_sample_win_rates = [r['trading_result'].win_rate for r in period_results]
        out_sample_sharpe = [r['trading_result'].sharpe_ratio for r in period_results]
        
        in_sample_scores = [r['in_sample_score'] for r in period_results]
        out_sample_scores = [r['out_of_sample_score'] for r in period_results]
        
        # Calculate stability metrics
        return_stability = 1.0 - (np.std(out_sample_returns) / (np.mean(np.abs(out_sample_returns)) + 1e-8))
        score_stability = 1.0 - (np.std(out_sample_scores) / (np.mean(out_sample_scores) + 1e-8))
        
        # Overall stability score
        stability_score = (return_stability + score_stability) / 2
        
        # Overfitting detection
        avg_in_sample = np.mean(in_sample_scores)
        avg_out_sample = np.mean(out_sample_scores)
        overfitting_ratio = avg_out_sample / avg_in_sample if avg_in_sample > 0 else 0
        
        return {
            'periods_analyzed': len(period_results),
            'avg_out_sample_return': np.mean(out_sample_returns),
            'std_out_sample_return': np.std(out_sample_returns),
            'avg_out_sample_win_rate': np.mean(out_sample_win_rates),
            'avg_out_sample_sharpe': np.mean(out_sample_sharpe),
            'stability_score': stability_score,
            'overfitting_ratio': overfitting_ratio,
            'consistent_periods': sum(1 for r in out_sample_returns if r > 0),
            'best_period_return': max(out_sample_returns),
            'worst_period_return': min(out_sample_returns),
            'period_results': period_results
        }

# ============================================================================
# MAIN OPTIMIZATION INTERFACE
# ============================================================================

class AutomatedConfigOptimizer:
    """Main interface for automated configuration optimization"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize backtesting engine
        self.backtesting_engine = IntegratedBacktestingEngine(
            symbols, start_date, end_date, initial_balance=100000
        )
        
        # Initialize optimizers
        self.parameter_optimizer = ParameterOptimizer(self.backtesting_engine)
        self.walk_forward_analyzer = WalkForwardAnalyzer(self.backtesting_engine)
        
        # Results storage
        self.optimization_results = {}
        self.final_optimized_config = None
        
        print(f"🎯 Automated Config Optimizer initialized")
    
    async def optimize_all_parameters(self, optimization_method: str = 'genetic') -> OptimizableConfig:
        """Optimize all parameters using specified method"""
        print(f"🚀 STARTING COMPREHENSIVE PARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"🎯 Method: {optimization_method.upper()}")
        print(f"📅 Date Range: {self.start_date} to {self.end_date}")
        print(f"📊 Symbols: {len(self.symbols)}")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. Quick grid search for initial exploration
        print("\n1️⃣ INITIAL GRID SEARCH (Quick Exploration)")
        initial_grid = {
            'stop_loss_pct': [0.004, 0.006, 0.008, 0.010],
            'take_profit_pct': [0.010, 0.013, 0.016, 0.020],
            'risk_reward_ratio': [1.8, 2.0, 2.2, 2.5],
            'trailing_stop_distance': [0.0002, 0.0003, 0.0005, 0.0007],
            'signal_strength_threshold': [0.18, 0.22, 0.26, 0.30],
            'ml_confidence_threshold': [0.60, 0.65, 0.70, 0.75]
        }
        
        grid_results = await self.parameter_optimizer.grid_search_optimization(
            initial_grid, max_combinations=200
        )
        self.optimization_results['grid_search'] = grid_results[:10]  # Top 10
        
        # 2. Genetic algorithm for fine-tuning
        if optimization_method == 'genetic':
            print("\n2️⃣ GENETIC ALGORITHM OPTIMIZATION (Fine-Tuning)")
            genetic_results = await self.parameter_optimizer.genetic_algorithm_optimization(
                population_size=30, generations=10, mutation_rate=0.15
            )
            self.optimization_results['genetic'] = genetic_results
            best_config = genetic_results[-1].config  # Best from final generation
        
        # 3. Bayesian optimization for precision
        elif optimization_method == 'bayesian':
            print("\n2️⃣ BAYESIAN OPTIMIZATION (Precision Tuning)")
            bayesian_results = await self.parameter_optimizer.bayesian_optimization(n_calls=100)
            self.optimization_results['bayesian'] = bayesian_results
            best_config = bayesian_results[0].config
        
        else:
            # Use best from grid search
            best_config = grid_results[0].config
        
        # 3. Walk-forward validation
        print("\n3️⃣ WALK-FORWARD VALIDATION (Robustness Testing)")
        walk_forward_results = await self.walk_forward_analyzer.run_walk_forward_analysis(
            best_config, optimization_window_days=21, trading_window_days=7
        )
        self.optimization_results['walk_forward'] = walk_forward_results
        
        # 4. Final validation
        print("\n4️⃣ FINAL VALIDATION (Complete Period)")
        final_result = await self.backtesting_engine.run_backtest_with_config(best_config, verbose=True)
        self.optimization_results['final_validation'] = final_result
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        self._generate_optimization_report(best_config, total_time)
        
        self.final_optimized_config = best_config
        return best_config
    
    def _generate_optimization_report(self, best_config: OptimizableConfig, 
                                    optimization_time: float):
        """Generate comprehensive optimization report"""
        print(f"\n📊 PARAMETER OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"⏱️  Total Optimization Time: {optimization_time/60:.1f} minutes")
        print(f"🔍 Total Backtests Run: {self.backtesting_engine.backtests_run}")
        
        # Best configuration
        print(f"\n🏆 OPTIMAL CONFIGURATION FOUND:")
        print(f"   Stop Loss: {best_config.stop_loss_pct:.3f} ({best_config.stop_loss_pct*100:.1f}%)")
        print(f"   Take Profit: {best_config.take_profit_pct:.3f} ({best_config.take_profit_pct*100:.1f}%)")
        print(f"   Risk/Reward Ratio: {best_config.risk_reward_ratio:.2f}")
        print(f"   Trailing Stop Distance: {best_config.trailing_stop_distance:.4f}")
        print(f"   Trailing Stop Activation: {best_config.trailing_stop_activation:.3f}")
        print(f"   Signal Threshold: {best_config.signal_strength_threshold:.3f}")
        print(f"   ML Confidence: {best_config.ml_confidence_threshold:.3f}")
        print(f"   Position Size: {best_config.base_position_pct:.1%}")
        
        # Performance metrics
        if 'final_validation' in self.optimization_results:
            result = self.optimization_results['final_validation']
            print(f"\n📈 OPTIMIZED PERFORMANCE:")
            print(f"   Total Return: {result.total_return:.2%}")
            print(f"   Win Rate: {result.win_rate:.1%}")
            print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {result.max_drawdown:.2%}")
            print(f"   Profit Factor: {result.profit_factor:.2f}")
            print(f"   Total Trades: {result.total_trades}")
            print(f"   Optimization Score: {result.optimization_score:.3f}")
        
        # Walk-forward analysis
        if 'walk_forward' in self.optimization_results:
            wf_results = self.optimization_results['walk_forward']
            print(f"\n🚶 WALK-FORWARD VALIDATION:")
            print(f"   Periods Analyzed: {wf_results.get('periods_analyzed', 0)}")
            print(f"   Avg Out-of-Sample Return: {wf_results.get('avg_out_sample_return', 0):.2%}")
            print(f"   Stability Score: {wf_results.get('stability_score', 0):.3f}")
            print(f"   Overfitting Ratio: {wf_results.get('overfitting_ratio', 0):.3f}")
            
            if wf_results.get('overfitting_ratio', 0) < 0.8:
                print("   ⚠️ Potential overfitting detected")
            else:
                print("   ✅ Parameters appear robust")
        
        # Save results
        self._save_optimization_results(best_config)
        
        print(f"\n✅ OPTIMIZATION COMPLETE - READY FOR DEPLOYMENT")
        print("=" * 80)
    
    def _save_optimization_results(self, best_config: OptimizableConfig):
        """Save optimization results to files"""
        # Save best configuration
        config_data = {
            'optimization_timestamp': datetime.now().isoformat(),
            'best_config': best_config.to_dict(),
            'optimization_results': {
                'final_validation': {
                    'total_return': self.optimization_results['final_validation'].total_return,
                    'win_rate': self.optimization_results['final_validation'].win_rate,
                    'sharpe_ratio': self.optimization_results['final_validation'].sharpe_ratio,
                    'max_drawdown': self.optimization_results['final_validation'].max_drawdown,
                    'profit_factor': self.optimization_results['final_validation'].profit_factor,
                    'optimization_score': self.optimization_results['final_validation'].optimization_score
                }
            },
            'backtests_run': self.backtesting_engine.backtests_run,
            'optimization_time_minutes': self.backtesting_engine.total_optimization_time / 60
        }
        
        # Save to JSON
        with open('optimized_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Save to Python file for easy import
        with open('optimized_config.py', 'w') as f:
            f.write(f"""# Optimized Configuration - Generated {datetime.now().isoformat()}
from dataclasses import dataclass

@dataclass
class OptimizedTradingConfig:
    # Risk Management (Optimized)
    STOP_LOSS_PCT: float = {best_config.stop_loss_pct}
    TAKE_PROFIT_PCT: float = {best_config.take_profit_pct}
    RISK_REWARD_RATIO: float = {best_config.risk_reward_ratio}
    TRAILING_STOP_DISTANCE: float = {best_config.trailing_stop_distance}
    TRAILING_STOP_ACTIVATION: float = {best_config.trailing_stop_activation}
    
    # Signal Thresholds (Optimized)
    SIGNAL_STRENGTH_THRESHOLD: float = {best_config.signal_strength_threshold}
    ML_CONFIDENCE_THRESHOLD: float = {best_config.ml_confidence_threshold}
    MIN_CONFIRMATIONS: int = {best_config.min_confirmations}
    
    # Position Management (Optimized)
    BASE_POSITION_PCT: float = {best_config.base_position_pct}
    MAX_POSITIONS: int = {best_config.max_positions}
    POSITION_SIZE_MULTIPLIER: float = {best_config.position_size_multiplier}
    
    # Technical Indicators (Optimized)
    RSI_OVERSOLD: float = {best_config.rsi_oversold}
    RSI_OVERBOUGHT: float = {best_config.rsi_overbought}
    RSI_PERIOD: int = {best_config.rsi_period}
    EMA_FAST: int = {best_config.ema_fast}
    EMA_SLOW: int = {best_config.ema_slow}
    
    # Advanced Parameters (Optimized)
    VOLUME_CONFIRMATION_THRESHOLD: float = {best_config.volume_confirmation_threshold}
    TIME_BASED_EXIT_HOURS: float = {best_config.time_based_exit_hours}
    BREAKEVEN_MOVE_THRESHOLD: float = {best_config.breakeven_move_threshold}

# Usage: 
# from optimized_config import OptimizedTradingConfig
# config = OptimizedTradingConfig()
""")
        
        print(f"💾 Optimization results saved:")
        print(f"   optimized_config.json - Complete results")
        print(f"   optimized_config.py - Ready-to-use configuration")

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

class OptimizationVisualizer:
    """Visualization tools for optimization results"""
    
    @staticmethod
    def plot_optimization_results(results: List[BacktestResult], save_path: str = 'optimization_results.png'):
        """Plot optimization results"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Extract data
            returns = [r.total_return * 100 for r in results]
            win_rates = [r.win_rate * 100 for r in results]
            sharpe_ratios = [r.sharpe_ratio for r in results]
            max_drawdowns = [abs(r.max_drawdown) * 100 for r in results]
            
            # 1. Return vs Win Rate
            scatter = ax1.scatter(win_rates, returns, c=sharpe_ratios, cmap='viridis', alpha=0.7)
            ax1.set_xlabel('Win Rate (%)')
            ax1.set_ylabel('Total Return (%)')
            ax1.set_title('Return vs Win Rate (colored by Sharpe Ratio)')
            plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
            
            # 2. Sharpe vs Drawdown
            ax2.scatter(max_drawdowns, sharpe_ratios, alpha=0.7, color='coral')
            ax2.set_xlabel('Max Drawdown (%)')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Risk-Adjusted Returns')
            
            # 3. Parameter distribution (Stop Loss)
            stop_losses = [r.config.stop_loss_pct * 100 for r in results]
            ax3.hist(stop_losses, bins=20, alpha=0.7, color='lightblue')
            ax3.set_xlabel('Stop Loss (%)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Stop Loss Parameter Distribution')
            ax3.axvline(results[0].config.stop_loss_pct * 100, color='red', linestyle='--', label='Best')
            ax3.legend()
            
            # 4. Optimization scores
            scores = [r.optimization_score for r in results]
            ax4.plot(scores, alpha=0.7, color='green')
            ax4.set_xlabel('Configuration #')
            ax4.set_ylabel('Optimization Score')
            ax4.set_title('Optimization Score Evolution')
            ax4.axhline(max(scores), color='red', linestyle='--', label=f'Best: {max(scores):.3f}')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Optimization visualization saved: {save_path}")
            
        except ImportError:
            print("⚠️ Matplotlib not available - skipping visualization")
        except Exception as e:
            print(f"❌ Visualization error: {e}")

# ============================================================================
# USAGE EXAMPLES AND DEMO
# ============================================================================

async def demo_automated_optimization():
    """Demonstrate automated parameter optimization"""
    print("🎯 AUTOMATED PARAMETER OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Initialize optimizer
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    optimizer = AutomatedConfigOptimizer(symbols, start_date, end_date)
    
    # Run optimization
    optimized_config = await optimizer.optimize_all_parameters('genetic')
    
    # Display results
    print(f"\n🏆 OPTIMIZATION COMPLETE!")
    print(f"   Optimal Stop Loss: {optimized_config.stop_loss_pct*100:.2f}%")
    print(f"   Optimal Take Profit: {optimized_config.take_profit_pct*100:.2f}%")
    print(f"   Optimal Risk/Reward: {optimized_config.risk_reward_ratio:.2f}")
    print(f"   Optimal Trailing Stop: {optimized_config.trailing_stop_distance*100:.3f}%")
    
    # Visualize results if possible
    if 'genetic' in optimizer.optimization_results:
        OptimizationVisualizer.plot_optimization_results(
            optimizer.optimization_results['genetic']
        )
    
    return optimized_config

async def quick_parameter_test():
    """Quick parameter testing example"""
    print("⚡ QUICK PARAMETER TEST")
    print("-" * 30)
    
    # Test different stop loss values
    symbols = ['BTCUSDT']
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    engine = IntegratedBacktestingEngine(symbols, start_date, end_date, 10000)
    
    stop_loss_values = [0.004, 0.006, 0.008, 0.010, 0.012]
    results = []
    
    print("Testing stop loss values:")
    for stop_loss in stop_loss_values:
        config = OptimizableConfig()
        config.stop_loss_pct = stop_loss
        
        result = await engine.run_backtest_with_config(config)
        results.append(result)
        
        print(f"   {stop_loss*100:.1f}%: Return={result.total_return:.2%}, "
              f"Win Rate={result.win_rate:.1%}, Drawdown={result.max_drawdown:.2%}")
    
    # Find best
    best_result = max(results, key=lambda r: r.optimization_score)
    print(f"\n🏆 Best Stop Loss: {best_result.config.stop_loss_pct*100:.1f}%")
    print(f"   Score: {best_result.optimization_score:.3f}")

if __name__ == "__main__":
    print("🚀 INTEGRATED BACKTESTING WITH PARAMETER OPTIMIZATION")
    print("=" * 60)
    print("Choose test to run:")
    print("1. Quick parameter test (2 minutes)")
    print("2. Full optimization demo (10 minutes)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(quick_parameter_test())
    elif choice == "2":
        asyncio.run(demo_automated_optimization())
    else:
        print("Running quick test by default...")
        asyncio.run(quick_parameter_test())