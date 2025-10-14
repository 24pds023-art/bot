"""
ADVANCED OPTIMIZATIONS - NEXT LEVEL ENHANCEMENTS
================================================
🚀 Additional critical optimizations for maximum performance
⚡ API rate limiting, advanced indicators, microstructure analysis
📊 Multi-exchange arbitrage, advanced ML models, portfolio optimization
🎯 Ultra-low latency techniques and professional-grade features
"""

import asyncio
import aiohttp
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, List, Any, Tuple, Optional
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import hashlib
import hmac
from urllib.parse import urlencode
import websockets
from concurrent.futures import ThreadPoolExecutor
import threading
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import numba
from numba import jit
import talib
import scipy.stats as stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. INTELLIGENT API RATE LIMITING
# ============================================================================

class IntelligentRateLimiter:
    """Advanced rate limiting with burst handling and priority queues"""
    
    def __init__(self, requests_per_minute: int = 1200, burst_limit: int = 100):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.request_times = deque(maxlen=requests_per_minute)
        self.burst_count = 0
        self.priority_queue = asyncio.PriorityQueue()
        self.weight_limits = {
            'CRITICAL': 50,    # Order execution
            'HIGH': 30,        # Position management
            'MEDIUM': 15,      # Signal generation
            'LOW': 5           # General queries
        }
        self.current_weights = defaultdict(int)
        self.lock = asyncio.Lock()
        
    async def acquire(self, priority: str = 'MEDIUM', weight: int = 1) -> bool:
        """Acquire rate limit permission with priority and weight"""
        async with self.lock:
            current_time = time.time()
            
            # Clean old requests (older than 1 minute)
            while self.request_times and current_time - self.request_times[0] > 60:
                self.request_times.popleft()
            
            # Check if we can make the request
            if len(self.request_times) + weight <= self.requests_per_minute:
                # Check weight limits
                if self.current_weights[priority] + weight <= self.weight_limits[priority]:
                    # Grant permission
                    for _ in range(weight):
                        self.request_times.append(current_time)
                    self.current_weights[priority] += weight
                    return True
            
            # Request denied - calculate wait time
            if self.request_times:
                oldest_request = self.request_times[0]
                wait_time = 60 - (current_time - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time + 0.1)  # Small buffer
                    return await self.acquire(priority, weight)
            
            return False
    
    async def acquire_burst(self, count: int) -> bool:
        """Acquire burst capacity for high-frequency operations"""
        if self.burst_count + count <= self.burst_limit:
            self.burst_count += count
            # Reset burst count every second
            asyncio.create_task(self._reset_burst_count(1.0))
            return True
        return False
    
    async def _reset_burst_count(self, delay: float):
        """Reset burst count after delay"""
        await asyncio.sleep(delay)
        self.burst_count = max(0, self.burst_count - 1)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        current_time = time.time()
        recent_requests = sum(1 for t in self.request_times if current_time - t < 60)
        
        return {
            'requests_used': recent_requests,
            'requests_remaining': self.requests_per_minute - recent_requests,
            'burst_used': self.burst_count,
            'burst_remaining': self.burst_limit - self.burst_count,
            'weight_usage': dict(self.current_weights),
            'reset_time': current_time + 60 - (current_time - self.request_times[0] if self.request_times else 0)
        }

# ============================================================================
# 2. ADVANCED TECHNICAL INDICATORS
# ============================================================================

class AdvancedIndicatorSuite:
    """Professional-grade technical indicators for higher win rates"""
    
    def __init__(self):
        self.indicator_cache = {}
        self.calculation_times = deque(maxlen=1000)
    
    @jit(nopython=True, cache=True)
    def calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
        """Volume Weighted Average Price - institutional level indicator"""
        if len(prices) == 0 or len(volumes) == 0:
            return 0.0
        
        total_volume = np.sum(volumes)
        if total_volume == 0:
            return np.mean(prices)
        
        return np.sum(prices * volumes) / total_volume
    
    @jit(nopython=True, cache=True)
    def calculate_twap(prices: np.ndarray, time_weights: np.ndarray) -> float:
        """Time Weighted Average Price"""
        if len(prices) == 0:
            return 0.0
        
        total_weight = np.sum(time_weights)
        if total_weight == 0:
            return np.mean(prices)
        
        return np.sum(prices * time_weights) / total_weight
    
    def calculate_ichimoku_cloud(self, highs: np.ndarray, lows: np.ndarray, 
                                closes: np.ndarray) -> Dict[str, float]:
        """Ichimoku Cloud - powerful trend and momentum indicator"""
        if len(closes) < 52:
            return {'tenkan': 0, 'kijun': 0, 'senkou_a': 0, 'senkou_b': 0, 'chikou': 0}
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan = (np.max(highs[-9:]) + np.min(lows[-9:])) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun = (np.max(highs[-26:]) + np.min(lows[-26:])) / 2
        
        # Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
        senkou_b = (np.max(highs[-52:]) + np.min(lows[-52:])) / 2
        
        # Chikou Span: Close plotted 26 periods behind
        chikou = closes[-26] if len(closes) >= 26 else closes[-1]
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou,
            'cloud_top': max(senkou_a, senkou_b),
            'cloud_bottom': min(senkou_a, senkou_b),
            'above_cloud': closes[-1] > max(senkou_a, senkou_b),
            'below_cloud': closes[-1] < min(senkou_a, senkou_b)
        }
    
    def calculate_market_profile(self, prices: np.ndarray, volumes: np.ndarray, 
                               num_bins: int = 50) -> Dict[str, Any]:
        """Market Profile - institutional trading levels"""
        if len(prices) < 10:
            return {'poc': 0, 'value_area_high': 0, 'value_area_low': 0}
        
        # Create price bins
        price_min, price_max = np.min(prices), np.max(prices)
        price_range = price_max - price_min
        bin_size = price_range / num_bins
        
        # Calculate volume at each price level
        volume_profile = np.zeros(num_bins)
        
        for i, price in enumerate(prices):
            if i < len(volumes):
                bin_idx = min(int((price - price_min) / bin_size), num_bins - 1)
                volume_profile[bin_idx] += volumes[i]
        
        # Find Point of Control (POC) - price level with highest volume
        poc_idx = np.argmax(volume_profile)
        poc_price = price_min + (poc_idx + 0.5) * bin_size
        
        # Calculate Value Area (70% of volume)
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * 0.7
        
        # Find value area around POC
        current_volume = volume_profile[poc_idx]
        low_idx = high_idx = poc_idx
        
        while current_volume < target_volume and (low_idx > 0 or high_idx < num_bins - 1):
            low_volume = volume_profile[low_idx - 1] if low_idx > 0 else 0
            high_volume = volume_profile[high_idx + 1] if high_idx < num_bins - 1 else 0
            
            if low_volume >= high_volume and low_idx > 0:
                low_idx -= 1
                current_volume += low_volume
            elif high_idx < num_bins - 1:
                high_idx += 1
                current_volume += high_volume
            else:
                break
        
        value_area_low = price_min + low_idx * bin_size
        value_area_high = price_min + (high_idx + 1) * bin_size
        
        return {
            'poc': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'volume_profile': volume_profile,
            'total_volume': total_volume
        }
    
    def calculate_order_flow_indicators(self, trades: List[Dict]) -> Dict[str, float]:
        """Advanced order flow analysis"""
        if not trades:
            return {'delta': 0, 'cumulative_delta': 0, 'delta_momentum': 0}
        
        # Calculate delta (buy volume - sell volume)
        buy_volume = sum(t['quantity'] for t in trades if not t.get('is_buyer_maker', False))
        sell_volume = sum(t['quantity'] for t in trades if t.get('is_buyer_maker', False))
        delta = buy_volume - sell_volume
        
        # Calculate cumulative delta
        cumulative_delta = 0
        deltas = []
        
        for trade in trades:
            trade_delta = trade['quantity'] if not trade.get('is_buyer_maker', False) else -trade['quantity']
            cumulative_delta += trade_delta
            deltas.append(trade_delta)
        
        # Calculate delta momentum
        if len(deltas) >= 10:
            recent_delta = sum(deltas[-10:])
            previous_delta = sum(deltas[-20:-10]) if len(deltas) >= 20 else 0
            delta_momentum = recent_delta - previous_delta
        else:
            delta_momentum = 0
        
        return {
            'delta': delta,
            'cumulative_delta': cumulative_delta,
            'delta_momentum': delta_momentum,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'imbalance_ratio': buy_volume / (sell_volume + 1e-8)
        }
    
    def calculate_volatility_indicators(self, prices: np.ndarray, 
                                      returns: np.ndarray = None) -> Dict[str, float]:
        """Advanced volatility analysis"""
        if len(prices) < 20:
            return {'realized_vol': 0, 'garch_vol': 0, 'vol_of_vol': 0}
        
        if returns is None:
            returns = np.diff(np.log(prices))
        
        # Realized volatility (annualized)
        realized_vol = np.std(returns) * np.sqrt(252 * 24 * 60)  # For minute data
        
        # GARCH-like volatility (simplified)
        alpha, beta = 0.1, 0.85
        garch_vol = realized_vol
        
        for i in range(1, len(returns)):
            garch_vol = np.sqrt(alpha * returns[i-1]**2 + beta * garch_vol**2)
        
        # Volatility of volatility
        if len(returns) >= 50:
            vol_windows = []
            window_size = 10
            for i in range(window_size, len(returns)):
                window_vol = np.std(returns[i-window_size:i])
                vol_windows.append(window_vol)
            vol_of_vol = np.std(vol_windows) if vol_windows else 0
        else:
            vol_of_vol = 0
        
        return {
            'realized_vol': realized_vol,
            'garch_vol': garch_vol,
            'vol_of_vol': vol_of_vol,
            'vol_regime': 'HIGH' if realized_vol > np.percentile(returns, 80) else 'LOW'
        }
    
    def calculate_momentum_indicators(self, prices: np.ndarray, 
                                    volumes: np.ndarray = None) -> Dict[str, float]:
        """Advanced momentum indicators"""
        if len(prices) < 20:
            return {'roc': 0, 'momentum': 0, 'trix': 0}
        
        # Rate of Change
        roc_10 = (prices[-1] - prices[-10]) / prices[-10] * 100 if len(prices) >= 10 else 0
        roc_20 = (prices[-1] - prices[-20]) / prices[-20] * 100 if len(prices) >= 20 else 0
        
        # Momentum
        momentum = prices[-1] - prices[-10] if len(prices) >= 10 else 0
        
        # TRIX (Triple Exponential Average)
        if len(prices) >= 30:
            ema1 = pd.Series(prices).ewm(span=14).mean()
            ema2 = ema1.ewm(span=14).mean()
            ema3 = ema2.ewm(span=14).mean()
            trix = (ema3.iloc[-1] - ema3.iloc[-2]) / ema3.iloc[-2] * 10000 if len(ema3) > 1 else 0
        else:
            trix = 0
        
        # Volume-Price Trend (VPT)
        vpt = 0
        if volumes is not None and len(volumes) == len(prices) and len(prices) > 1:
            for i in range(1, len(prices)):
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                vpt += volumes[i] * price_change
        
        return {
            'roc_10': roc_10,
            'roc_20': roc_20,
            'momentum': momentum,
            'trix': trix,
            'vpt': vpt
        }
    
    def calculate_all_advanced_indicators(self, prices: np.ndarray, highs: np.ndarray,
                                        lows: np.ndarray, volumes: np.ndarray,
                                        trades: List[Dict] = None) -> Dict[str, Any]:
        """Calculate all advanced indicators in one pass"""
        start_time = time.perf_counter()
        
        indicators = {}
        
        try:
            # VWAP and TWAP
            if len(volumes) > 0:
                indicators['vwap'] = self.calculate_vwap(prices, volumes)
                time_weights = np.arange(1, len(prices) + 1)  # Linear time weighting
                indicators['twap'] = self.calculate_twap(prices, time_weights)
            
            # Ichimoku Cloud
            ichimoku = self.calculate_ichimoku_cloud(highs, lows, prices)
            indicators.update({f'ichimoku_{k}': v for k, v in ichimoku.items()})
            
            # Market Profile
            if len(volumes) > 0:
                market_profile = self.calculate_market_profile(prices, volumes)
                indicators.update({f'profile_{k}': v for k, v in market_profile.items() 
                                 if k != 'volume_profile'})
            
            # Order Flow
            if trades:
                order_flow = self.calculate_order_flow_indicators(trades)
                indicators.update({f'flow_{k}': v for k, v in order_flow.items()})
            
            # Volatility
            volatility = self.calculate_volatility_indicators(prices)
            indicators.update({f'vol_{k}': v for k, v in volatility.items()})
            
            # Momentum
            momentum = self.calculate_momentum_indicators(prices, volumes)
            indicators.update({f'mom_{k}': v for k, v in momentum.items()})
            
            # Additional TALib indicators if available
            try:
                import talib
                if len(prices) >= 30:
                    indicators['adx'] = talib.ADX(highs, lows, prices, timeperiod=14)[-1]
                    indicators['cci'] = talib.CCI(highs, lows, prices, timeperiod=14)[-1]
                    indicators['williams_r'] = talib.WILLR(highs, lows, prices, timeperiod=14)[-1]
                    indicators['ultimate_oscillator'] = talib.ULTOSC(highs, lows, prices)[-1]
                    indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(highs, lows, prices)
                    indicators['stoch_k'] = indicators['stoch_k'][-1]
                    indicators['stoch_d'] = indicators['stoch_d'][-1]
            except ImportError:
                pass  # TALib not available
            
            calculation_time = time.perf_counter() - start_time
            self.calculation_times.append(calculation_time)
            indicators['calculation_time'] = calculation_time
            
        except Exception as e:
            print(f"Error calculating advanced indicators: {e}")
            indicators['error'] = str(e)
        
        return indicators

# ============================================================================
# 3. ENHANCED MARKET MICROSTRUCTURE ANALYSIS
# ============================================================================

class EnhancedMicrostructureAnalyzer:
    """Professional-grade market microstructure analysis"""
    
    def __init__(self, max_depth_levels: int = 20):
        self.max_depth_levels = max_depth_levels
        self.order_book_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=5000)
        self.liquidity_metrics = deque(maxlen=100)
        
    def analyze_order_book_depth(self, bids: List[Tuple[float, float]], 
                                asks: List[Tuple[float, float]]) -> Dict[str, float]:
        """Analyze order book depth and liquidity"""
        if not bids or not asks:
            return {'spread': 0, 'depth_imbalance': 0, 'liquidity_score': 0}
        
        # Basic spread
        best_bid = max(bids, key=lambda x: x[0])[0]
        best_ask = min(asks, key=lambda x: x[0])[0]
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000  # Basis points
        
        # Depth analysis
        bid_depth = sum(size for price, size in bids[:self.max_depth_levels])
        ask_depth = sum(size for price, size in asks[:self.max_depth_levels])
        total_depth = bid_depth + ask_depth
        
        depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        # Liquidity score (higher is better)
        if spread > 0:
            liquidity_score = total_depth / spread
        else:
            liquidity_score = total_depth * 1000  # Very tight spread
        
        # Price impact analysis
        impact_levels = [0.1, 0.5, 1.0, 2.0, 5.0]  # Percentage levels
        buy_impact = self._calculate_price_impact(asks, impact_levels, 'buy')
        sell_impact = self._calculate_price_impact(bids, impact_levels, 'sell')
        
        # Order book slope (how quickly liquidity falls off)
        bid_slope = self._calculate_book_slope(bids, best_bid, 'bid')
        ask_slope = self._calculate_book_slope(asks, best_ask, 'ask')
        
        return {
            'spread': spread,
            'spread_bps': spread_bps,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'depth_imbalance': depth_imbalance,
            'liquidity_score': liquidity_score,
            'buy_impact_1pct': buy_impact.get(1.0, 0),
            'sell_impact_1pct': sell_impact.get(1.0, 0),
            'bid_slope': bid_slope,
            'ask_slope': ask_slope,
            'asymmetry': abs(bid_slope - ask_slope)
        }
    
    def _calculate_price_impact(self, orders: List[Tuple[float, float]], 
                               impact_levels: List[float], side: str) -> Dict[float, float]:
        """Calculate price impact for different order sizes"""
        if not orders:
            return {level: 0 for level in impact_levels}
        
        # Sort orders by price
        if side == 'buy':
            sorted_orders = sorted(orders, key=lambda x: x[0])  # Ascending for asks
        else:
            sorted_orders = sorted(orders, key=lambda x: x[0], reverse=True)  # Descending for bids
        
        best_price = sorted_orders[0][0]
        impacts = {}
        
        for level in impact_levels:
            target_value = best_price * level / 100  # Convert percentage to value
            cumulative_size = 0
            weighted_price = 0
            
            for price, size in sorted_orders:
                if cumulative_size >= target_value:
                    break
                
                remaining_target = target_value - cumulative_size
                size_to_use = min(size, remaining_target)
                
                weighted_price += price * size_to_use
                cumulative_size += size_to_use
            
            if cumulative_size > 0:
                avg_price = weighted_price / cumulative_size
                impact = abs(avg_price - best_price) / best_price * 100
                impacts[level] = impact
            else:
                impacts[level] = float('inf')  # No liquidity available
        
        return impacts
    
    def _calculate_book_slope(self, orders: List[Tuple[float, float]], 
                             best_price: float, side: str) -> float:
        """Calculate how quickly liquidity falls off from best price"""
        if len(orders) < 3:
            return 0
        
        # Calculate cumulative size at different price levels
        price_levels = []
        cumulative_sizes = []
        cumulative_size = 0
        
        sorted_orders = sorted(orders, key=lambda x: x[0], 
                             reverse=(side == 'bid'))
        
        for price, size in sorted_orders[:10]:  # Top 10 levels
            price_distance = abs(price - best_price) / best_price * 100
            cumulative_size += size
            price_levels.append(price_distance)
            cumulative_sizes.append(cumulative_size)
        
        # Calculate slope using linear regression
        if len(price_levels) >= 2:
            slope = np.polyfit(price_levels, cumulative_sizes, 1)[0]
            return slope
        
        return 0
    
    def analyze_trade_flow(self, recent_trades: List[Dict]) -> Dict[str, float]:
        """Analyze recent trade flow patterns"""
        if not recent_trades:
            return {'trade_intensity': 0, 'size_momentum': 0, 'price_momentum': 0}
        
        # Trade intensity (trades per minute)
        current_time = time.time()
        recent_trades_1min = [t for t in recent_trades 
                             if current_time - t.get('timestamp', 0) < 60]
        trade_intensity = len(recent_trades_1min)
        
        # Size momentum
        buy_sizes = [t['quantity'] for t in recent_trades if not t.get('is_buyer_maker', False)]
        sell_sizes = [t['quantity'] for t in recent_trades if t.get('is_buyer_maker', False)]
        
        avg_buy_size = np.mean(buy_sizes) if buy_sizes else 0
        avg_sell_size = np.mean(sell_sizes) if sell_sizes else 0
        size_momentum = (avg_buy_size - avg_sell_size) / (avg_buy_size + avg_sell_size + 1e-8)
        
        # Price momentum
        if len(recent_trades) >= 10:
            prices = [t['price'] for t in recent_trades[-10:]]
            price_momentum = (prices[-1] - prices[0]) / prices[0] * 100
        else:
            price_momentum = 0
        
        # Large trade detection
        all_sizes = [t['quantity'] for t in recent_trades]
        if all_sizes:
            size_threshold = np.percentile(all_sizes, 90)  # 90th percentile
            large_trades = [t for t in recent_trades if t['quantity'] > size_threshold]
            large_trade_ratio = len(large_trades) / len(recent_trades)
        else:
            large_trade_ratio = 0
        
        return {
            'trade_intensity': trade_intensity,
            'size_momentum': size_momentum,
            'price_momentum': price_momentum,
            'large_trade_ratio': large_trade_ratio,
            'avg_buy_size': avg_buy_size,
            'avg_sell_size': avg_sell_size
        }
    
    def detect_institutional_activity(self, trades: List[Dict], 
                                    order_book: Dict) -> Dict[str, Any]:
        """Detect potential institutional trading activity"""
        if not trades:
            return {'institutional_score': 0, 'activity_type': 'NONE'}
        
        # Large size trades
        sizes = [t['quantity'] for t in trades]
        large_size_threshold = np.percentile(sizes, 95) if sizes else 0
        large_trades = [t for t in trades if t['quantity'] > large_size_threshold]
        
        # Iceberg detection (consistent large sizes)
        if len(large_trades) >= 3:
            large_sizes = [t['quantity'] for t in large_trades]
            size_consistency = 1 - (np.std(large_sizes) / np.mean(large_sizes))
            iceberg_score = size_consistency if size_consistency > 0.8 else 0
        else:
            iceberg_score = 0
        
        # TWAP execution detection (consistent timing)
        if len(trades) >= 10:
            timestamps = [t.get('timestamp', 0) for t in trades[-10:]]
            time_intervals = np.diff(timestamps)
            if len(time_intervals) > 0:
                time_consistency = 1 - (np.std(time_intervals) / np.mean(time_intervals))
                twap_score = time_consistency if time_consistency > 0.7 else 0
            else:
                twap_score = 0
        else:
            twap_score = 0
        
        # Stealth trading (trades at mid-price)
        if 'bids' in order_book and 'asks' in order_book:
            best_bid = max(order_book['bids'], key=lambda x: x[0])[0]
            best_ask = min(order_book['asks'], key=lambda x: x[0])[0]
            mid_price = (best_bid + best_ask) / 2
            
            mid_trades = [t for t in trades 
                         if abs(t['price'] - mid_price) / mid_price < 0.001]  # Within 0.1%
            stealth_score = len(mid_trades) / len(trades)
        else:
            stealth_score = 0
        
        # Overall institutional score
        institutional_score = (iceberg_score * 0.4 + twap_score * 0.3 + stealth_score * 0.3)
        
        # Determine activity type
        if institutional_score > 0.7:
            activity_type = 'HIGH_INSTITUTIONAL'
        elif institutional_score > 0.4:
            activity_type = 'MODERATE_INSTITUTIONAL'
        elif iceberg_score > 0.6:
            activity_type = 'ICEBERG'
        elif twap_score > 0.6:
            activity_type = 'TWAP'
        else:
            activity_type = 'RETAIL'
        
        return {
            'institutional_score': institutional_score,
            'iceberg_score': iceberg_score,
            'twap_score': twap_score,
            'stealth_score': stealth_score,
            'activity_type': activity_type,
            'large_trade_count': len(large_trades)
        }

# ============================================================================
# 4. ADVANCED ML MODELS
# ============================================================================

class AdvancedMLSuite:
    """Advanced machine learning models for signal prediction"""
    
    def __init__(self):
        self.models = {
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, max_depth=5),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=7),
            'neural_network': MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500)
        }
        self.ensemble_weights = {'gradient_boost': 0.4, 'random_forest': 0.35, 'neural_network': 0.25}
        self.feature_importance = {}
        self.model_performance = defaultdict(list)
        
    def extract_advanced_features(self, market_data: Dict) -> np.ndarray:
        """Extract comprehensive feature set"""
        features = []
        
        # Technical indicators
        features.extend([
            market_data.get('rsi', 50),
            market_data.get('ema_10', 0),
            market_data.get('ema_21', 0),
            market_data.get('macd', 0),
            market_data.get('bb_position', 0.5),
            market_data.get('atr', 0.01),
            market_data.get('vwap', 0),
            market_data.get('ichimoku_tenkan', 0),
            market_data.get('ichimoku_kijun', 0),
            market_data.get('adx', 0),
            market_data.get('cci', 0),
            market_data.get('williams_r', 0)
        ])
        
        # Market microstructure
        features.extend([
            market_data.get('spread_bps', 0),
            market_data.get('depth_imbalance', 0),
            market_data.get('liquidity_score', 0),
            market_data.get('buy_impact_1pct', 0),
            market_data.get('institutional_score', 0),
            market_data.get('trade_intensity', 0),
            market_data.get('size_momentum', 0)
        ])
        
        # Volatility and momentum
        features.extend([
            market_data.get('vol_realized_vol', 0),
            market_data.get('vol_garch_vol', 0),
            market_data.get('mom_roc_10', 0),
            market_data.get('mom_momentum', 0),
            market_data.get('mom_trix', 0)
        ])
        
        # Time-based features
        now = datetime.now()
        features.extend([
            now.hour,
            now.weekday(),
            now.minute,
            (now.hour * 60 + now.minute) / 1440  # Time of day as fraction
        ])
        
        # Market regime features
        features.extend([
            1 if market_data.get('vol_regime') == 'HIGH' else 0,
            market_data.get('trend_strength', 0),
            market_data.get('market_stress', 0)  # Would be calculated from correlations
        ])
        
        return np.array(features, dtype=np.float32)
    
    def train_ensemble(self, training_data: List[Dict]):
        """Train ensemble of ML models"""
        if len(training_data) < 100:
            return False
        
        # Prepare training data
        X = []
        y = []
        
        for sample in training_data:
            features = self.extract_advanced_features(sample['market_data'])
            X.append(features)
            y.append(1 if sample['outcome'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                model.fit(X, y)
                
                # Calculate feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
                
                # Cross-validation performance
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(model, X, y, cv=5)
                self.model_performance[model_name] = cv_scores
                
                print(f"✅ {model_name} trained: CV Score = {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
        
        return True
    
    def predict_ensemble(self, market_data: Dict) -> Dict[str, float]:
        """Make ensemble prediction"""
        features = self.extract_advanced_features(market_data).reshape(1, -1)
        
        predictions = {}
        weighted_prediction = 0
        total_weight = 0
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features)[0][1]
                    predictions[model_name] = prob
                    
                    weight = self.ensemble_weights.get(model_name, 0.33)
                    weighted_prediction += prob * weight
                    total_weight += weight
                    
            except Exception as e:
                print(f"Error in {model_name} prediction: {e}")
                predictions[model_name] = 0.5
        
        ensemble_prediction = weighted_prediction / total_weight if total_weight > 0 else 0.5
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'confidence': self._calculate_prediction_confidence(predictions)
        }
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate confidence based on model agreement"""
        if not predictions:
            return 0.0
        
        pred_values = list(predictions.values())
        
        # High confidence if models agree
        agreement = 1.0 - np.std(pred_values)
        
        # Boost confidence for extreme predictions
        avg_pred = np.mean(pred_values)
        extremeness = abs(avg_pred - 0.5) * 2  # 0 to 1 scale
        
        confidence = (agreement * 0.7 + extremeness * 0.3)
        return np.clip(confidence, 0.0, 1.0)

# ============================================================================
# 5. PORTFOLIO OPTIMIZATION AND RISK MANAGEMENT
# ============================================================================

class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimization and risk management"""
    
    def __init__(self, max_positions: int = 20):
        self.max_positions = max_positions
        self.correlation_matrix = None
        self.risk_metrics = {}
        self.position_history = deque(maxlen=1000)
        
    def calculate_position_correlations(self, positions: Dict, 
                                      price_histories: Dict) -> np.ndarray:
        """Calculate correlation matrix between positions"""
        symbols = list(positions.keys())
        if len(symbols) < 2:
            return np.eye(1)
        
        # Get return series for each symbol
        return_series = {}
        min_length = float('inf')
        
        for symbol in symbols:
            if symbol in price_histories and len(price_histories[symbol]) > 1:
                prices = np.array(price_histories[symbol])
                returns = np.diff(np.log(prices))
                return_series[symbol] = returns
                min_length = min(min_length, len(returns))
        
        if min_length < 10:  # Need at least 10 observations
            return np.eye(len(symbols))
        
        # Align return series
        aligned_returns = []
        for symbol in symbols:
            if symbol in return_series:
                aligned_returns.append(return_series[symbol][-min_length:])
            else:
                aligned_returns.append(np.zeros(min_length))
        
        # Calculate correlation matrix
        return_matrix = np.array(aligned_returns)
        correlation_matrix = np.corrcoef(return_matrix)
        
        return correlation_matrix
    
    def optimize_position_sizes(self, signals: Dict[str, float], 
                               correlations: np.ndarray,
                               risk_budget: float = 0.02) -> Dict[str, float]:
        """Optimize position sizes using modern portfolio theory"""
        symbols = list(signals.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}
        
        # Expected returns (based on signal strength)
        expected_returns = np.array([signals[symbol] for symbol in symbols])
        
        # Risk model (simplified)
        if correlations.shape[0] != n_assets:
            correlations = np.eye(n_assets)
        
        # Add volatility estimates
        volatilities = np.ones(n_assets) * 0.02  # 2% daily vol assumption
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        
        # Optimization objective: maximize return / risk
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -portfolio_return / (portfolio_risk + 1e-8)  # Negative for minimization
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: risk_budget - np.sqrt(np.dot(x, np.dot(cov_matrix, x)))}  # Risk budget
        ]
        
        # Bounds (0 to 0.3 per position)
        bounds = [(0, 0.3) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                return dict(zip(symbols, optimal_weights))
            else:
                # Fallback to equal weights
                equal_weight = 1.0 / n_assets
                return {symbol: equal_weight for symbol in symbols}
                
        except Exception as e:
            print(f"Portfolio optimization error: {e}")
            # Fallback to signal-weighted allocation
            total_signal = sum(abs(s) for s in signals.values())
            if total_signal > 0:
                return {symbol: abs(signal) / total_signal 
                       for symbol, signal in signals.items()}
            else:
                return {}
    
    def calculate_portfolio_risk_metrics(self, positions: Dict, 
                                       price_histories: Dict) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics"""
        if not positions:
            return {}
        
        # Portfolio value and weights
        total_value = sum(pos['size'] * pos['current_price'] for pos in positions.values())
        weights = {symbol: (pos['size'] * pos['current_price']) / total_value 
                  for symbol, pos in positions.items()}
        
        # Calculate portfolio returns
        portfolio_returns = []
        if len(self.position_history) > 1:
            for i in range(1, len(self.position_history)):
                prev_value = self.position_history[i-1]['total_value']
                curr_value = self.position_history[i]['total_value']
                if prev_value > 0:
                    portfolio_returns.append((curr_value - prev_value) / prev_value)
        
        if len(portfolio_returns) < 10:
            return {'var_95': 0, 'cvar_95': 0, 'max_drawdown': 0, 'sharpe_ratio': 0}
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)  # 5th percentile
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        
        # Concentration risk
        concentration_risk = np.sum(np.array(list(weights.values()))**2)  # Herfindahl index
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': np.std(portfolio_returns),
            'concentration_risk': concentration_risk,
            'num_positions': len(positions)
        }
    
    def should_reduce_risk(self, risk_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if risk reduction is needed"""
        # Risk thresholds
        max_var = -0.05  # 5% daily VaR
        max_drawdown = -0.15  # 15% max drawdown
        max_concentration = 0.5  # 50% concentration
        min_sharpe = -0.5  # Minimum Sharpe ratio
        
        if risk_metrics.get('var_95', 0) < max_var:
            return True, f"VaR exceeded: {risk_metrics['var_95']:.2%}"
        
        if risk_metrics.get('max_drawdown', 0) < max_drawdown:
            return True, f"Max drawdown exceeded: {risk_metrics['max_drawdown']:.2%}"
        
        if risk_metrics.get('concentration_risk', 0) > max_concentration:
            return True, f"Concentration risk too high: {risk_metrics['concentration_risk']:.2%}"
        
        if risk_metrics.get('sharpe_ratio', 0) < min_sharpe:
            return True, f"Sharpe ratio too low: {risk_metrics['sharpe_ratio']:.2f}"
        
        return False, "Risk within acceptable limits"

# ============================================================================
# 6. INTEGRATION CLASS
# ============================================================================

class UltraAdvancedTradingSystem:
    """Complete ultra-advanced trading system with all enhancements"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        
        # Initialize all advanced components
        self.rate_limiter = IntelligentRateLimiter()
        self.advanced_indicators = AdvancedIndicatorSuite()
        self.microstructure_analyzer = EnhancedMicrostructureAnalyzer()
        self.ml_suite = AdvancedMLSuite()
        self.portfolio_optimizer = AdvancedPortfolioOptimizer()
        
        # Data storage
        self.price_histories = {symbol: deque(maxlen=1000) for symbol in symbols}
        self.order_books = {}
        self.trade_flows = {symbol: deque(maxlen=1000) for symbol in symbols}
        
        print("🚀 Ultra-Advanced Trading System initialized with all enhancements")
    
    async def process_market_data_advanced(self, symbol: str, price: float, 
                                         volume: float, order_book: Dict = None,
                                         trades: List[Dict] = None) -> Dict[str, Any]:
        """Process market data with all advanced analysis"""
        
        # Update price history
        self.price_histories[symbol].append(price)
        
        if len(self.price_histories[symbol]) < 50:
            return {'signal': 'NONE', 'confidence': 0}
        
        # Get price arrays
        prices = np.array(list(self.price_histories[symbol]))
        highs = prices * 1.001  # Simplified
        lows = prices * 0.999
        volumes = np.ones(len(prices)) * volume
        
        # Calculate advanced indicators
        advanced_indicators = self.advanced_indicators.calculate_all_advanced_indicators(
            prices, highs, lows, volumes, trades
        )
        
        # Microstructure analysis
        microstructure_metrics = {}
        if order_book:
            depth_analysis = self.microstructure_analyzer.analyze_order_book_depth(
                order_book.get('bids', []), order_book.get('asks', [])
            )
            microstructure_metrics.update(depth_analysis)
            
            if trades:
                trade_flow = self.microstructure_analyzer.analyze_trade_flow(trades)
                institutional = self.microstructure_analyzer.detect_institutional_activity(
                    trades, order_book
                )
                microstructure_metrics.update(trade_flow)
                microstructure_metrics.update(institutional)
        
        # Combine all market data
        complete_market_data = {
            **advanced_indicators,
            **microstructure_metrics,
            'current_price': price,
            'volume': volume
        }
        
        # ML prediction
        ml_prediction = self.ml_suite.predict_ensemble(complete_market_data)
        
        # Generate signal
        signal_strength = self._calculate_advanced_signal_strength(complete_market_data)
        
        return {
            'signal': 'BUY' if signal_strength > 0.3 else ('SELL' if signal_strength < -0.3 else 'NONE'),
            'strength': abs(signal_strength),
            'ml_confidence': ml_prediction['ensemble_prediction'],
            'ml_individual': ml_prediction['individual_predictions'],
            'market_data': complete_market_data,
            'microstructure_score': microstructure_metrics.get('institutional_score', 0)
        }
    
    def _calculate_advanced_signal_strength(self, market_data: Dict) -> float:
        """Calculate signal strength using all available indicators"""
        signals = []
        weights = []
        
        # Technical indicators
        rsi = market_data.get('rsi', 50)
        if rsi < 25:
            signals.append(0.8)
            weights.append(0.15)
        elif rsi > 75:
            signals.append(-0.8)
            weights.append(0.15)
        
        # Ichimoku Cloud
        if market_data.get('ichimoku_above_cloud', False):
            signals.append(0.6)
            weights.append(0.12)
        elif market_data.get('ichimoku_below_cloud', False):
            signals.append(-0.6)
            weights.append(0.12)
        
        # Market Profile
        current_price = market_data.get('current_price', 0)
        poc = market_data.get('profile_poc', 0)
        if current_price > 0 and poc > 0:
            if current_price > poc * 1.002:  # Above POC
                signals.append(0.4)
                weights.append(0.10)
            elif current_price < poc * 0.998:  # Below POC
                signals.append(-0.4)
                weights.append(0.10)
        
        # Order Flow
        delta = market_data.get('flow_delta', 0)
        if abs(delta) > 1000:  # Significant order flow imbalance
            signals.append(0.5 if delta > 0 else -0.5)
            weights.append(0.15)
        
        # Microstructure
        institutional_score = market_data.get('institutional_score', 0)
        if institutional_score > 0.7:
            depth_imbalance = market_data.get('depth_imbalance', 0)
            signals.append(depth_imbalance * 0.6)
            weights.append(0.18)
        
        # Volatility regime
        vol_regime = market_data.get('vol_regime', 'LOW')
        vol_adjustment = 1.2 if vol_regime == 'HIGH' else 1.0
        
        # Calculate weighted signal
        if signals and weights:
            weighted_signal = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
            return weighted_signal * vol_adjustment
        
        return 0.0
    
    def get_system_performance_advanced(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        return {
            'rate_limiter': self.rate_limiter.get_rate_limit_status(),
            'indicator_performance': {
                'avg_calculation_time': np.mean(self.advanced_indicators.calculation_times) if self.advanced_indicators.calculation_times else 0,
                'calculations_per_second': 1 / np.mean(self.advanced_indicators.calculation_times) if self.advanced_indicators.calculation_times else 0
            },
            'ml_performance': {
                'models_trained': len([m for m in self.ml_suite.models.values() if hasattr(m, 'n_features_in_')]),
                'feature_count': len(self.ml_suite.extract_advanced_features({'rsi': 50})),
                'model_performance': dict(self.ml_suite.model_performance)
            },
            'microstructure_metrics': {
                'order_books_analyzed': len(self.order_books),
                'trade_flows_tracked': sum(len(flow) for flow in self.trade_flows.values())
            }
        }

# Example usage
async def demo_advanced_features():
    """Demonstrate advanced features"""
    print("🚀 Advanced Features Demonstration")
    print("=" * 50)
    
    # Initialize system
    system = UltraAdvancedTradingSystem(['BTCUSDT', 'ETHUSDT'])
    
    # Simulate market data
    for i in range(100):
        price = 45000 + np.random.normal(0, 500)
        volume = 1000 + np.random.normal(0, 200)
        
        # Simulate order book
        order_book = {
            'bids': [(price - j, 100 + np.random.normal(0, 20)) for j in range(1, 11)],
            'asks': [(price + j, 100 + np.random.normal(0, 20)) for j in range(1, 11)]
        }
        
        # Simulate trades
        trades = [
            {
                'price': price + np.random.normal(0, 10),
                'quantity': 50 + np.random.normal(0, 10),
                'is_buyer_maker': np.random.random() > 0.5,
                'timestamp': time.time()
            }
            for _ in range(5)
        ]
        
        # Process with advanced system
        result = await system.process_market_data_advanced(
            'BTCUSDT', price, volume, order_book, trades
        )
        
        if i % 20 == 0:
            print(f"Signal: {result['signal']}, Strength: {result['strength']:.3f}, "
                  f"ML Confidence: {result['ml_confidence']:.3f}")
    
    # Get performance stats
    performance = system.get_system_performance_advanced()
    print(f"\nAdvanced System Performance:")
    print(f"Rate Limiter: {performance['rate_limiter']['requests_remaining']} requests remaining")
    print(f"Indicator Speed: {performance['indicator_performance']['calculations_per_second']:.0f} calc/sec")
    print(f"ML Models: {performance['ml_performance']['models_trained']} trained")

if __name__ == "__main__":
    asyncio.run(demo_advanced_features())