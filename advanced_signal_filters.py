"""
ADVANCED SIGNAL FILTERING SYSTEM FOR ENHANCED WIN RATE
====================================================
🎯 Multi-layer signal validation with ML-inspired techniques
📊 Dynamic threshold adjustment based on market conditions
🔍 Signal quality scoring with confidence intervals
⚡ Real-time market regime detection
💎 Advanced pattern recognition for higher accuracy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SignalFilter:
    """Advanced signal filter configuration"""
    min_confidence: float = 0.75
    min_pattern_strength: float = 0.65
    max_noise_ratio: float = 0.3
    regime_sensitivity: float = 0.8
    volatility_adjustment: bool = True
    volume_confirmation: bool = True

class MarketRegimeDetector:
    """Real-time market regime detection for adaptive filtering"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.price_history = deque(maxlen=lookback_period)
        self.volume_history = deque(maxlen=lookback_period)
        self.volatility_history = deque(maxlen=50)
        
        # Regime states
        self.current_regime = "NEUTRAL"  # TRENDING, RANGING, VOLATILE, NEUTRAL
        self.regime_confidence = 0.0
        self.regime_duration = 0
        
    def update(self, price: float, volume: float, volatility: float):
        """Update regime detection with new data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.volatility_history.append(volatility)
        
        if len(self.price_history) >= 50:
            self._detect_regime()
    
    def _detect_regime(self):
        """Detect current market regime using statistical analysis"""
        try:
            prices = np.array(list(self.price_history))
            volumes = np.array(list(self.volume_history))
            volatilities = np.array(list(self.volatility_history))
            
            # Calculate regime indicators
            price_trend = self._calculate_trend_strength(prices)
            volatility_level = np.mean(volatilities[-20:]) if len(volatilities) >= 20 else 0.02
            volume_pattern = self._analyze_volume_pattern(volumes)
            
            # Regime classification
            if abs(price_trend) > 0.7 and volatility_level < 0.03:
                new_regime = "TRENDING"
                confidence = min(abs(price_trend) + (1 - volatility_level/0.05), 1.0)
            elif volatility_level > 0.05:
                new_regime = "VOLATILE"
                confidence = min(volatility_level / 0.08, 1.0)
            elif abs(price_trend) < 0.3 and volatility_level < 0.025:
                new_regime = "RANGING"
                confidence = 1.0 - abs(price_trend) - volatility_level/0.025
            else:
                new_regime = "NEUTRAL"
                confidence = 0.5
            
            # Update regime with hysteresis to prevent flickering
            if new_regime != self.current_regime:
                if confidence > 0.7:
                    self.current_regime = new_regime
                    self.regime_confidence = confidence
                    self.regime_duration = 1
                else:
                    self.regime_duration += 1
            else:
                self.regime_duration += 1
                self.regime_confidence = max(self.regime_confidence * 0.99, confidence)
                
        except Exception as e:
            print(f"Error in regime detection: {e}")
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 20:
            return 0.0
        
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Normalize slope by price level
        normalized_slope = slope / np.mean(prices) * len(prices)
        trend_strength = normalized_slope * (r_value ** 2)  # Weight by R-squared
        
        return np.clip(trend_strength, -1.0, 1.0)
    
    def _analyze_volume_pattern(self, volumes: np.ndarray) -> str:
        """Analyze volume patterns"""
        if len(volumes) < 20:
            return "NORMAL"
        
        recent_vol = np.mean(volumes[-10:])
        historical_vol = np.mean(volumes[:-10])
        
        if recent_vol > historical_vol * 1.5:
            return "INCREASING"
        elif recent_vol < historical_vol * 0.7:
            return "DECREASING"
        else:
            return "NORMAL"
    
    def get_regime_info(self) -> Dict[str, any]:
        """Get current regime information"""
        return {
            'regime': self.current_regime,
            'confidence': self.regime_confidence,
            'duration': self.regime_duration
        }

class PatternRecognitionEngine:
    """Advanced pattern recognition for signal validation"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.pattern_success_rates = {
            'double_bottom': 0.72,
            'double_top': 0.68,
            'ascending_triangle': 0.75,
            'descending_triangle': 0.71,
            'head_shoulders': 0.78,
            'inverse_head_shoulders': 0.76,
            'flag': 0.69,
            'pennant': 0.67,
            'wedge': 0.73
        }
    
    def detect_patterns(self, prices: np.ndarray, highs: np.ndarray, 
                       lows: np.ndarray) -> Dict[str, float]:
        """Detect chart patterns and return confidence scores"""
        if len(prices) < 50:
            return {}
        
        patterns = {}
        
        try:
            # Double bottom/top detection
            patterns.update(self._detect_double_patterns(prices, lows, highs))
            
            # Triangle patterns
            patterns.update(self._detect_triangle_patterns(prices, highs, lows))
            
            # Head and shoulders
            patterns.update(self._detect_head_shoulders(prices, highs, lows))
            
            # Flag and pennant patterns
            patterns.update(self._detect_flag_pennant(prices))
            
            # Wedge patterns
            patterns.update(self._detect_wedge_patterns(prices, highs, lows))
            
        except Exception as e:
            print(f"Error in pattern detection: {e}")
        
        return patterns
    
    def _detect_double_patterns(self, prices: np.ndarray, lows: np.ndarray, 
                               highs: np.ndarray) -> Dict[str, float]:
        """Detect double bottom and double top patterns"""
        patterns = {}
        
        if len(prices) < 30:
            return patterns
        
        # Double bottom detection
        recent_lows = lows[-30:]
        low_indices = self._find_local_extrema(recent_lows, 'min')
        
        if len(low_indices) >= 2:
            last_two_lows = recent_lows[low_indices[-2:]]
            if abs(last_two_lows[0] - last_two_lows[1]) / np.mean(last_two_lows) < 0.02:
                # Check for price recovery between lows
                between_high = np.max(recent_lows[low_indices[-2]:low_indices[-1]])
                if between_high > np.mean(last_two_lows) * 1.01:
                    patterns['double_bottom'] = 0.8
        
        # Double top detection
        recent_highs = highs[-30:]
        high_indices = self._find_local_extrema(recent_highs, 'max')
        
        if len(high_indices) >= 2:
            last_two_highs = recent_highs[high_indices[-2:]]
            if abs(last_two_highs[0] - last_two_highs[1]) / np.mean(last_two_highs) < 0.02:
                # Check for price decline between highs
                between_low = np.min(recent_highs[high_indices[-2]:high_indices[-1]])
                if between_low < np.mean(last_two_highs) * 0.99:
                    patterns['double_top'] = 0.8
        
        return patterns
    
    def _detect_triangle_patterns(self, prices: np.ndarray, highs: np.ndarray, 
                                 lows: np.ndarray) -> Dict[str, float]:
        """Detect triangle patterns"""
        patterns = {}
        
        if len(prices) < 40:
            return patterns
        
        recent_highs = highs[-40:]
        recent_lows = lows[-40:]
        
        # Calculate trend lines
        high_trend = self._calculate_trendline(recent_highs, 'descending')
        low_trend = self._calculate_trendline(recent_lows, 'ascending')
        
        # Ascending triangle
        if abs(high_trend) < 0.001 and low_trend > 0.001:
            patterns['ascending_triangle'] = 0.75
        
        # Descending triangle
        elif abs(low_trend) < 0.001 and high_trend < -0.001:
            patterns['descending_triangle'] = 0.71
        
        # Symmetrical triangle
        elif high_trend < -0.001 and low_trend > 0.001:
            convergence = abs(high_trend) + low_trend
            if convergence > 0.002:
                patterns['symmetrical_triangle'] = 0.68
        
        return patterns
    
    def _detect_head_shoulders(self, prices: np.ndarray, highs: np.ndarray, 
                              lows: np.ndarray) -> Dict[str, float]:
        """Detect head and shoulders patterns"""
        patterns = {}
        
        if len(highs) < 50:
            return patterns
        
        # Find peaks for head and shoulders
        peaks = self._find_local_extrema(highs[-50:], 'max')
        
        if len(peaks) >= 3:
            # Check for head and shoulders formation
            last_three_peaks = highs[-50:][peaks[-3:]]
            
            # Head should be higher than shoulders
            if (last_three_peaks[1] > last_three_peaks[0] and 
                last_three_peaks[1] > last_three_peaks[2]):
                
                shoulder_symmetry = abs(last_three_peaks[0] - last_three_peaks[2]) / last_three_peaks[1]
                
                if shoulder_symmetry < 0.05:  # Shoulders roughly equal
                    patterns['head_shoulders'] = 0.78
        
        # Inverse head and shoulders (using lows)
        troughs = self._find_local_extrema(lows[-50:], 'min')
        
        if len(troughs) >= 3:
            last_three_troughs = lows[-50:][troughs[-3:]]
            
            # Head should be lower than shoulders
            if (last_three_troughs[1] < last_three_troughs[0] and 
                last_three_troughs[1] < last_three_troughs[2]):
                
                shoulder_symmetry = abs(last_three_troughs[0] - last_three_troughs[2]) / abs(last_three_troughs[1])
                
                if shoulder_symmetry < 0.05:
                    patterns['inverse_head_shoulders'] = 0.76
        
        return patterns
    
    def _detect_flag_pennant(self, prices: np.ndarray) -> Dict[str, float]:
        """Detect flag and pennant patterns"""
        patterns = {}
        
        if len(prices) < 30:
            return patterns
        
        # Look for strong move followed by consolidation
        recent_prices = prices[-30:]
        
        # Check for strong initial move
        initial_move = (recent_prices[10] - recent_prices[0]) / recent_prices[0]
        
        if abs(initial_move) > 0.03:  # Strong move > 3%
            # Check for consolidation
            consolidation_prices = recent_prices[10:]
            volatility = np.std(consolidation_prices) / np.mean(consolidation_prices)
            
            if volatility < 0.02:  # Low volatility consolidation
                if initial_move > 0:
                    patterns['bull_flag'] = 0.69
                else:
                    patterns['bear_flag'] = 0.69
        
        return patterns
    
    def _detect_wedge_patterns(self, prices: np.ndarray, highs: np.ndarray, 
                              lows: np.ndarray) -> Dict[str, float]:
        """Detect wedge patterns"""
        patterns = {}
        
        if len(prices) < 40:
            return patterns
        
        # Calculate converging trend lines
        high_slope = self._calculate_trendline(highs[-40:], 'any')
        low_slope = self._calculate_trendline(lows[-40:], 'any')
        
        # Rising wedge (both slopes positive, high slope < low slope)
        if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
            patterns['rising_wedge'] = 0.73
        
        # Falling wedge (both slopes negative, high slope > low slope)
        elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
            patterns['falling_wedge'] = 0.73
        
        return patterns
    
    def _find_local_extrema(self, data: np.ndarray, extrema_type: str) -> List[int]:
        """Find local extrema in data"""
        extrema = []
        
        for i in range(1, len(data) - 1):
            if extrema_type == 'max':
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    extrema.append(i)
            elif extrema_type == 'min':
                if data[i] < data[i-1] and data[i] < data[i+1]:
                    extrema.append(i)
        
        return extrema
    
    def _calculate_trendline(self, data: np.ndarray, direction: str) -> float:
        """Calculate trendline slope"""
        if len(data) < 10:
            return 0.0
        
        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)
        
        # Weight slope by R-squared for reliability
        weighted_slope = slope * (r_value ** 2)
        
        return weighted_slope

class AdvancedSignalFilter:
    """Main advanced signal filtering system"""
    
    def __init__(self, config: SignalFilter = None):
        self.config = config or SignalFilter()
        self.regime_detector = MarketRegimeDetector()
        self.pattern_engine = PatternRecognitionEngine()
        
        # Signal quality tracking
        self.signal_history = deque(maxlen=1000)
        self.success_rate_tracker = {}
        
        # Adaptive thresholds
        self.dynamic_thresholds = {
            'TRENDING': {'min_confidence': 0.65, 'min_strength': 0.6},
            'RANGING': {'min_confidence': 0.8, 'min_strength': 0.75},
            'VOLATILE': {'min_confidence': 0.85, 'min_strength': 0.8},
            'NEUTRAL': {'min_confidence': 0.75, 'min_strength': 0.7}
        }
    
    def filter_signal(self, signal_data: Dict, market_data: Dict) -> Dict[str, any]:
        """Apply advanced filtering to trading signal"""
        try:
            # Update regime detection
            self.regime_detector.update(
                market_data.get('price', 0),
                market_data.get('volume', 0),
                market_data.get('volatility', 0.02)
            )
            
            # Get current market regime
            regime_info = self.regime_detector.get_regime_info()
            current_regime = regime_info['regime']
            
            # Apply regime-specific thresholds
            thresholds = self.dynamic_thresholds.get(current_regime, self.dynamic_thresholds['NEUTRAL'])
            
            # Initial signal validation
            if signal_data.get('strength', 0) < thresholds['min_strength']:
                return self._create_filter_result(False, "Insufficient signal strength for current regime")
            
            # Pattern recognition validation
            patterns = self.pattern_engine.detect_patterns(
                market_data.get('prices', np.array([])),
                market_data.get('highs', np.array([])),
                market_data.get('lows', np.array([]))
            )
            
            # Calculate pattern confidence
            pattern_confidence = self._calculate_pattern_confidence(patterns, signal_data.get('signal', 'NONE'))
            
            # Volume confirmation
            volume_confirmation = self._validate_volume_pattern(market_data)
            
            # Volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment(market_data.get('volatility', 0.02))
            
            # Noise ratio analysis
            noise_ratio = self._calculate_noise_ratio(market_data)
            
            # Combine all factors
            final_confidence = self._calculate_final_confidence(
                signal_data.get('strength', 0),
                pattern_confidence,
                volume_confirmation,
                volatility_adjustment,
                regime_info['confidence'],
                noise_ratio
            )
            
            # Final decision
            should_trade = (
                final_confidence >= thresholds['min_confidence'] and
                pattern_confidence >= 0.5 and
                noise_ratio <= self.config.max_noise_ratio and
                volume_confirmation >= 0.6
            )
            
            # Create detailed result
            result = self._create_filter_result(
                should_trade,
                f"Regime: {current_regime}, Confidence: {final_confidence:.3f}",
                {
                    'final_confidence': final_confidence,
                    'pattern_confidence': pattern_confidence,
                    'volume_confirmation': volume_confirmation,
                    'volatility_adjustment': volatility_adjustment,
                    'noise_ratio': noise_ratio,
                    'regime': current_regime,
                    'regime_confidence': regime_info['confidence'],
                    'detected_patterns': patterns,
                    'thresholds_used': thresholds
                }
            )
            
            # Track signal for learning
            self._track_signal_quality(signal_data, result)
            
            return result
            
        except Exception as e:
            print(f"Error in advanced signal filtering: {e}")
            return self._create_filter_result(False, f"Filter error: {e}")
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, float], signal_direction: str) -> float:
        """Calculate confidence based on detected patterns"""
        if not patterns:
            return 0.5  # Neutral if no patterns detected
        
        # Weight patterns by their historical success rates and alignment with signal
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for pattern, strength in patterns.items():
            success_rate = self.pattern_engine.pattern_success_rates.get(pattern, 0.6)
            
            # Check if pattern aligns with signal direction
            bullish_patterns = ['double_bottom', 'ascending_triangle', 'inverse_head_shoulders', 'bull_flag', 'falling_wedge']
            bearish_patterns = ['double_top', 'descending_triangle', 'head_shoulders', 'bear_flag', 'rising_wedge']
            
            alignment_bonus = 1.0
            if signal_direction == 'BUY' and pattern in bullish_patterns:
                alignment_bonus = 1.2
            elif signal_direction == 'SELL' and pattern in bearish_patterns:
                alignment_bonus = 1.2
            elif signal_direction == 'BUY' and pattern in bearish_patterns:
                alignment_bonus = 0.7
            elif signal_direction == 'SELL' and pattern in bullish_patterns:
                alignment_bonus = 0.7
            
            weight = strength * success_rate * alignment_bonus
            weighted_confidence += weight
            total_weight += weight
        
        return min(weighted_confidence / total_weight if total_weight > 0 else 0.5, 1.0)
    
    def _validate_volume_pattern(self, market_data: Dict) -> float:
        """Validate volume patterns"""
        volumes = market_data.get('volumes', np.array([]))
        
        if len(volumes) < 20:
            return 0.6  # Neutral if insufficient data
        
        # Check for volume confirmation
        recent_volume = np.mean(volumes[-5:])
        historical_volume = np.mean(volumes[-20:-5])
        
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
        
        # Higher volume during signal = higher confidence
        if volume_ratio > 1.5:
            return 0.9
        elif volume_ratio > 1.2:
            return 0.8
        elif volume_ratio > 0.8:
            return 0.7
        else:
            return 0.5
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate volatility-based adjustment factor"""
        # Moderate volatility is optimal for trading
        if 0.015 <= volatility <= 0.035:
            return 1.0  # Optimal range
        elif volatility < 0.01:
            return 0.7  # Too low volatility
        elif volatility > 0.05:
            return 0.6  # Too high volatility
        else:
            return 0.85  # Suboptimal but acceptable
    
    def _calculate_noise_ratio(self, market_data: Dict) -> float:
        """Calculate market noise ratio"""
        prices = market_data.get('prices', np.array([]))
        
        if len(prices) < 20:
            return 0.3  # Default moderate noise
        
        # Calculate price efficiency (trend vs noise)
        price_changes = np.diff(prices[-20:])
        total_movement = np.sum(np.abs(price_changes))
        net_movement = abs(prices[-1] - prices[-20])
        
        if total_movement == 0:
            return 0.5
        
        efficiency = net_movement / total_movement
        noise_ratio = 1.0 - efficiency
        
        return np.clip(noise_ratio, 0.0, 1.0)
    
    def _calculate_final_confidence(self, signal_strength: float, pattern_confidence: float,
                                   volume_confirmation: float, volatility_adjustment: float,
                                   regime_confidence: float, noise_ratio: float) -> float:
        """Calculate final confidence score"""
        # Weighted combination of all factors
        weights = {
            'signal': 0.25,
            'pattern': 0.20,
            'volume': 0.15,
            'volatility': 0.15,
            'regime': 0.15,
            'noise': 0.10
        }
        
        final_confidence = (
            signal_strength * weights['signal'] +
            pattern_confidence * weights['pattern'] +
            volume_confirmation * weights['volume'] +
            volatility_adjustment * weights['volatility'] +
            regime_confidence * weights['regime'] +
            (1.0 - noise_ratio) * weights['noise']  # Lower noise = higher confidence
        )
        
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _create_filter_result(self, should_trade: bool, reason: str, 
                             details: Dict = None) -> Dict[str, any]:
        """Create standardized filter result"""
        return {
            'should_trade': should_trade,
            'reason': reason,
            'timestamp': time.time(),
            'details': details or {}
        }
    
    def _track_signal_quality(self, signal_data: Dict, filter_result: Dict):
        """Track signal quality for continuous improvement"""
        self.signal_history.append({
            'signal': signal_data,
            'filter_result': filter_result,
            'timestamp': time.time()
        })
    
    def get_filter_performance(self) -> Dict[str, float]:
        """Get filter performance statistics"""
        if not self.signal_history:
            return {}
        
        total_signals = len(self.signal_history)
        approved_signals = sum(1 for s in self.signal_history if s['filter_result']['should_trade'])
        
        return {
            'total_signals_processed': total_signals,
            'approval_rate': approved_signals / total_signals,
            'average_confidence': np.mean([s['filter_result']['details'].get('final_confidence', 0) 
                                         for s in self.signal_history if s['filter_result']['details']]),
            'regime_distribution': self._calculate_regime_distribution()
        }
    
    def _calculate_regime_distribution(self) -> Dict[str, float]:
        """Calculate distribution of market regimes"""
        regimes = [s['filter_result']['details'].get('regime', 'UNKNOWN') 
                  for s in self.signal_history if s['filter_result']['details']]
        
        if not regimes:
            return {}
        
        unique_regimes, counts = np.unique(regimes, return_counts=True)
        total = len(regimes)
        
        return {regime: count/total for regime, count in zip(unique_regimes, counts)}

# Example usage and testing
if __name__ == "__main__":
    # Initialize advanced filter
    filter_config = SignalFilter(
        min_confidence=0.75,
        min_pattern_strength=0.65,
        max_noise_ratio=0.3
    )
    
    advanced_filter = AdvancedSignalFilter(filter_config)
    
    # Example signal data
    signal_data = {
        'signal': 'BUY',
        'strength': 0.8,
        'quality': 0.75,
        'confirmations': 3
    }
    
    # Example market data
    market_data = {
        'price': 45000.0,
        'volume': 1000000.0,
        'volatility': 0.025,
        'prices': np.random.random(100) * 1000 + 44000,
        'highs': np.random.random(100) * 1000 + 44500,
        'lows': np.random.random(100) * 1000 + 43500,
        'volumes': np.random.random(100) * 2000000 + 500000
    }
    
    # Apply filter
    result = advanced_filter.filter_signal(signal_data, market_data)
    
    print("Advanced Signal Filter Result:")
    print(f"Should Trade: {result['should_trade']}")
    print(f"Reason: {result['reason']}")
    print(f"Details: {result['details']}")