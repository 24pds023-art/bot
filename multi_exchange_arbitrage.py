"""
MULTI-EXCHANGE ARBITRAGE SYSTEM
===============================
🔄 Cross-exchange price monitoring and arbitrage opportunities
⚡ Real-time price feeds from multiple exchanges
💰 Automated arbitrage execution with risk management
📊 Latency optimization for maximum profit capture
"""

import asyncio
import aiohttp
import websockets
import json
import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import statistics

# Exchange configurations
EXCHANGES = {
    'binance': {
        'name': 'Binance',
        'websocket_url': 'wss://stream.binance.com:9443/ws/',
        'rest_api': 'https://api.binance.com/api/v3/',
        'maker_fee': 0.001,  # 0.1%
        'taker_fee': 0.001,
        'min_profit_threshold': 0.002  # 0.2% minimum profit
    },
    'coinbase': {
        'name': 'Coinbase Pro',
        'websocket_url': 'wss://ws-feed.pro.coinbase.com',
        'rest_api': 'https://api.pro.coinbase.com/',
        'maker_fee': 0.005,  # 0.5%
        'taker_fee': 0.005,
        'min_profit_threshold': 0.012  # 1.2% minimum profit
    },
    'kraken': {
        'name': 'Kraken',
        'websocket_url': 'wss://ws.kraken.com',
        'rest_api': 'https://api.kraken.com/0/public/',
        'maker_fee': 0.0016,  # 0.16%
        'taker_fee': 0.0026,  # 0.26%
        'min_profit_threshold': 0.005  # 0.5% minimum profit
    },
    'ftx': {
        'name': 'FTX',
        'websocket_url': 'wss://ftx.com/ws/',
        'rest_api': 'https://ftx.com/api/',
        'maker_fee': 0.0002,  # 0.02%
        'taker_fee': 0.0007,  # 0.07%
        'min_profit_threshold': 0.002  # 0.2% minimum profit
    }
}

@dataclass
class PriceData:
    """Price data structure"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    timestamp: float
    volume: float = 0.0
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity structure"""
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: float
    sell_price: float
    profit_percentage: float
    profit_absolute: float
    volume_limit: float
    timestamp: float
    latency_ms: float
    
    @property
    def is_profitable(self) -> bool:
        return self.profit_percentage > 0

class ExchangeConnector:
    """Base class for exchange connections"""
    
    def __init__(self, exchange_name: str, config: Dict):
        self.exchange_name = exchange_name
        self.config = config
        self.is_connected = False
        self.price_data = {}
        self.connection_stats = {
            'messages_received': 0,
            'connection_errors': 0,
            'last_message_time': 0,
            'average_latency': 0
        }
        
    async def connect(self):
        """Connect to exchange WebSocket"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from exchange"""
        self.is_connected = False
    
    def get_latest_price(self, symbol: str) -> Optional[PriceData]:
        """Get latest price for symbol"""
        return self.price_data.get(symbol)

class BinanceConnector(ExchangeConnector):
    """Binance exchange connector"""
    
    def __init__(self):
        super().__init__('binance', EXCHANGES['binance'])
        self.symbol_streams = []
        
    async def connect(self, symbols: List[str]):
        """Connect to Binance WebSocket streams"""
        # Convert symbols to Binance format
        binance_symbols = [s.lower() for s in symbols]
        
        # Create stream names
        streams = []
        for symbol in binance_symbols:
            streams.append(f"{symbol}@bookTicker")
        
        stream_url = f"{self.config['websocket_url']}{'stream?streams=' + '/'.join(streams)}"
        
        try:
            async with websockets.connect(stream_url) as websocket:
                self.is_connected = True
                print(f"✅ Connected to {self.config['name']}")
                
                async for message in websocket:
                    await self._process_message(json.loads(message))
                    
        except Exception as e:
            print(f"❌ Binance connection error: {e}")
            self.connection_stats['connection_errors'] += 1
            self.is_connected = False
    
    async def _process_message(self, message: Dict):
        """Process incoming WebSocket message"""
        try:
            if 'data' in message:
                data = message['data']
                symbol = data['s']  # Symbol
                
                price_data = PriceData(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    bid=float(data['b']),  # Best bid price
                    ask=float(data['a']),  # Best ask price
                    timestamp=time.time(),
                    volume=float(data.get('B', 0))  # Best bid quantity
                )
                
                self.price_data[symbol] = price_data
                self.connection_stats['messages_received'] += 1
                self.connection_stats['last_message_time'] = time.time()
                
        except Exception as e:
            print(f"Error processing Binance message: {e}")

class CoinbaseConnector(ExchangeConnector):
    """Coinbase Pro exchange connector"""
    
    def __init__(self):
        super().__init__('coinbase', EXCHANGES['coinbase'])
    
    async def connect(self, symbols: List[str]):
        """Connect to Coinbase Pro WebSocket"""
        # Convert symbols to Coinbase format (e.g., BTCUSDT -> BTC-USD)
        coinbase_symbols = [s.replace('USDT', '-USD') for s in symbols]
        
        subscribe_message = {
            "type": "subscribe",
            "product_ids": coinbase_symbols,
            "channels": ["ticker"]
        }
        
        try:
            async with websockets.connect(self.config['websocket_url']) as websocket:
                await websocket.send(json.dumps(subscribe_message))
                self.is_connected = True
                print(f"✅ Connected to {self.config['name']}")
                
                async for message in websocket:
                    await self._process_message(json.loads(message))
                    
        except Exception as e:
            print(f"❌ Coinbase connection error: {e}")
            self.connection_stats['connection_errors'] += 1
            self.is_connected = False
    
    async def _process_message(self, message: Dict):
        """Process incoming WebSocket message"""
        try:
            if message.get('type') == 'ticker':
                symbol = message['product_id'].replace('-USD', 'USDT')
                
                price_data = PriceData(
                    exchange=self.exchange_name,
                    symbol=symbol,
                    bid=float(message['best_bid']),
                    ask=float(message['best_ask']),
                    timestamp=time.time(),
                    volume=float(message.get('volume_24h', 0))
                )
                
                self.price_data[symbol] = price_data
                self.connection_stats['messages_received'] += 1
                self.connection_stats['last_message_time'] = time.time()
                
        except Exception as e:
            print(f"Error processing Coinbase message: {e}")

class ArbitrageEngine:
    """Main arbitrage detection and execution engine"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.connectors = {
            'binance': BinanceConnector(),
            'coinbase': CoinbaseConnector()
        }
        
        # Arbitrage tracking
        self.opportunities = deque(maxlen=1000)
        self.executed_trades = []
        self.price_history = defaultdict(lambda: defaultdict(deque))
        
        # Performance metrics
        self.detection_times = deque(maxlen=1000)
        self.opportunity_count = 0
        self.profitable_opportunities = 0
        self.total_profit = 0.0
        
        # Risk management
        self.max_position_size = 1000  # USD
        self.max_daily_trades = 50
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        
        print(f"🔄 Arbitrage Engine initialized for {len(symbols)} symbols")
    
    async def start_monitoring(self):
        """Start monitoring all exchanges"""
        print("🚀 Starting multi-exchange monitoring...")
        
        # Start all exchange connections
        tasks = []
        for name, connector in self.connectors.items():
            task = asyncio.create_task(connector.connect(self.symbols))
            tasks.append(task)
        
        # Start arbitrage detection
        detection_task = asyncio.create_task(self._arbitrage_detection_loop())
        tasks.append(detection_task)
        
        # Start performance monitoring
        monitor_task = asyncio.create_task(self._performance_monitoring_loop())
        tasks.append(monitor_task)
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except KeyboardInterrupt:
            print("\n⏹️ Stopping arbitrage monitoring...")
            await self._cleanup()
    
    async def _arbitrage_detection_loop(self):
        """Main arbitrage detection loop"""
        while True:
            try:
                start_time = time.perf_counter()
                
                # Check for arbitrage opportunities
                opportunities = await self._detect_arbitrage_opportunities()
                
                # Process opportunities
                for opportunity in opportunities:
                    await self._evaluate_opportunity(opportunity)
                
                # Record detection time
                detection_time = time.perf_counter() - start_time
                self.detection_times.append(detection_time)
                
                # Sleep briefly to prevent excessive CPU usage
                await asyncio.sleep(0.1)  # 100ms detection cycle
                
            except Exception as e:
                print(f"❌ Error in arbitrage detection: {e}")
                await asyncio.sleep(1)
    
    async def _detect_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across exchanges"""
        opportunities = []
        
        for symbol in self.symbols:
            # Get prices from all exchanges
            exchange_prices = {}
            
            for exchange_name, connector in self.connectors.items():
                if connector.is_connected:
                    price_data = connector.get_latest_price(symbol)
                    if price_data and price_data.timestamp > time.time() - 5:  # Fresh data only
                        exchange_prices[exchange_name] = price_data
            
            # Find arbitrage opportunities
            if len(exchange_prices) >= 2:
                opportunities.extend(self._find_arbitrage_pairs(symbol, exchange_prices))
        
        return opportunities
    
    def _find_arbitrage_pairs(self, symbol: str, exchange_prices: Dict[str, PriceData]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities between exchange pairs"""
        opportunities = []
        exchanges = list(exchange_prices.keys())
        
        # Check all exchange pairs
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange1, exchange2 = exchanges[i], exchanges[j]
                price1, price2 = exchange_prices[exchange1], exchange_prices[exchange2]
                
                # Calculate potential profits in both directions
                
                # Direction 1: Buy on exchange1, sell on exchange2
                profit1 = self._calculate_arbitrage_profit(
                    buy_exchange=exchange1,
                    sell_exchange=exchange2,
                    buy_price=price1.ask,
                    sell_price=price2.bid,
                    symbol=symbol
                )
                
                if profit1 and profit1.is_profitable:
                    opportunities.append(profit1)
                
                # Direction 2: Buy on exchange2, sell on exchange1
                profit2 = self._calculate_arbitrage_profit(
                    buy_exchange=exchange2,
                    sell_exchange=exchange1,
                    buy_price=price2.ask,
                    sell_price=price1.bid,
                    symbol=symbol
                )
                
                if profit2 and profit2.is_profitable:
                    opportunities.append(profit2)
        
        return opportunities
    
    def _calculate_arbitrage_profit(self, buy_exchange: str, sell_exchange: str,
                                  buy_price: float, sell_price: float, symbol: str) -> Optional[ArbitrageOpportunity]:
        """Calculate arbitrage profit after fees"""
        
        # Get exchange configurations
        buy_config = EXCHANGES.get(buy_exchange, {})
        sell_config = EXCHANGES.get(sell_exchange, {})
        
        # Calculate fees
        buy_fee = buy_config.get('taker_fee', 0.001)
        sell_fee = sell_config.get('maker_fee', 0.001)  # Assume we can place limit orders
        
        # Calculate net prices after fees
        net_buy_price = buy_price * (1 + buy_fee)
        net_sell_price = sell_price * (1 - sell_fee)
        
        # Calculate profit
        if net_sell_price > net_buy_price:
            profit_absolute = net_sell_price - net_buy_price
            profit_percentage = profit_absolute / net_buy_price
            
            # Check minimum profit threshold
            min_threshold = max(
                buy_config.get('min_profit_threshold', 0.002),
                sell_config.get('min_profit_threshold', 0.002)
            )
            
            if profit_percentage >= min_threshold:
                # Calculate volume limit (simplified)
                volume_limit = min(self.max_position_size / buy_price, 10.0)  # Max 10 units
                
                return ArbitrageOpportunity(
                    buy_exchange=buy_exchange,
                    sell_exchange=sell_exchange,
                    symbol=symbol,
                    buy_price=net_buy_price,
                    sell_price=net_sell_price,
                    profit_percentage=profit_percentage,
                    profit_absolute=profit_absolute,
                    volume_limit=volume_limit,
                    timestamp=time.time(),
                    latency_ms=0  # Would be calculated from actual execution
                )
        
        return None
    
    async def _evaluate_opportunity(self, opportunity: ArbitrageOpportunity):
        """Evaluate and potentially execute arbitrage opportunity"""
        self.opportunity_count += 1
        
        # Risk checks
        if not self._risk_check(opportunity):
            return
        
        # Profitability check
        if opportunity.profit_percentage < 0.005:  # 0.5% minimum
            return
        
        # Record profitable opportunity
        self.profitable_opportunities += 1
        self.opportunities.append(opportunity)
        
        print(f"💰 Arbitrage Opportunity: {opportunity.symbol}")
        print(f"   Buy: {opportunity.buy_exchange} @ ${opportunity.buy_price:.4f}")
        print(f"   Sell: {opportunity.sell_exchange} @ ${opportunity.sell_price:.4f}")
        print(f"   Profit: {opportunity.profit_percentage:.2%} (${opportunity.profit_absolute:.4f})")
        
        # Execute if conditions are met
        if self._should_execute(opportunity):
            await self._execute_arbitrage(opportunity)
    
    def _risk_check(self, opportunity: ArbitrageOpportunity) -> bool:
        """Perform risk checks on opportunity"""
        
        # Daily trade limit
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = current_date
        
        if self.daily_trade_count >= self.max_daily_trades:
            return False
        
        # Position size check
        position_value = opportunity.buy_price * opportunity.volume_limit
        if position_value > self.max_position_size:
            return False
        
        # Exchange connectivity check
        buy_connector = self.connectors.get(opportunity.buy_exchange)
        sell_connector = self.connectors.get(opportunity.sell_exchange)
        
        if not (buy_connector and buy_connector.is_connected and 
                sell_connector and sell_connector.is_connected):
            return False
        
        # Latency check (opportunity might be stale)
        if time.time() - opportunity.timestamp > 1.0:  # 1 second max age
            return False
        
        return True
    
    def _should_execute(self, opportunity: ArbitrageOpportunity) -> bool:
        """Determine if opportunity should be executed"""
        
        # Conservative execution criteria
        min_profit_for_execution = 0.008  # 0.8%
        
        if opportunity.profit_percentage < min_profit_for_execution:
            return False
        
        # Check recent execution history to avoid overtrading
        recent_executions = [
            trade for trade in self.executed_trades
            if time.time() - trade['timestamp'] < 300  # Last 5 minutes
            and trade['symbol'] == opportunity.symbol
        ]
        
        if len(recent_executions) >= 3:  # Max 3 trades per symbol per 5 minutes
            return False
        
        return True
    
    async def _execute_arbitrage(self, opportunity: ArbitrageOpportunity):
        """Execute arbitrage trade (simulation)"""
        
        # In a real implementation, this would:
        # 1. Place buy order on buy_exchange
        # 2. Place sell order on sell_exchange
        # 3. Monitor execution
        # 4. Handle partial fills
        # 5. Manage inventory
        
        print(f"🚀 EXECUTING ARBITRAGE: {opportunity.symbol}")
        print(f"   Expected Profit: ${opportunity.profit_absolute:.4f}")
        
        # Simulate execution
        execution_success = np.random.random() > 0.1  # 90% success rate
        
        if execution_success:
            # Record successful trade
            trade_record = {
                'symbol': opportunity.symbol,
                'buy_exchange': opportunity.buy_exchange,
                'sell_exchange': opportunity.sell_exchange,
                'profit': opportunity.profit_absolute,
                'profit_percentage': opportunity.profit_percentage,
                'volume': opportunity.volume_limit,
                'timestamp': time.time()
            }
            
            self.executed_trades.append(trade_record)
            self.total_profit += opportunity.profit_absolute
            self.daily_trade_count += 1
            
            print(f"✅ Arbitrage executed successfully!")
            print(f"   Profit: ${opportunity.profit_absolute:.4f}")
            
        else:
            print(f"❌ Arbitrage execution failed")
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Log performance every 5 minutes
                if int(time.time()) % 300 == 0:
                    await self._log_performance_summary()
                    
            except Exception as e:
                print(f"❌ Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _log_performance_summary(self):
        """Log comprehensive performance summary"""
        print(f"\n📊 ARBITRAGE PERFORMANCE SUMMARY")
        print(f"   Opportunities Detected: {self.opportunity_count}")
        print(f"   Profitable Opportunities: {self.profitable_opportunities}")
        print(f"   Success Rate: {self.profitable_opportunities/max(self.opportunity_count, 1):.1%}")
        print(f"   Trades Executed: {len(self.executed_trades)}")
        print(f"   Total Profit: ${self.total_profit:.2f}")
        
        # Detection performance
        if self.detection_times:
            avg_detection_time = statistics.mean(self.detection_times) * 1000
            print(f"   Avg Detection Time: {avg_detection_time:.1f}ms")
        
        # Exchange connectivity
        print(f"   Exchange Status:")
        for name, connector in self.connectors.items():
            status = "🟢 Connected" if connector.is_connected else "🔴 Disconnected"
            messages = connector.connection_stats['messages_received']
            print(f"     {name}: {status} ({messages} messages)")
        
        # Recent opportunities
        recent_opportunities = [
            opp for opp in self.opportunities
            if time.time() - opp.timestamp < 300  # Last 5 minutes
        ]
        
        if recent_opportunities:
            avg_profit = statistics.mean([opp.profit_percentage for opp in recent_opportunities])
            best_profit = max([opp.profit_percentage for opp in recent_opportunities])
            print(f"   Recent Opportunities (5min):")
            print(f"     Count: {len(recent_opportunities)}")
            print(f"     Avg Profit: {avg_profit:.2%}")
            print(f"     Best Profit: {best_profit:.2%}")
        
        print("=" * 60)
    
    def get_arbitrage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive arbitrage statistics"""
        
        # Calculate success metrics
        if self.executed_trades:
            profits = [trade['profit'] for trade in self.executed_trades]
            profit_percentages = [trade['profit_percentage'] for trade in self.executed_trades]
            
            avg_profit = statistics.mean(profits)
            total_profit = sum(profits)
            avg_profit_pct = statistics.mean(profit_percentages)
            best_trade = max(self.executed_trades, key=lambda x: x['profit'])
        else:
            avg_profit = total_profit = avg_profit_pct = 0
            best_trade = None
        
        # Exchange performance
        exchange_stats = {}
        for name, connector in self.connectors.items():
            exchange_stats[name] = {
                'connected': connector.is_connected,
                'messages_received': connector.connection_stats['messages_received'],
                'connection_errors': connector.connection_stats['connection_errors'],
                'last_message_age': time.time() - connector.connection_stats.get('last_message_time', 0)
            }
        
        return {
            'opportunities_detected': self.opportunity_count,
            'profitable_opportunities': self.profitable_opportunities,
            'success_rate': self.profitable_opportunities / max(self.opportunity_count, 1),
            'trades_executed': len(self.executed_trades),
            'total_profit': total_profit,
            'average_profit': avg_profit,
            'average_profit_percentage': avg_profit_pct,
            'best_trade': best_trade,
            'daily_trade_count': self.daily_trade_count,
            'exchange_statistics': exchange_stats,
            'detection_performance': {
                'avg_detection_time_ms': statistics.mean(self.detection_times) * 1000 if self.detection_times else 0,
                'detection_cycles': len(self.detection_times)
            }
        }
    
    async def _cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up arbitrage engine...")
        
        # Disconnect all exchanges
        for connector in self.connectors.values():
            await connector.disconnect()
        
        print("✅ Arbitrage engine cleanup complete")

# ============================================================================
# DEMO AND TESTING
# ============================================================================

async def demo_arbitrage_system():
    """Demonstrate arbitrage system"""
    print("🔄 Multi-Exchange Arbitrage System Demo")
    print("=" * 50)
    
    # Test symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Initialize arbitrage engine
    engine = ArbitrageEngine(symbols)
    
    # Simulate price data for demo
    async def simulate_price_feeds():
        """Simulate price feeds from exchanges"""
        base_prices = {'BTCUSDT': 45000, 'ETHUSDT': 3000, 'BNBUSDT': 300}
        
        while True:
            for symbol in symbols:
                base_price = base_prices[symbol]
                
                # Simulate price differences between exchanges
                for exchange_name, connector in engine.connectors.items():
                    # Add some random variation and exchange-specific bias
                    price_variation = np.random.normal(0, 0.001)  # 0.1% variation
                    exchange_bias = {'binance': 0, 'coinbase': 0.002}  # Coinbase slightly higher
                    
                    adjusted_price = base_price * (1 + price_variation + exchange_bias.get(exchange_name, 0))
                    
                    # Create price data
                    spread = adjusted_price * 0.001  # 0.1% spread
                    price_data = PriceData(
                        exchange=exchange_name,
                        symbol=symbol,
                        bid=adjusted_price - spread/2,
                        ask=adjusted_price + spread/2,
                        timestamp=time.time(),
                        volume=1000
                    )
                    
                    connector.price_data[symbol] = price_data
                    connector.is_connected = True
                    connector.connection_stats['messages_received'] += 1
                    connector.connection_stats['last_message_time'] = time.time()
            
            await asyncio.sleep(0.5)  # Update every 500ms
    
    # Start simulation
    simulation_task = asyncio.create_task(simulate_price_feeds())
    
    # Run arbitrage detection for 30 seconds
    print("🚀 Starting arbitrage detection (30 seconds)...")
    
    try:
        await asyncio.wait_for(engine._arbitrage_detection_loop(), timeout=30.0)
    except asyncio.TimeoutError:
        print("⏰ Demo completed")
    
    # Stop simulation
    simulation_task.cancel()
    
    # Get final statistics
    stats = engine.get_arbitrage_statistics()
    
    print(f"\n📊 DEMO RESULTS:")
    print(f"   Opportunities Detected: {stats['opportunities_detected']}")
    print(f"   Profitable Opportunities: {stats['profitable_opportunities']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")
    print(f"   Trades Executed: {stats['trades_executed']}")
    print(f"   Total Simulated Profit: ${stats['total_profit']:.2f}")
    
    if stats['best_trade']:
        best = stats['best_trade']
        print(f"   Best Trade: {best['symbol']} - {best['profit_percentage']:.2%} profit")
    
    return engine

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_arbitrage_system())