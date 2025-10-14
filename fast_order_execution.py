"""
ULTRA-FAST ORDER EXECUTION MODULE
=================================
🚀 Pre-authorized order templates for instant execution
⚡ Connection pooling and persistent sessions
📊 Order batching and pipelining
🎯 Sub-50ms execution times
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, getcontext
import json
import hmac
import hashlib
from urllib.parse import urlencode
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, deque

# Set decimal precision for financial calculations
getcontext().prec = 10

@dataclass
class OrderTemplate:
    """Pre-computed order template for instant execution"""
    symbol: str
    type: str = 'MARKET'
    timeInForce: str = 'GTC'
    quantity_precision: int = 3
    price_precision: int = 2
    min_qty: float = 0.001
    max_qty: float = 1000000.0
    min_notional: float = 5.0
    tick_size: float = 0.01
    step_size: float = 0.001

class UltraFastOrderExecution:
    """Ultra-fast order execution with pre-cached templates and connection pooling"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # API endpoints
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
        
        self.order_endpoint = f"{self.base_url}/fapi/v1/order"
        self.batch_order_endpoint = f"{self.base_url}/fapi/v1/batchOrders"
        
        # Pre-cached templates and precision data
        self.order_templates: Dict[str, OrderTemplate] = {}
        self.symbol_filters: Dict[str, Dict] = {}
        
        # Connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        self.connection_pool_size = 100
        self.request_timeout = 5.0
        
        # Performance tracking
        self.execution_times = deque(maxlen=1000)
        self.success_count = 0
        self.error_count = 0
        self.total_orders = 0
        
        # Order batching
        self.batch_queue = asyncio.Queue(maxsize=100)
        self.batch_size = 5
        self.batch_timeout = 0.1  # 100ms
        
        print("🚀 Ultra-Fast Order Execution initialized")
    
    async def initialize(self, binance_client):
        """Initialize execution engine with symbol information"""
        print("🔧 Initializing Ultra-Fast Order Execution...")
        
        # Setup persistent HTTP session
        await self._setup_persistent_session()
        
        # Cache all symbol information
        await self._cache_symbol_info(binance_client)
        
        # Start batch processing
        asyncio.create_task(self._batch_processor())
        
        print(f"✅ Order execution initialized with {len(self.order_templates)} symbols")
    
    async def _setup_persistent_session(self):
        """Setup persistent HTTP session with optimized settings"""
        connector = aiohttp.TCPConnector(
            limit=self.connection_pool_size,
            limit_per_host=50,
            ttl_dns_cache=600,  # 10 minutes DNS cache
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.request_timeout,
            connect=2.0,
            sock_read=3.0
        )
        
        headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'UltraFastTrader/1.0'
        }
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        print("✅ Persistent HTTP session established")
    
    async def _cache_symbol_info(self, binance_client):
        """Cache all symbol information for instant order creation"""
        try:
            exchange_info = await binance_client.futures_exchange_info()
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                
                # Skip symbols we're not trading
                if symbol not in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 
                                'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'LTCUSDT',
                                'UNIUSDT', 'ATOMUSDT', 'VETUSDT', 'ALGOUSDT', 'DOGEUSDT',
                                'NEARUSDT', 'SANDUSDT', 'MANAUSDT', 'ARBUSDT', 'OPUSDT',
                                'FILUSDT', 'ETCUSDT', 'AAVEUSDT', 'COMPUSDT', 'SNXUSDT',
                                'INJUSDT', 'SUIUSDT', 'APTUSDT', 'ARKMUSDT', 'IMXUSDT']:
                    continue
                
                # Extract filter information
                filters = {}
                template = OrderTemplate(symbol=symbol)
                
                for filter_info in symbol_info['filters']:
                    filter_type = filter_info['filterType']
                    filters[filter_type] = filter_info
                    
                    if filter_type == 'LOT_SIZE':
                        template.step_size = float(filter_info['stepSize'])
                        template.min_qty = float(filter_info['minQty'])
                        template.max_qty = float(filter_info['maxQty'])
                        template.quantity_precision = self._get_precision(filter_info['stepSize'])
                    
                    elif filter_type == 'PRICE_FILTER':
                        template.tick_size = float(filter_info['tickSize'])
                        template.price_precision = self._get_precision(filter_info['tickSize'])
                    
                    elif filter_type == 'MIN_NOTIONAL':
                        template.min_notional = float(filter_info['minNotional'])
                
                self.order_templates[symbol] = template
                self.symbol_filters[symbol] = filters
            
            print(f"✅ Cached order templates for {len(self.order_templates)} symbols")
            
        except Exception as e:
            print(f"❌ Error caching symbol info: {e}")
            raise
    
    def _get_precision(self, step_size: str) -> int:
        """Calculate precision from step size string"""
        if '.' not in step_size:
            return 0
        return len(step_size.rstrip('0').split('.')[-1])
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to symbol precision"""
        if symbol not in self.order_templates:
            return round(quantity, 3)  # Default precision
        
        template = self.order_templates[symbol]
        
        # Round to step size
        rounded = round(quantity / template.step_size) * template.step_size
        
        # Apply precision
        rounded = round(rounded, template.quantity_precision)
        
        # Ensure minimum quantity
        if rounded < template.min_qty:
            rounded = template.min_qty
        
        # Ensure maximum quantity
        if rounded > template.max_qty:
            rounded = template.max_qty
        
        return rounded
    
    def _validate_order(self, symbol: str, side: str, quantity: float) -> Tuple[bool, str, float]:
        """Validate order parameters"""
        if symbol not in self.order_templates:
            return False, f"Symbol {symbol} not supported", quantity
        
        template = self.order_templates[symbol]
        
        # Round quantity
        rounded_qty = self._round_quantity(symbol, quantity)
        
        # Check minimum quantity
        if rounded_qty < template.min_qty:
            return False, f"Quantity {rounded_qty} below minimum {template.min_qty}", rounded_qty
        
        # Check maximum quantity
        if rounded_qty > template.max_qty:
            return False, f"Quantity {rounded_qty} above maximum {template.max_qty}", rounded_qty
        
        # Validate side
        if side not in ['BUY', 'SELL']:
            return False, f"Invalid side: {side}", rounded_qty
        
        return True, "Valid", rounded_qty
    
    def _create_signature(self, params: Dict[str, Any]) -> str:
        """Create HMAC signature for Binance API"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def execute_market_order(self, symbol: str, side: str, quantity: float, 
                                 client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute single market order with ultra-fast execution"""
        start_time = time.perf_counter()
        
        try:
            # Validate order
            is_valid, error_msg, rounded_qty = self._validate_order(symbol, side, quantity)
            if not is_valid:
                return {
                    'success': False,
                    'error': error_msg,
                    'execution_time': time.perf_counter() - start_time
                }
            
            # Create order parameters
            timestamp = int(time.time() * 1000)
            params = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': str(rounded_qty),
                'timestamp': timestamp
            }
            
            if client_order_id:
                params['newClientOrderId'] = client_order_id
            
            # Add signature
            params['signature'] = self._create_signature(params)
            
            # Execute order
            async with self.session.post(self.order_endpoint, data=params) as response:
                response_data = await response.json()
                
                execution_time = time.perf_counter() - start_time
                self.execution_times.append(execution_time)
                
                if response.status == 200:
                    self.success_count += 1
                    self.total_orders += 1
                    
                    return {
                        'success': True,
                        'orderId': response_data.get('orderId'),
                        'symbol': symbol,
                        'side': side,
                        'quantity': rounded_qty,
                        'status': response_data.get('status'),
                        'fills': response_data.get('fills', []),
                        'avgFillPrice': self._calculate_avg_fill_price(response_data.get('fills', [])),
                        'execution_time': execution_time
                    }
                else:
                    self.error_count += 1
                    self.total_orders += 1
                    
                    return {
                        'success': False,
                        'error': response_data.get('msg', 'Unknown error'),
                        'code': response_data.get('code'),
                        'execution_time': execution_time
                    }
        
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.error_count += 1
            self.total_orders += 1
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _calculate_avg_fill_price(self, fills: List[Dict]) -> float:
        """Calculate average fill price from fills"""
        if not fills:
            return 0.0
        
        total_qty = 0.0
        total_value = 0.0
        
        for fill in fills:
            price = float(fill['price'])
            qty = float(fill['qty'])
            total_value += price * qty
            total_qty += qty
        
        return total_value / total_qty if total_qty > 0 else 0.0
    
    async def execute_batch_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple orders in a single batch request"""
        if not orders:
            return []
        
        start_time = time.perf_counter()
        
        try:
            # Validate and prepare all orders
            batch_orders = []
            results = []
            
            for i, order in enumerate(orders):
                symbol = order['symbol']
                side = order['side']
                quantity = order['quantity']
                
                # Validate order
                is_valid, error_msg, rounded_qty = self._validate_order(symbol, side, quantity)
                if not is_valid:
                    results.append({
                        'success': False,
                        'error': error_msg,
                        'order_index': i
                    })
                    continue
                
                # Create batch order
                batch_order = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'MARKET',
                    'quantity': str(rounded_qty)
                }
                
                if 'clientOrderId' in order:
                    batch_order['newClientOrderId'] = order['clientOrderId']
                
                batch_orders.append(batch_order)
            
            if not batch_orders:
                return results
            
            # Create batch request parameters
            timestamp = int(time.time() * 1000)
            params = {
                'batchOrders': json.dumps(batch_orders),
                'timestamp': timestamp
            }
            
            # Add signature
            params['signature'] = self._create_signature(params)
            
            # Execute batch order
            async with self.session.post(self.batch_order_endpoint, data=params) as response:
                response_data = await response.json()
                
                execution_time = time.perf_counter() - start_time
                
                if response.status == 200:
                    # Process batch results
                    for i, order_result in enumerate(response_data):
                        if 'orderId' in order_result:
                            self.success_count += 1
                            results.append({
                                'success': True,
                                'orderId': order_result.get('orderId'),
                                'symbol': order_result.get('symbol'),
                                'side': order_result.get('side'),
                                'quantity': float(order_result.get('origQty', 0)),
                                'status': order_result.get('status'),
                                'avgFillPrice': self._calculate_avg_fill_price(order_result.get('fills', [])),
                                'order_index': i,
                                'execution_time': execution_time
                            })
                        else:
                            self.error_count += 1
                            results.append({
                                'success': False,
                                'error': order_result.get('msg', 'Unknown error'),
                                'code': order_result.get('code'),
                                'order_index': i,
                                'execution_time': execution_time
                            })
                    
                    self.total_orders += len(batch_orders)
                else:
                    # Batch request failed
                    error_msg = response_data.get('msg', 'Batch order failed')
                    for i in range(len(batch_orders)):
                        results.append({
                            'success': False,
                            'error': error_msg,
                            'order_index': i,
                            'execution_time': execution_time
                        })
                    
                    self.error_count += len(batch_orders)
                    self.total_orders += len(batch_orders)
                
                return results
        
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            error_results = []
            
            for i in range(len(orders)):
                error_results.append({
                    'success': False,
                    'error': str(e),
                    'order_index': i,
                    'execution_time': execution_time
                })
            
            self.error_count += len(orders)
            self.total_orders += len(orders)
            
            return error_results
    
    async def queue_order_for_batch(self, symbol: str, side: str, quantity: float, 
                                   client_order_id: Optional[str] = None):
        """Queue order for batch execution"""
        order = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity
        }
        
        if client_order_id:
            order['clientOrderId'] = client_order_id
        
        try:
            await asyncio.wait_for(self.batch_queue.put(order), timeout=1.0)
        except asyncio.TimeoutError:
            print("⚠️ Batch queue full, executing immediately")
            return await self.execute_market_order(symbol, side, quantity, client_order_id)
    
    async def _batch_processor(self):
        """Process batched orders periodically"""
        while True:
            try:
                batch = []
                batch_start_time = time.time()
                
                # Collect orders for batch
                while len(batch) < self.batch_size:
                    try:
                        # Wait for order with timeout
                        remaining_time = self.batch_timeout - (time.time() - batch_start_time)
                        if remaining_time <= 0:
                            break
                        
                        order = await asyncio.wait_for(
                            self.batch_queue.get(), 
                            timeout=remaining_time
                        )
                        batch.append(order)
                        
                    except asyncio.TimeoutError:
                        break
                
                # Execute batch if we have orders
                if batch:
                    results = await self.execute_batch_orders(batch)
                    
                    # Log batch execution
                    successful = sum(1 for r in results if r.get('success', False))
                    print(f"📦 Batch executed: {successful}/{len(batch)} successful")
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Error in batch processor: {e}")
                await asyncio.sleep(1.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        if not self.execution_times:
            return {
                'total_orders': self.total_orders,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'success_rate': 0.0
            }
        
        execution_times_ms = [t * 1000 for t in self.execution_times]  # Convert to milliseconds
        
        return {
            'total_orders': self.total_orders,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / max(self.total_orders, 1),
            'avg_execution_time_ms': np.mean(execution_times_ms),
            'p50_execution_time_ms': np.percentile(execution_times_ms, 50),
            'p95_execution_time_ms': np.percentile(execution_times_ms, 95),
            'p99_execution_time_ms': np.percentile(execution_times_ms, 99),
            'min_execution_time_ms': np.min(execution_times_ms),
            'max_execution_time_ms': np.max(execution_times_ms),
            'cached_symbols': len(self.order_templates)
        }
    
    async def close(self):
        """Close the execution engine and cleanup resources"""
        if self.session:
            await self.session.close()
        print("✅ Ultra-Fast Order Execution closed")

# Example usage and testing
async def test_fast_execution():
    """Test the fast execution system"""
    # This would use actual API credentials in production
    api_key = "test_key"
    api_secret = "test_secret"
    
    executor = UltraFastOrderExecution(api_key, api_secret, testnet=True)
    
    # Simulate initialization (would use real Binance client)
    class MockBinanceClient:
        async def futures_exchange_info(self):
            return {
                'symbols': [
                    {
                        'symbol': 'BTCUSDT',
                        'filters': [
                            {'filterType': 'LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001', 'maxQty': '1000'},
                            {'filterType': 'PRICE_FILTER', 'tickSize': '0.10'},
                            {'filterType': 'MIN_NOTIONAL', 'minNotional': '5.0'}
                        ]
                    }
                ]
            }
    
    try:
        await executor.initialize(MockBinanceClient())
        
        # Test single order
        result = await executor.execute_market_order('BTCUSDT', 'BUY', 0.001)
        print(f"Single order result: {result}")
        
        # Test batch orders
        batch_orders = [
            {'symbol': 'BTCUSDT', 'side': 'BUY', 'quantity': 0.001},
            {'symbol': 'BTCUSDT', 'side': 'SELL', 'quantity': 0.001}
        ]
        batch_results = await executor.execute_batch_orders(batch_orders)
        print(f"Batch order results: {batch_results}")
        
        # Get performance stats
        stats = executor.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    finally:
        await executor.close()

if __name__ == "__main__":
    asyncio.run(test_fast_execution())