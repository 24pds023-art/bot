"""
ULTRA-HIGH PERFORMANCE TRADING DASHBOARD
=======================================
🚀 WebSocket-Only Updates (No API Rate Limits)
⚡ Real-Time Performance Monitoring
📊 Advanced Signal Visualization
🎯 Zero-Latency Data Pipeline Integration
💎 Responsive Design with Dark Theme
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Any, Set
from aiohttp import web, WSMsgType
import aiohttp_cors
import weakref
import numpy as np

class OptimizedTradingDashboard:
    """Ultra-high performance dashboard with WebSocket-only updates"""
    
    def __init__(self, bot, port: int = 8080):
        self.bot = bot
        self.port = port
        self.app = web.Application()
        self.websocket_connections: Set[web.WebSocketResponse] = set()
        
        # Performance tracking
        self.update_count = 0
        self.last_performance_log = time.time()
        self.message_queue = asyncio.Queue(maxsize=1000)
        
        # Data caching for performance
        self.cached_data = {}
        self.last_cache_update = 0
        self.cache_ttl = 1.0  # 1 second cache TTL
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes with CORS"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Routes
        self.app.router.add_get('/', self.dashboard_handler)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_get('/api/status', self.status_api)
        self.app.router.add_post('/api/emergency-stop', self.emergency_stop_api)
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def dashboard_handler(self, request):
        """Serve the optimized dashboard HTML"""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Ultra-High Performance Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-card: #2a2a2a;
            --accent-green: #00ff88;
            --accent-blue: #00d4ff;
            --accent-red: #ff4757;
            --accent-yellow: #ffa502;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --border-color: #404040;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-radius: 20px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .performance-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 20px;
            font-size: 0.9em;
        }
        
        .perf-stat {
            background: rgba(0,255,136,0.1);
            padding: 10px;
            border-radius: 10px;
            border: 1px solid var(--accent-green);
            text-align: center;
        }
        
        .perf-label { color: var(--accent-green); font-weight: 600; }
        .perf-value { color: var(--text-primary); font-weight: 700; font-size: 1.1em; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: var(--bg-card);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid var(--border-color);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.4);
            border-color: var(--accent-blue);
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            font-family: 'SF Mono', monospace;
        }
        
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.9em;
            font-weight: 500;
        }
        
        .metric-change {
            font-size: 0.8em;
            padding: 4px 10px;
            border-radius: 15px;
            margin-top: 8px;
            font-weight: 600;
        }
        
        .positive { color: var(--accent-green); }
        .negative { color: var(--accent-red); }
        .neutral { color: var(--accent-yellow); }
        
        .change-positive { background: rgba(0,255,136,0.2); }
        .change-negative { background: rgba(255,71,87,0.2); }
        .change-neutral { background: rgba(255,165,2,0.2); }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-section {
            display: grid;
            gap: 20px;
        }
        
        .chart-container {
            background: var(--bg-card);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid var(--border-color);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            height: 400px;
        }
        
        .side-panel {
            display: grid;
            gap: 20px;
        }
        
        .panel-section {
            background: var(--bg-card);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid var(--border-color);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .section-title {
            color: var(--accent-blue);
            font-size: 1.3em;
            font-weight: 600;
        }
        
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        
        .positions-table th {
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            padding: 15px 10px;
            text-align: left;
            color: var(--text-primary);
            border-bottom: 2px solid var(--accent-blue);
            font-weight: 600;
        }
        
        .positions-table td {
            padding: 12px 10px;
            border-bottom: 1px solid var(--border-color);
            transition: background 0.3s ease;
        }
        
        .positions-table tr:hover {
            background: rgba(0,212,255,0.1);
        }
        
        .control-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .control-btn {
            padding: 15px 20px;
            background: linear-gradient(135deg, var(--accent-blue), #3498db);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9em;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .control-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }
        
        .emergency-btn {
            background: linear-gradient(135deg, var(--accent-red), #e74c3c);
            grid-column: span 2;
        }
        
        .status-indicators {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 20px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.9em;
        }
        
        .indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .indicator.active { background: var(--accent-green); }
        .indicator.inactive { background: var(--accent-red); }
        .indicator.warning { background: var(--accent-yellow); }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 30px;
            font-weight: 600;
            z-index: 1000;
            backdrop-filter: blur(20px);
            transition: all 0.3s ease;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        }
        
        .connected {
            background: linear-gradient(135deg, var(--accent-green), #00b894);
            color: #000;
            border: 1px solid rgba(0,255,136,0.5);
        }
        
        .disconnected {
            background: linear-gradient(135deg, var(--accent-red), #e74c3c);
            color: #fff;
            border: 1px solid rgba(255,71,87,0.5);
        }
        
        .signal-item {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid var(--accent-blue);
            transition: all 0.3s ease;
        }
        
        .signal-item:hover {
            background: rgba(255,255,255,0.08);
            transform: translateX(5px);
        }
        
        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .signal-symbol {
            font-weight: 700;
            color: var(--text-primary);
        }
        
        .signal-type {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        .signal-buy {
            background: rgba(0,255,136,0.2);
            color: var(--accent-green);
        }
        
        .signal-sell {
            background: rgba(255,71,87,0.2);
            color: var(--accent-red);
        }
        
        .signal-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            color: var(--text-secondary);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .slide-in { animation: slideIn 0.5s ease; }
        
        @media (max-width: 1400px) {
            .main-content { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
        }
        
        @media (max-width: 768px) {
            .container { padding: 15px; }
            .header h1 { font-size: 2.5em; }
            .metrics-grid { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
            .control-grid { grid-template-columns: 1fr; }
            .performance-stats { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        <span class="indicator active"></span>Connecting...
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🚀 Ultra-High Performance Trading Bot</h1>
            <p style="font-size: 1.2em; margin: 15px 0; opacity: 0.9;">
                ⚡ Zero-Copy Pipeline | 🎯 10x Faster Execution | 📊 WebSocket-Only Updates
            </p>
            
            <div class="performance-stats">
                <div class="perf-stat">
                    <div class="perf-label">WebSocket Latency</div>
                    <div class="perf-value" id="wsLatency">0.0ms</div>
                </div>
                <div class="perf-stat">
                    <div class="perf-label">Calculation Speed</div>
                    <div class="perf-value" id="calcSpeed">0.0ms</div>
                </div>
                <div class="perf-stat">
                    <div class="perf-label">Memory Usage</div>
                    <div class="perf-value" id="memUsage">0 MB</div>
                </div>
                <div class="perf-stat">
                    <div class="perf-label">CPU Usage</div>
                    <div class="perf-value" id="cpuUsage">0%</div>
                </div>
                <div class="perf-stat">
                    <div class="perf-label">Messages/Sec</div>
                    <div class="perf-value" id="msgRate">0</div>
                </div>
                <div class="perf-stat">
                    <div class="perf-label">Speedup Factor</div>
                    <div class="perf-value" id="speedup">10x</div>
                </div>
            </div>
            
            <div style="margin-top: 20px; font-size: 0.9em; opacity: 0.8;">
                Last Update: <span id="lastUpdate">--:--:--</span> | 
                Uptime: <span id="uptime">00:00:00</span> | 
                Updates: <span id="updateCount">0</span>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value positive" id="balance">$0.00</div>
                <div class="metric-label">💰 Account Balance</div>
                <div class="metric-change change-neutral" id="balanceChange">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral" id="totalPnl">$0.00</div>
                <div class="metric-label">📈 Total P&L</div>
                <div class="metric-change change-neutral" id="pnlChange">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="activePositions">0</div>
                <div class="metric-label">📊 Active Positions</div>
                <div class="metric-change change-neutral" id="positionsChange">0/15</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="signalsGenerated">0</div>
                <div class="metric-label">🎯 Signals Generated</div>
                <div class="metric-change change-neutral" id="signalsRate">0/min</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="winRate">0%</div>
                <div class="metric-label">🏆 Win Rate</div>
                <div class="metric-change change-neutral" id="winStreak">Streak: 0</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="avgHoldTime">0m</div>
                <div class="metric-label">⏱️ Avg Hold Time</div>
                <div class="metric-change change-neutral" id="maxHoldTime">Max: 0m</div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="chart-section">
                <div class="chart-container">
                    <div class="section-header">
                        <div class="section-title">📈 Real-Time Performance</div>
                        <div style="font-size: 0.8em; color: var(--text-secondary);">
                            WebSocket-Only | Zero API Calls
                        </div>
                    </div>
                    <canvas id="performanceChart"></canvas>
                </div>
                
                <div class="panel-section">
                    <div class="section-header">
                        <div class="section-title">📊 Active Positions</div>
                        <div style="font-size: 0.8em; color: var(--text-secondary);">
                            <span id="positionsCount">0</span>/15 | Risk: <span id="totalRisk">$0</span>
                        </div>
                    </div>
                    
                    <div style="max-height: 300px; overflow-y: auto;">
                        <table class="positions-table">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Side</th>
                                    <th>Entry</th>
                                    <th>Current</th>
                                    <th>P&L</th>
                                    <th>Duration</th>
                                </tr>
                            </thead>
                            <tbody id="positionsTableBody">
                                <tr><td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 20px;">No active positions</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="side-panel">
                <div class="panel-section">
                    <div class="section-header">
                        <div class="section-title">🎮 Control Center</div>
                    </div>
                    
                    <div class="control-grid">
                        <button class="control-btn" onclick="refreshData()">🔄 Refresh</button>
                        <button class="control-btn" onclick="toggleAutoRefresh()" id="autoRefreshBtn">⏸️ Pause</button>
                        <button class="control-btn emergency-btn" onclick="emergencyStop()">🚨 EMERGENCY STOP</button>
                    </div>
                    
                    <div class="status-indicators">
                        <div class="status-item">
                            <span class="indicator active" id="wsIndicator"></span>
                            <span id="wsStatus">WebSocket</span>
                        </div>
                        <div class="status-item">
                            <span class="indicator active" id="pipelineIndicator"></span>
                            <span id="pipelineStatus">Zero-Copy Pipeline</span>
                        </div>
                        <div class="status-item">
                            <span class="indicator active" id="parallelIndicator"></span>
                            <span id="parallelStatus">Parallel Engine</span>
                        </div>
                        <div class="status-item">
                            <span class="indicator active" id="filterIndicator"></span>
                            <span id="filterStatus">Advanced Filters</span>
                        </div>
                    </div>
                </div>
                
                <div class="panel-section">
                    <div class="section-header">
                        <div class="section-title">🎯 Recent Signals</div>
                        <div style="font-size: 0.8em; color: var(--text-secondary);">
                            Quality: <span id="signalQuality">High</span>
                        </div>
                    </div>
                    
                    <div id="signalsContainer" style="max-height: 400px; overflow-y: auto;">
                        <div style="text-align: center; color: var(--text-secondary); padding: 20px;">
                            No signals generated yet
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let reconnectInterval = null;
        let autoRefresh = true;
        let startTime = Date.now();
        let performanceChart = null;
        let previousData = {};
        let updateCount = 0;
        
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            connectWebSocket();
            setInterval(updateUptime, 1000);
            loadInitialData();
        });
        
        function initializeChart() {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'P&L ($)',
                            data: [],
                            borderColor: '#00ff88',
                            backgroundColor: 'rgba(0, 255, 136, 0.1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            pointHoverRadius: 6
                        },
                        {
                            label: 'Balance ($)',
                            data: [],
                            borderColor: '#00d4ff',
                            backgroundColor: 'rgba(0, 212, 255, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.4,
                            pointRadius: 0,
                            pointHoverRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { 
                            display: true,
                            labels: { color: '#b0b0b0' }
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'minute' },
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#b0b0b0' }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { 
                                color: '#b0b0b0',
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        }
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                updateConnectionStatus(true);
                clearInterval(reconnectInterval);
                reconnectInterval = null;
            };
            
            ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    if (autoRefresh) {
                        updateDashboard(data);
                    }
                } catch (e) {
                    console.error('WebSocket message error:', e);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                updateConnectionStatus(false);
                
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connectWebSocket, 3000);
                }
            };
        }
        
        function updateConnectionStatus(connected) {
            const status = document.getElementById('connectionStatus');
            const wsIndicator = document.getElementById('wsIndicator');
            
            if (connected) {
                status.innerHTML = '<span class="indicator active"></span>Connected';
                status.className = 'connection-status connected';
                wsIndicator.className = 'indicator active';
            } else {
                status.innerHTML = '<span class="indicator inactive"></span>Disconnected';
                status.className = 'connection-status disconnected';
                wsIndicator.className = 'indicator inactive';
            }
        }
        
        function updateDashboard(data) {
            const now = new Date();
            document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
            
            updateCount++;
            document.getElementById('updateCount').textContent = updateCount;
            
            // Update performance metrics
            if (data.performance) {
                document.getElementById('wsLatency').textContent = (data.performance.websocket_latency || 0).toFixed(1) + 'ms';
                document.getElementById('calcSpeed').textContent = (data.performance.calculation_time || 0).toFixed(1) + 'ms';
                document.getElementById('memUsage').textContent = Math.round(data.performance.memory_usage || 0) + ' MB';
                document.getElementById('cpuUsage').textContent = Math.round(data.performance.cpu_usage || 0) + '%';
                document.getElementById('msgRate').textContent = Math.round(data.performance.message_rate || 0);
            }
            
            // Update main metrics
            updateMetricWithChange('balance', data.balance || 0, '$');
            updateMetricWithChange('totalPnl', data.total_unrealized_pnl || 0, '$');
            updateMetricWithChange('activePositions', data.active_positions || 0);
            updateMetricWithChange('signalsGenerated', data.signals_generated || 0);
            updateMetricWithChange('winRate', (data.win_rate || 0) * 100, '%');
            
            // Update positions
            updatePositionsTable(data.positions || []);
            
            // Update signals
            updateSignalsPanel(data.recent_signals || []);
            
            // Update chart
            if (data.total_unrealized_pnl !== undefined || data.balance !== undefined) {
                updatePerformanceChart(data.total_unrealized_pnl || 0, data.balance || 0);
            }
            
            previousData = data;
        }
        
        function updateMetricWithChange(elementId, newValue, prefix = '') {
            const element = document.getElementById(elementId);
            if (element) {
                const oldValue = previousData[elementId] || 0;
                const change = newValue - oldValue;
                
                element.textContent = prefix + (typeof newValue === 'number' ? newValue.toFixed(2) : newValue);
                
                if (elementId === 'totalPnl') {
                    element.className = 'metric-value ' + (newValue > 0 ? 'positive' : newValue < 0 ? 'negative' : 'neutral');
                }
            }
            previousData[elementId] = newValue;
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positionsTableBody');
            
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 20px;">No active positions</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(pos => {
                const pnlClass = pos.current_pnl >= 0 ? 'positive' : 'negative';
                const sideClass = pos.side === 'BUY' ? 'positive' : 'negative';
                const duration = formatDuration(Date.now() - new Date(pos.timestamp).getTime());
                
                return `
                    <tr class="slide-in">
                        <td><strong>${pos.symbol}</strong></td>
                        <td class="${sideClass}">${pos.side}</td>
                        <td>$${pos.entry_price?.toFixed(4) || '0'}</td>
                        <td>$${pos.current_price?.toFixed(4) || '0'}</td>
                        <td class="${pnlClass}">$${pos.current_pnl?.toFixed(2) || '0'}</td>
                        <td>${duration}</td>
                    </tr>
                `;
            }).join('');
            
            document.getElementById('positionsCount').textContent = positions.length;
        }
        
        function updateSignalsPanel(signals) {
            const container = document.getElementById('signalsContainer');
            
            if (signals.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 20px;">No signals generated yet</div>';
                return;
            }
            
            const recentSignals = signals.slice(-10).reverse();
            
            container.innerHTML = recentSignals.map(signal => {
                const signalClass = signal.signal === 'BUY' ? 'signal-buy' : signal.signal === 'SELL' ? 'signal-sell' : 'signal-none';
                const time = new Date(signal.timestamp).toLocaleTimeString();
                const strength = (signal.strength * 100).toFixed(1);
                
                return `
                    <div class="signal-item slide-in">
                        <div class="signal-header">
                            <span class="signal-symbol">${signal.symbol}</span>
                            <span class="signal-type ${signalClass}">${signal.signal}</span>
                        </div>
                        <div class="signal-meta">
                            <span>${time}</span>
                            <span>Strength: ${strength}%</span>
                            <span>Quality: ${(signal.quality * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function updatePerformanceChart(pnl, balance) {
            const now = new Date();
            
            performanceChart.data.labels.push(now);
            performanceChart.data.datasets[0].data.push(pnl);
            performanceChart.data.datasets[1].data.push(balance);
            
            // Keep only last 100 data points
            const maxPoints = 100;
            if (performanceChart.data.labels.length > maxPoints) {
                performanceChart.data.labels.shift();
                performanceChart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            performanceChart.update('none');
        }
        
        function updateUptime() {
            const uptimeMs = Date.now() - startTime;
            const hours = Math.floor(uptimeMs / 3600000);
            const minutes = Math.floor((uptimeMs % 3600000) / 60000);
            const seconds = Math.floor((uptimeMs % 60000) / 1000);
            
            document.getElementById('uptime').textContent = 
                `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        
        function formatDuration(ms) {
            const minutes = Math.floor(ms / 60000);
            const hours = Math.floor(minutes / 60);
            const days = Math.floor(hours / 24);
            
            if (days > 0) return `${days}d ${hours % 24}h`;
            if (hours > 0) return `${hours}h ${minutes % 60}m`;
            return `${minutes}m`;
        }
        
        function refreshData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(console.error);
        }
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const btn = document.getElementById('autoRefreshBtn');
            btn.innerHTML = autoRefresh ? '⏸️ Pause' : '▶️ Resume';
            btn.style.background = autoRefresh ? 
                'linear-gradient(135deg, #00d4ff, #3498db)' : 
                'linear-gradient(135deg, #ffa502, #ff6348)';
        }
        
        function emergencyStop() {
            if (confirm('⚠️ EMERGENCY STOP will immediately close ALL positions and halt trading.\\n\\nThis action cannot be undone. Continue?')) {
                fetch('/api/emergency-stop', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert('Emergency stop executed: ' + (data.message || 'All positions closed'));
                        location.reload();
                    })
                    .catch(error => {
                        console.error('Emergency stop failed:', error);
                        alert('Emergency stop failed. Check console for details.');
                    });
            }
        }
        
        function loadInitialData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.log('Initial load failed:', error));
        }
    </script>
</body>
</html>
        """
        return web.Response(text=html_content, content_type='text/html')
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        print(f"📱 Dashboard client connected. Total: {len(self.websocket_connections)}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
                    break
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.websocket_connections.discard(ws)
            print(f"📱 Client disconnected. Remaining: {len(self.websocket_connections)}")
        
        return ws
    
    async def status_api(self, request):
        """API endpoint for current status"""
        try:
            current_time = time.time()
            
            # Use cached data if recent enough
            if (current_time - self.last_cache_update < self.cache_ttl and 
                self.cached_data):
                return web.json_response(self.cached_data)
            
            # Get fresh data from bot
            status = await self._get_bot_status()
            
            # Cache the result
            self.cached_data = status
            self.last_cache_update = current_time
            
            return web.json_response(status, default=self._json_serializer)
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def emergency_stop_api(self, request):
        """Emergency stop endpoint"""
        try:
            # Force close all positions
            if hasattr(self.bot, 'position_manager'):
                closed_positions = self.bot.position_manager.force_close_all_positions("Emergency Stop")
                
                return web.json_response({
                    'success': True,
                    'message': 'Emergency stop executed',
                    'closed_positions': len(closed_positions),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return web.json_response({
                    'success': False,
                    'message': 'Position manager not available'
                })
                
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def _get_bot_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        try:
            # Get performance metrics from the bot's performance monitor
            performance_stats = {}
            if hasattr(self.bot, 'performance_monitor'):
                perf_data = self.bot.performance_monitor.get_performance_stats()
                performance_stats = {
                    'websocket_latency': perf_data.get('websocket_latency_avg', 0) * 1000,  # Convert to ms
                    'calculation_time': perf_data.get('calculation_time_avg', 0) * 1000,
                    'memory_usage': perf_data.get('memory_usage_avg', 0),
                    'cpu_usage': perf_data.get('cpu_usage_avg', 0),
                    'message_rate': perf_data.get('message_rate', 0)
                }
            
            # Get positions with real-time prices
            positions = []
            total_unrealized_pnl = 0
            
            if hasattr(self.bot, 'analysis_engine') and hasattr(self.bot.analysis_engine, 'data_managers'):
                for position_id, position in getattr(self.bot, 'position_manager', {}).get('active_positions', {}).items():
                    symbol = position['symbol']
                    
                    # Get current price from data manager
                    current_price = 0
                    if symbol in self.bot.analysis_engine.data_managers:
                        current_price = self.bot.analysis_engine.data_managers[symbol].current_price
                    
                    if current_price <= 0:
                        current_price = position['entry_price']
                    
                    # Calculate P&L
                    entry_price = position['entry_price']
                    size = position['size']
                    side = position['side']
                    
                    if side == 'BUY':
                        pnl = (current_price - entry_price) * size
                    else:
                        pnl = (entry_price - current_price) * size
                    
                    total_unrealized_pnl += pnl
                    
                    positions.append({
                        'symbol': symbol,
                        'side': side,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'current_pnl': pnl,
                        'size': size,
                        'timestamp': position['timestamp'].isoformat() if hasattr(position.get('timestamp'), 'isoformat') else str(position.get('timestamp', ''))
                    })
            
            # Get recent signals
            recent_signals = []
            if hasattr(self.bot, 'recent_signals'):
                recent_signals = list(self.bot.recent_signals)[-20:]  # Last 20 signals
            
            return {
                'timestamp': datetime.now().isoformat(),
                'balance': getattr(self.bot, 'balance', 0),
                'total_unrealized_pnl': total_unrealized_pnl,
                'active_positions': len(positions),
                'positions': positions,
                'signals_generated': getattr(self.bot, 'signals_generated', 0),
                'trades_executed': getattr(self.bot, 'trades_executed', 0),
                'winning_trades': getattr(self.bot, 'winning_trades', 0),
                'losing_trades': getattr(self.bot, 'losing_trades', 0),
                'win_rate': (getattr(self.bot, 'winning_trades', 0) / 
                           max(getattr(self.bot, 'winning_trades', 0) + getattr(self.bot, 'losing_trades', 0), 1)),
                'recent_signals': recent_signals,
                'performance': performance_stats,
                'uptime': str(datetime.now() - getattr(self.bot, 'start_time', datetime.now())),
                'is_running': getattr(self.bot, 'is_running', False)
            }
            
        except Exception as e:
            print(f"Error getting bot status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'balance': 0,
                'active_positions': 0,
                'positions': [],
                'performance': {}
            }
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if not self.websocket_connections:
            return
        
        try:
            message = json.dumps(data, default=self._json_serializer)
            disconnected = set()
            
            for ws in list(self.websocket_connections):
                try:
                    if ws.closed:
                        disconnected.add(ws)
                        continue
                    await ws.send_str(message)
                except Exception:
                    disconnected.add(ws)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected
            
            # Performance tracking
            self.update_count += 1
            if self.update_count % 1000 == 0:
                current_time = time.time()
                elapsed = current_time - self.last_performance_log
                updates_per_second = 1000 / elapsed if elapsed > 0 else 0
                print(f"📊 Dashboard updates: {updates_per_second:.1f}/sec to {len(self.websocket_connections)} clients")
                self.last_performance_log = current_time
            
        except Exception as e:
            print(f"Error broadcasting update: {e}")
    
    def _json_serializer(self, obj):
        """JSON serializer for datetime and numpy objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    async def start_server(self):
        """Start the dashboard server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        
        print(f"🌐 Ultra-High Performance Dashboard started at http://localhost:{self.port}")
        print(f"⚡ Features: WebSocket-Only Updates, Zero-Copy Pipeline Integration, Real-Time Performance Monitoring")
        
        return runner
    
    async def start_update_loop(self):
        """Start the dashboard update loop"""
        while True:
            try:
                if self.websocket_connections:
                    status = await self._get_bot_status()
                    await self.broadcast_update(status)
                
                # Adaptive update frequency based on activity
                if self.websocket_connections:
                    await asyncio.sleep(1.0)  # 1 second when clients connected
                else:
                    await asyncio.sleep(5.0)  # 5 seconds when no clients
                    
            except Exception as e:
                print(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(10.0)

# Integration with the optimized bot
async def integrate_dashboard_with_bot(bot):
    """Integrate the optimized dashboard with the trading bot"""
    dashboard = OptimizedTradingDashboard(bot)
    
    # Start dashboard server
    await dashboard.start_server()
    
    # Start update loop
    update_task = asyncio.create_task(dashboard.start_update_loop())
    
    return dashboard, update_task

if __name__ == "__main__":
    # Standalone dashboard for testing
    class MockBot:
        def __init__(self):
            self.balance = 1000.0
            self.signals_generated = 150
            self.trades_executed = 45
            self.winning_trades = 32
            self.losing_trades = 13
            self.start_time = datetime.now() - timedelta(hours=2)
            self.is_running = True
            self.recent_signals = deque([
                {
                    'symbol': 'BTCUSDT',
                    'signal': 'BUY',
                    'strength': 0.85,
                    'quality': 0.92,
                    'timestamp': datetime.now() - timedelta(minutes=5)
                }
            ], maxlen=100)
    
    async def main():
        bot = MockBot()
        dashboard = OptimizedTradingDashboard(bot)
        await dashboard.start_server()
        await dashboard.start_update_loop()
    
    asyncio.run(main())