"""
DEEP LEARNING MODELS FOR TRADING
================================
🧠 Advanced neural networks: LSTM, GRU, Transformer, CNN
⚡ Time series prediction with attention mechanisms
📊 Multi-modal learning combining price, volume, and order book data
🎯 State-of-the-art architectures for maximum prediction accuracy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import deque
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# ============================================================================
# 1. LSTM-BASED PRICE PREDICTION MODEL
# ============================================================================

class LSTMPricePredictor(nn.Module):
    """Advanced LSTM model for price prediction"""
    
    def __init__(self, input_size: int = 50, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2, output_size: int = 1):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden_size)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # Back to (batch, seq_len, hidden_size)
        
        # Layer normalization
        attn_out = self.layer_norm(attn_out)
        
        # Take the last output
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

# ============================================================================
# 2. TRANSFORMER-BASED SIGNAL PREDICTION
# ============================================================================

class TransformerSignalPredictor(nn.Module):
    """Transformer model for trading signal prediction"""
    
    def __init__(self, input_dim: int = 50, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1, output_dim: int = 3):
        super(TransformerSignalPredictor, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)  # BUY, HOLD, SELL
        )
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Take the last token for classification
        last_token = transformer_out[-1]  # (batch, d_model)
        
        # Classify
        output = self.classifier(last_token)
        
        return F.softmax(output, dim=1)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# ============================================================================
# 3. CNN FOR PATTERN RECOGNITION
# ============================================================================

class CNNPatternRecognizer(nn.Module):
    """CNN for recognizing price patterns"""
    
    def __init__(self, input_channels: int = 5, sequence_length: int = 100, 
                 num_classes: int = 10):
        super(CNNPatternRecognizer, self).__init__()
        
        # 1D Convolution layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Convolution blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)

# ============================================================================
# 4. MULTI-MODAL FUSION MODEL
# ============================================================================

class MultiModalTradingModel(nn.Module):
    """Multi-modal model combining price, volume, and order book data"""
    
    def __init__(self, price_dim: int = 100, volume_dim: int = 100, 
                 orderbook_dim: int = 40, hidden_dim: int = 256):
        super(MultiModalTradingModel, self).__init__()
        
        # Price branch (LSTM)
        self.price_lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
        self.price_fc = nn.Linear(64, hidden_dim // 3)
        
        # Volume branch (CNN)
        self.volume_conv = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.volume_pool = nn.AdaptiveAvgPool1d(1)
        self.volume_fc = nn.Linear(32, hidden_dim // 3)
        
        # Order book branch (Dense)
        self.orderbook_fc1 = nn.Linear(orderbook_dim, 128)
        self.orderbook_fc2 = nn.Linear(128, hidden_dim // 3)
        
        # Fusion layers
        self.fusion_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1)  # Probability output
        )
        
    def forward(self, price_data, volume_data, orderbook_data):
        # Price branch
        price_out, _ = self.price_lstm(price_data.unsqueeze(-1))
        price_features = self.price_fc(price_out[:, -1, :])
        
        # Volume branch
        volume_out = F.relu(self.volume_conv(volume_data.unsqueeze(1)))
        volume_out = self.volume_pool(volume_out).squeeze(-1)
        volume_features = self.volume_fc(volume_out)
        
        # Order book branch
        orderbook_out = F.relu(self.orderbook_fc1(orderbook_data))
        orderbook_features = self.orderbook_fc2(orderbook_out)
        
        # Concatenate features
        fused_features = torch.cat([price_features, volume_features, orderbook_features], dim=1)
        
        # Apply attention
        fused_features = fused_features.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.fusion_attention(fused_features, fused_features, fused_features)
        attn_out = attn_out.squeeze(1)  # Remove sequence dimension
        
        # Final prediction
        output = torch.sigmoid(self.fusion_fc(attn_out))
        
        return output

# ============================================================================
# 5. DEEP LEARNING TRAINING MANAGER
# ============================================================================

class DeepLearningTrainingManager:
    """Manages training of all deep learning models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        self.device = device
        
        # Initialize models
        self.models['lstm'] = LSTMPricePredictor().to(self.device)
        self.models['transformer'] = TransformerSignalPredictor().to(self.device)
        self.models['cnn'] = CNNPatternRecognizer().to(self.device)
        self.models['multimodal'] = MultiModalTradingModel().to(self.device)
        
        # Initialize scalers
        self.scalers['price'] = MinMaxScaler()
        self.scalers['volume'] = StandardScaler()
        self.scalers['features'] = StandardScaler()
        
        print(f"🧠 Deep Learning models initialized on {self.device}")
    
    def prepare_lstm_data(self, price_history: List[float], 
                         sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        if len(price_history) < sequence_length + 1:
            return None, None
        
        # Scale prices
        prices = np.array(price_history).reshape(-1, 1)
        scaled_prices = self.scalers['price'].fit_transform(prices).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_prices) - sequence_length):
            X.append(scaled_prices[i:i+sequence_length])
            y.append(scaled_prices[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_transformer_data(self, market_features: List[Dict], 
                                sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for Transformer training"""
        if len(market_features) < sequence_length + 1:
            return None, None
        
        # Extract features
        feature_arrays = []
        labels = []
        
        for i, features in enumerate(market_features):
            feature_vector = [
                features.get('rsi', 50),
                features.get('macd', 0),
                features.get('bb_position', 0.5),
                features.get('volume_ratio', 1.0),
                features.get('price_momentum', 0),
                # Add more features as needed
            ]
            feature_arrays.append(feature_vector)
            
            # Label: future price direction (simplified)
            if i < len(market_features) - 1:
                current_price = features.get('current_price', 0)
                future_price = market_features[i+1].get('current_price', 0)
                
                if future_price > current_price * 1.001:  # 0.1% threshold
                    labels.append([1, 0, 0])  # BUY
                elif future_price < current_price * 0.999:
                    labels.append([0, 0, 1])  # SELL
                else:
                    labels.append([0, 1, 0])  # HOLD
        
        if len(feature_arrays) < sequence_length:
            return None, None
        
        # Scale features
        features_array = np.array(feature_arrays)
        scaled_features = self.scalers['features'].fit_transform(features_array)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:i+sequence_length])
            if i < len(labels):
                y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, price_history: List[float], 
                        epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """Train LSTM price prediction model"""
        print("🔥 Training LSTM model...")
        
        # Prepare data
        X, y = self.prepare_lstm_data(price_history)
        if X is None:
            return {'error': 'Insufficient data'}
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)  # Add feature dimension
        y_tensor = torch.FloatTensor(y).unsqueeze(-1).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        model = self.models['lstm']
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.training_history['lstm'] = losses
        
        return {
            'final_loss': losses[-1],
            'epochs_trained': epochs,
            'min_loss': min(losses)
        }
    
    def train_transformer_model(self, market_features: List[Dict], 
                              epochs: int = 50, batch_size: int = 16) -> Dict[str, float]:
        """Train Transformer signal prediction model"""
        print("🤖 Training Transformer model...")
        
        # Prepare data
        X, y = self.prepare_transformer_data(market_features)
        if X is None:
            return {'error': 'Insufficient data'}
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        model = self.models['transformer']
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        model.train()
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.argmax(dim=1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(batch_y.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        self.training_history['transformer'] = {'losses': losses, 'accuracies': accuracies}
        
        return {
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1],
            'best_accuracy': max(accuracies),
            'epochs_trained': epochs
        }
    
    def predict_price_lstm(self, recent_prices: List[float], 
                          steps_ahead: int = 1) -> List[float]:
        """Predict future prices using LSTM"""
        if len(recent_prices) < 50:
            return [recent_prices[-1]] * steps_ahead
        
        model = self.models['lstm']
        model.eval()
        
        # Prepare input
        scaled_prices = self.scalers['price'].transform(
            np.array(recent_prices).reshape(-1, 1)
        ).flatten()
        
        # Take last 50 prices
        input_seq = scaled_prices[-50:]
        predictions = []
        
        with torch.no_grad():
            for _ in range(steps_ahead):
                # Prepare input tensor
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(self.device)
                
                # Predict
                pred = model(input_tensor)
                pred_value = pred.cpu().numpy()[0, 0]
                
                # Inverse transform
                pred_price = self.scalers['price'].inverse_transform([[pred_value]])[0, 0]
                predictions.append(pred_price)
                
                # Update sequence for next prediction
                input_seq = np.append(input_seq[1:], pred_value)
        
        return predictions
    
    def predict_signal_transformer(self, market_features: List[Dict]) -> Dict[str, float]:
        """Predict trading signal using Transformer"""
        if len(market_features) < 30:
            return {'buy': 0.33, 'hold': 0.34, 'sell': 0.33}
        
        model = self.models['transformer']
        model.eval()
        
        # Prepare features
        feature_arrays = []
        for features in market_features[-30:]:  # Last 30 observations
            feature_vector = [
                features.get('rsi', 50),
                features.get('macd', 0),
                features.get('bb_position', 0.5),
                features.get('volume_ratio', 1.0),
                features.get('price_momentum', 0),
            ]
            feature_arrays.append(feature_vector)
        
        # Scale and convert to tensor
        features_array = np.array(feature_arrays)
        scaled_features = self.scalers['features'].transform(features_array)
        input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = outputs.cpu().numpy()[0]
        
        return {
            'buy': float(probabilities[0]),
            'hold': float(probabilities[1]),
            'sell': float(probabilities[2])
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        performance = {}
        
        for model_name, model in self.models.items():
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            performance[model_name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
                'training_history': self.training_history.get(model_name, {})
            }
        
        return performance
    
    def save_models(self, path_prefix: str = 'model'):
        """Save all trained models"""
        for model_name, model in self.models.items():
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': self.scalers.get(model_name.split('_')[0], None)
            }, f"{path_prefix}_{model_name}.pth")
        
        print(f"💾 Models saved with prefix: {path_prefix}")
    
    def load_models(self, path_prefix: str = 'model'):
        """Load pre-trained models"""
        for model_name, model in self.models.items():
            try:
                checkpoint = torch.load(f"{path_prefix}_{model_name}.pth", map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded {model_name} model")
            except FileNotFoundError:
                print(f"⚠️ No saved model found for {model_name}")

# ============================================================================
# 6. EXAMPLE USAGE AND DEMO
# ============================================================================

def demo_deep_learning_models():
    """Demonstrate deep learning models"""
    print("🧠 Deep Learning Models Demonstration")
    print("=" * 50)
    
    # Initialize training manager
    trainer = DeepLearningTrainingManager()
    
    # Generate sample data
    np.random.seed(42)
    
    # Sample price history
    price_history = []
    base_price = 45000
    for i in range(500):
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        base_price *= (1 + price_change)
        price_history.append(base_price)
    
    # Sample market features
    market_features = []
    for i in range(200):
        features = {
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.normal(0, 5),
            'bb_position': np.random.uniform(0, 1),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'price_momentum': np.random.normal(0, 0.01),
            'current_price': price_history[i] if i < len(price_history) else 45000
        }
        market_features.append(features)
    
    # Train LSTM model
    print("\n1. Training LSTM Price Predictor...")
    lstm_results = trainer.train_lstm_model(price_history, epochs=50, batch_size=16)
    print(f"LSTM Results: {lstm_results}")
    
    # Train Transformer model
    print("\n2. Training Transformer Signal Predictor...")
    transformer_results = trainer.train_transformer_model(market_features, epochs=30, batch_size=8)
    print(f"Transformer Results: {transformer_results}")
    
    # Make predictions
    print("\n3. Making Predictions...")
    
    # LSTM price prediction
    future_prices = trainer.predict_price_lstm(price_history[-100:], steps_ahead=5)
    print(f"LSTM Price Predictions (next 5 steps): {[f'{p:.2f}' for p in future_prices]}")
    
    # Transformer signal prediction
    signal_probs = trainer.predict_signal_transformer(market_features[-50:])
    print(f"Transformer Signal Probabilities: {signal_probs}")
    
    # Get model performance
    performance = trainer.get_model_performance()
    print(f"\n4. Model Performance:")
    for model_name, perf in performance.items():
        print(f"   {model_name}: {perf['total_parameters']:,} parameters, "
              f"{perf['model_size_mb']:.1f}MB")
    
    # Save models
    trainer.save_models('demo_model')
    
    return trainer

if __name__ == "__main__":
    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run demonstration
    trainer = demo_deep_learning_models()
    
    print("\n🎯 Deep Learning Models Ready!")
    print("   • LSTM for price prediction")
    print("   • Transformer for signal classification") 
    print("   • CNN for pattern recognition")
    print("   • Multi-modal fusion model")
    print("   • All models optimized for GPU acceleration")