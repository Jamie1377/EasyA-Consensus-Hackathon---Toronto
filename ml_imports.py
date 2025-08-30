"""
ML Strategy Import Configuration
Handles importing and configuring ML models and technical analysis libraries
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import TA library
try:
    import ta
    HAS_TA = True
    print("✓ TA library imported successfully")
except ImportError:
    HAS_TA = False
    print("⚠ TA library not available, using fallback implementations")

# Try to import custom models from STA410 package
try:
    import sys
    # Try to import from local stock_prediction package first
    from stock_prediction.core.models import ARIMAXGBoost, GradientDescentRegressor
    HAS_CUSTOM_MODELS = True
    print("✓ Custom models imported successfully from local package")
except ImportError:
    try:
        # Fallback: try to import from external STA410 package
        sys.path.append('/Users/jamie/Downloads/ml/STA410_Package')
        from stock_prediction.core.models import ARIMAXGBoost, GradientDescentRegressor
        HAS_CUSTOM_MODELS = True
        print("✓ Custom STA410 models imported successfully from external package")
    except ImportError as e:
        HAS_CUSTOM_MODELS = False
        print(f"⚠ Custom models not available: {e}")

# Standard ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import SGDRegressor


# Fallback implementations
class SimpleGradientDescentRegressor:
    """Fallback implementation when custom GD model is not available"""
    
    def __init__(self, n_iter=1000, lr=0.01, alpha=0.01, l1_ratio=0.01, 
                 momentum=0.9, random_state=42):
        self.n_iter = n_iter
        self.lr = lr
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.momentum = momentum
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0
        
    def optimize_hyperparameters_optuna(self, X, y, n_trials=20):
        """Placeholder optimization"""
        print(f"Mock optimization with {n_trials} trials")
        return {}
        
    def fit(self, X, y):
        """Simple SGD implementation"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.coef_ = np.random.randn(n_features) * 0.01
        self.intercept_ = 0
        
        # Simple gradient descent
        for i in range(self.n_iter):
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y
            
            # Gradients
            dw = (2/n_samples) * X.T @ error + self.alpha * self.coef_
            db = (2/n_samples) * np.sum(error)
            
            # Update with learning rate decay
            lr_current = self.lr / (1 + 0.001 * i)
            self.coef_ -= lr_current * dw
            self.intercept_ -= lr_current * db
            
        return self
        
    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class SimpleARIMAXGBoost:
    """Fallback implementation when custom ARIMA model is not available"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)


# Technical indicator fallbacks
def calculate_rsi(prices, window=14):
    """Simple RSI calculation"""
    if len(prices) < window + 1:
        return 50  # Neutral RSI
        
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Simple MACD calculation"""
    if len(prices) < slow:
        return 0, 0, 0
        
    prices_series = pd.Series(prices)
    exp1 = prices_series.ewm(span=fast).mean()
    exp2 = prices_series.ewm(span=slow).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    
    return macd.iloc[-1], macd_signal.iloc[-1], macd_hist.iloc[-1]


def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
    """Simple Stochastic oscillator"""
    if len(closes) < k_period:
        return 50, 50
        
    lowest_low = pd.Series(lows).rolling(k_period).min()
    highest_high = pd.Series(highs).rolling(k_period).max()
    
    k_percent = 100 * ((pd.Series(closes) - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(d_period).mean()
    
    return k_percent.iloc[-1], d_percent.iloc[-1]


# Configure which models to use
if HAS_CUSTOM_MODELS:
    MLGradientDescent = GradientDescentRegressor
    MLEnsemble = ARIMAXGBoost
else:
    MLGradientDescent = SimpleGradientDescentRegressor
    MLEnsemble = SimpleARIMAXGBoost

print(f"Model configuration: Custom={HAS_CUSTOM_MODELS}, TA={HAS_TA}")
