import os
import sys
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import logging
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from typing import List, Dict, Tuple, Optional

# Import ML configuration and models
from ml_imports import (
    HAS_TA, HAS_CUSTOM_MODELS, 
    MLGradientDescent, MLEnsemble,
    calculate_rsi, calculate_macd, calculate_stochastic
)

# Scientific computing
from scipy import stats
from scipy.optimize import minimize

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import SGDRegressor

# Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Backtesting infrastructure
from backtester import AptosBacktester, create_multi_stock_signal_generator
from predictor import StockPredictor

# Market complexity measures
from scipy.spatial.distance import correlation
from scipy.stats import entropy

# Visualization
import matplotlib.pyplot as plt

# Configure logging
log_directory = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_directory, f"ml_strategy_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Optional numba acceleration for heavy numeric loops
try:
    import numba as nb
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False


if HAS_NUMBA:
    @nb.njit
    def _fractaldim_numba_window(normalized, start_idx, window, scales):
        # normalized: 1D array of floats
        n_scales = scales.shape[0]
        counts = np.zeros(n_scales, dtype=np.int64)
        N = normalized.shape[0]
        for s_idx in range(n_scales):
            scale = scales[s_idx]
            # grid size computed similar to Python version
            if scale <= 0:
                counts[s_idx] = 0
                continue
            grid_size = int(1.0 / scale)
            if grid_size < 2:
                counts[s_idx] = 0
                continue

            # small occupancy grid
            arr = np.zeros((grid_size, grid_size), dtype=np.uint8)
            for k in range(window):
                idx = start_idx + k
                # map time index within window to x coordinate
                x_box = int((k) / float(window) * grid_size)
                y_val = normalized[idx]
                y_box = int(y_val * grid_size)
                if x_box < 0:
                    x_box = 0
                if y_box < 0:
                    y_box = 0
                if x_box >= grid_size:
                    x_box = grid_size - 1
                if y_box >= grid_size:
                    y_box = grid_size - 1
                arr[x_box, y_box] = 1

            # count occupied boxes
            s = 0
            for a in range(grid_size):
                for b in range(grid_size):
                    s += arr[a, b]
            counts[s_idx] = s

        return counts



class MarketComplexityMeasures:
    """
    Calculate market complexity measures like fractal dimension and approximate entropy
    """
    
    @staticmethod
    def fractal_dimension(series, window=50):
        """Calculate fractal dimension using box counting method"""
        if len(series) < window:
            return 1.5  # Default value
        
        # Normalize the series
        normalized = (series - series.min()) / (series.max() - series.min() + 1e-8)
        
        # Box counting approach
        scales = np.logspace(0.01, 1, 20)

        # Use numba-accelerated routine when available
        try:
            if HAS_NUMBA:
                arr = normalized.values if hasattr(normalized, 'values') else np.array(normalized)
                # ensure float64
                arr = arr.astype(np.float64)
                n = arr.shape[0]
                start_idx = n - window
                scales_arr = np.array(scales, dtype=np.float64)
                counts_arr = _fractaldim_numba_window(arr, start_idx, window, scales_arr)
                counts = [int(c) for c in counts_arr if c > 0]
            else:
                counts = []
                for scale in scales:
                    # Grid size
                    grid_size = int(1 / scale)
                    if grid_size < 2:
                        continue
                    
                    # Count boxes that contain data points
                    boxes = set()
                    window_vals = normalized[-window:]
                    for i, val in enumerate(window_vals):
                        x_box = int(i / len(window_vals) * grid_size)
                        y_box = int(val * grid_size)
                        boxes.add((x_box, y_box))
                    counts.append(len(boxes))
        except Exception:
            # fallback to previous python approach on any failure
            counts = []
            for scale in scales:
                grid_size = int(1 / scale)
                if grid_size < 2:
                    continue
                boxes = set()
                window_vals = normalized[-window:]
                for i, val in enumerate(window_vals):
                    x_box = int(i / len(window_vals) * grid_size)
                    y_box = int(val * grid_size)
                    boxes.add((x_box, y_box))
                counts.append(len(boxes))
        
        if len(counts) < 2:
            return 1.5
        
        # Linear regression to find fractal dimension
        log_scales = np.log(scales[:len(counts)])
        log_counts = np.log(counts)
        
        try:
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return abs(slope)
        except:
            return 1.5
    
    @staticmethod
    def approximate_entropy(series, m=2, r=None, window=100):
        """Calculate approximate entropy of a time series"""
        if len(series) < window:
            series_data = series.values if hasattr(series, 'values') else series
        else:
            series_data = series[-window:].values if hasattr(series, 'values') else series[-window:]
        
        N = len(series_data)
        
        if r is None:
            r = 0.2 * np.std(series_data)
        
        def _maxdist(xi, xj, N, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([series_data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j], N, m) <= r:
                        C[i] += 1.0
                        
            C = C / (N - m + 1.0)
            C = C[C > 0]  # Remove zeros to avoid log(0)
            
            if len(C) == 0:
                return 0
            
            phi = np.sum(np.log(C)) / len(C)
            return phi
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.5  # Default value


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for ML strategy
    """
    
    @staticmethod
    def create_technical_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators"""
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Moving averages and momentum
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI and other indicators
        if HAS_TA:
            import ta
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            stoch_osc = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch_osc.stoch()
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
        else:
            # Fallback implementations
            df['RSI'] = df['Close'].rolling(14).apply(lambda x: calculate_rsi(x.values) if len(x) >= 14 else 50)
            
            # Simple Stochastic
            if 'High' in df.columns and 'Low' in df.columns:
                stoch_k, stoch_d = calculate_stochastic(df['High'], df['Low'], df['Close'])
                df['Stoch_K'] = stoch_k
            else:
                df['Stoch_K'] = 50  # Placeholder
            
            # Simple MACD calculation
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['BB_middle'] = df['Close'].rolling(bb_period).mean()
        bb_std_val = df['Close'].rolling(bb_period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std_val * bb_std)
        df['BB_lower'] = df['BB_middle'] - (bb_std_val * bb_std)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        else:
            df['Volume_MA'] = 1
            df['Volume_ratio'] = 1
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
        df['Doji'] = (abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-8) < 0.1).astype(int)
        
        return df
    
    @staticmethod
    def create_statistical_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical and complexity features"""
        df = data.copy()
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'kurt_{window}'] = df['returns'].rolling(window).kurt()
            df[f'var_{window}'] = df['returns'].rolling(window).var()
        
        # Market complexity measures
        complexity = MarketComplexityMeasures()
        
        # Calculate fractal dimension (rolling)
        fractal_dims = []
        for i in range(len(df)):
            if i < 50:
                fractal_dims.append(1.5)
            else:
                fd = complexity.fractal_dimension(df['Close'].iloc[i-50:i+1])
                fractal_dims.append(fd)
        df['fractal_dimension'] = fractal_dims
        
        # Calculate approximate entropy (rolling)
        approx_entropies = []
        for i in range(len(df)):
            if i < 50:
                approx_entropies.append(0.5)
            else:
                ae = complexity.approximate_entropy(df['returns'].iloc[i-50:i+1])
                approx_entropies.append(ae)
        df['approx_entropy'] = approx_entropies
        
        # Hurst exponent (simplified)
        df['hurst_exponent'] = df['returns'].rolling(50).apply(
            lambda x: 0.5 if len(x) < 20 else min(0.99, max(0.01, 
                0.5 + np.corrcoef(np.arange(len(x)), np.cumsum(x))[0,1] * 0.5))
        )
        
        return df


class MLStrategyOptimizer:
    """
    Optuna-based optimizer for ML strategy parameters
    """
    
    def __init__(self, objective_metric='geometric_expectancy_mdd'):
        self.objective_metric = objective_metric
        self.study = None
        
    def optimize_sl_tp_parameters(self, 
                                 backtest_func,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str,
                                 n_trials: int = 50) -> Dict:
        """
        Optimize stop-loss and take-profit parameters using Optuna
        """
        
        def objective(trial):
            # Suggest hyperparameters
            stop_loss = trial.suggest_float('stop_loss', 0.005, 0.05, step=0.005)  # 0.5% to 5%
            take_profit = trial.suggest_float('take_profit', 0.01, 0.1, step=0.005)  # 1% to 10%
            trailing_stop = trial.suggest_float('trailing_stop', 0.005, 0.03, step=0.005)
            
            # ML model parameters
            model_type = trial.suggest_categorical('model_type', ['gradient_descent', 'arimax_boost', 'ensemble'])
            lookback_days = trial.suggest_int('lookback_days', 30, 252)
            
            # Feature selection
            use_technical = trial.suggest_categorical('use_technical', [True, False])
            use_statistical = trial.suggest_categorical('use_statistical', [True, False])
            use_complexity = trial.suggest_categorical('use_complexity', [True, False])
            
            try:
                # Run backtest with these parameters
                results = backtest_func(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=trailing_stop,
                    model_type=model_type,
                    lookback_days=lookback_days,
                    use_technical=use_technical,
                    use_statistical=use_statistical,
                    use_complexity=use_complexity
                )
                
                # Calculate objective based on selected metric
                if self.objective_metric == 'geometric_expectancy_mdd':
                    if results['max_drawdown'] == 0:
                        return float('inf')  # Avoid division by zero
                    return -results['geometric_expectancy'] / results['max_drawdown']
                
                elif self.objective_metric == 'sharpe_sortino':
                    return -(results['sharpe_ratio'] + results['sortino_ratio']) / 2
                
                else:  # Default to negative total return
                    return -results['total_return']
                    
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('inf')
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"ml_strategy_optimization_{symbol}"
        )
        
        # Optimize
        self.study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour timeout
        
        return self.study.best_params


class WalkForwardMLStrategy:
    """
    Main ML Strategy class with walk-forward backtesting
    """
    
    def __init__(self, 
                 training_window_years: float = 10,  # More aggressive: 3 months for faster adaptation
                 rebalance_frequency: str = 'daily',
                 initial_capital: float = 1000,  # $1K for realistic leverage
                 max_features: int = 30,  # Reduce from ~79 to ~50 features
                 retrain_frequency_days: int = 1):  # Retrain every 5 days instead of daily
        
        self.training_window_years = training_window_years
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital
        self.max_features = max_features
        self.retrain_frequency_days = retrain_frequency_days
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}  # Store feature selectors per symbol
        self.last_retrain_dates = {}  # Track when models were last retrained
        
        # Prediction history for trend-based signals
        self.prediction_history = {}  # Store recent predictions per symbol
        self.prediction_window = 5  # Number of recent predictions to keep
        
        # ML confidence for position sizing
        self.ml_confidence = 0.0
        
        # Feature engineering
        self.feature_engineer = AdvancedFeatureEngineering()
        
        # Optimizer
        self.optimizer = MLStrategyOptimizer()
        
        # Performance tracking
        self.performance_metrics = {}
    
    def prepare_features(self, data: pd.DataFrame, 
                        use_technical: bool = True,
                        use_statistical: bool = True,
                        use_complexity: bool = True) -> pd.DataFrame:
        """
        Prepare features for ML models
        """
        df = data.copy()
        
        if use_technical:
            df = self.feature_engineer.create_technical_features(df)
            
        if use_statistical:
            df = self.feature_engineer.create_statistical_features(df)
            
        # Remove NaN values (more conservative)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaNs, fill with reasonable defaults
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if 'returns' in col.lower():
                    df[col] = df[col].fillna(0.0)  # Zero return for missing values
                elif 'volatility' in col.lower():
                    df[col] = df[col].fillna(0.02)  # Default 2% volatility
                elif 'rsi' in col.lower():
                    df[col] = df[col].fillna(50.0)  # Neutral RSI
                elif 'ma_' in col.lower() or 'ema_' in col.lower():
                    df[col] = df[col].fillna(df['Close'].mean())  # Use mean price
                else:
                    df[col] = df[col].fillna(df[col].median())  # Use median for other features
        
        # Create a safe column selection that only includes existing columns
        available_cols = df.columns.tolist() # current available
        logger.info(f"Number of available columns for feature selection: {len(available_cols)}")
        
        # Try to get a reasonable subset of columns, but only if they exist
        selected_cols = []
        
        # Add basic OHLCV columns if they exist
        basic_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in basic_cols:
            if col in available_cols:
                selected_cols.append(col)
        
        # Add all other available columns (technical and statistical features)'
    
        for i, col in enumerate(available_cols):
            if col not in selected_cols and i % 2==0:
                selected_cols.append(col)
        
        # Ensure we don't exceed our feature limit and avoid empty selection
        if not selected_cols:
            selected_cols = available_cols  # Fallback to all columns
            
        return df[selected_cols]
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str], object]:
        """
        Select top features using mutual information and variance thresholds
        """
        from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
        from sklearn.pipeline import Pipeline
        
        # Remove low variance features first
        variance_selector = VarianceThreshold(threshold=0.01)
        
        # Select top k features using mutual information
        k_best_selector = SelectKBest(
            score_func=mutual_info_regression, 
            k=min(self.max_features, len(feature_names))
        )
        
        # Create pipeline
        feature_selector = Pipeline([
            ('variance', variance_selector),
            ('k_best', k_best_selector)
        ])
        
        # Fit and transform
        X_selected = feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        # First get features that passed variance threshold
        variance_mask = variance_selector.get_support()
        variance_features = [feature_names[i] for i, selected in enumerate(variance_mask) if selected]
        
        # Then get k-best features from those
        k_best_mask = k_best_selector.get_support()
        selected_features = [variance_features[i] for i, selected in enumerate(k_best_mask) if selected]
        
        logger.info(f"Feature selection: {len(feature_names)} -> {len(selected_features)} features")
        logger.info(f"Selected features: {selected_features[:10]}...")  # Log first 10
        
        return X_selected, selected_features, feature_selector
    
    def create_target_variable(self, data: pd.DataFrame, 
                              prediction_horizon: int = 1) -> pd.DataFrame:
        """
        Create target variable for ML prediction
        """
        df = data.copy()
        
        # Tomorrow's return as target
        df['target'] = df['Close'].shift(-prediction_horizon) / df['Close'] - 1 # Percentage of change OR return vs previous value
        
        # Classification target (direction)
        df['target_direction'] = (df['target'] > 0).astype(int)
        
        return df
    
    def train_ml_models(self, 
                       training_data: pd.DataFrame,
                       model_type: str = 'gradient_descent',
                       target_col: str = 'target',
                       feature_selection: bool = True) -> Dict:
        """
        Train ML models on historical data
        """
        # Prepare features
        feature_cols = [col for col in training_data.columns 
                       if col not in ['target', 'target_direction', 'Date']]
        
        X = training_data[feature_cols].values
        y = training_data[target_col].values
        
        # Remove any remaining NaN values (more conservative approach)
        # First, fill NaNs with forward fill and backward fill
        training_data_filled = training_data.fillna(method='ffill').fillna(method='bfill')
        
        X = training_data_filled[feature_cols].values
        y = training_data_filled[target_col].values
        
        # Only remove rows where target is NaN (more critical)
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Also remove rows where too many features are NaN (>50% of features)
        nan_count = np.isnan(X).sum(axis=1)
        max_allowed_nans = len(feature_cols) * 0.5
        valid_indices = nan_count <= max_allowed_nans
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Fill remaining NaNs with column means
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        logger.info(f"Training data before feature selection: {len(X)} samples, {X.shape[1]} features")
        
        if len(X) < 20:  # Reduced minimum requirement
            logger.warning(f"Insufficient training data: only {len(X)} samples available")
            return None
        
        
        # Feature selection
        if feature_selection:
            X_selected, selected_features, feature_selector = self.select_features(X, y, feature_cols)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
        
            logger.info(f"Training data after feature selection: {len(X_scaled)} samples, {X_scaled.shape[1]} features")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            selected_features = feature_cols
            feature_selector = None
            logger.info(f"Training data without feature selection: {len(X_scaled)} samples, {X_scaled.shape[1]} features")
        models = {}
        
        try:
            if model_type in ['gradient_descent', 'ensemble']:
                # Custom Gradient Descent model
                gd_model = MLGradientDescent(
                    n_iter=1000,
                    lr=0.01,
                    alpha=0.01,
                    l1_ratio=0.01,
                    momentum=0.9,
                    random_state=42
                )
                
                # Optimize hyperparameters
                gd_model.optimize_hyperparameters_optuna(X_scaled, y, n_trials=20)
                gd_model.fit(X_scaled, y)
                models['gradient_descent'] = gd_model
                
        except Exception as e:
            logger.error(f"Error training gradient descent model: {e}")
        
        try:
            if model_type in ['arimax_boost', 'ensemble']:
                # ARIMA + XGBoost ensemble
                arimax_model = MLEnsemble()
                arimax_model.fit(X_scaled, y)
                models['arimax_boost'] = arimax_model
                
        except Exception as e:
            logger.error(f"Error training ARIMAX model: {e}")
        
        # Fallback models
        if not models:
            # Simple Random Forest as fallback
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_scaled, y)
            models['random_forest'] = rf_model
        
        return {
            'models': models,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'selected_features': selected_features,
            'feature_selector': feature_selector
        }
    
    def predict_next_return(self, 
                           current_data: pd.DataFrame,
                           model_dict: Dict) -> float:
        """
        Predict next period's return using trained models
        """
        if not model_dict or 'models' not in model_dict:
            return 0.0
        
        try:
            # Prepare features
            feature_cols = model_dict['feature_cols']
            
            # Check if all required feature columns exist
            missing_cols = [col for col in feature_cols if col not in current_data.columns]
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}, using available features only")
                # Use only available features
                available_features = [col for col in feature_cols if col in current_data.columns]
                if not available_features:
                    logger.error("No available features for prediction")
                    return 0.0
                feature_cols = available_features
            
            # Check if we have enough data
            if len(current_data) == 0:
                logger.error("No current data available for prediction")
                return 0.0
                
            X = current_data[feature_cols].iloc[-1:].values
            
            # Apply feature selection if available
            if 'feature_selector' in model_dict and model_dict['feature_selector'] is not None:
                feature_selector = model_dict['feature_selector']
                try:
                    X = feature_selector.transform(X)
                except Exception as e:
                    logger.warning(f"Feature selector transform failed: {e}, using original features")
            
            # Scale features
            scaler = model_dict['scaler']
            X_scaled = scaler.transform(X)
            
            # Get predictions from all models
            predictions = []
            models = model_dict['models']
            
            for model_name, model in models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Error predicting with {model_name}: {e}")
            
            if predictions:
                # Ensemble prediction (mean)
                return np.mean(predictions)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.0
    
    def get_prediction_trend_summary(self, symbol: str) -> Dict:
        """
        Get a summary of prediction trends for a symbol
        """
        if symbol not in self.prediction_history:
            return {}
        
        pred_history = self.prediction_history[symbol]
        if len(pred_history) < 2:
            return {}
        
        predictions = [p['prediction'] for p in pred_history]
        dates = [p['date'] for p in pred_history]
        
        # Calculate trend metrics
        changes = np.diff(predictions)
        
        return {
            'symbol': symbol,
            'recent_predictions': predictions,
            'dates': [str(d.date()) for d in dates],
            'prediction_changes': changes.tolist(),
            'current_momentum': changes[-1] if len(changes) > 0 else 0, # up or down
            'average_trend': np.mean(changes) if len(changes) > 0 else 0,
            'trend_strength': np.std(changes) if len(changes) > 1 else 0,
            'prediction_range': [min(predictions), max(predictions)]
        }
    
    def generate_trading_signal(self, 
                               predicted_return: float,
                               current_data: pd.DataFrame,
                               symbol: str,
                               current_date: pd.Timestamp,
                               stop_loss: float = 0.02,
                               take_profit: float = 0.04) -> str:
        """
        Trend-based signal generation: Compare current prediction with previous prediction.
        
        Logic: If predicted return is increasing → BUY
               If predicted return is decreasing → SELL
               If no previous prediction → HOLD
        
        This avoids the issue of model bias/offset by focusing on trend direction.
        """
        # Store current prediction in history
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []
        
        # Add current prediction to history
        self.prediction_history[symbol].append({
            'prediction': predicted_return,
            'date': current_date
        })
        
        # Keep only recent predictions (window size)
        if len(self.prediction_history[symbol]) > self.prediction_window:
            self.prediction_history[symbol] = self.prediction_history[symbol][-self.prediction_window:]
        
        # Need at least 2 predictions to compare trend
        if len(self.prediction_history[symbol]) < 2:
            signal = 'HOLD'
            prediction_change = 0.0
            logger.info(f"Trend signal for {symbol}: HOLD (insufficient history)")
        else:
            # Compare current vs previous prediction
            current_pred = self.prediction_history[symbol][-1]['prediction']
            previous_pred = self.prediction_history[symbol][-2]['prediction']
            prediction_change = current_pred - previous_pred
            
            # Use percentage-based threshold for trend significance
            threshold = 0.0125  # 1.5% change threshold for trend significance

            if prediction_change >= threshold:
                signal = 'BUY'  # Prediction trending up significantly
            elif prediction_change <= -threshold:
                signal = 'SELL'  # Prediction trending down significantly
            else:
                signal = 'HOLD'  # Trend change not significant enough
            
            # Log trend analysis
            logger.info(f"Trend signal for {symbol}:")
            logger.info(f"  Previous prediction: {previous_pred:.6f} ({previous_pred*100:.3f}%)")
            logger.info(f"  Current prediction: {current_pred:.6f} ({current_pred*100:.3f}%)")
            logger.info(f"  Prediction change: {prediction_change:.6f} ({prediction_change*100:.3f}%)")
            logger.info(f"  Signal: {signal}")
        
        return signal, prediction_change
    
    def run_single_day_backtest(self, 
                               symbol: str,
                               current_date: pd.Timestamp,
                               price_data: pd.DataFrame,
                               feature_selection = False,
                               **params) -> Dict:
        """
        Run backtest for a single day with walk-forward approach
        """
        try:
            # Handle multi-level columns from yfinance
            if price_data.columns.nlevels > 1:
                # Flatten multi-level columns - take the symbol-specific columns
                price_data = price_data.xs(symbol, level=1, axis=1)
            
            # Get training window
            training_end = current_date
            # Convert fractional years to days (365.25 days per year on average)
            training_days = int(self.training_window_years * 365.25)
            training_start = training_end - pd.DateOffset(days=training_days)
            
            # Filter training data
            training_mask = (price_data.index >= training_start) & (price_data.index <= training_end)
            training_data = price_data[training_mask].copy()
            
            logger.info(f"Training data for {symbol}: {len(training_data)} rows from {training_start} to {training_end}")
            
            if len(training_data) < 200:  # Further reduced minimum requirement
                logger.warning(f"Insufficient raw data for {symbol}: only {len(training_data)} rows")
                return {'signal': 'HOLD', 'confidence': 0.0}
            
            # Check if we need to retrain based on frequency
            model_key = f"{symbol}_{params.get('model_type', 'ensemble')}"
            last_retrain = self.last_retrain_dates.get(model_key, None)
            need_retrain = True
            
            if last_retrain is not None:
                days_since_retrain = (current_date - last_retrain).days
                need_retrain = days_since_retrain >= self.retrain_frequency_days
                logger.info(f"Days since last retrain for {symbol}: {days_since_retrain}, need_retrain: {need_retrain}")
            
            # Get or train models
            if need_retrain or model_key not in self.models:
                logger.info(f"Retraining models for {symbol}")
                
                # Prepare features and target
                training_data = self.prepare_features(
                    training_data,
                    use_technical=params.get('use_technical', True),
                    use_statistical=params.get('use_statistical', True),
                    use_complexity=params.get('use_complexity', True)
                )
                training_data = self.create_target_variable(training_data)
                
                # Remove last row (no target available)
                training_data = training_data[:-1]
                
                # Train models
                model_dict = self.train_ml_models(
                    training_data,
                    model_type=params.get('model_type', 'ensemble'),
                    feature_selection=feature_selection
                )
                
                if not model_dict:
                    return {'signal': 'HOLD', 'confidence': 0.0}
                
                # Cache the models
                self.models[model_key] = model_dict
                self.last_retrain_dates[model_key] = current_date
            else:
                logger.info(f"Using cached models for {symbol}")
                model_dict = self.models[model_key]
            
            # Get current data for prediction
            current_mask = price_data.index <= current_date
            current_data = price_data[current_mask].copy()
            current_data = self.prepare_features(
                current_data,
                use_technical=params.get('use_technical', True),
                use_statistical=params.get('use_statistical', True),
                use_complexity=params.get('use_complexity', True)
            )
            
            # Make prediction
            predicted_return = self.predict_next_return(current_data, model_dict)
            
            # Generate signal using trend-based approach
            signal_result = self.generate_trading_signal(
                predicted_return,
                current_data,
                symbol,
                current_date,
                stop_loss=params.get('stop_loss', 0.02),
                take_profit=params.get('take_profit', 0.04)
            )
            
            # Handle signal result (could be string or tuple)
            if isinstance(signal_result, tuple):
                signal, prediction_change = signal_result
            else:
                signal = signal_result
                prediction_change = 0.0
            
            # Store ML confidence for position sizing
            self.ml_confidence = min(15.0, abs(prediction_change) * 100)  # Cap at 10%
            
            return {
                'signal': signal,
                'predicted_return': predicted_return,
                'prediction_change': prediction_change,
                'confidence': abs(predicted_return) * 100
            }
            
        except Exception as e:
            logger.error(f"Error in single day backtest for {symbol} on {current_date}: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}


def parallel_backtest_worker(args):
    """
    Worker function for parallel backtesting
    """
    symbol, date_chunk, price_data, strategy, feature_selection, params = args
    
    results = []
    for current_date in date_chunk:
        try:
            result = strategy.run_single_day_backtest(
                symbol=symbol,
                current_date=current_date,
                price_data=price_data,
                feature_selection=feature_selection,
                **params
            )
            result['date'] = current_date
            result['symbol'] = symbol
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error in worker for {symbol} on {current_date}: {e}")
    
    return results


def symbol_backtest_worker(args):
    """
    Worker that runs a full backtest for a single symbol. This is top-level so it can be pickled
    and executed in a separate process. Returns a tuple of (symbol, history_df, performance_metrics, execution_time, error)
    """
    symbol, start_date, end_date, params, training_window_years, initial_capital = args
    start_time = time.time()
    try:
        # Local imports to avoid pickling issues
        from backtester import AptosBacktester
        # Create a fresh strategy instance per process
        strategy = WalkForwardMLStrategy(training_window_years=training_window_years)

        def ml_signal_generator(historical_data):
            # historical_data passed by the backtester for the symbol
            # Use last available date
            current_date = historical_data.index[-1]
            result = strategy.run_single_day_backtest(
                symbol=symbol,
                current_date=current_date,
                price_data=historical_data,
                **params
            )
            return result.get('signal', 'HOLD')

        backtester = AptosBacktester(symbols=[symbol], initial_capital=initial_capital)
        history_df, performance_metrics = backtester.run_backtest(
            start_date=start_date,
            end_date=end_date,
            signal_generator=ml_signal_generator
        )

        exec_time = time.time() - start_time
        return (symbol, history_df, performance_metrics, exec_time, None)

    except Exception as e:
        exec_time = time.time() - start_time
        return (symbol, pd.DataFrame(), {}, exec_time, str(e))


class ParallelWalkForwardBacktest:
    """
    Parallel implementation of walk-forward backtesting
    """
    
    def __init__(self, n_processes: int = None):
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        self.strategy = WalkForwardMLStrategy()
        
    def load_market_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load market data for backtesting
        """
        try:
            # Use existing predictor infrastructure
            predictor = StockPredictor(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            predictor.load_data()
            
            if predictor.data is not None and not predictor.data.empty:
                return predictor.data
            else:
                logger.warning(f"No data loaded for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_parallel_backtest(self, 
                             symbols: List[str],
                             start_date: str,
                             end_date: str,
                             **params) -> Dict:
        """
        Run parallel walk-forward backtest across multiple symbols using existing AptosBacktester
        """
        logger.info(f"Starting parallel backtest with {self.n_processes} processes")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        start_time = time.time()
        
        # Create signal generator using ML strategy
        def ml_signal_generator(historical_data):
            """ML-based signal generator for the backtester"""
            symbol = historical_data.name if hasattr(historical_data, 'name') else 'Unknown'
            
            # Use the last available date as current date
            current_date = historical_data.index[-1]
            
            # Run single day backtest
            result = self.strategy.run_single_day_backtest(
                symbol=symbol,
                current_date=current_date,
                price_data=historical_data,
                feature_selection=False,
                **params
            )
            
            # Store ML confidence for position sizing (use prediction change magnitude)
            prediction_change = abs(result.get('prediction_change', 0)) * 100  # Convert to percentage
            self.strategy.ml_confidence = min(10.0, prediction_change)  # Cap at 10% for position sizing
            
            return result.get('signal', 'HOLD')
        
        # If only one symbol, run single-process for faster startup/debugging
        if len(symbols) == 1:
            symbol = symbols[0]
            backtester = AptosBacktester(symbols=[symbol], initial_capital=self.strategy.initial_capital)
            # Connect ML strategy to backtester for dynamic position sizing
            backtester.ml_strategy = self.strategy
            logger.info(f"Running single-symbol backtest for {symbol} (no parallelism)")
            try:
                history_df, performance_metrics = backtester.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    signal_generator=ml_signal_generator
                )

                elapsed_time = time.time() - start_time
                return {
                    'results': history_df,
                    'performance': performance_metrics,
                    'execution_time': elapsed_time,
                    'history_df': history_df,
                    'backtester': backtester
                }
            except Exception as e:
                logger.error(f"Error in single-symbol backtesting: {e}")
                return {'results': pd.DataFrame(), 'performance': {}, 'execution_time': time.time() - start_time, 'error': str(e)}

        # Multi-symbol UNIFIED PORTFOLIO execution (not parallel)
        logger.info(f"Running unified multi-symbol portfolio backtest for {symbols}")
        
        # Create single backtester with ALL symbols for unified portfolio
        backtester = AptosBacktester(symbols=symbols, initial_capital=self.strategy.initial_capital)
        # Connect ML strategy to backtester for dynamic position sizing
        backtester.ml_strategy = self.strategy
        
        try:
            history_df, performance_metrics = backtester.run_backtest(
                start_date=start_date,
                end_date=end_date,
                signal_generator=ml_signal_generator
            )

            elapsed_time = time.time() - start_time
            return {
                'results': history_df,
                'performance': performance_metrics,
                'execution_time': elapsed_time,
                'history_df': history_df,
                'backtester': backtester
            }
        except Exception as e:
            logger.error(f"Error in unified portfolio backtesting: {e}")
            return {'results': pd.DataFrame(), 'performance': {}, 'execution_time': time.time() - start_time, 'error': str(e)}

        # OLD: Multi-symbol parallel execution (each symbol gets separate portfolio)
        # Keeping this code commented for reference if needed
        """
        logger.info(f"Running multi-symbol parallel backtest across up to {self.n_processes} processes")

        # Prepare args for workers
        worker_args = []
        for symbol in symbols:
            worker_args.append((symbol, start_date, end_date, params, self.strategy.training_window_years, self.strategy.initial_capital))

        results = []
        with ProcessPoolExecutor(max_workers=min(self.n_processes, len(worker_args))) as executor:
            futures = {executor.submit(symbol_backtest_worker, arg): arg[0] for arg in worker_args}
            for fut in futures:
                symbol = futures[fut]
                try:
                    sym, history_df, performance_metrics, exec_time, error = fut.result()
                    logger.info(f"Completed backtest for {sym} in {exec_time:.2f}s")
                    results.append({
                        'symbol': sym,
                        'history_df': history_df,
                        'performance': performance_metrics,
                        'execution_time': exec_time,
                        'error': error
                    })
                except Exception as e:
                    logger.error(f"Worker for {symbol} raised: {e}")
                    results.append({'symbol': symbol, 'history_df': pd.DataFrame(), 'performance': {}, 'execution_time': 0.0, 'error': str(e)})

        total_time = time.time() - start_time
        logger.info(f"Parallel backtest finished in {total_time:.2f}s for {len(results)} symbols")

        # Aggregate results simplistically: return list of per-symbol results
        return {
            'results': results,
            'execution_time': total_time,
            'num_symbols': len(results)
        }
        """
    
    def calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if results_df.empty:
            return {}
        
        try:
            # Simulate portfolio performance based on signals
            # This method is now deprecated - AptosBacktester handles performance calculation
            # with dynamic position sizing. Keeping minimal implementation for compatibility.
            
            portfolio_value = [self.strategy.initial_capital]
            
            # Simple approximation since actual position sizing is handled by AptosBacktester
            total_return = 0
            num_trades = len(results_df)
            win_rate = len(results_df[results_df['predicted_return'] > 0]) / len(results_df) * 100 if len(results_df) > 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': 0.0,  # Placeholder - AptosBacktester calculates this properly
                'sortino_ratio': 0.0,  # Placeholder
                'max_drawdown': 0.0,  # Placeholder  
                'win_rate': win_rate,
                'geometric_expectancy': 0.0,  # Placeholder
                'custom_metric': 0.0,  # Placeholder
                'total_trades': num_trades,
                'final_portfolio_value': self.strategy.initial_capital,
                'note': 'Performance calculated by AptosBacktester with dynamic position sizing'
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}


def main_strategy_backtest():
    """
    Main function to run the ML strategy backtest
    """
    # MAXIMUM EARNINGS STRATEGY - Optimized for leverage and volatility
    strategy_params = {
        'stop_loss': 0.015,        # Tighter stops for higher frequency trading
        'take_profit': 0.06,       # Higher profit targets to let ML confidence work longer
        'trailing_stop': 0.012,    # Tighter trailing stops for better risk management
        'model_type': 'gradient_descent',
        'lookback_days': 100,
        'use_technical': True,
        'use_statistical': True,
        'use_complexity': True
    }
    
    # HIGH LEVERAGE PORTFOLIO - Volatile stocks for maximum ML confidence leverage
    test_symbols = [
        # 'NVDA',      # High volatility tech (leverage multipliers work best)
        # 'TSLA',      # High volatility EV (maximum ML confidence opportunities)
        # 'QQQ',       # Tech ETF (consistent trends for leverage)
        # 'SPY',       # Baseline ETF for stability
        # 'AAPL',      # Large cap tech (liquidity for large positions)
        # 'AMZN',      # E-commerce giant (volatile but liquid)
        # 'MSFT',      # Stable tech (balance for portfolio)
        # 'META',      # Social media giant (high volatility)
        # 'GME',       # Meme stock (extreme volatility for testing)
        # 'XAUUSD'     # Gold (hedge against volatility)
        # 'JPM',        # Banking giant (stable with growth potential)
        # want high volatility stock
        'NFLX',      # Streaming giant (high volatility)
        'BA',        # Boeing (volatile aerospace)
        'DIS',      # Disney (entertainment with volatility)
        'NKE',      # Nike (consumer goods with high volatility)

    ]
    
    # Date range (3 years for faster testing, extendable to 5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    # Initialize parallel backtester
    parallel_backtester = ParallelWalkForwardBacktest(n_processes=4)
    
    # Run backtest
    logger.info("Starting ML Strategy Backtest (mimicking Reddit XAUUSD strategy)")
    
    results = parallel_backtester.run_parallel_backtest(
        symbols=test_symbols,
        start_date=start_date,
        end_date=end_date,
        **strategy_params
    )
    
    # Display results
    if 'performance' in results and results['performance']:
        # Single symbol case
        perf = results['performance']
        logger.info("="*50)
        logger.info("BACKTEST RESULTS")
        logger.info("="*50)
        logger.info(f"Total Return: {perf.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {perf.get('sharpe', 0):.2f}")
        logger.info(f"Sortino Ratio: {perf.get('sortino', 0):.2f}")
        logger.info(f"Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
        logger.info(f"Win Rate: {perf.get('win_rate', 0):.2%}")
        logger.info(f"Total Trades: {perf.get('num_trades', 0)}")
        logger.info(f"Custom Metric (Geo Expectancy/MDD): {perf.get('custom_metric', 0):.4f}")
        logger.info(f"Execution Time: {results['execution_time']:.2f} seconds")
        logger.info("="*50)
    elif 'results' in results and results['results']:
        # Multi-symbol case 
        logger.info("="*50)
        logger.info("MULTI-SYMBOL BACKTEST RESULTS")
        logger.info("="*50)
        logger.info(f"Number of symbols: {results.get('num_symbols', 0)}")
        logger.info(f"Total execution time: {results['execution_time']:.2f} seconds")
        
        # Show results for each symbol
        for result in results['results']:
            if result.get('performance'):
                perf = result['performance']
                logger.info(f"\n{result['symbol']}:")
                logger.info(f"  Total Trades: {perf.get('num_trades', 0)}")
                logger.info(f"  Total Return: {perf.get('total_return', 0):.2%}")
                logger.info(f"  Sharpe Ratio: {perf.get('sharpe', 0):.2f}")
                logger.info(f"  Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
                logger.info(f"  Win Rate: {perf.get('win_rate', 0):.2%}")
                logger.info(f"  Execution Time: {result.get('execution_time', 0):.2f}s")
            else:
                logger.info(f"\n{result['symbol']}: No performance data")
        logger.info("="*50)
    else:
        logger.info("No results to display")
    
    return results


if __name__ == "__main__":
    # Run the main backtest
    results = main_strategy_backtest()
    
    # Optionally save results
    output_data = None
    if 'results' in results and hasattr(results['results'], 'empty') and not results['results'].empty:
        # Single symbol case - results['results'] is a DataFrame
        output_data = results['results']
    elif 'results' in results and isinstance(results['results'], list) and results['results']:
        # Multi-symbol case - results['results'] is a list, try to combine DataFrames
        dataframes = []
        for result in results['results']:
            if 'history_df' in result and not result['history_df'].empty:
                df = result['history_df'].copy()
                df['symbol'] = result['symbol']
                dataframes.append(df)
        if dataframes:
            output_data = pd.concat(dataframes, ignore_index=True)
    
    if output_data is not None:
        output_file = f"ml_strategy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_data.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Results saved to {output_file}")
