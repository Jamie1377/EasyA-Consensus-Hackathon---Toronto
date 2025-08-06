import warnings

warnings.filterwarnings("ignore")

import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from numba import jit, njit, prange, float64


# Sample Dataset
stock_data = yf.download("AAPL", start="2024-01-01", end=date.today())
stock_data.columns = stock_data.columns.droplevel(1)
stock_data

# Add to models.py
import requests
import pandas as pd
from datetime import timedelta
import hashlib


# Alpaca API imports
import os
import requests
import json


def get_cached_data(
    tickers, start_date, end_date, interval="1d", cache_dir="data_cache"
):
    """
    Fetch data from Yahoo Finance with caching to avoid rate limits

    Args:
        tickers: String or list of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        interval: Data interval (1d, 1h, etc)
        cache_dir: Directory to store cached data

    Returns:
        DataFrame with the requested data
    """
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Convert tickers to list if it's a string
    if isinstance(tickers, str):
        tickers = [tickers]

    # Create a unique cache key for this request
    cache_key = f"{'-'.join(sorted(tickers))}_{start_date}_{end_date}_{interval}"
    hash_key = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{hash_key}.parquet")
    metadata_file = os.path.join(cache_dir, f"{hash_key}.json")

    # Check if cached data exists and is fresh (less than 1 day old for end_date == today)
    use_cache = False
    if os.path.exists(cache_file) and os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        cache_date = datetime.strptime(metadata["cache_date"], "%Y-%m-%d %H:%M:%S")
        is_recent_query = (
            datetime.strptime(end_date, "%Y-%m-%d").date()
            >= (datetime.now() - timedelta(days=7)).date()
        )

        # Cache is valid if it's less than 1 day old for recent queries
        # or any age for historical queries
        if is_recent_query:
            use_cache = (datetime.now() - cache_date) < timedelta(days=1)
        else:
            use_cache = True

    if use_cache:
        print(f"Using cached data for {tickers} ({start_date} to {end_date})")
        data = pd.read_parquet(cache_file)
    else:
        # Need to download fresh data
        print(f"Downloading fresh data for {tickers} ({start_date} to {end_date})")
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by="ticker" if len(tickers) > 1 else "column",
                auto_adjust=True,
                progress=False,
            )

            # Save to cache
            if not data.empty:
                data.to_parquet(cache_file)

                # Save metadata
                metadata = {
                    "tickers": tickers,
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval": interval,
                    "cache_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)

        except Exception as e:
            print(f"Error downloading data: {str(e)}")

            # If cache exists but is outdated, use it as fallback
            if os.path.exists(cache_file):
                print(f"Using outdated cache as fallback")
                data = pd.read_parquet(cache_file)
            else:
                # No cache and download failed
                raise e

    return data


# Example directory structure for cache organization
def organize_cache(cache_dir="data_cache"):
    """Organize cache by ticker and timeframe"""
    if not os.path.isdir(cache_dir):
        return

    # List all cache files
    files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]

    for file in files:
        metadata_path = os.path.join(cache_dir, file)
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Create organized structure
            tickers_dir = os.path.join(cache_dir, "-".join(metadata["tickers"]))
            if not os.path.exists(tickers_dir):
                os.makedirs(tickers_dir)

            # Move files to organized location
            cache_file = file.replace(".json", ".parquet")
            if os.path.exists(os.path.join(cache_dir, cache_file)):
                new_path = os.path.join(tickers_dir, cache_file)
                new_metadata_path = os.path.join(tickers_dir, file)

                # Only move if destination doesn't exist
                if not os.path.exists(new_path):
                    os.rename(os.path.join(cache_dir, cache_file), new_path)
                    os.rename(metadata_path, new_metadata_path)

        except Exception as e:
            print(f"Error organizing {file}: {str(e)}")


class StockPredictor:
    """Stock price prediction pipeline

    Parameters:
        symbol (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        interval (str): Data interval (1d, 1h)
    """

    def __init__(self, symbol, start_date, end_date=None, interval="1d",
                 include_fourier=False, include_fft_pca=False, include_rolling_stats=True,
                 include_economic=True,
                 include_crypto_liquidity=True, include_crypto_volatility=True,
                 include_crypto_structure=True, include_onchain=True):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else date.today()
        # Feature inclusion flags
        self.include_fourier = include_fourier
        self.include_fft_pca = include_fft_pca
        self.include_rolling_stats = include_rolling_stats
        self.include_economic = include_economic
        self.include_crypto_liquidity = include_crypto_liquidity
        self.include_crypto_volatility = include_crypto_volatility
        self.include_crypto_structure = include_crypto_structure
        self.include_onchain = include_onchain
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.best_params = {}
        self.data = None
        self.feature_sets = {
            "Close": {"target": "Close", "features": None},
            "Low": {"target": "Low", "features": None},
            "Daily Returns": {"target": "Daily Returns", "features": None},
            "Volatility": {"target": "Volatility", "features": None},
            "TNX": {"target": "TNX", "features": None},
            "Treasury_Yield": {"target": "Treasury_Yield", "features": None},
            "SP500": {"target": "SP500", "features": None},
            "USDCAD=X": {"target": "USDCAD=X", "features": None},
        }
        self.scalers = {}
        self.transformers = {}
        valid_intervals = ["1m", "5m", "1d"]
        if interval not in valid_intervals:
            print(
                f"Warning: Interval '{interval}' not supported. Must be one of {valid_intervals}"
            )
            print(f"Defaulting to '1d' interval")
            self.interval = "1d"
        else:
            self.interval = interval
        self.history = []  # New attribute for error correction
        self.risk_params = {
            "max_portfolio_risk": 0.05,  # 5% total portfolio risk
            "per_trade_risk": 0.025,  # 2.5% risk per trade
            "stop_loss_pct": 0.03,  # 3% trailing stop
            "take_profit_pct": 0.003,  # 1.5% take profit
            "max_sector_exposure": 0.4,  # 40% max energy sector exposure
            "daily_loss_limit": -0.03,  # -3% daily loss threshold
        }
        self.model_cache_dir = f"model_cache/{self.symbol}"
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.data_hash = None
        self.forecast_record = {}

    def _compute_rsi(self, window=14):
        """Custom RSI implementation using NumPy for better performance"""
        # Convert Close prices to numpy array
        close_values = self.data["Close"].values

        # Calculate price changes
        delta = np.zeros(len(close_values))
        delta[1:] = close_values[1:] - close_values[:-1]

        # Split gains and losses
        gain = np.zeros(len(delta))
        loss = np.zeros(len(delta))

        gain[delta > 0] = delta[delta > 0]
        loss[delta < 0] = -delta[delta < 0]

        # Calculate average gains and losses with the rolling window
        avg_gain = np.full(len(gain), np.nan)
        avg_loss = np.full(len(loss), np.nan)

        # Initial averages
        if len(gain) >= window:
            avg_gain[window - 1] = np.mean(gain[:window])
            avg_loss[window - 1] = np.mean(loss[:window])

        # Calculate subsequent values using EMA formula
        for i in range(window, len(gain)):
            avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
            avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window

        # Calculate RSI - handle division by zero
        rs = np.zeros(len(avg_gain))
        rsi = np.zeros(len(avg_gain))

        non_zero_loss = avg_loss > 0
        rs[non_zero_loss] = avg_gain[non_zero_loss] / avg_loss[non_zero_loss]

        rsi = 100 - (100 / (1 + rs))

        # Return as pandas Series with the same index
        return pd.Series(rsi, index=self.data.index)
    
    def _compute_atr(self, window=14):
        """Average True Range"""
        high_low = self.data["High"] - self.data["Low"]
        high_close = (self.data["High"] - self.data["Close"].shift()).abs()
        low_close = (self.data["Low"] - self.data["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # Using NumPy for rolling mean
        return pd.Series(self._rolling_mean_numpy(tr, window), index=tr.index)

    def _rolling_mean_numpy(self, data: pd.Series, window: int):
        """
        Calculate rolling mean using NumPy's convolve function for better performance
        """
        # Convert to numpy array for faster calculation
        values = data.values if hasattr(data, "values") else np.array(data)
        weights = np.ones(window) / window
        # Calculate the rolling mean using np.convolve
        convolved = np.convolve(values, weights, mode="valid")
        # Create array with NaN for the initial window-1 elements
        result = np.full(len(values), np.nan)
        result[window - 1 :] = convolved
        return result

    def _add_technical_indicators(
        self, window=15, include_fourier=False, include_fft_pca=False
    ):
        """Add all technical indicators with optional features"""
        ### 1. Add rolling indicators - optimized with NumPy
        close_values = self.data["Close"].values

        # Calculate moving averages using NumPy
        self.data["MA_50"] = pd.Series(
            self._rolling_mean_numpy(close_values, 50), index=self.data.index
        )
        self.data["MA_200"] = pd.Series(
            self._rolling_mean_numpy(close_values, 200), index=self.data.index
        )
        self.data["MA_7"] = pd.Series(
            self._rolling_mean_numpy(close_values, 7), index=self.data.index
        )
        self.data["MA_21"] = pd.Series(
            self._rolling_mean_numpy(close_values, 21), index=self.data.index
        )

        # Optional Fourier transform features
        if include_fourier:
            data_FT = self.data.copy().reset_index()[["Date", "Close"]]
            close_fft = np.fft.fft(np.asarray(data_FT["Close"].tolist()))
            self.data["FT_real"] = np.real(close_fft)
            self.data["FT_img"] = np.imag(close_fft)

        # Optional PCA on FFT
        if include_fft_pca:
            from sklearn.decomposition import PCA
            X_fft = np.column_stack([np.real(close_fft), np.imag(close_fft)])
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_fft)
            for i in range(X_pca.shape[1]):
                self.data[f"Fourier_PCA_{i}"] = X_pca[:, i]

        ### 3. Add rolling statistics
        # Use NumPy for better performance
        close_values = self.data["Close"].values

        # Standard deviation using NumPy's rolling window
        def np_rolling_std(values: np.ndarray, window: int):
            result = np.full(len(values), np.nan)
            for i in range(window - 1, len(values)):
                result[i] = np.std(values[i - window + 1 : i + 1], ddof=1)
            return result

        @jit
        def np_rolling_min(values: np.ndarray, window: int):
            result = np.full(len(values), np.nan)
            for i in range(window - 1, len(values)):
                result[i] = np.min(values[i - window + 1 : i + 1])
            return result

        @jit
        def np_rolling_median(values, window):
            result = np.full(len(values), np.nan)
            for i in range(window - 1, len(values)):
                result[i] = np.median(values[i - window + 1 : i + 1])
            return result

        @jit
        def np_rolling_sum(values, window):
            result = np.full(len(values), np.nan)
            weights = np.ones(window)
            convolved = np.convolve(values, weights, mode="valid")
            result[window - 1 :] = convolved
            return result

        def np_rolling_var(values: np.ndarray, window: int):
            result = np.full(len(values), np.nan)
            for i in range(window - 1, len(values)):
                result[i] = np.var(values[i - window + 1 : i + 1], ddof=1)
            return result

        @jit(nopython=True, parallel=True)
        def np_rolling_ema(values: np.ndarray, span: int):
            alpha = 2 / (span + 1)
            result = np.full(len(values), np.nan)
            result[0] = values[0]
            for i in range(1, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
            return result

        self.data["rolling_std"] = pd.Series(
            np_rolling_std(close_values, 50), index=self.data.index
        )
        self.data["rolling_min"] = pd.Series(
            np_rolling_min(close_values, 50), index=self.data.index
        )
        # self.data['rolling_max'] = pd.Series(np_rolling_min(close_values, window), index=self.data.index)
        self.data["rolling_median"] = pd.Series(
            np_rolling_median(close_values, 50), index=self.data.index
        )
        self.data["rolling_sum"] = pd.Series(
            np_rolling_sum(close_values, 50), index=self.data.index
        )
        self.data["rolling_var"] = pd.Series(
            np_rolling_var(close_values, 50), index=self.data.index
        )
        self.data["rolling_ema"] = pd.Series(
            np_rolling_ema(close_values, 50), index=self.data.index
        )  # Exponential Moving Average

        # Add rolling quantiles (25th and 75th percentiles) using NumPy
        @jit
        def np_rolling_quantile(values, window, quantile):
            result = np.full(len(values), np.nan)
            for i in range(window - 1, len(values)):
                result[i] = np.quantile(values[i - window + 1 : i + 1], quantile)
            return result

        self.data["rolling_25p"] = pd.Series(
            np_rolling_quantile(close_values, 50, 0.25), index=self.data.index
        )
        self.data["rolling_75p"] = pd.Series(
            np_rolling_quantile(close_values, 50, 0.75), index=self.data.index
        )
        # Drop rows with NaN values (due to rolling window)
        
        # stock_data.index.name = "Date"  # Ensure the index is named "Date"

        ### 4. Advanced Momentum
        self.data["RSI"] = self._compute_rsi(window=14)

        # Calculate MACD using the np_rolling_ema function
        ema_12 = np_rolling_ema(close_values, 12)
        ema_26 = np_rolling_ema(close_values, 26)
        macd = ema_12 - ema_26

        self.data["MACD"] = pd.Series(macd, index=self.data.index)
        self.data.dropna(inplace=True)
        ### 5. Williams %R with NumPy optimization
        @jit
        def np_rolling_max(values, window):
            result = np.full(len(values), np.nan)
            for i in range(window - 1, len(values)):
                result[i] = np.max(values[i - window + 1 : i + 1])
            return result

        high_values = self.data["High"].values
        low_values = self.data["Low"].values
        close_values = self.data["Close"].values

        high_max_np = np_rolling_max(high_values, window)
        low_min_np = np_rolling_min(low_values, window)

        # Calculate Williams %R using NumPy operations
        williams_r = np.full(len(close_values), np.nan)
        valid_idx = (
            ~np.isnan(high_max_np) & ~np.isnan(low_min_np) & (high_max_np != low_min_np)
        )
        williams_r[valid_idx] = (
            (high_max_np[valid_idx] - close_values[valid_idx])
            / (high_max_np[valid_idx] - low_min_np[valid_idx])
        ) * -100

        self.data["Williams_%R"] = pd.Series(williams_r, index=self.data.index)

        ### 6. Stochastic Oscillator with NumPy optimization
        stochastic_k = np.full(len(close_values), np.nan)
        valid_idx = (
            ~np.isnan(high_max_np) & ~np.isnan(low_min_np) & (high_max_np != low_min_np)
        )
        stochastic_k[valid_idx] = (
            (close_values[valid_idx] - low_min_np[valid_idx])
            / (high_max_np[valid_idx] - low_min_np[valid_idx])
        ) * 100

        self.data["Stochastic_%K"] = pd.Series(stochastic_k, index=self.data.index)
        self.data["Stochastic_%D"] = pd.Series(
            self._rolling_mean_numpy(stochastic_k, 3), index=self.data.index
        )

        ### 7. Momentum Divergence Detection
        self.data["Price_Change"] = self.data["Close"].diff()
        self.data["Momentum_Divergence"] = (
            (self.data["Price_Change"] * self.data["MACD"].diff()).rolling(5).sum()
        )

        ### 8. Volatility-adjusted Channels
        self.data["ATR"] = self._compute_atr(window=14)
        self.data["Upper_Bollinger"] = (
            self.data["MA_21"] + 2 * self.data["Close"].rolling(50).std()
        )
        self.data["Lower_Bollinger"] = (
            self.data["MA_21"] - 2 * self.data["Close"].rolling(50).std()
        )

        ### 9. Volume-based Features
        # self.data['OBV'] = self._compute_obv()
        if self.data["Volume"].cumsum()[-1] != 0:
            self.data["VWAP"] = (
                self.data["Volume"]
                * (self.data["High"] + self.data["Low"] + self.data["Close"])
                / 3
            ).cumsum() / self.data["Volume"].cumsum()
        # 11. Whether the next or previous day is a non-trading day
        # nyse = mcal.get_calendar("NYSE")
        # schedule = nyse.schedule(start_date=self.start_date, end_date=self.end_date)
        # economic_data["is_next_non_trading_day"] = economic_data.index.shift(
        #     -1, freq="1d"
        # ).isin(schedule.index).astype(int) + economic_data.index.shift(
        #     1, freq="1d"
        # ).isin(
        #     schedule.index
        # ).astype(
        #     int
        # )
        ### 12. Volatility and Momentum
        # self.data["Daily Returns"] = self.data["Close"].pct_change() # Percentage change
        self.data["Daily Returns"] = (
            self.data["Close"].pct_change(window) * 100
        )  # Percentage change in the standard window for the momentum
        self.data["Volatility"] = self.data["Daily Returns"].rolling(window=20).std()
        # Adaptive Momentum Score
        vol_weight = self.data["Volatility"] * 100
        self.data["Momentum_Score"] = (
            self.data["RSI"] * 0.4
            + self.data["Daily Returns"] * 0.3
            + self.data["Williams_%R"] * 0.3
        ) / (1 + vol_weight)
        # Drop rows with NaN values
        self.data["Momentum_Interaction"] = (
            self.data["RSI"] * self.data["Daily Returns"]
        )
        self.data["Volatility_Adj_Momentum"] = self.data["Momentum_Score"] / (
            1 + self.data["Volatility"]
        )
        self.data["Volatility_Adj_Momentum"] = self.data[
            "Volatility_Adj_Momentum"
        ].clip(lower=0.1)
        self.data["Volatility_Adj_Momentum"] = self.data[
            "Volatility_Adj_Momentum"
        ].clip(upper=10.0)
        self.data["Volatility_Adj_Momentum"] = self.data[
            "Volatility_Adj_Momentum"
        ].fillna(0.0)

        ### 13. Market Regime Detection by HMM
        # hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        # hmm.fit(self.data["Close"].pct_change().dropna().values.reshape(-1, 1))
        # # Predict hidden states
        # market_state = hmm.predict(
        #     self.data["Close"].pct_change().dropna().values.reshape(-1, 1)
        # )
        # hmm_sp = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        # hmm_sp.fit(self.data["SP500"].pct_change().dropna().values.reshape(-1, 1))
        # market_state_sp500 = hmm_sp.predict(
        #     self.data["SP500"].pct_change().dropna().values.reshape(-1, 1)
        # )
        # # Initialize the Market_State column
        # self.data["Market_State"] = np.zeros(len(self.data))
        # if (
        #     len(set(list(market_state))) != 1
        #     and len(set(list(market_state_sp500))) != 1
        # ):
        #     self.data["Market_State"][0] = 0
        #     self.data.iloc[1:]["Market_State"] = market_state + market_state_sp500

        # ### 14. Sentiment Analysis (Computationally expensive)
        # self.data["Market_Sentiment"] = 0.0
        # sentimement = MarketSentimentAnalyzer().get_historical_sentiment(
        #     self.symbol, self.data.shape[0]
        # )
        # self.data["Market_Sentiment"] = sentimement

    def _add_economic_indicators(self):
        """Add economic indicators for stock trading"""
        # Batch download all economic indicators at once
        economic_tickers = [
            "^GSPC",
            "^TNX",
            "IEF",
            "USDCAD=X",
            "XLK",
            "XLF",
            "XLE",
            "^VIX",
        ]
        try:
            print("Trying to fetch economic data from cache...")
            economic_data = get_cached_data(
                economic_tickers,
                start_date=self.start_date,
                end_date=self.end_date,
                interval=self.interval,
                cache_dir="data_cache",
            )
        except Exception as e:
            print(f"Cache fetch failed: {str(e)}. Downloading fresh data...")
            economic_data = yf.download(
                economic_tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                group_by="ticker",  # Group results by ticker
                auto_adjust=True,  # Auto-adjust data
                progress=False,  # Disable progress bar
                prepost=True,  # Include pre/post market data
            )

        # Extract each indicator from the batched data
        try:
            sp500 = (
                economic_data["^GSPC"]["Close"] - economic_data["^GSPC"]["Close"].mean()
            )
            tnx = economic_data["^TNX"]["Close"]
            tnx_len = len(tnx)
            treasury_yield = economic_data["IEF"]["Close"]
            exchange_rate = economic_data["USDCAD=X"]["Close"]
            technology_sector = economic_data["XLK"]["Close"]
            financials_sector = economic_data["XLF"]["Close"]
            energy_sector = economic_data["XLE"]["Close"]
            vix = economic_data["^VIX"]["Close"]

            # Additional defensive code to handle missing data
            for series_name, series in [
                ("TNX", tnx),
                ("Treasury_Yield", treasury_yield),
                ("Exchange Rate", exchange_rate),
                ("Technology Sector", technology_sector),
                ("Financial Sector", financials_sector),
                ("Energy Sector", energy_sector),
                ("VIX", vix),
            ]:
                if series.empty:
                    print(f"Warning: {series_name} data is empty, filling with zeros")
                    if series_name == "TNX":
                        tnx = pd.Series(0, index=self.data.index)
                        tnx_len = 0
        except KeyError as e:
            print(f"Warning: One or more economic indicators missing: {e}")
            # Provide fallback values or skip the missing indicators

        economic_data = (
            pd.concat(
                [
                    sp500,
                    tnx,
                    treasury_yield,
                    exchange_rate,
                    technology_sector,
                    financials_sector,
                    vix,
                    energy_sector,
                ],
                axis=1,
                keys=[
                    "SP500",
                    "TNX",
                    "Treasury_Yield",
                    "USDCAD=X",
                    "Tech",
                    "Fin",
                    "VIX",
                    "Energy",
                ],
            )
            .reset_index()
            .rename(columns={"index": "Date"})
            # .dropna()
        )
        economic_data.columns = economic_data.columns.get_level_values(0)
        if self.interval == "1m" or self.interval == "5m":
            # or self.interval == "15m" or self.interval == "30m" or self.interval == "60m" or self.interval == "90m":

            economic_data["Datetime"] = pd.to_datetime(economic_data["Datetime"])
            economic_data.set_index("Datetime", inplace=True)
        else:
            economic_data["Date"] = pd.to_datetime(economic_data["Date"])
            economic_data.set_index("Date", inplace=True)

        # Issue of Yfinance API of USDCAD=X
        # Fill missing values with the mean
        economic_data["USDCAD=X"] = economic_data["USDCAD=X"].fillna(
            economic_data["USDCAD=X"].mean()
        )
        # Merge with stock data
        if tnx_len < len(self.data):
            economic_data = economic_data.drop(columns="TNX")
        if self.interval in ["1m", "5m"]:
            self.data = pd.merge(self.data, economic_data, on="Datetime", how="left")
        else:
            self.data = pd.merge(self.data, economic_data, on="Date", how="left")

    def _add_crypto_liquidity_indicators(self):
        """Add crypto-specific liquidity indicators"""

        # Volume-based liquidity metrics
        self.data["Volume_MA"] = self.data["Volume"].rolling(24).mean()
        self.data["Relative_Volume"] = self.data["Volume"] / self.data["Volume_MA"]

        # Liquidity ratio (higher values indicate better liquidity)
        self.data["Liquidity_Ratio"] = self.data["Volume"] / (
            self.data["High"] - self.data["Low"]
        ).replace(0, 0.001)

        # Volume-weighted volatility (measures how efficiently price moves with volume)
        self.data["Vol_Weighted_Volatility"] = (
            self.data["Close"].pct_change().abs()
            * self.data["Volume"]
            / self.data["Volume_MA"]
        )

        # VWAP and deviation from VWAP (measure of buying/selling pressure)
        self.data["VWAP"] = (self.data["Volume"] * self.data["Close"]).rolling(
            24
        ).sum() / self.data["Volume"].rolling(24).sum()
        self.data["VWAP_Deviation"] = (
            (self.data["Close"] - self.data["VWAP"]) / self.data["VWAP"]
        ) * 100

        # Flash crash detector (sudden volume spike with price drop)
        vol_spike = self.data["Volume"] > (self.data["Volume"].rolling(12).mean() * 3)
        price_drop = self.data["Close"].pct_change() < -0.03
        self.data["Flash_Crash_Signal"] = vol_spike & price_drop

    def _add_crypto_volatility_indicators(self):
        """Add specialized crypto volatility indicators"""

        # True Range and ATR variations
        self.data["TR"] = np.maximum(
            self.data["High"] - self.data["Low"],
            np.maximum(
                abs(self.data["High"] - self.data["Close"].shift(1)),
                abs(self.data["Low"] - self.data["Close"].shift(1)),
            ),
        )
        self.data["ATR_1h"] = (
            self.data["TR"].rolling(60).mean()
        )  # 1-hour ATR (for minute data)
        self.data["ATR_24h"] = self.data["TR"].rolling(1440).mean()  # 24-hour ATR

        # Volatility ratio (short-term vs long-term)
        self.data["Volatility_Ratio"] = self.data["ATR_1h"] / self.data["ATR_24h"]

        # Bollinger Band Width (normalized)
        bb_period = 20
        std_dev = 2
        self.data["BB_Middle"] = self.data["Close"].rolling(bb_period).mean()
        self.data["BB_Width"] = (
            (self.data["Close"].rolling(bb_period).std() * std_dev * 2)
            / self.data["BB_Middle"]
        ) * 100

        # Historical Volatility (annualized)
        self.data["HV_1h"] = self.data["Close"].pct_change().rolling(
            60
        ).std() * np.sqrt(
            525600
        )  # Minutes in a year
        self.data["HV_24h"] = self.data["Close"].pct_change().rolling(
            1440
        ).std() * np.sqrt(525600)

        # Volatility Regime (1=low, 2=medium, 3=high)
        self.data["Volatility_Regime"] = 1  # Default to low
        vol_75th = self.data["HV_24h"].quantile(0.75)
        vol_25th = self.data["HV_24h"].quantile(0.25)
        self.data.loc[self.data["HV_24h"] > vol_25th, "Volatility_Regime"] = 2  # Medium
        self.data.loc[self.data["HV_24h"] > vol_75th, "Volatility_Regime"] = 3  # High

        # Guppy Multiple Moving Average (GMMA) Compression/Expansion
        # A measure of trend strength and potential volatility expansion
        ema_short = [3, 5, 8, 10, 12, 15]
        ema_long = [30, 35, 40, 45, 50, 60]

        for period in ema_short + ema_long:
            self.data[f"EMA_{period}"] = (
                self.data["Close"].ewm(span=period, adjust=False).mean()
            )

        # Calculate average distance between short EMAs
        short_emas = [self.data[f"EMA_{p}"] for p in ema_short]
        long_emas = [self.data[f"EMA_{p}"] for p in ema_long]

        self.data["GMMA_Short_Spread"] = (
            max([short_emas[i].iloc[-1] for i in range(len(short_emas))])
            - min([short_emas[i].iloc[-1] for i in range(len(short_emas))])
        ) / self.data["Close"]
        self.data["GMMA_Long_Spread"] = (
            max([long_emas[i].iloc[-1] for i in range(len(long_emas))])
            - min([long_emas[i].iloc[-1] for i in range(len(long_emas))])
        ) / self.data["Close"]

    def _add_crypto_market_structure_indicators(self):
        """Add market structure indicators specifically for crypto"""

        # Detect pivot points (swing highs and lows)
        pivot_length = 5
        self.data["Pivot_High"] = 0
        self.data["Pivot_Low"] = 0

        for i in range(pivot_length, len(self.data) - pivot_length):
            # Check for pivot high
            if (
                self.data["High"].iloc[i]
                > self.data["High"].iloc[i - pivot_length : i].max()
                and self.data["High"].iloc[i]
                > self.data["High"].iloc[i + 1 : i + pivot_length + 1].max()
            ):
                self.data.loc[self.data.index[i], "Pivot_High"] = 1

            # Check for pivot low
            if (
                self.data["Low"].iloc[i]
                < self.data["Low"].iloc[i - pivot_length : i].min()
                and self.data["Low"].iloc[i]
                < self.data["Low"].iloc[i + 1 : i + pivot_length + 1].min()
            ):
                self.data.loc[self.data.index[i], "Pivot_Low"] = 1

        # Market Structure Shift Detection
        self.data["Structure_Bullish"] = 0
        self.data["Structure_Bearish"] = 0

        pivot_highs = self.data[self.data["Pivot_High"] == 1]
        pivot_lows = self.data[self.data["Pivot_Low"] == 1]

        if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
            # Check for Higher Highs & Higher Lows (Bullish Structure)
            last_two_highs = pivot_highs.iloc[-2:]["High"].values
            last_two_lows = pivot_lows.iloc[-2:]["Low"].values

            if len(last_two_highs) == 2 and len(last_two_lows) == 2:
                if (
                    last_two_highs[1] > last_two_highs[0]
                    and last_two_lows[1] > last_two_lows[0]
                ):
                    self.data.loc[self.data.index[-1], "Structure_Bullish"] = 1

                # Check for Lower Highs & Lower Lows (Bearish Structure)
                if (
                    last_two_highs[1] < last_two_highs[0]
                    and last_two_lows[1] < last_two_lows[0]
                ):
                    self.data.loc[self.data.index[-1], "Structure_Bearish"] = 1

        # Ichimoku Cloud for trend structure and support/resistance
        high_9 = self.data["High"].rolling(window=9).max()
        low_9 = self.data["Low"].rolling(window=9).min()
        self.data["Tenkan_Sen"] = (high_9 + low_9) / 2  # Conversion Line

        high_26 = self.data["High"].rolling(window=26).max()
        low_26 = self.data["Low"].rolling(window=26).min()
        self.data["Kijun_Sen"] = (high_26 + low_26) / 2  # Base Line

        self.data["Senkou_Span_A"] = (
            (self.data["Tenkan_Sen"] + self.data["Kijun_Sen"]) / 2
        ).shift(
            26
        )  # Leading Span A
        self.data["Senkou_Span_B"] = (
            (
                self.data["High"].rolling(window=52).max()
                + self.data["Low"].rolling(window=52).min()
            )
            / 2
        ).shift(
            26
        )  # Leading Span B

        # Cloud state (Above/Below/In cloud)
        self.data["Cloud_State"] = (
            0  # 0 = in cloud, 1 = above cloud (bullish), -1 = below cloud (bearish)
        )

        for i in range(len(self.data)):
            if i > 26:  # Ensure we have cloud data
                if self.data["Close"].iloc[i] > max(
                    self.data["Senkou_Span_A"].iloc[i],
                    self.data["Senkou_Span_B"].iloc[i],
                ):
                    self.data.loc[self.data.index[i], "Cloud_State"] = 1
                elif self.data["Close"].iloc[i] < min(
                    self.data["Senkou_Span_A"].iloc[i],
                    self.data["Senkou_Span_B"].iloc[i],
                ):
                    self.data.loc[self.data.index[i], "Cloud_State"] = -1

    def _add_crypto_onchain_indicators(self):
        """Add crypto-specific indicators that mimic on-chain and exchange data"""

        # Simulate exchange inflow/outflow with price and volume
        self.data["Exchange_Flow"] = 0
        price_change = self.data["Close"].pct_change()
        volume_change = self.data["Volume"].pct_change()

        # When price drops but volume increases = potential exchange inflow (selling pressure)
        self.data.loc[
            (price_change < -0.01) & (volume_change > 0.2), "Exchange_Flow"
        ] = -1

        # When price increases with volume = potential exchange outflow (buying pressure)
        self.data.loc[
            (price_change > 0.01) & (volume_change > 0.2), "Exchange_Flow"
        ] = 1

        # Average True Range Volatility Bands (wider during high volatility)
        self.data["ATR_5"] = self.data["TR"].rolling(5).mean()
        self.data["Upper_Band"] = self.data["Close"] + (self.data["ATR_5"] * 2)
        self.data["Lower_Band"] = self.data["Close"] - (self.data["ATR_5"] * 2)

        # Whale activity detection (large volume spikes)
        median_vol = self.data["Volume"].rolling(50).median()
        self.data["Whale_Activity"] = (self.data["Volume"] > median_vol * 3).astype(int)

        # Create Buy/Sell imbalance ratio
        close_change = self.data["Close"].diff()
        self.data["Buy_Volume"] = self.data["Volume"]
        self.data["Sell_Volume"] = self.data["Volume"]

        self.data.loc[close_change > 0, "Sell_Volume"] = (
            self.data.loc[close_change > 0, "Volume"] * 0.4
        )
        self.data.loc[close_change < 0, "Buy_Volume"] = (
            self.data.loc[close_change < 0, "Volume"] * 0.4
        )

        # Replace the problematic line with this more robust implementation
        buy_vol_sum = self.data["Buy_Volume"].rolling(24).sum()
        sell_vol_sum = self.data["Sell_Volume"].rolling(24).sum()
        # Handle potential zeros in denominator and handle NaNs
        self.data["Buy_Sell_Ratio"] = buy_vol_sum / sell_vol_sum.replace(0, 0.001)
        # Fill NaN values with a neutral ratio of 1.0
        self.data["Buy_Sell_Ratio"] = self.data["Buy_Sell_Ratio"].fillna(1.0)

    def load_data(self):
        """Load and prepare stock data with features"""
        # Add momentum-specific features
        window = 15  # Standard momentum window
        # Check if this is a cryptocurrency ticker
        is_crypto = (
            "-USD" in self.symbol
            or "USD" in self.symbol
            or "/USD" in self.symbol
            or self.symbol.endswith("BTC")
        )
        try:
            yf_symbol = (
                self.symbol.replace("/", "-") if "/" in self.symbol else self.symbol
            )
            data = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False,
                # prepost=True  # Include pre/post market data
            )
            # Check if we got valid data
            if data.empty or len(data) < 5:
                raise Exception(f"Insufficient data from Yahoo Finance for {yf_symbol}")

            # Process the data
            data.columns = data.columns.get_level_values(0)  # Remove multi-index
            self.data = data
            print(f"Successfully loaded data from Yahoo Finance for {yf_symbol}")
        except Exception as e:  # Handle  incompatiable ticker symbols or other issues
            print(f"Failed to get data from Yahoo Finance for {self.symbol}: {str(e)}")
            if is_crypto:
                print(f"Attempting to get crypto data from Alpaca for {self.symbol}")
                self._load_crypto_data_from_alpaca()
            else:
                print(f"Attempting to get stock data from Alpaca for {self.symbol}")
                self._load_stock_data_from_alpaca(current=True)

        # self.data = self.data.ffill()  # Forward fill missing values
        # self.data = self.data.dropna()  # Remove rows with missing values
        # Defensive check for empty data after cleaning
        if self.data.empty:
            print(f"Warning: Data for {self.symbol} is empty after cleaning")
            return self

        # Add technical indicators with optional features
        self._add_technical_indicators(
            window,
            include_fourier=self.include_fourier,
            include_fft_pca=self.include_fft_pca,
        )
        # Add additional indicators based on asset type and inclusion flags
        if is_crypto:
            if self.include_crypto_liquidity:
                self._add_crypto_liquidity_indicators()
            if self.include_crypto_volatility:
                self._add_crypto_volatility_indicators()
            if self.include_crypto_structure:
                self._add_crypto_market_structure_indicators()
            if self.include_onchain:
                self._add_crypto_onchain_indicators()
        else:
            if self.include_economic:
                self._add_economic_indicators()

        # Final cleaning
        # convert timezone to AMErican/New_York
        if self.interval in ["1m", "5m"]:
            self.data.index = self.data.index.tz_convert("America/New_York")
        # Drop the columns if most of the values are NaN
        self.data = self.data.dropna(axis=1, thresh=len(self.data) * 0.1)
        self.data = self.data.dropna()
        if len(self.data) < 50:

            print("Not enough data to train the model.")
            raise ValueError(
                "Not enough data to train the model. Number of rows: {}".format(
                    len(self.data)
                )
            )

        # Defensive check for empty data after all processing
        else:
            print(f"Data loaded for {self.symbol} with {len(self.data)} rows.")

        return self

    def _load_crypto_data_from_alpaca(self):
        """Get crypto data from Alpaca API when yfinance fails"""
        try:
            from alpaca.data import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            # Initialize crypto data client
            crypto_client = CryptoHistoricalDataClient()

            # Convert interval to Alpaca timeframe
            timeframe_map = {
                "1m": TimeFrame.Minute,
                "5m": TimeFrame(5, TimeFrameUnit.Minute),
                # "15m": TimeFrame.Minute(15),
                # "30m": TimeFrame.Minute(30),
                # "60m": TimeFrame.Hour,
                # "90m": TimeFrame.Hour,  # Alpaca doesn't have 90m, use Hour
                "1h": TimeFrame.Hour,
                "1d": TimeFrame.Day,
            }

            timeframe = timeframe_map.get(self.interval, TimeFrame.Day)

            # Prepare symbol for Alpaca (ensure it has the right format)
            alpaca_symbol = self.symbol
            if "-" in alpaca_symbol and "/" not in alpaca_symbol:
                alpaca_symbol = alpaca_symbol.replace("-", "/")

            # Create request for crypto bars
            request_params = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=timeframe,
                start=pd.Timestamp(self.start_date).tz_localize("America/New_York"),
                end=pd.Timestamp(self.end_date).tz_localize("America/New_York"),
            )

            # Get the bars
            bars = crypto_client.get_crypto_bars(request_params)

            # Convert to DataFrame
            df = bars.df

            # Reset multi-level index and format similar to yfinance output
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)

            # Rename columns to match yfinance format
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    # 'trade_count': 'Trade_Count',
                    # 'vwap': 'VWAP'
                }
            )

            self.data = df
            # logger.info(f"Successfully loaded crypto data from Alpaca for {alpaca_symbol}")
            return df

        except Exception as e:
            print(f"Failed to get crypto data from Alpaca for {self.symbol}: {str(e)}")
            self.data = pd.DataFrame()  # Empty DataFrame
            return self.data

    def _load_stock_data_from_alpaca(self, current=False):
        """Get stock data from Alpaca API when yfinance fails"""
        try:
            from alpaca.data import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
            from alpaca.data.live import StockDataStream
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            # Init stock data client (assuming API keys are available)
            api_key = "PKXPBKCIK15IBA4G84P4"
            secret_key = "aJHuDphvn8S6M69F0Vrc0EAudEgob2xc5ltXc0bA"
            if current:
                # Get live/latest data

                print(f"Fetching live data for {self.symbol}")

                stock_client = StockHistoricalDataClient(api_key, secret_key)

                # Get latest quote
                latest_quote_request = StockLatestQuoteRequest(
                    symbol_or_symbols=self.symbol
                )
                latest_quote = stock_client.get_stock_latest_quote(latest_quote_request)

                # Convert to DataFrame format - fix timezone handling
                current_time = pd.Timestamp.now()
                # Check if current_time already has timezone info
                # if current_time.tz is None:
                #     current_time = current_time.tz_localize("America/New_York")
                # else:
                current_time = current_time.tz_convert("America/New_York")

                # Create a single row DataFrame with current data
                live_data = {
                    "Open": latest_quote[
                        self.symbol
                    ].bid_price,  # Use bid as proxy for open
                    "High": max(
                        latest_quote[self.symbol].ask_price,
                        latest_quote[self.symbol].bid_price,
                    ),
                    "Low": min(
                        latest_quote[self.symbol].ask_price,
                        latest_quote[self.symbol].bid_price,
                    ),
                    "Close": latest_quote[
                        self.symbol
                    ].bid_price,  # Use bid as current price
                    "Volume": latest_quote[self.symbol].bid_size
                    + latest_quote[self.symbol].ask_size,
                }

                df = pd.DataFrame([live_data], index=[current_time])

                # If we need more historical context, get recent bars too
                if self.interval in ["1m", "5m"]:
                    # Get last few hours of data for context
                    bars_request = StockBarsRequest(
                        symbol_or_symbols=self.symbol,
                        timeframe=(
                            TimeFrame.Minute
                            if self.interval == "1m"
                            else TimeFrame(5, TimeFrameUnit.Minute)
                        ),
                        start=(current_time - pd.Timedelta(hours=6)).tz_localize(
                            "America/New_York"
                        ),
                        end=current_time,
                    )
                    bars = stock_client.get_stock_bars(bars_request)
                    bars_df = bars.df

                    if isinstance(bars_df.index, pd.MultiIndex):
                        bars_df = bars_df.reset_index(level=0, drop=True)

                    bars_df = bars_df.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        }
                    )

                    # Combine historical bars with current live data
                    df = pd.concat([bars_df, df]).drop_duplicates()

            else:
                stock_client = StockHistoricalDataClient(api_key, secret_key)

                # Convert interval to Alpaca timeframe
                timeframe_map = {
                    "1m": TimeFrame.Minute,
                    "5m": TimeFrame(5, TimeFrameUnit.Minute),
                    # "15m": TimeFrame.Minute(15),
                    # "30m": TimeFrame.Minute(30),
                    # "60m": TimeFrame.Hour,
                    # "90m": TimeFrame.Hour,  # Alpaca doesn't have 90m, use Hour
                    "1h": TimeFrame.Hour,
                    "1d": TimeFrame.Day,
                }

                timeframe = timeframe_map.get(self.interval, TimeFrame.Day)

                # Create request for stock bars
                request_params = StockBarsRequest(
                    symbol_or_symbols=self.symbol,
                    timeframe=timeframe,
                    start=pd.Timestamp(self.start_date).tz_localize("America/New_York"),
                    end=pd.Timestamp(self.end_date).tz_localize("America/New_York"),
                )

                # Get the bars
                bars = stock_client.get_stock_bars(request_params)

                # Convert to DataFrame
                df = bars.df

                # Reset multi-level index and format similar to yfinance output
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index(level=0, drop=True)

                # Rename columns to match yfinance format
                df = df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                        # 'trade_count': 'Trade_Count',
                        # 'vwap': 'VWAP'
                    }
                )

            self.data = df
            # print(f"Successfully loaded stock data from Alpaca for {self.symbol}")
            return df

        except Exception as e:
            print(f"Failed to get stock data from Alpaca for {self.symbol}: {str(e)}")
            self.data = pd.DataFrame()  # Empty DataFrame
            return self.data


class Backtester:
    """Integrated backtesting engine that works with your StockPredictor"""

    def __init__(self, predictor, initial_capital=100000):
        self.predictor = predictor
        self.initial_capital = initial_capital
        self.portfolio = {
            "cash": initial_capital,
            "positions": {},
            "value_history": [],
            "transactions": [],
        }
        self.slippage = 0.0005  # 5bps
        self.commission = 0.0001  # $0.01 per share
        self.full_data = None  # Placeholder for full data

    def _calculate_position_size(self, current_price):
        """Use your existing risk parameters"""
        # risk_per_trade = self.initial_capital * self.predictor.risk_params['per_trade_risk']
        # atr = self.predictor.data['ATR'].iloc[-1]
        # return risk_per_trade / (atr * current_price)
        risk_per_trade = (
            self.portfolio["cash"] * self.predictor.risk_params["per_trade_risk"]
        )
        return risk_per_trade / current_price

    def run_backtest(self, start_date, end_date):
        """More robust date handling"""
        try:
            import pandas_market_calendars as mcal
            import pandas as pd
            import numpy as np

            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=start_date, end_date=end_date)
            if schedule.empty:
                print(f"No trading days between {start_date} and {end_date}")
                return pd.DataFrame(), {"error": "No trading days"}

            dates = schedule.index.tz_localize(None)
            print("First date:", dates[0])
            print("Last date:", dates[-1])
        except Exception as e:
            print(f"Date error: {str(e)}")
            return pd.DataFrame(), {"error": str(e)}

        # if rebalance_frequency == 'weekly':
        #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        # elif rebalance_frequency == 'monthly':
        #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='BM')
        # elif rebalance_frequency == 'quarterly':
        #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='BQ')
        # else:
        #     raise ValueError("rebalance_frequency must be 'weekly', 'monthly', or 'quarterly'")

        # Store original full data
        full_data = (
            self.predictor.data.copy()
        )  # data till today and so no end date is needed for the stock predictor
        self.full_data = full_data

        # Get signal from new model once three days
        i = 0
        first_date = dates[0]
        for date in dates:
            # Make sure date exists in our data
            if date not in full_data.index:
                print(f"Date {date} not in data. Skipping.")
                continue

            # is_rebalance_day = pd.to_datetime(date) in rebalance_dates

            # if is_rebalance_day:
            #     print(f"Running model on rebalance date: {date}")

            if i % 3 == 0 and i != 0:  # regenerate signal every 10 days
                first_date = date

            # self.predictor.end_date = date - pd.Timedelta(days=1)
            self.predictor.end_date = first_date
            self.predictor.load_data()  # Fresh load with cutoff
            # self.predictor.data = self.predictor.data.loc[:date]
            print(f"last data of predictor data: {self.predictor.data.index[-1]}")
            i += 1

            # Filter data up to current date
            # self.predictor.data = full_data.loc[:date].copy()

            # Generate signal using existing code
            signal = self.predictor.generate_trading_signal(
                self.predictor.symbol, horizon=5
            )

            # Execute trade
            try:

                current_price = full_data["Close"].loc[date]
                position_size = self._calculate_position_size(current_price)
            except (KeyError, IndexError) as e:
                print(f"Data not available for {date}. Error: {e}. Skipping.")
                continue

            # Apply slippage and commission
            executed_price = (
                current_price * (1 + self.slippage)
                if signal == "BUY"
                else current_price * (1 - self.slippage)
            )

            if (
                signal == "BUY"
                and self.portfolio["cash"] > executed_price * position_size
            ):
                self._execute_buy(executed_price, position_size, date)
            elif (
                signal == "SELL"
                and self.predictor.symbol in self.portfolio["positions"]
            ):
                self._execute_sell(executed_price, date)
            else:
                print(
                    f"Signal is hold so no trade executed for {self.predictor.symbol} on {date} "
                )

            # Update portfolio value
            self._update_portfolio_value(date)

            # Check risk limits
            if self._check_daily_loss():
                print(f"Daily loss limit hit on {date}. Stopping backtest.")
                break

        # Restore original data
        self.predictor.data = full_data

        return self._generate_report()

    # ------------------------------------------------------------------------------------------------------------

    def _execute_buy(self, price, qty, date):
        cost = price * qty + self.commission * qty
        self.portfolio["cash"] -= cost
        # Want to ensure we don't overwrite existing positions but add to them
        if self.predictor.symbol in self.portfolio["positions"]:
            self.portfolio["positions"][self.predictor.symbol]["qty"] += qty
            self.portfolio["positions"][self.predictor.symbol]["Avg_entry_price"] = (
                self.portfolio["positions"][self.predictor.symbol]["Avg_entry_price"]
                * self.portfolio["positions"][self.predictor.symbol]["qty"]
                + price * qty
            ) / (self.portfolio["positions"][self.predictor.symbol]["qty"] + qty)
        else:
            # Initialize new position
            self.portfolio["positions"][self.predictor.symbol] = {
                "qty": qty,
                "Avg_entry_price": price,
                # 'entry_date': date
            }

        self.portfolio["transactions"].append(("BUY", price, qty, date))
        print(f"BUY executed on {date}: {qty} shares at ${price:.2f}")

    def _execute_sell(self, price, date):
        if self.predictor.symbol not in self.portfolio["positions"]:
            print(f"No position to sell for {self.predictor.symbol} but want to short")
            # Naked shorting
            qty = self._calculate_position_size(price)
            self.portfolio["cash"] += price * qty - self.commission * qty
            self.portfolio["transactions"].append(("SELL", price, qty, date))
            self.portfolio["positions"][self.predictor.symbol] = {
                "qty": -qty,
                "Avg_entry_price": price,
                # 'entry_date': date
            }
            print(f"SELL executed on {date}: {qty} shares at ${price:.2f}")
            return
        # Option 1: liquidate all positions
        # position = self.portfolio['positions'].pop(self.predictor.symbol)
        # proceeds = price * position['qty'] - self.commission * position['qty']
        # self.portfolio['cash'] += proceeds
        # profit = proceeds - (position['Avg_entry_price'] * position['qty'] + self.commission * position['qty'])
        # self.portfolio['transactions'].append(('SELL', price, position['qty'], date))
        # print(f"SELL executed on {date}: {position['qty']} shares at ${price:.2f}, profit: ${profit:.2f}")

        # Option 2: partial liquidation from the postion by amount of shares calculated
        position = self.portfolio["positions"][self.predictor.symbol]
        qty = self._calculate_position_size(price)
        if qty >= position["qty"]:
            qty = position["qty"]
            self.portfolio["positions"].pop(self.predictor.symbol)

        position["qty"] -= qty
        proceeds = price * qty - self.commission * qty
        self.portfolio["cash"] += proceeds
        profit = proceeds - (position["Avg_entry_price"] * qty + self.commission * qty)
        self.portfolio["transactions"].append(("SELL", price, qty, date))
        print(
            f"SELL executed on {date}: {qty} shares at ${price:.2f}, profit: ${profit:.2f}"
        )

    def _check_daily_loss(self):
        """Use your existing risk management"""
        if len(self.portfolio["value_history"]) < 2:
            return False
        daily_pct = (
            self.portfolio["value_history"][-1]["value"]
            / self.portfolio["value_history"][-2]["value"]
        ) - 1
        return daily_pct < self.predictor.risk_params["daily_loss_limit"]

    def _update_portfolio_value(self, date):
        position_value = 0
        for sym, pos in self.portfolio["positions"].items():
            try:
                if sym == self.predictor.symbol:  # We're only tracking one symbol
                    # current_price = self.predictor.data['Close'].iloc[-1]
                    # the current price at the data, not the last date
                    # current_price = self.predictor.data['Close'].loc[date]
                    current_price = self.full_data["Close"].loc[date]
                    if pos["qty"] < 0:
                        position_value -= -pos["qty"] * current_price
                    else:
                        position_value += pos["qty"] * current_price
            except (KeyError, IndexError) as e:
                print(f"Error updating portfolio value: {e}")

        total_value = self.portfolio["cash"] + position_value

        # Ensure consistent data format
        self.portfolio["value_history"].append(
            {
                "date": pd.to_datetime(date),
                "value": total_value,
                "cash": self.portfolio["cash"],
            }
        )
        print(f"Portfolio value on {date}: ${total_value:.2f}")

    def _generate_report(self):
        """More robust report generation"""
        import pandas as pd
        import numpy as np

        if not self.portfolio["value_history"]:
            print("No portfolio history to generate report")
            return pd.DataFrame(), {"error": "No trades executed"}

        try:
            df = pd.DataFrame(self.portfolio["value_history"])
            df = df.set_index("date").sort_index()

            if df.empty:
                return df, {"error": "Empty portfolio history"}

            returns = df["value"].pct_change().dropna()

            if len(returns) < 2:
                return df, {"error": "Insufficient data for metrics"}

            report = {
                "sharpe": returns.mean() / returns.std() * np.sqrt(252),
                "max_drawdown": (df["value"] / df["value"].cummax() - 1).min(),
                "total_return": df["value"].iloc[-1] / self.initial_capital - 1,
                "num_trades": len(self.portfolio["transactions"]),
                "win_rate": self._calculate_win_rate(),
            }
            print(f"Report generated: {report}")
            return df, report

        except Exception as e:
            print(f"Report generation error: {str(e)}")
            return pd.DataFrame(), {"error": str(e)}

    # def _calculate_win_rate(self, history_df):
    #     """Safer win rate calculation"""
    #     # Definition: Winning trades means the trade makes the value of profolio higher than the previous trade (whether or not the position is liquidated)
    #     # We can use the history_df to calculate the win rate
    #     return (history_df['value'].diff().dropna()>0).astype(int).sum() / len(self.portfolio['transactions']) if len(self.portfolio['transactions']) > 0 else 0.0
    def _calculate_win_rate(self):
        """Calculate win rate from completed trades. Only count winning trades when the position is liquidated"""
        buy_trades = [
            (p, d, q) for t, p, q, d in self.portfolio["transactions"] if t == "BUY"
        ]
        sell_trades = [
            (p, d, q) for t, p, q, d in self.portfolio["transactions"] if t == "SELL"
        ]

        if not sell_trades:
            return 0.0

        winning_trades = 0

        # Match buys with sells sequentially (FIFO)
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i][0]
            sell_price = sell_trades[i][0]

            if sell_price > buy_price:
                winning_trades += 1

        return winning_trades / len(sell_trades)  # only count winning


class StressTester(Backtester):
    """Stress tests using your existing strategy"""

    def _apply_market_crash(self, date):
        """Simulate flash crash scenario"""
        if np.random.rand() < 0.05:  # 5% chance daily
            self.predictor.data.loc[date:, "Close"] *= 0.9  # 10% drop
            self.predictor.data["Volatility"] *= 2  # Spike volatility

    def _apply_liquidity_crisis(self, date):
        """Simulate bid-ask spread widening"""
        if np.random.rand() < 0.03:  # 3% chance daily
            self.slippage = 0.01  # 1% slippage
            self.commission = 0.001  # $0.1 per share

    def run_stress_test(self, start_date, end_date):
        """Run stress test using your existing strategy"""
        nyse = mcal.get_calendar("NYSE")
        dates = nyse.schedule(start_date=start_date, end_date=end_date).index
        for date in dates:
            # Apply stress events
            self._apply_market_crash(date)
            self._apply_liquidity_crisis(date)

            # Run normal backtest
            super().run_backtest(date, date)

        return self._generate_report()

    def _run_stress_tests(self, history_df):
        """Run stress tests on the strategy"""
        if len(history_df) < 30:  # Need sufficient data
            return {"stress_test": "Insufficient data"}

        results = {}
        returns = history_df["value"].pct_change().dropna()

        # Test 1: Worst week performance
        weekly_returns = history_df["value"].resample("W").last().pct_change().dropna()
        results["worst_week"] = weekly_returns.min()

        # Test 2: Performance in high volatility periods
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        high_vol_returns = returns[rolling_vol > rolling_vol.quantile(0.75)]
        results["high_vol_performance"] = (
            high_vol_returns.mean() * 252 if not high_vol_returns.empty else 0
        )

        # Test 3: Monte Carlo simulation - 100 paths
        mc_results = self._monte_carlo_simulation(returns, paths=100)
        results["mc_5pct_var"] = mc_results["5pct_var"]
        results["mc_worst_drawdown"] = mc_results["worst_drawdown"]

        return {"stress_tests": results}

    def _monte_carlo_simulation(self, returns, paths=100, horizon=252):
        """Run Monte Carlo simulation to test strategy robustness"""
        sim_returns = np.random.choice(
            returns.values, size=(paths, horizon), replace=True
        )

        # Convert returns to paths
        sim_paths = np.cumprod(1 + sim_returns, axis=1)

        # Calculate metrics
        final_values = sim_paths[:, -1]
        drawdowns = np.zeros(paths)

        for i in range(paths):
            drawdowns[i] = (
                np.min(sim_paths[i] / np.maximum.accumulate(sim_paths[i])) - 1
            )

        return {
            "5pct_var": np.percentile(final_values, 5) - 1,  # 5% VaR
            "worst_drawdown": np.min(drawdowns),  # Worst drawdown across all sims
        }


# Example usage
if __name__ == "__main__":
    predictor = StockPredictor("AAPL", start_date="2020-01-01")
    predictor.load_data()
    print(predictor.data.head())
