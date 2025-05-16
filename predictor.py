import warnings
warnings.filterwarnings('ignore')

import numpy as np
import yfinance as yf

import pandas as pd
from datetime import date, timedelta, datetime


import matplotlib.pyplot as plt
import pandas_market_calendars as mcal


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

 


class StockPredictor:
    """Stock price prediction pipeline

    Parameters:
        symbol (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        interval (str): Data interval (1d, 1h, etc)
    """

    def __init__(self, symbol, start_date, end_date=None, interval="1d"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else date.today()
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
    
    
    def load_data(self):
        """Load and prepare stock data with features"""
        # Add momentum-specific features
        window = 15  # Standard momentum window
        self.data = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            timeout=0.5,
        )
        self.data.columns = self.data.columns.get_level_values(0)  # Remove multi-index
        self.data.ffill()
        self.data.dropna()

        ### 1. Add rolling indicators
        self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()
        self.data["MA_200"] = self.data["Close"].rolling(window=200).mean()
        self.data["MA_7"] = self.data["Close"].rolling(window=7).mean()
        self.data["MA_21"] = self.data["Close"].rolling(window=21).mean()

        ### 2. Fourier transform
        # data_FT = self.data.copy().reset_index()[["Date", "Close"]]
        # close_fft = np.fft.fft(np.asarray(data_FT["Close"].tolist()))
        # self.data["FT_real"] = np.real(close_fft)
        # self.data["FT_img"] = np.imag(close_fft)

        # # Fourier Transformation is not used
        # fft_df = pd.DataFrame({'fft': close_fft})
        # fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        # fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
        # fft_list = np.asarray(fft_df['fft'].tolist())
        # for num_ in [3, 6, 9, 100]:
        #     fft_list_m10 = np.copy(fft_list)
        #     fft_list_m10[num_:-num_] = 0
        #     complex_num = np.fft.ifft(fft_list_m10)
        #     self.data[f'Fourier_trans_{num_}_comp_real'] = np.real(complex_num)
        #     self.data[f'Fourier_trans_{num_}_comp_img'] = np.imag(complex_num)

        # ### Fourier Transformation PCA
        # X_fft = np.column_stack([np.real(close_fft), np.imag(close_fft)])
        # pca = PCA(n_components=2)  # Keep top 2 components
        # X_pca = pca.fit_transform(X_fft)
        # for i in range(X_pca.shape[1]):
        #     self.data[f"Fourier_PCA_{i}"] = X_pca[:, i]

        ### 3. Add rolling statistics
        self.data["rolling_std"] = self.data["Close"].rolling(window=50).std()
        self.data["rolling_min"] = self.data["Close"].rolling(window=50).min()
        # self.data['rolling_max'] = self.data['Close'].rolling(window=window).max()
        self.data["rolling_median"] = self.data["Close"].rolling(window=50).median()
        self.data["rolling_sum"] = self.data["Close"].rolling(window=50).sum()
        self.data["rolling_var"] = self.data["Close"].rolling(window=50).var()
        self.data["rolling_ema"] = (
            self.data["Close"].ewm(span=50, adjust=False).mean()
        )  # Exponential Moving Average
        # Add rolling quantiles (25th and 75th percentiles)
        self.data["rolling_25p"] = self.data["Close"].rolling(window=50).quantile(0.25)
        self.data["rolling_75p"] = self.data["Close"].rolling(window=50).quantile(0.75)
        # Drop rows with NaN values (due to rolling window)
        self.data.dropna(inplace=True)
        stock_data.index.name = "Date"  # Ensure the index is named "Date"

        ### 4. Advanced Momentum
        self.data["RSI"] = self._compute_rsi(window=14)
        self.data["MACD"] = (
            self.data["Close"].ewm(span=12).mean()
            - self.data["Close"].ewm(span=26).mean()
        )
        ### 5. Williams %R
        high_max = self.data["High"].rolling(window).max()
        low_min = self.data["Low"].rolling(window).min()
        self.data["Williams_%R"] = (
            (high_max - self.data["Close"]) / (high_max - low_min)
        ) * -100

        ### 6. Stochastic Oscillator
        self.data["Stochastic_%K"] = (
            (self.data["Close"] - low_min) / (high_max - low_min)
        ) * 100
        self.data["Stochastic_%D"] = self.data["Stochastic_%K"].rolling(3).mean()

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
        self.data["VWAP"] = (
            self.data["Volume"]
            * (self.data["High"] + self.data["Low"] + self.data["Close"])
            / 3
        ).cumsum() / self.data["Volume"].cumsum()

        ### 10. Economic Indicators
        sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date, interval=self.interval,)["Close"]
        # Fetch S&P 500 Index (GSPC) and Treasury Yield ETF (IEF) from Yahoo Finance
        sp500 = sp500 - sp500.mean()
        tnx = yf.download(
            "^TNX", start=self.start_date, end=self.end_date, interval=self.interval, timeout=0.5,
        )["Close"]
        tnx_len = len(tnx)
        treasury_yield = yf.download(
            "IEF", start=self.start_date, end=self.end_date, interval=self.interval, timeout=0.5,
        )["Close"]
        exchange_rate = yf.download(
            "USDCAD=X", start=self.start_date, end=self.end_date, interval=self.interval, timeout=0.5,
        )["Close"]
        technology_sector = yf.download(
            "XLK", start=self.start_date, end=self.end_date, interval=self.interval, timeout=0.5,
        )["Close"]
        financials_sector = yf.download(
            "XLF", start=self.start_date, end=self.end_date, interval=self.interval, timeout=0.5,
        )["Close"]
        energy_sector = yf.download(
            "XLE", start=self.start_date, end=self.end_date, interval=self.interval, timeout=0.5,
        )["Close"]
        vix = yf.download(
            "^VIX", start=self.start_date, end=self.end_date, interval=self.interval, timeout=0.5,
        )["Close"]

        # self.data["SP500"] = sp500
        # self.data["TNX"] = tnx
        # self.data["Treasury_Yield"] = treasury_yield
        # self.data["USDCAD=X"] = exchange_rate
        # self.data["Tech"] = technology_sector
        # self.data["Fin"] = financials_sector
        # self.data["VIX"] = vix
        # self.data["Energy"] = energy_sector

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
        if self.interval == "1m":

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

        # Merge with stock data
        if tnx_len < len(self.data):
            economic_data = economic_data.drop(columns='TNX')
        if self.interval == "1m":
            self.data = pd.merge(self.data, economic_data, on="Datetime", how="left")
        else:
            self.data = pd.merge(self.data, economic_data, on="Date", how="left")

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

        # Final cleaning
        # convert timezone to AMErican/New_York
        if self.interval == "1m":
            self.data.index = self.data.index.tz_convert("America/New_York")
        self.data = self.data.dropna()
        if len(self.data) < 50:
            print("Not enough data to train the model.")
            raise ValueError("Not enough data to train the model.")

        return self


class Backtester:
    """Integrated backtesting engine that works with your StockPredictor"""
    
    def __init__(self, predictor, initial_capital=100000):
        self.predictor = predictor
        self.initial_capital = initial_capital
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},
            'value_history': [],
            'transactions': []
        }
        self.slippage = 0.0005  # 5bps
        self.commission = 0.0001  # $0.01 per share
        self.full_data = None  # Placeholder for full data
    
    def _calculate_position_size(self, current_price):
        """Use your existing risk parameters"""
        # risk_per_trade = self.initial_capital * self.predictor.risk_params['per_trade_risk']
        # atr = self.predictor.data['ATR'].iloc[-1]
        # return risk_per_trade / (atr * current_price)
        risk_per_trade = self.portfolio['cash'] * self.predictor.risk_params['per_trade_risk']
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
                return pd.DataFrame(), {'error': 'No trading days'}
                
            dates = schedule.index.tz_localize(None)
            print('First date:', dates[0])
            print('Last date:', dates[-1])
        except Exception as e:
            print(f"Date error: {str(e)}")
            return pd.DataFrame(), {'error': str(e)}
    
        # if rebalance_frequency == 'weekly':
        #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        # elif rebalance_frequency == 'monthly':
        #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='BM')
        # elif rebalance_frequency == 'quarterly':
        #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='BQ')
        # else:
        #     raise ValueError("rebalance_frequency must be 'weekly', 'monthly', or 'quarterly'")
        

        # Store original full data
        full_data = self.predictor.data.copy()# data till today and so no end date is needed for the stock predictor
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



            
            if i % 3 == 0 and i != 0: # regenerate signal every 10 days
                first_date = date 



                
            # self.predictor.end_date = date - pd.Timedelta(days=1)
            self.predictor.end_date = first_date
            self.predictor.load_data()  # Fresh load with cutoff
            # self.predictor.data = self.predictor.data.loc[:date]  
            print(f'last data of predictor data: {self.predictor.data.index[-1]}')
            i += 1
        
            # Filter data up to current date
            # self.predictor.data = full_data.loc[:date].copy()
            
            # Generate signal using existing code
            signal = self.predictor.generate_trading_signal(self.predictor.symbol, horizon = 5)
            
            # Execute trade
            try:

                current_price = full_data['Close'].loc[date]
                position_size = self._calculate_position_size(current_price)
            except (KeyError, IndexError) as e:
                print(f"Data not available for {date}. Error: {e}. Skipping.")
                continue
            
            # Apply slippage and commission
            executed_price = current_price * (1 + self.slippage) if signal == 'BUY' \
                else current_price * (1 - self.slippage)
            
            if signal == 'BUY' and self.portfolio['cash'] > executed_price * position_size:
                self._execute_buy(executed_price, position_size, date)
            elif signal == 'SELL' and self.predictor.symbol in self.portfolio['positions']:
                self._execute_sell(executed_price, date)
            else:
                print(f"Signal is hold so no trade executed for {self.predictor.symbol} on {date} ")
            
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
        self.portfolio['cash'] -= cost
        # Want to ensure we don't overwrite existing positions but add to them
        if self.predictor.symbol in self.portfolio['positions']:
            self.portfolio['positions'][self.predictor.symbol]['qty'] += qty
            self.portfolio['positions'][self.predictor.symbol]['Avg_entry_price'] = (
                self.portfolio['positions'][self.predictor.symbol]['Avg_entry_price'] *
                self.portfolio['positions'][self.predictor.symbol]['qty'] + price * qty 
            ) / (self.portfolio['positions'][self.predictor.symbol]['qty'] + qty)
        else:
            # Initialize new position
            self.portfolio['positions'][self.predictor.symbol] = {
                'qty': qty, 
                'Avg_entry_price': price,
                # 'entry_date': date
            }



        self.portfolio['transactions'].append(('BUY', price, qty, date))
        print(f"BUY executed on {date}: {qty} shares at ${price:.2f}")



    def _execute_sell(self, price, date):
        if self.predictor.symbol not in self.portfolio['positions']:
            print(f"No position to sell for {self.predictor.symbol} but want to short")
            # Naked shorting
            qty = self._calculate_position_size(price)
            self.portfolio['cash'] += price * qty - self.commission * qty
            self.portfolio['transactions'].append(('SELL', price, qty, date))
            self.portfolio['positions'][self.predictor.symbol] = {
                'qty': -qty, 
                'Avg_entry_price': price,
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
        position = self.portfolio['positions'][self.predictor.symbol]
        qty = self._calculate_position_size(price)
        if qty >= position['qty']:
            qty = position['qty']
            self.portfolio['positions'].pop(self.predictor.symbol)

        position['qty'] -= qty
        proceeds = price * qty - self.commission * qty
        self.portfolio['cash'] += proceeds
        profit = proceeds - (position['Avg_entry_price'] * qty + self.commission * qty)
        self.portfolio['transactions'].append(('SELL', price, qty, date))
        print(f"SELL executed on {date}: {qty} shares at ${price:.2f}, profit: ${profit:.2f}")





    def _check_daily_loss(self):
        """Use your existing risk management"""
        if len(self.portfolio['value_history']) < 2:
            return False
        daily_pct = (self.portfolio['value_history'][-1]['value'] / 
                    self.portfolio['value_history'][-2]['value']) - 1
        return daily_pct < self.predictor.risk_params['daily_loss_limit']
 
    def _update_portfolio_value(self, date):
        position_value = 0
        for sym, pos in self.portfolio['positions'].items():
            try:
                if sym == self.predictor.symbol:  # We're only tracking one symbol
                    # current_price = self.predictor.data['Close'].iloc[-1]
                    # the current price at the data, not the last date
                    # current_price = self.predictor.data['Close'].loc[date]
                    current_price = self.full_data['Close'].loc[date]
                    if pos['qty'] < 0:
                        position_value -= -pos['qty'] * current_price
                    else:
                        position_value += pos['qty'] * current_price
            except (KeyError, IndexError) as e:
                print(f"Error updating portfolio value: {e}")
        
        total_value = self.portfolio['cash'] + position_value
        
        # Ensure consistent data format
        self.portfolio['value_history'].append({
            'date': pd.to_datetime(date),
            'value': total_value,
            'cash': self.portfolio['cash'],
        })
        print(f"Portfolio value on {date}: ${total_value:.2f}")

    def _generate_report(self):
        """More robust report generation"""
        import pandas as pd
        import numpy as np
        
        if not self.portfolio['value_history']:
            print("No portfolio history to generate report")
            return pd.DataFrame(), {'error': 'No trades executed'}
        
        try:
            df = pd.DataFrame(self.portfolio['value_history'])
            df = df.set_index('date').sort_index()
            
            if df.empty:
                return df, {'error': 'Empty portfolio history'}
                
            returns = df['value'].pct_change().dropna()
            
            if len(returns) < 2:
                return df, {'error': 'Insufficient data for metrics'}
            
            report = {
                'sharpe': returns.mean() / returns.std() * np.sqrt(252),
                'max_drawdown': (df['value'] / df['value'].cummax() - 1).min(),
                'total_return': df['value'].iloc[-1] / self.initial_capital - 1,
                'num_trades': len(self.portfolio['transactions']),
                'win_rate': self._calculate_win_rate()
            }
            print(f"Report generated: {report}")
            return df, report
            
        except Exception as e:
            print(f"Report generation error: {str(e)}")
            return pd.DataFrame(), {'error': str(e)}

    # def _calculate_win_rate(self, history_df):
    #     """Safer win rate calculation"""
    #     # Definition: Winning trades means the trade makes the value of profolio higher than the previous trade (whether or not the position is liquidated)
    #     # We can use the history_df to calculate the win rate
    #     return (history_df['value'].diff().dropna()>0).astype(int).sum() / len(self.portfolio['transactions']) if len(self.portfolio['transactions']) > 0 else 0.0
    def _calculate_win_rate(self):
        """Calculate win rate from completed trades. Only count winning trades when the position is liquidated"""
        buy_trades = [(p, d, q) for t, p, q, d in self.portfolio['transactions'] if t == 'BUY']
        sell_trades = [(p, d, q) for t, p, q, d in self.portfolio['transactions'] if t == 'SELL']
        
        if not sell_trades:
            return 0.0
        
        winning_trades = 0
        
        # Match buys with sells sequentially (FIFO)
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i][0]
            sell_price = sell_trades[i][0]
            
            if sell_price > buy_price:
                winning_trades += 1
        
        return winning_trades / len(sell_trades) #only count winning
        

        

class StressTester(Backtester):
    """Stress tests using your existing strategy"""
    
    def _apply_market_crash(self, date):
        """Simulate flash crash scenario"""
        if np.random.rand() < 0.05:  # 5% chance daily
            self.predictor.data.loc[date:, 'Close'] *= 0.9  # 10% drop
            self.predictor.data['Volatility'] *= 2  # Spike volatility
            
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
            return {'stress_test': 'Insufficient data'}
        
        results = {}
        returns = history_df['value'].pct_change().dropna()
        
        # Test 1: Worst week performance
        weekly_returns = (history_df['value'].resample('W').last().pct_change().dropna())
        results['worst_week'] = weekly_returns.min()
        
        # Test 2: Performance in high volatility periods
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        high_vol_returns = returns[rolling_vol > rolling_vol.quantile(0.75)]
        results['high_vol_performance'] = high_vol_returns.mean() * 252 if not high_vol_returns.empty else 0
        
        # Test 3: Monte Carlo simulation - 100 paths
        mc_results = self._monte_carlo_simulation(returns, paths=100)
        results['mc_5pct_var'] = mc_results['5pct_var']
        results['mc_worst_drawdown'] = mc_results['worst_drawdown']
        
        return {'stress_tests': results}
    
    def _monte_carlo_simulation(self, returns, paths=100, horizon=252):
        """Run Monte Carlo simulation to test strategy robustness"""
        sim_returns = np.random.choice(
            returns.values,
            size=(paths, horizon),
            replace=True
        )
        
        # Convert returns to paths
        sim_paths = np.cumprod(1 + sim_returns, axis=1)
        
        # Calculate metrics
        final_values = sim_paths[:, -1]
        drawdowns = np.zeros(paths)
        
        for i in range(paths):
            drawdowns[i] = np.min(sim_paths[i] / np.maximum.accumulate(sim_paths[i])) - 1
        
        return {
            '5pct_var': np.percentile(final_values, 5) - 1,  # 5% VaR
            'worst_drawdown': np.min(drawdowns)  # Worst drawdown across all sims
        }






# Example usage
if __name__ == "__main__":
    predictor = StockPredictor("AAPL", start_date="2020-01-01")

