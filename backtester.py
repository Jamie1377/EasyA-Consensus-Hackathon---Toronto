from aptos_sdk.account import Account
from aptos_sdk.account_address import AccountAddress
from aptos_sdk.async_client import FaucetClient, RestClient
from aptos_sdk.transactions import (
    EntryFunction,
    TransactionPayload,
    TransactionArgument,
    RawTransaction,
)
from aptos_sdk.bcs import Serializer
import os
import json
import requests
import asyncio
import time
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import random
from pytickersymbols import PyTickerSymbols


# Configure a shared logger for the backtester module
log_directory = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    log_directory, f"aptos_backtest_{date.today().strftime('%Y%m%d')}.log"
)

# Create the main backtester logger
logger = logging.getLogger("aptos_backtest")
logger.setLevel(logging.INFO)

# Ensure we only add handlers once
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)


class AptosBacktester:
    """
    Portfolio tracker for Aptos transactions with functionality similar to
    the Backtester in stock_prediction/core/predictor.py
    """

    def __init__(self, symbols=None, initial_capital=100):
        """
        Initialize the backtester

        Args:
            symbols (list or str): Token/stock symbols to track (single str or list)
            initial_capital (float): Initial capital in USD
        """
        # Handle both single symbol string and list of symbols
        if symbols is None:
            self.symbols = ["APT21794-USD"]
        elif isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = symbols

        self.initial_capital = initial_capital
        self.portfolio = {
            "cash": initial_capital,
            "positions": {},  # symbol -> {qty, entry_price}
            "value_history": [],  # [{timestamp, value, cash}]
            "transactions": [],  # [type, symbol, price, qty, timestamp]
        }

        # Current prices for all symbols
        self.current_prices = {symbol: 0.0 for symbol in self.symbols}

        # Trade parameters
        self.slippage = 0.002  # 10 basis points
        self.commission = 0.001  # 0.1% per transaction and usually fixed

        # Add a reference index for the symbol
        self.reference_ticker = "SPY"
        self.reference_data = None

        # Configure logging
        log_directory = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(
            log_directory, f"aptos_backtest_{date.today().strftime('%Y%m%d')}.log"
        )

        self.logger = logging.getLogger("aptos_backtest")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)

        # Setup transaction log files
        self.transaction_log_file = os.path.join(
            log_directory, "aptos_transactions.csv"
        )
        self.portfolio_log_file = os.path.join(log_directory, "aptos_portfolio.csv")

        # Initialize transaction log if it doesn't exist
        if not os.path.exists(self.transaction_log_file):
            pd.DataFrame(
                columns=["timestamp", "type", "symbol", "price", "quantity", "value"]
            ).to_csv(self.transaction_log_file, index=False)

        # Initialize portfolio log if it doesn't exist
        if not os.path.exists(self.portfolio_log_file):
            pd.DataFrame(
                columns=["timestamp", "total_value", "cash", "positions"]
            ).to_csv(self.portfolio_log_file, index=False)

        self.logger.info(
            f"AptosBacktester initialized with {initial_capital} USD for {len(self.symbols)} symbols: {', '.join(self.symbols)}"
        )

    def record_transaction(
        self, transaction_type, symbol, price, quantity, timestamp=None
    ):
        """Record a buy or sell transaction"""
        if timestamp is None:
            timestamp = datetime.now()
        value = price * quantity

        # Add to in-memory transaction list
        self.portfolio["transactions"].append(
            (transaction_type, symbol, price, quantity, timestamp)
        )

        # Log transaction to file
        transaction_df = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "type": transaction_type,
                    "symbol": symbol,
                    "price": price,
                    "quantity": quantity,
                    "value": value,
                }
            ]
        )

        transaction_df.to_csv(
            self.transaction_log_file, mode="a", header=False, index=False
        )
        self.logger.info(
            f"{transaction_type} executed: {quantity:.6f} {symbol} at ${price:.4f}"
        )

        return True

    def execute_buy(self, symbol, price, quantity, timestamp=None):
        """Execute a buy order for a specific symbol"""
        if symbol not in self.symbols:
            self.logger.warning(f"Symbol {symbol} not in tracked symbols list")
            return False

        if timestamp is None:
            timestamp = datetime.now()

        # Apply slippage to buy price
        executed_price = price * (1 + self.slippage)

        # Calculate transaction cost
        cost = executed_price * quantity
        commission = cost * self.commission
        total_cost = cost + commission

        # Check if enough cash
        if total_cost > self.portfolio["cash"]:
            self.logger.warning(
                f"Insufficient funds for purchase. Required: ${total_cost:.2f}, Available: ${self.portfolio['cash']:.2f}"
            )
            # Adjust quantity to available cash
            adjusted_quantity = (
                self.portfolio["cash"] * 0.025
            ) / executed_price  # Leave 90% buffer
            if adjusted_quantity <= 0:
                return False

            quantity = adjusted_quantity
            cost = executed_price * quantity
            commission = cost * self.commission
            total_cost = cost + commission
            self.logger.info(
                f"Adjusted buy quantity to {quantity:.6f} based on available funds"
            )

        # Simulate on-chain smart contract call for buy
        # In production, replace this with actual contract interaction
        try:
            # Simulate sending transaction to smart contract
            # tx_hash = aptos_sdk_wrapper.buy_token(symbol, executed_price, quantity)
            import random, string

            tx_hash = "0x" + "".join(random.choices(string.hexdigits, k=64))
            onchain_status = "SUCCESS"  # Simulate status
            self.logger.info(
                f"On-chain BUY tx sent: {tx_hash} | Status: {onchain_status}"
            )
        except Exception as e:
            self.logger.error(f"On-chain BUY tx failed: {e}")
            return False

        # Update portfolio
        self.portfolio["cash"] -= total_cost

        if symbol in self.portfolio["positions"]:
            # Update existing position
            position = self.portfolio["positions"][symbol]
            total_quantity = position["qty"] + quantity
            avg_price = (
                (position["qty"] * position["entry_price"])
                + (quantity * executed_price)
            ) / total_quantity

            self.portfolio["positions"][symbol] = {
                "qty": total_quantity,
                "entry_price": avg_price,
            }
        else:
            # Create new position
            self.portfolio["positions"][symbol] = {
                "qty": quantity,
                "entry_price": executed_price,
            }

        # Update current price for this symbol
        self.current_prices[symbol] = price

        # Record the transaction (include tx_hash in log)
        self.record_transaction("BUY", symbol, executed_price, quantity, timestamp)
        self.logger.info(f"On-chain BUY transaction hash: {tx_hash}")

        return True

    def execute_sell(self, symbol, price, quantity=None, timestamp=None):
        """Execute a sell order with support for naked short selling"""
        if symbol not in self.symbols:
            self.logger.warning(f"Symbol {symbol} not in tracked symbols list")
            return False

        if timestamp is None:
            timestamp = datetime.now()

        # Apply slippage to sell price (negative for sells)
        executed_price = price * (1 - self.slippage)

        # If no quantity specified, sell all holdings or create standard short position
        if quantity is None:
            if symbol in self.portfolio["positions"]:
                quantity = (
                    self.portfolio["positions"][symbol]["qty"] * 0.025
                )  # 0.3 % of current position
            else:
                # Default short position size (10% of cash value)
                quantity = (self.portfolio["cash"] * 0.025) / executed_price
                self.logger.info(
                    f"No position to sell, creating short position of {quantity:.6f} shares"
                )

        # Simulate on-chain smart contract call for sell
        # In production, replace this with actual contract interaction
        try:
            # Simulate sending transaction to smart contract
            # tx_hash = aptos_sdk_wrapper.sell_token(symbol, executed_price, quantity)
            import random, string

            tx_hash = "0x" + "".join(random.choices(string.hexdigits, k=64))
            onchain_status = "SUCCESS"  # Simulate status
            self.logger.info(
                f"On-chain SELL tx sent: {tx_hash} | Status: {onchain_status}"
            )
        except Exception as e:
            self.logger.error(f"On-chain SELL tx failed: {e}")
            return False

        # Calculate transaction value and fees
        value = executed_price * quantity
        commission = value * self.commission
        net_proceeds = value - commission

        if symbol in self.portfolio["positions"]:
            # We have an existing position
            position = self.portfolio["positions"][symbol]

            if position["qty"] >= quantity:
                # Normal sell - we have enough shares
                position["qty"] -= quantity
                if position["qty"] <= 0:
                    # Remove position if sold out completely
                    self.portfolio["positions"].pop(symbol)

                # Add proceeds to cash
                self.portfolio["cash"] += net_proceeds

            else:
                # Selling more than we own - partial short
                # First sell all existing shares
                existing_qty = position["qty"]
                remaining_qty = quantity - existing_qty

                # Add proceeds from existing shares
                self.portfolio["cash"] += (executed_price * existing_qty) - (
                    commission * existing_qty / quantity
                )

                # Create short position with remaining quantity
                self.portfolio["positions"][symbol] = {
                    "qty": -remaining_qty,  # Negative quantity indicates short
                    "entry_price": executed_price,
                }

                # Add proceeds from short sale (these are held as cash but may be restricted in real trading)
                self.portfolio["cash"] += (executed_price * remaining_qty) - (
                    commission * remaining_qty / quantity
                )

                self.logger.info(
                    f"Partial short created: sold {existing_qty:.6f} owned shares of {symbol} and shorted {remaining_qty:.6f} additional shares"
                )

        else:
            # No existing position - creating a pure short position
            self.portfolio["positions"][symbol] = {
                "qty": -quantity,  # Negative quantity indicates short
                "entry_price": executed_price,
            }

            # Add proceeds to cash (in real trading this might be held as margin)
            self.portfolio["cash"] += net_proceeds

            self.logger.info(
                f"Created new short position of {quantity:.6f} shares of {symbol} at ${executed_price:.2f}"
            )

        # Update current price for this symbol
        self.current_prices[symbol] = price

        # Record the transaction (include tx_hash in log)
        self.record_transaction("SELL", symbol, executed_price, quantity, timestamp)
        self.logger.info(f"On-chain SELL transaction hash: {tx_hash}")

        return True

    def update_portfolio_value(self, symbol_prices=None, timestamp=None):
        """Calculate current portfolio value and record to history, supporting short positions and multiple symbols"""
        if timestamp is None:
            timestamp = datetime.now()

        # If no prices provided, initialize an empty dict
        if symbol_prices is None:
            symbol_prices = {}

        # Calculate position value, handling both long and short positions
        position_value = 0
        position_values = {}

        for symbol, position in self.portfolio["positions"].items():
            # Get price for this symbol (use provided price, current price, or entry price as fallback)
            if symbol in symbol_prices:
                price = symbol_prices[symbol]
            elif symbol in self.current_prices and self.current_prices[symbol] > 0:
                price = self.current_prices[symbol]
            else:
                price = position["entry_price"]

                # Try to get current price if we don't have one
                try:
                    current_price = float(
                        yf.download(symbol, period="1d", interval="1m", timeout=10)[
                            "Close"
                        ].iloc[-1]
                    )
                    price = current_price
                    self.current_prices[symbol] = current_price
                except Exception as e:
                    self.logger.error(f"Failed to get current price for {symbol}: {e}")

            # Calculate value based on position type
            if position["qty"] < 0:  # Short position
                liability = -position["qty"] * price
                position_value -= liability
                position_values[symbol] = -liability
            else:  # Long position
                value = position["qty"] * price
                position_value += value
                position_values[symbol] = value

        total_value = self.portfolio["cash"] + position_value

        # Record to history
        value_entry = {
            "timestamp": timestamp,
            "value": total_value,
            "cash": self.portfolio["cash"],
            "positions": position_values,
        }

        for symbol in self.symbols:
            value_entry[f"{symbol}_price"] = symbol_prices.get(
                symbol, self.current_prices.get(symbol, 0)
            )

        self.portfolio["value_history"].append(value_entry)

        # Log to portfolio file
        portfolio_record = {
            "timestamp": timestamp,
            "total_value": total_value,
            "cash": self.portfolio["cash"],
            "positions": str(self.portfolio["positions"]),
        }

        pd.DataFrame([portfolio_record]).to_csv(
            self.portfolio_log_file, mode="a", header=False, index=False
        )

        return total_value

    def run_backtest(
        self, start_date, end_date, price_data=None, signal_generator=None
    ):
        """
        Run a backtest over a date range for multiple symbols

        Args:
            start_date (str): Start date for backtest (YYYY-MM-DD)
            end_date (str): End date for backtest (YYYY-MM-DD)
            price_data (dict): Dict of {symbol: DataFrame} with historical price data (if None, will be downloaded)
            signal_generator (callable): Function that generates trading signals
                                        Should return "BUY", "SELL", or "HOLD"

        Returns:
            tuple: (history_df, performance_metrics)
        """

        self.logger.info(
            f"Starting backtest from {start_date} to {end_date} for {len(self.symbols)} symbols"
        )

        # Create a StockPredictor for reference data
        from predictor import StockPredictor

        # Get reference index data
        self.reference_data = yf.download(
            self.reference_ticker, start=start_date, end=end_date, interval="1d"
        )
        
            
            
        # Initialize data for all symbols
        if price_data is None:
            price_data = {}

            # Download data for each symbol
            for symbol in self.symbols:
                self.logger.info(f"Downloading price data for {symbol}")
                try:
                    # Create symbol-specific predictor to get proper data
                    predictor = StockPredictor(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval="1d",
                    )
                    predictor.load_data()

                    if predictor.data.empty:
                        self.logger.error(f"No data available for {symbol}")
                        continue

                    # Store data for this symbol
                    price_data[symbol] = predictor.data

                    # Add volatility calculation
                    if "Volatility" not in price_data[symbol].columns:
                        price_data[symbol]["Volatility"] = (
                            price_data[symbol]["Close"]
                            .pct_change()
                            .rolling(window=20)
                            .std()
                        )

                    self.logger.info(
                        f"Successfully loaded {len(price_data[symbol])} data points for {symbol}"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to download data for {symbol}: {e}")

        # Check if we have any valid data
        if not price_data:
            self.logger.error("No valid price data available for any symbols")
            return pd.DataFrame(), {"error": "No data available"}

        # Reset portfolio for backtest
        self.portfolio = {
            "cash": self.initial_capital,
            "positions": {},
            "value_history": [],
            "transactions": [],
        }

        # Find the common date range across all symbols
        all_dates = set()
        for symbol, data in price_data.items():
            all_dates.update(data.index)
        common_dates = sorted(list(all_dates))

        self.logger.info(f"Running backtest over {len(common_dates)} trading days")
        min_portfolio_value = self.initial_capital * 0.1

        # Daily loop for backtest
        for i, date in enumerate(common_dates):
            # Skip first few days (need previous data for signals)
            if i <= 5:
                # Just record initial portfolio value
                symbol_prices = {}
                for symbol, data in price_data.items():
                    if date in data.index:
                        symbol_prices[symbol] = float(data["Close"].loc[date])

                self.update_portfolio_value(symbol_prices, timestamp=date)
                continue

            # Check if portfolio value is below minimum threshold
            if (
                self.portfolio["value_history"]
                and self.portfolio["value_history"][-1]["value"] <= min_portfolio_value
            ):
                self.logger.warning(
                    f"Portfolio value fell below minimum threshold ({min_portfolio_value:.2f}). "
                    f"Stopping backtest at {date}"
                )
                break

            # Check if we're completely out of cash and have no positions
            if self.portfolio["cash"] <= 0 and not self.portfolio["positions"]:
                self.logger.warning(
                    f"Portfolio is bankrupt! Stopping backtest at {date}"
                )
                break

            # Process each symbol on this date
            current_prices = {}
            for symbol in self.symbols:
                # Skip symbols with no data for this date
                if symbol not in price_data or date not in price_data[symbol].index:
                    continue

                # Get symbol's historical data up to this date
                symbol_data = price_data[symbol].loc[:date]

                # Calculate recent trend
                if len(symbol_data) > 20:
                    recent_trend = (
                        symbol_data["Close"].iloc[-20:].pct_change().mean() * 100
                    )
                else:
                    recent_trend = 0

                # Get current price for this symbol
                current_price = float(price_data[symbol].loc[date, "Open"])
                current_prices[symbol] = current_price

                # Generate trading signal
                if signal_generator:
                    # Use provided signal generator for this symbol
                    signal = signal_generator(price_data[symbol].loc[:date])
                else:
                    # Simple momentum strategy as default
                    if len(symbol_data) > 1:
                        prev_price = float(
                            symbol_data["Close"].iloc[-2]
                        )  # Previous close

                        if current_price > prev_price * 1.01:  # 1% increase
                            signal = "BUY"
                        elif current_price < prev_price * 0.99:  # 1% decrease
                            signal = "SELL"
                        else:
                            signal = "HOLD"
                    else:
                        signal = "HOLD"

                # Execute trades based on signal
                if signal == "BUY" and self.portfolio["cash"] > 0:
                    # Calculate position size dynamically
                    position_size_factor = self.calculate_dynamic_position_size(
                        price_data=symbol_data,
                        current_price=current_price,
                        recent_trend=recent_trend,
                    )

                    # Adjust for portfolio diversification
                    position_size_factor /= len(self.symbols)

                    position_size = (
                        self.portfolio["cash"] * position_size_factor / current_price
                    )

                    if position_size > 1e-3:  # Minimum tradeable amount
                        if (
                            self.portfolio["cash"] >= self.initial_capital * 0.1
                        ):  # Reduced from 0.15 to 0.1
                            self.execute_buy(
                                symbol, current_price, position_size, timestamp=date
                            )
                            self.logger.info(
                                f"{date} - {symbol}: BUY signal at ${current_price:.2f}, bought {position_size:.4f} units"
                            )

                elif signal == "SELL":
                    # Calculate market conditions for sell aggressiveness
                    symbol_data_recent = (
                        symbol_data.iloc[-50:]
                        if len(symbol_data) >= 50
                        else symbol_data
                    )

                    # Calculate trend strength for this symbol
                    trend_strength = 0
                    if len(symbol_data_recent) >= 21:
                        ma10 = symbol_data_recent["Close"].iloc[-10:].mean()
                        ma21 = symbol_data_recent["Close"].iloc[-21:].mean()
                        ma50 = (
                            symbol_data_recent["Close"].iloc[-50:].mean()
                            if len(symbol_data_recent) >= 50
                            else ma21
                        )

                        short_trend = 1 if ma10 > ma21 else -1
                        medium_trend = 1 if ma21 > ma50 else -1
                        long_trend = 1 if current_price > ma50 else -1
                        trend_strength = short_trend + medium_trend + long_trend

                    if symbol in self.portfolio["positions"]:
                        position_qty = self.portfolio["positions"][symbol]["qty"]

                        # ENHANCED SELL LOGIC - More aggressive in bear markets
                        position_size_factor = self.calculate_dynamic_position_size(
                            price_data=symbol_data,
                            current_price=current_price,
                            recent_trend=recent_trend,
                            is_sell=True,
                        )

                        position_size = position_qty * position_size_factor

                        # In strong bear markets, allow overselling (exit long + create short)
                        if trend_strength <= -2 and position_size_factor > 1.0:
                            # This will trigger partial short logic in execute_sell
                            self.logger.info(
                                f"{date} - {symbol}: AGGRESSIVE BEAR SELL - exiting {position_qty:.4f} and creating short"
                            )

                        min_quantity = 1e-2
                        if position_size < min_quantity:
                            position_size = min_quantity

                        self.execute_sell(
                            symbol, current_price, position_size, timestamp=date
                        )
                        self.logger.info(
                            f"{date} - {symbol}: SELL signal at ${current_price:.2f}, sold {position_size:.4f} units (trend: {trend_strength})"
                        )
                    else:
                        # Create a short position - more aggressive approach
                        position_size_factor = self.calculate_dynamic_position_size(
                            price_data=symbol_data,
                            current_price=current_price,
                            recent_trend=recent_trend,
                            is_sell=True,
                        )

                        # Enhanced shorting logic - don't require strict downtrend
                        if (
                            trend_strength <= -1 or recent_trend < -0.3
                        ):  # Relaxed conditions
                            position_size = (
                                self.portfolio["cash"]
                                * position_size_factor
                                / current_price
                                / len(self.symbols)
                            )

                            if position_size > 1e-3:
                                self.execute_sell(
                                    symbol, current_price, position_size, timestamp=date
                                )
                                self.logger.info(
                                    f"{date} - {symbol}: Created short position of {position_size:.4f} units (trend: {trend_strength})"
                                )

            # Update portfolio value using current prices
            self.update_portfolio_value(
                {
                    symbol: float(data.loc[date, "Close"])
                    for symbol, data in price_data.items()
                    if date in data.index
                },
                timestamp=date,
            )

            # Run risk management for all positions
            self._run_risk_management(price_data, date)

        # Generate report
        self.logger.info("Backtest completed, generating report...")
        return self.generate_report()

    def _run_risk_management(self, price_data, current_date):
        """Run adaptive risk management checks across the portfolio based on market regime"""
        for symbol, position in list(self.portfolio["positions"].items()):
            # Skip if we don't have data for this symbol/date
            if symbol not in price_data or current_date not in price_data[symbol].index:
                continue

            symbol_data = price_data[symbol]
            entry_price = position["entry_price"]
            current_price = float(symbol_data.loc[current_date, "Open"])
            position_qty = position["qty"]

            # Calculate trend strength for market regime detection
            trend_strength = 0
            if len(symbol_data) >= 50:
                # Short-term trend (10-day vs 21-day MA)
                if len(symbol_data) >= 21:
                    ma10 = symbol_data["Close"].iloc[-10:].mean()
                    ma21 = symbol_data["Close"].iloc[-21:].mean()
                    short_trend = 1 if ma10 > ma21 else -1
                else:
                    short_trend = 0

                # Medium-term trend (21-day vs 50-day MA)
                if len(symbol_data) >= 50:
                    ma50 = symbol_data["Close"].iloc[-50:].mean()
                    medium_trend = 1 if ma21 > ma50 else -1
                else:
                    medium_trend = 0

                # Long-term trend (current price vs 50-day MA)
                long_trend = 1 if current_price > ma50 else -1

                trend_strength = short_trend + medium_trend + long_trend

            # Calculate momentum for additional context
            momentum_score = 0
            if len(symbol_data) >= 30:
                returns_1m = (
                    (symbol_data["Close"].iloc[-1] / symbol_data["Close"].iloc[-21] - 1)
                    if len(symbol_data) >= 21
                    else 0
                )
                returns_3m = (
                    (symbol_data["Close"].iloc[-1] / symbol_data["Close"].iloc[-63] - 1)
                    if len(symbol_data) >= 63
                    else 0
                )
                momentum_score = (returns_1m + returns_3m) / 2

            if position_qty > 0:  # Long position
                profit_pct = (current_price - entry_price) / entry_price

                # TREND-ADAPTIVE PROFIT TAKING
                if trend_strength >= 2:  # Strong bull market
                    # Much higher profit targets in bull markets
                    if profit_pct > 1.0:  # 100% profit (instead of 40%)
                        sell_qty = position_qty * 0.15  # Smaller sale (15% vs 20%)
                        if sell_qty > 1e-3:
                            self.execute_sell(
                                symbol, current_price, sell_qty, timestamp=current_date
                            )
                            self.logger.info(
                                f"{current_date} - {symbol}: Bull market 100% take-profit at ${current_price:.2f}"
                            )
                    elif profit_pct > 0.7:  # 70% profit
                        sell_qty = position_qty * 0.1  # Small partial sale
                        if sell_qty > 1e-3:
                            self.execute_sell(
                                symbol, current_price, sell_qty, timestamp=current_date
                            )
                            self.logger.info(
                                f"{current_date} - {symbol}: Bull market 70% take-profit at ${current_price:.2f}"
                            )
                elif trend_strength <= -1:  # Weak/bear market
                    # Keep conservative targets in bear markets
                    if profit_pct > 0.4:  # 40% profit
                        sell_qty = position_qty * 0.2
                        if sell_qty > 1e-3:
                            self.execute_sell(
                                symbol, current_price, sell_qty, timestamp=current_date
                            )
                            self.logger.info(
                                f"{current_date} - {symbol}: Bear market 40% take-profit at ${current_price:.2f}"
                            )
                    elif profit_pct > 0.25:  # 25% profit
                        sell_qty = position_qty * 0.15
                        if sell_qty > 1e-3:
                            self.execute_sell(
                                symbol, current_price, sell_qty, timestamp=current_date
                            )
                            self.logger.info(
                                f"{current_date} - {symbol}: Bear market 25% take-profit at ${current_price:.2f}"
                            )
                else:  # Neutral market
                    # Standard profit taking
                    if profit_pct > 0.6:  # 60% profit
                        sell_qty = position_qty * 0.18
                        if sell_qty > 1e-3:
                            self.execute_sell(
                                symbol, current_price, sell_qty, timestamp=current_date
                            )
                            self.logger.info(
                                f"{current_date} - {symbol}: Neutral market 60% take-profit at ${current_price:.2f}"
                            )

                # TREND-ADAPTIVE TRAILING STOPS
                if profit_pct > 0.25:
                    recent_prices = symbol_data.loc[:current_date]
                    if len(recent_prices) >= 10:
                        recent_high = recent_prices["Close"].iloc[-10:].max()

                        if trend_strength >= 2:  # Bull market - wider stops
                            trailing_stop_pct = 0.15  # 15% instead of 7%
                            if current_price < recent_high * (1 - trailing_stop_pct):
                                sell_qty = (
                                    position_qty * 0.25
                                )  # Smaller sale in bull markets
                                if sell_qty > 1e-3:
                                    self.execute_sell(
                                        symbol,
                                        current_price,
                                        sell_qty,
                                        timestamp=current_date,
                                    )
                                    self.logger.info(
                                        f"{current_date} - {symbol}: Bull market trailing stop (15%) at ${current_price:.2f}"
                                    )
                        else:  # Bear/neutral market - tighter stops
                            trailing_stop_pct = 0.08  # 8%
                            if current_price < recent_high * (1 - trailing_stop_pct):
                                sell_qty = (
                                    position_qty * 0.5
                                )  # Larger sale in weak markets
                                if sell_qty > 1e-3:
                                    self.execute_sell(
                                        symbol,
                                        current_price,
                                        sell_qty,
                                        timestamp=current_date,
                                    )
                                    self.logger.info(
                                        f"{current_date} - {symbol}: Standard trailing stop (8%) at ${current_price:.2f}"
                                    )

                # MOMENTUM-BASED POSITION SCALING
                if (
                    momentum_score > 0.15 and trend_strength >= 1
                ):  # Strong positive momentum in uptrend
                    # Consider adding to position (pyramid up)
                    if profit_pct > 0.1 and profit_pct < 0.3:  # Sweet spot for adding
                        # Small addition to winning position
                        additional_size = min(
                            position_qty * 0.2,
                            self.portfolio["cash"] * 0.03 / current_price,
                        )
                        if (
                            additional_size > 1e-3
                            and self.portfolio["cash"] > additional_size * current_price
                        ):
                            self.execute_buy(
                                symbol,
                                current_price,
                                additional_size,
                                timestamp=current_date,
                            )
                            self.logger.info(
                                f"{current_date} - {symbol}: Pyramiding up - added {additional_size:.4f} shares at ${current_price:.2f}"
                            )

                # Hard stop loss - also trend adaptive
                if trend_strength >= 1:  # Uptrend - give more room
                    stop_loss_threshold = -0.25  # 25% loss tolerance in uptrends
                else:  # Downtrend/sideways - tighter stops
                    stop_loss_threshold = -0.15  # 15% loss tolerance

                if profit_pct < stop_loss_threshold:
                    sell_qty = position_qty * 0.75
                    if sell_qty > 1e-3:
                        self.execute_sell(
                            symbol, current_price, sell_qty, timestamp=current_date
                        )
                        self.logger.info(
                            f"{current_date} - {symbol}: Adaptive stop-loss triggered at ${current_price:.2f} (trend: {trend_strength})"
                        )

            elif position_qty < 0:  # Short position
                short_profit_pct = (entry_price - current_price) / entry_price

                # Trend-adaptive short management
                if trend_strength <= -2:  # Strong bear market - let shorts run longer
                    if short_profit_pct > 0.4:  # 40% profit on shorts
                        cover_qty = (
                            abs(position_qty) * 0.3
                        )  # Smaller cover in bear markets
                        if cover_qty > 1e-3:
                            self.execute_buy(
                                symbol, current_price, cover_qty, timestamp=current_date
                            )
                            self.logger.info(
                                f"{current_date} - {symbol}: Bear market short take-profit at ${current_price:.2f}"
                            )
                else:  # Bull/neutral market - cover shorts more quickly
                    if short_profit_pct > 0.2:  # 20% profit
                        cover_qty = abs(position_qty) * 0.4
                        if cover_qty > 1e-3:
                            self.execute_buy(
                                symbol, current_price, cover_qty, timestamp=current_date
                            )
                            self.logger.info(
                                f"{current_date} - {symbol}: Quick short cover in bull market at ${current_price:.2f}"
                            )

                # Stop loss for shorts - also trend adaptive
                if trend_strength >= 1:  # Uptrend - cover shorts quickly
                    short_stop_threshold = -0.10  # 10% loss tolerance
                else:  # Downtrend - give shorts more room
                    short_stop_threshold = -0.20  # 20% loss tolerance

                if short_profit_pct < short_stop_threshold:
                    cover_qty = abs(position_qty) * 0.8  # Cover most of position
                    if cover_qty > 1e-3:
                        self.execute_buy(
                            symbol, current_price, cover_qty, timestamp=current_date
                        )
                        self.logger.info(
                            f"{current_date} - {symbol}: Adaptive short stop-loss at ${current_price:.2f} (trend: {trend_strength})"
                        )

    def calculate_dynamic_position_size(
        self, price_data, current_price, recent_trend=0, is_sell=False
    ):
        """
        Calculate optimal position size based on market conditions and trend strength

        Args:
            price_data (DataFrame): Historical price data for the symbol
            current_price (float): Current price of the symbol
            recent_trend (float): Recent trend percentage (default 0)
            is_sell (bool): Whether this is a sell signal (default False)
        """

        # Base volatility adjustment
        volatility = (
            price_data["Volatility"].iloc[-1]
            if "Volatility" in price_data.columns and len(price_data) > 0
            else 0.02
        )
        volatility = max(0.01, min(0.05, volatility))  # Bound volatility between 1-5%

        # Calculate comprehensive trend strength (same as in risk management)
        trend_strength = 0
        if len(price_data) >= 50:
            try:
                # Short-term trend (10-day vs 21-day MA)
                if len(price_data) >= 21:
                    ma10 = price_data["Close"].iloc[-10:].mean()
                    ma21 = price_data["Close"].iloc[-21:].mean()
                    short_trend = 1 if ma10 > ma21 else -1
                else:
                    short_trend = 0

                # Medium-term trend (21-day vs 50-day MA)
                if len(price_data) >= 50:
                    ma50 = price_data["Close"].iloc[-50:].mean()
                    medium_trend = 1 if ma21 > ma50 else -1
                else:
                    medium_trend = 0

                # Long-term trend (current price vs 50-day MA)
                long_trend = 1 if current_price > ma50 else -1

                trend_strength = short_trend + medium_trend + long_trend
            except:
                trend_strength = 0

        # Calculate momentum for additional context
        momentum_score = 0
        if len(price_data) >= 30:
            try:
                returns_1m = (
                    (price_data["Close"].iloc[-1] / price_data["Close"].iloc[-21] - 1)
                    if len(price_data) >= 21
                    else 0
                )
                returns_3m = (
                    (price_data["Close"].iloc[-1] / price_data["Close"].iloc[-63] - 1)
                    if len(price_data) >= 63
                    else 0
                )
                momentum_score = (returns_1m + returns_3m) / 2
            except:
                momentum_score = 0

        # Detect bottoming pattern
        bottoming_pattern = False
        if len(price_data) > 20:
            try:
                # Look for higher lows after a downtrend
                recent_lows = [
                    min(price_data["Low"].iloc[j - 5 : j])
                    for j in range(len(price_data) - 15, len(price_data))
                ]
                if (
                    len(recent_lows) >= 3
                    and recent_lows[-3] < recent_lows[-2] < recent_lows[-1]
                ):
                    # Check for volume expansion if available
                    if "Volume" in price_data.columns:
                        if (
                            price_data["Volume"].iloc[-1]
                            > price_data["Volume"].rolling(20).mean().iloc[-1] * 1.2
                        ):
                            bottoming_pattern = True
                            self.logger.info(f"Bottoming pattern detected")
            except:
                pass

        # BULL MARKET POSITION SIZING - Base size adaptation
        if trend_strength >= 2:  # Strong bull market
            base_position_factor = 0.12  # Larger base size (6% vs 3%)
        elif trend_strength >= 1:  # Moderate bull market
            base_position_factor = 0.09  # Medium size (4.5%)
        elif trend_strength <= -2:  # Strong bear market
            base_position_factor = 0.04  # Smaller base size (2%)
        else:  # Neutral/weak markets
            base_position_factor = 0.07  # Standard size (3%)

        # Base position size factor (inversely related to volatility)
        position_size_factor = min(
            0.08, max(0.02, base_position_factor / (volatility * 6))
        )

        # TREND-SPECIFIC ADJUSTMENTS
        if not is_sell:  # Buying adjustments
            if (
                trend_strength >= 2 and momentum_score > 0.1
            ):  # Strong bull with momentum
                position_size_factor *= (
                    2.2  # 80% larger positions in strong bull markets
                )
                self.logger.info(
                    f"Bull market position boost: trend={trend_strength}, momentum={momentum_score:.3f}"
                )
            elif trend_strength >= 1 and momentum_score > 0.05:  # Moderate bull
                position_size_factor *= 1.7  # 40% larger
                self.logger.info(
                    f"Moderate bull market position boost: trend={trend_strength}"
                )
            elif trend_strength <= -1:  # Bear market - reduce size
                position_size_factor *= 0.8  # 30% smaller
                self.logger.info(
                    f"Bear market position reduction: trend={trend_strength}"
                )
        else:  # Selling adjustments
            if trend_strength <= -2:  # Strong bear - sell more aggressively
                position_size_factor *= 3.5
                self.logger.info(f"Bear market sell boost: trend={trend_strength}")
            elif trend_strength >= 2:  # Strong bull - sell less aggressively
                position_size_factor *= 0.9
                self.logger.info(f"Bull market sell reduction: trend={trend_strength}")

        # Adjust for market conditions
        if bottoming_pattern and not is_sell:
            if trend_strength <= 0:  # Only boost if not already in uptrend
                position_size_factor *= (
                    1.5  # 50% larger positions on bottoming patterns
                )
                self.logger.info(
                    f"Increased position size due to bottoming pattern: {position_size_factor:.4f}"
                )

        # Calculate RSI if available
        if "RSI" in price_data.columns and len(price_data) > 0:
            try:
                # Use RSI to adjust position size
                current_rsi = price_data["RSI"].iloc[-1]

                if not is_sell and current_rsi < 30:  # Oversold
                    # In bull markets, oversold is more bullish
                    rsi_multiplier = 1.8 if trend_strength >= 1 else 1.5
                    position_size_factor *= rsi_multiplier
                    self.logger.info(
                        f"Increased buy size due to oversold RSI: {current_rsi:.1f} (trend: {trend_strength})"
                    )
                elif (
                    not is_sell and current_rsi < 40 and trend_strength >= 2
                ):  # Mild oversold in bull market
                    position_size_factor *= 1.5
                    self.logger.info(f"Bull market RSI dip buy: {current_rsi:.1f}")

                elif is_sell and current_rsi > 70:  # Overbought
                    # In bear markets, overbought is more bearish
                    rsi_multiplier = 1.8 if trend_strength <= -1 else 1.5
                    position_size_factor *= rsi_multiplier
                    self.logger.info(
                        f"Increased sell size due to overbought RSI: {current_rsi:.1f} (trend: {trend_strength})"
                    )
            except:
                pass

        # Trend-based adjustment (legacy - now mostly handled above)
        if recent_trend > 0 and not is_sell:
            trend_boost = min(recent_trend * 0.3, 0.3)  # Reduced since handled above
            position_size_factor *= 1 + trend_boost
        elif recent_trend < 0 and is_sell:
            trend_boost = min(abs(recent_trend) * 0.3, 0.3)
            position_size_factor *= 1 + trend_boost

        # BULL MARKET LEVERAGE - Apply higher leverage in strong uptrends
        if trend_strength >= 2 and not is_sell:
            leverage_multiplier = 2.8  # Higher leverage in bull markets
        elif trend_strength >= 1 and not is_sell:
            leverage_multiplier = 2.4  # Moderate leverage
        elif trend_strength <= -1:
            leverage_multiplier = 1.8  # Lower leverage in bear markets
        else:
            leverage_multiplier = 2.1  # Standard leverage

        position_size_factor *= leverage_multiplier

        # ENHANCED POSITION SIZING - Allow up to 100% of positions/cash
        if is_sell:
            # AGGRESSIVE SELLING in bear markets - can sell entire position + create shorts
            if trend_strength <= -2:
                position_size_factor = np.mean(
                    [1.2, position_size_factor]
                )  # Can sell 120% (exit longs + short)
                self.logger.info(
                    f"Strong bear market - allowing oversell: {position_size_factor:.3f}"
                )
            elif trend_strength <= -1:
                position_size_factor = np.mean(
                    [1.0, position_size_factor]
                )  # Can sell 100% (full exit)
                self.logger.info(
                    f"Bear market - allowing full exit: {position_size_factor:.3f}"
                )
            else:
                position_size_factor = np.mean(
                    [0.6, position_size_factor]
                )  # Cap at 40% for sells
        else:
            # AGGRESSIVE BUYING in bull markets - can use all available cash
            if trend_strength >= 2:
                position_size_factor = min(
                    0.4, position_size_factor
                )  # Up to 40% of cash in strong bull
                self.logger.info(
                    f"Strong bull market - large positions: {position_size_factor:.3f}"
                )
            elif trend_strength >= 1:
                position_size_factor = min(
                    0.3, position_size_factor
                )  # Up to 30% in moderate bull
            else:
                position_size_factor = min(
                    0.2, position_size_factor
                )  # Max 20% in neutral/bear

        # REDUCED diversification penalty - allow more concentrated positions
        num_symbols = len(self.symbols)
        if num_symbols > 1:
            # Less aggressive reduction for multi-stock portfolio
            diversification_factor = max(
                0.7, 1.0 - (num_symbols - 1) * 0.1
            )  # Gentler reduction
            position_size_factor *= diversification_factor

        return position_size_factor

    def generate_report(self):
        """Generate performance report for multi-symbol portfolio"""
        # Convert history to DataFrame
        if not self.portfolio["value_history"]:
            self.logger.warning("No history data to generate report")
            return pd.DataFrame(), {"error": "No history data"}

        history_df = pd.DataFrame(self.portfolio["value_history"])
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        history_df = history_df.set_index("timestamp")

        # Calculate metrics
        metrics = {}

        # Total return
        initial_value = self.initial_capital
        final_value = history_df["value"].iloc[-1]
        total_return = (final_value / initial_value) - 1
        metrics["total_return"] = total_return

        # Get daily returns
        history_df["daily_return"] = history_df["value"].pct_change()

        # Sharpe ratio (annualized, assuming risk-free rate of 0)
        if len(history_df) > 1:
            daily_returns = history_df["daily_return"].dropna()
            if not daily_returns.empty and daily_returns.std() != 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            metrics["sharpe"] = sharpe

        # Max drawdown
        if len(history_df) > 1:
            history_df["peak"] = history_df["value"].cummax()
            history_df["drawdown"] = (history_df["value"] / history_df["peak"]) - 1
            max_drawdown = history_df["drawdown"].min()
            metrics["max_drawdown"] = max_drawdown

        # Improved win rate calculation for multi-symbol portfolio
        # Track buys/sells by symbol
        buys = {}
        sells = {}

        for transaction in self.portfolio["transactions"]:
            tx_type, symbol, price, qty, timestamp = transaction

            if tx_type == "BUY":
                if symbol not in buys:
                    buys[symbol] = []
                buys[symbol].append((price, qty, timestamp))
            else:  # SELL
                if symbol not in sells:
                    sells[symbol] = []
                sells[symbol].append((price, qty, timestamp))

        # Calculate win rate using FIFO across all symbols
        buy_queues = {symbol: [] for symbol in self.symbols}
        winning_trades = 0
        losing_trades = 0

        for transaction in self.portfolio["transactions"]:
            tx_type, symbol, price, qty, timestamp = transaction

            if tx_type == "BUY":
                # Add to buy queue for this symbol
                if symbol not in buy_queues:
                    buy_queues[symbol] = []
                buy_queues[symbol].append((price, qty))

            elif tx_type == "SELL" and symbol in buy_queues and buy_queues[symbol]:
                # Process sell against available buys
                remaining_sell_qty = qty

                while remaining_sell_qty > 0 and buy_queues[symbol]:
                    buy_price, buy_qty = buy_queues[symbol][0]

                    # Determine how much of this buy is being sold
                    match_qty = min(remaining_sell_qty, buy_qty)

                    # Count profit/loss
                    if price > buy_price:
                        winning_trades += 1
                    else:
                        losing_trades += 1

                    # Update quantities
                    remaining_sell_qty -= match_qty

                    if match_qty >= buy_qty:
                        # Consumed entire buy
                        buy_queues[symbol].pop(0)
                    else:
                        # Partially consumed
                        buy_queues[symbol][0] = (buy_price, buy_qty - match_qty)
                        break

        # Calculate win rate
        total_closed_trades = winning_trades + losing_trades
        win_rate = (
            winning_trades / total_closed_trades if total_closed_trades > 0 else 0
        )

        metrics["win_rate"] = win_rate
        metrics["total_closed_trades"] = total_closed_trades

        # Count total trades
        num_buys = sum(len(b) for b in buys.values() if isinstance(b, list))
        num_sells = sum(len(s) for s in sells.values() if isinstance(s, list))
        metrics["num_trades"] = num_buys + num_sells
        metrics["num_buy_trades"] = num_buys
        metrics["num_sell_trades"] = num_sells

        # Calculate symbols with positions and their contribution
        final_positions = {}
        for symbol in self.symbols:
            if symbol in self.portfolio["positions"]:
                pos = self.portfolio["positions"][symbol]
                if symbol in self.current_prices:
                    value = pos["qty"] * self.current_prices[symbol]
                    final_positions[symbol] = {
                        "qty": pos["qty"],
                        "value": value,
                        "weight": value / final_value if final_value > 0 else 0,
                    }

        metrics["final_positions"] = final_positions
        metrics["cash_weight"] = (
            self.portfolio["cash"] / final_value if final_value > 0 else 0
        )

        # Calculate CAGR
        if len(history_df) > 0:
            days = (history_df.index[-1] - history_df.index[0]).days
            years = days / 365.0 if days > 0 else 1.0
            cagr = (final_value / self.initial_capital) ** (1 / years) - 1
            metrics["cagr"] = cagr

            # Sortino ratio
            downside_returns = history_df["daily_return"].copy()
            downside_returns[downside_returns > 0] = 0
            sortino = (
                history_df["daily_return"].mean()
                / downside_returns.std()
                * np.sqrt(252)
                if not downside_returns.empty and downside_returns.std() > 0
                else 0
            )
            metrics["sortino"] = sortino

            # FIXED: Proper annualized return calculation
            years = days / 365.25
            if years > 0:
                metrics["yearly return"] = (final_value / self.initial_capital) ** (
                    1 / years
                ) - 1
            else:
                metrics["yearly return"] = 0

        days = (history_df.index[-1] - history_df.index[0]).days
        # Log report summary
        self.logger.info(
            f"Backtest Report: Total Return: {total_return:.2%}, Sharpe: {metrics.get('sharpe', 0):.2f}, "
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}, Win Rate: {metrics.get('win_rate', 0):.2%}, "
            f"Number of Trades: {metrics.get('num_trades', 0)}, CAGR: {metrics.get('cagr', 0):.2%}, "
            f"Sortino: {metrics.get('sortino', 0):.2f}, Yearly Return: {metrics["yearly return"]:.4%}"
        )

        return history_df, metrics

    def plot_results(self, history_df=None):
        """Plot backtest results with multiple symbols"""
        if history_df is None:
            history_df = pd.DataFrame(self.portfolio["value_history"])
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
            history_df = history_df.set_index("timestamp")

        if history_df.empty:
            self.logger.warning("No data to plot")
            return

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 15), gridspec_kw={"height_ratios": [2, 1, 1.5]}
        )

        # Plot portfolio value
        ax1.plot(
            history_df.index,
            history_df["value"] / history_df["value"].iloc[0],
            label="Portfolio Value",
            color="blue",
            linewidth=2,
        )
        self.logger.info(f"Plotted PnL for {len(history_df)} data points")

        # Plot reference portfolio if available
        if self.reference_data is not None and not self.reference_data.empty:
            # Make sure reference data aligns with our history dates
            aligned_ref = pd.DataFrame(index=history_df.index)
            try:
                aligned_ref = aligned_ref.join(self.reference_data["Close"], how="left")
            except Exception as e:
                self.logger.error(f"Error aligning reference data: {e}")
                aligned_ref = aligned_ref.join(self.reference_data["Adj Close"], how="left")
            aligned_ref = aligned_ref.fillna(method="ffill")

            if not aligned_ref.empty and not aligned_ref.iloc[:, 0].isna().all():
                normalized_ref = aligned_ref / aligned_ref.iloc[0]
                ax1.plot(
                    history_df.index,
                    normalized_ref,
                    label=f"{self.reference_ticker}",
                    color="orange",
                    alpha=0.7,
                    linewidth=1.5,
                )
                self.logger.info(
                    f"Plotted benchmark for {len(normalized_ref)} data points"
                )

        # Mark buy and sell points on the portfolio value chart
        for transaction in self.portfolio["transactions"]:
            tx_type, symbol, price, qty, timestamp = transaction
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            # Find corresponding portfolio value at this timestamp
            if timestamp in history_df.index:
                portfolio_value = (
                    history_df.loc[timestamp, "value"] / history_df["value"].iloc[0]
                )

                if tx_type == "BUY":
                    marker_color = "green"
                    marker_style = "^"
                    marker_label = f"Buy {symbol}"
                else:  # SELL
                    marker_color = "red"
                    marker_style = "v"
                    marker_label = f"Sell {symbol}"

                ax1.scatter(
                    timestamp,
                    portfolio_value,
                    color=marker_color,
                    marker=marker_style,
                    s=80,
                    alpha=0.7,
                )

        # Add legend for portfolio value chart
        buy_marker = Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="Buy",
        )
        sell_marker = Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Sell",
        )
        portfolio_line = Line2D([0], [0], color="blue", linewidth=2, label="Portfolio")

        legend_elements = [portfolio_line, buy_marker, sell_marker]
        if self.reference_data is not None and not self.reference_data.empty:
            reference_line = Line2D(
                [0], [0], color="orange", linewidth=1.5, label=self.reference_ticker
            )
            legend_elements.append(reference_line)

        ax1.legend(handles=legend_elements, loc="upper left")

        # Plot second chart: cash vs. positions allocation
        if len(history_df) > 0:
            # Extract cash values
            x = history_df.index
            cash_values = history_df["cash"]

            # Extract position values for each symbol
            position_values = {}
            for symbol in self.symbols:
                position_values[symbol] = []

            # Go through each history entry and extract position values by symbol
            for entry in self.portfolio["value_history"]:
                if "positions" in entry:
                    positions = entry["positions"]
                    for symbol in self.symbols:
                        position_values[symbol].append(positions.get(symbol, 0))
                else:
                    # If no positions in this entry, add zeros
                    for symbol in self.symbols:
                        position_values[symbol].append(0)

            # Filter to symbols that had non-zero positions
            active_symbols = []
            for symbol in self.symbols:
                if any(val != 0 for val in position_values[symbol]):
                    active_symbols.append(symbol)

            # Create stacked area chart for positions
            if active_symbols:
                ax2.stackplot(
                    x,
                    cash_values,
                    *[position_values[s] for s in active_symbols],
                    labels=["Cash"] + active_symbols,
                    alpha=0.7,
                )
                ax2.legend(loc="upper left")
            else:
                ax2.plot(x, cash_values, label="Cash")
                ax2.legend(loc="upper left")

        # Third chart: Individual symbol prices
        symbol_price_columns = [
            col for col in history_df.columns if col.endswith("_price")
        ]
        stock_num = len(symbol_price_columns)

        # Store normalized prices for average calculation
        normalized_prices = {}

        for col in symbol_price_columns:
            symbol = col.replace("_price", "")
            prices = history_df[col].dropna()

            if len(prices) > 0:
                # Normalize to starting value
                normalized = prices / prices.iloc[0]
                normalized_prices[symbol] = normalized
                ax3.plot(prices.index, normalized, label=symbol, alpha=0.8)

        # Calculate and plot average only if we have price data
        if normalized_prices and stock_num > 0:
            # Find common index across all normalized prices
            common_index = None
            for symbol, norm_prices in normalized_prices.items():
                if common_index is None:
                    common_index = norm_prices.index
                else:
                    common_index = common_index.intersection(
                        norm_prices.index
                    )  # overlapped index

            if len(common_index) > 0:
                # Calculate average on common dates
                stock_sum = pd.Series(0.0, index=common_index)
                valid_count = pd.Series(0, index=common_index)

                for symbol, norm_prices in normalized_prices.items():
                    # Add values where we have data
                    aligned_prices = norm_prices.reindex(
                        common_index, fill_value=np.nan
                    )
                    mask = ~aligned_prices.isna()
                    stock_sum[mask] += aligned_prices[mask]
                    valid_count[mask] += 1

                # Calculate average (avoid division by zero)
                average_prices = stock_sum / valid_count.replace(0, np.nan)
                average_prices = average_prices.dropna()

                if len(average_prices) > 0:
                    ax3.plot(
                        average_prices.index,
                        average_prices,
                        label="Average",
                        color="black",
                        linewidth=2.5,
                        alpha=0.9,
                    )
                    self.logger.info(
                        f"Plotted average for {len(average_prices)} data points"
                    )
                else:
                    self.logger.warning("No valid average data to plot")
            else:
                self.logger.warning(
                    "No common dates found across symbols for average calculation"
                )

        if symbol_price_columns:
            ax3.legend(loc="upper left")

        # Configure plots
        ax1.set_title("Portfolio Performance vs Benchmark")
        ax1.set_ylabel("Normalized Value")
        ax1.grid(True, linestyle="--", alpha=0.7)

        ax2.set_title("Portfolio Composition")
        ax2.set_ylabel("Value ($)")
        ax2.grid(True, linestyle="--", alpha=0.7)

        ax3.set_title("Individual Symbol Performance (Normalized)")
        ax3.set_ylabel("Normalized Price")
        ax3.set_xlabel("Date")
        ax3.grid(True, linestyle="--", alpha=0.7)

        # Improve date formatting
        import matplotlib.dates as mdates

        date_format = mdates.DateFormatter("%Y-%m-%d")

        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

        return fig

    def whether_beat_market(self) -> bool:
        if (
            pd.to_datetime(self.portfolio["value_history"][-1]["timestamp"])
            == self.reference_data.index[-1]
        ):
            normalized_ref = (
                self.reference_data["Close"].iloc[-1].values[0]
                / self.reference_data["Close"].iloc[0].values[0]
                - 1
            )
            normalized_portfolio = (
                self.portfolio["value_history"][-1]["value"]
                / self.portfolio["value_history"][0]["value"]
            ) - 1
            self.logger.info("index matched")
            self.logger.info(
                f"Return of portfolio: {normalized_portfolio:.2%} vs\nreference data: {normalized_ref:.2%}"
            )
            return normalized_portfolio > normalized_ref

        else:
            self.logger.info(
                f"index not same: {pd.to_datetime(self.portfolio['value_history'][-1]['timestamp'])} and {self.reference_data.index[-1]}"
            )
            return False

    def whether_beat_risk_free(self, metrics) -> bool:
        """
        Compare portfolio's annualized return to risk-free rate
        Returns True if portfolio beats risk-free rate
        """
        try:
            # Download 13-week Treasury bill rate (^IRX) - already annualized
            risk_free_data = yf.download(
                "^IRX",
                start=self.reference_data.index[0],
                end=self.reference_data.index[-1],
                interval="1d",
            )

            if risk_free_data.empty:
                self.logger.warning(
                    "Could not download risk-free rate data, using default 2%"
                )
                risk_free_rate = 0.02
            else:
                # IRX is already in percentage, convert to decimal
                risk_free_rate = risk_free_data["Close"].mean().values[0] / 100

            portfolio_annual_return = metrics.get("yearly return", 0)

            self.logger.info(
                f"Risk-free rate: {risk_free_rate:.4%} vs Portfolio annual return: {portfolio_annual_return:.4%}"
            )

            return portfolio_annual_return > risk_free_rate

        except Exception as e:
            self.logger.error(f"Error in risk-free comparison: {e}")
            # Fallback to simple comparison with 2% risk-free rate
            return metrics.get("yearly return", 0) > 0.02



def select_stocks(
    num_stocks=5,
    universe="NASDAQ 100",
    lookback_days=90,
    strategy="momentum",
    end_date=None,
    interval="1d",
    override_list = None
):
    """
    Select promising stocks from a universe based on fundamental and technical metrics

    Args:
        num_stocks (int): Number of stocks to select
        universe (str or list): Stock universe to select from ("S&P 100", "NASDAQ 100", etc.)
        lookback_days (int): Days to look back for performance metrics
        strategy (str): Selection strategy - "momentum", "value", "quality", or "blend"
        end_date (str or datetime): End date for historical data (default: today)
        interval (str): Data interval for historical data (default: "1d")
        override_list (list or str): List of stock symbols to override universe selection

    Returns:
        list: List of selected stock symbols
    """

    try:
        from pytickersymbols import PyTickerSymbols
        import yfinance as yf
        from datetime import datetime, timedelta
        import numpy as np
        import pandas as pd

        # Get stocks from the specified index
        ticker_data = PyTickerSymbols()
        symbols = []
        if override_list is not None:
            if isinstance(override_list, str):
                override_list = [override_list]
            symbols = override_list
        if universe == "S&P 100":
            stocks = ticker_data.get_stocks_by_index("S&P 100")
        elif universe == "NASDAQ 100":
            stocks = ticker_data.get_stocks_by_index("NASDAQ 100")
        elif universe == "DOW JONES":
            stocks = ticker_data.get_stocks_by_index("DOW JONES")
        elif universe == "S&P 500":
            stocks = ticker_data.get_stocks_by_index("S&P 500")
        elif not isinstance(universe, list):  # not in a list of tickers
            stocks = ticker_data.get_stocks_by_index("S&P 100")  # Default
        
        # Extract Yahoo Finance symbols
        if len(symbols) == 0:
            for stock in stocks:
                for symbol in stock["symbols"]:
                    if symbol["currency"] == "USD":
                        symbols.append(symbol["yahoo"])

        if universe == "crypto":
            from autotrade_aptos import get_alpaca_tradable_cryptos

            alpaca_cryptos, symbols = get_alpaca_tradable_cryptos()

        # Remove duplicates and sort
        symbols = sorted(list(set(symbols)))

        # Define time periods for analysis
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)
        start_date = end_date - timedelta(days=lookback_days)

        print(f"Analyzing {len(symbols)} stocks from {universe}...")

        # Download historical data for all symbols
        # Use a batch approach to avoid timeouts
        stock_data = {}
        batch_size = 25

        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i : i + batch_size]
            try:
                batch_data = yf.download(
                    batch_symbols,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    interval=interval,
                )

                # Handle single stock case where yfinance doesn't return MultiIndex
                if isinstance(batch_data.columns, pd.MultiIndex):
                    for symbol in batch_symbols:
                        if ("Close", symbol) in batch_data.columns:
                            # Extract data for this symbol
                            symbol_data = batch_data.xs(symbol, axis=1, level=1)
                            if (
                                not symbol_data.empty
                                and len(symbol_data) > lookback_days // 2
                            ):
                                stock_data[symbol] = symbol_data
                else:
                    # Single stock case
                    if len(batch_data) > lookback_days // 2 and not batch_data.empty:
                        symbol = batch_symbols[0]
                        stock_data[symbol] = batch_data
            except Exception as e:
                print(f"Error downloading batch {i//batch_size + 1}: {e}")

        # Calculate metrics for each stock
        metrics = []

        for symbol, data in stock_data.items():
            try:
                if len(data) < 20:  # Need at least 20 data points
                    continue

                # Get price data
                closes = data["Close"]
                volumes = data["Volume"] if "Volume" in data else None

                # Convert to numpy arrays for faster calculations
                closes_np = closes.values
                data_length = len(closes_np)

                # 1. Momentum metrics - use direct indexing for speed
                returns_1m = (closes_np[-1] / closes_np[-min(21, data_length)]) - 1
                returns_3m = (closes_np[-1] / closes_np[-min(63, data_length)]) - 1

                # 2. Volatility metrics - use numpy for faster standard deviation
                returns_np = np.diff(closes_np) / closes_np[:-1]  # pct_change in numpy
                volatility = np.nanstd(returns_np) * np.sqrt(252)  # Annualized

                # 3. Technical indicators
                # a. Moving averages using numpy's convolve function
                def np_rolling_mean(values, window):
                    """Calculate rolling mean using NumPy's convolve"""
                    window_size = min(window, len(values))
                    weights = np.ones(window_size) / window_size
                    # Pad the signal for the valid convolution mode
                    return np.convolve(values, weights, "valid")

                # Calculate moving averages
                ma7_np = np_rolling_mean(closes_np, min(7, data_length))
                ma21_np = np_rolling_mean(closes_np, min(21, data_length))

                # Get last values of MAs
                ma7 = ma7_np[-1] if len(ma7_np) > 0 else closes_np[-1]
                ma21 = ma21_np[-1] if len(ma21_np) > 0 else closes_np[-1]

                price_to_ma7 = closes_np[-1] / ma7 - 1  # % diff from 7d MA
                price_to_ma21 = closes_np[-1] / ma21 - 1  # % diff from 21d MA

                # b. RSI (14-day) using NumPy for better performance
                delta_np = np.zeros(data_length)
                delta_np[1:] = closes_np[1:] - closes_np[:-1]

                # Split gains and losses
                gain_np = np.zeros_like(delta_np)
                loss_np = np.zeros_like(delta_np)

                gain_np[delta_np > 0] = delta_np[delta_np > 0]
                loss_np[delta_np < 0] = -delta_np[delta_np < 0]

                # Calculate average gains and losses with NumPy
                window_size = min(14, data_length)

                # Simple moving average for initial values
                avg_gain_np = np.zeros_like(gain_np)
                avg_loss_np = np.zeros_like(loss_np)

                if data_length >= window_size:
                    # Initial averages
                    avg_gain_np[window_size - 1] = np.mean(gain_np[:window_size])
                    avg_loss_np[window_size - 1] = np.mean(loss_np[:window_size])

                    # Calculate subsequent values
                    for i in range(window_size, data_length):
                        avg_gain_np[i] = (
                            avg_gain_np[i - 1] * (window_size - 1) + gain_np[i]
                        ) / window_size
                        avg_loss_np[i] = (
                            avg_loss_np[i - 1] * (window_size - 1) + loss_np[i]
                        ) / window_size

                # Calculate RS and RSI
                rs_np = np.zeros_like(avg_gain_np)
                rsi_np = np.zeros_like(avg_gain_np)

                # Avoid division by zero
                valid_indices = avg_loss_np > 0
                rs_np[valid_indices] = (
                    avg_gain_np[valid_indices] / avg_loss_np[valid_indices]
                )

                # Calculate RSI
                rsi_np = 100 - (100 / (1 + rs_np))

                # Fill NaN values with 50
                rsi_np = np.nan_to_num(rsi_np, nan=50.0)
                current_rsi = rsi_np[-1]

                # c. Volume trends - using NumPy for faster calculations
                if volumes is not None and not volumes.empty:
                    volumes_np = volumes.values
                    if len(volumes_np) >= 5:
                        recent_volume = np.mean(volumes_np[-5:])
                        if len(volumes_np) >= 20:
                            past_volume = np.mean(volumes_np[-20:-5])
                        else:
                            past_volume = np.mean(volumes_np)
                        volume_change = (
                            (recent_volume / past_volume) - 1 if past_volume > 0 else 0
                        )
                    else:
                        volume_change = 0
                else:
                    volume_change = 0

                # d. Rate of change (ROC) - using direct numpy array indexing
                roc_5 = (closes_np[-1] / closes_np[-min(6, data_length)]) - 1
                roc_10 = (closes_np[-1] / closes_np[-min(11, data_length)]) - 1

                # Calculate a composite score based on strategy
                if strategy == "momentum":
                    # Momentum strategy prioritizes recent performance and uptrends
                    score = (
                        0.3 * returns_1m
                        + 0.3 * returns_3m
                        + 0.2 * price_to_ma7
                        + 0.1 * (1 - volatility)  # Lower volatility is better
                        + 0.1
                        * (
                            volume_change if volume_change > 0 else 0
                        )  # Increasing volume is good
                    )

                elif strategy == "value":
                    # Value strategy looks for undervalued stocks (lower relative to MAs, oversold)
                    # Better if using P/E ratios etc., but using technical proxies here
                    score = (
                        0.3
                        * (0.3 - current_rsi / 100)  # Lower RSI is better (oversold)
                        + 0.3 * (-price_to_ma21)  # Lower price vs 200MA
                        + 0.2 * (-price_to_ma7)  # Lower price vs 50MA
                        + 0.2 * (1 - volatility)  # Lower volatility is better
                    )

                elif strategy == "quality":
                    # Quality focuses on stable trends and lower volatility
                    score = (
                        0.3 * (1 - volatility)
                        + 0.2
                        * (50 - abs(current_rsi - 50))
                        / 50  # Closer to RSI 50 is better (stable)
                        + 0.2 * (0.05 - abs(price_to_ma7))  # Near but above 50MA
                        + 0.2 * (0.05 - abs(price_to_ma21))  # Near but above 200MA
                        + 0.1 * ((returns_3m > 0) * returns_3m)  # Positive returns only
                    )

                else:  # "blend" or default
                    # Balanced approach - use NumPy operations for faster calculations
                    momentum_score = 0.5 * returns_1m + 0.5 * returns_3m
                    trend_score = 0.7 * float(price_to_ma7 > 0) + 0.3 * float(
                        price_to_ma21 > 0
                    )
                    volatility_score = 1 - min(1, volatility)
                    rsi_score = 0

                    # Use NumPy conditions for better performance
                    if current_rsi < 30:  # Oversold
                        rsi_score = 0.8  # Good buying opportunity
                    elif current_rsi > 70:  # Overbought
                        rsi_score = 0.2  # Caution
                    else:
                        rsi_score = 0.5  # Neutral

                    score = (
                        0.4 * momentum_score
                        + 0.3 * trend_score
                        + 0.2 * volatility_score
                        + 0.1 * rsi_score
                    )

                # Store metrics
                metrics.append(
                    {
                        "symbol": symbol,
                        "score": score,
                        "returns_1m": returns_1m,
                        "returns_3m": returns_3m,
                        "volatility": volatility,
                        "rsi": current_rsi,
                        "price_to_ma50": price_to_ma7,
                        "price_to_ma200": price_to_ma21,
                        "volume_change": volume_change,
                    }
                )

            except Exception as e:
                print(f"Error calculating metrics for {symbol}: {e}")
                continue

        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)

        if metrics_df.empty:
            print("No valid stocks with metrics found, returning random selection")
            # Fallback to random selection
            import random

            return random.sample(symbols, min(num_stocks, len(symbols)))

        # Select top stocks by score
        selected_symbols = (
            metrics_df.sort_values("score", ascending=False)
            .head(num_stocks)["symbol"]
            .tolist()
        )

        # Print selected stocks with metrics
        print(
            f"\nSelected {len(selected_symbols)} stocks based on {strategy} strategy:"
        )
        summary = metrics_df[metrics_df["symbol"].isin(selected_symbols)].set_index(
            "symbol"
        )

        # Format the summary for better display
        formatted_summary = summary.copy()
        for col in [
            "returns_1m",
            "returns_3m",
            "price_to_ma50",
            "price_to_ma200",
            "volume_change",
        ]:
            if col in formatted_summary.columns:
                formatted_summary[col] = formatted_summary[col].apply(
                    lambda x: f"{x:.2%}"
                )

        formatted_summary["volatility"] = formatted_summary["volatility"].apply(
            lambda x: f"{x:.2%}"
        )
        formatted_summary["rsi"] = formatted_summary["rsi"].apply(lambda x: f"{x:.1f}")
        formatted_summary["score"] = formatted_summary["score"].apply(
            lambda x: f"{x:.3f}"
        )

        print(formatted_summary)
        return list(set(selected_symbols))

    except Exception as e:
        print(f"Error in stock selection: {e}")
        # Fallback to default stocks
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"][:num_stocks]

def create_multi_stock_signal_generator(
    predictor_class,
    always_reverse=False,
    autodetect_reversal=False,
    strat=None,
    indicators_to_drop=None,
):
    """
    Create a signal generator that works with multiple stocks

    Args:
        predictor_class: The StockPredictor class to use for each symbol
        always_reverse: If True, always reverse signals
        autodetect_reversal: If True, dynamically decide when to reverse signals
        strat: Allow use diff strat

    Returns:
        function: A function that generates signals for any stock
    """

    def generate_signal(historical_data):
        """Generate a trading signal based on historical data"""
        symbol = historical_data.name if hasattr(historical_data, "name") else "Unknown"

        # Create temporary predictor for this symbol and data
        predictor = predictor_class(
            symbol=symbol, start_date=None, end_date=None
        )  # not fixed
        predictor.data = historical_data.copy()

        # Calculate features the predictor might need
        if hasattr(predictor, "load_features"):
            try:
                predictor.load_features()
            except:
                pass

        # Determine whether to use reversed signals
        use_reversal = always_reverse

        if not always_reverse and autodetect_reversal:
            # Calculate market direction over past 20 days
            market_trend = historical_data["Close"].pct_change(20).mean()

            # Simple reversal logic - reverse in downtrends
            # if market_trend < 0:
            #     use_reversal = True

            price = historical_data["Close"].iloc[-1]
            ma50 = historical_data["MA_50"].iloc[-1]
            ma200 = historical_data["MA_200"].iloc[-1]

            # Logic: In uptrends, normal signals work better; in downtrends, reversed signals work better
            if market_trend > 0 and price > ma200:  # Solid uptrend
                use_reversal = True  # reverse in strong uptrends
            elif market_trend < 0 and price < ma200:  # Solid downtrend
                use_reversal = False  # Don't Reverse in downtrends
            else:  # Sideways market
                use_reversal = True  # Default to reversal in uncertain conditions

        # Get the signal
        try:
            # Try to use custom entry signal function
            from private_strat import get_entry_signal

            if strat is not None:
                decision, confidence, rationale, levels = strat(
                    predictor,
                    symbol=symbol,
                    current_price=float(historical_data["Close"].iloc[-1]),
                    reverse_signals=use_reversal,
                    indicators_to_drop=indicators_to_drop,
                )

            else:
                decision, confidence, rationale, levels = get_entry_signal(
                    predictor,
                    symbol=symbol,
                    current_price=float(historical_data["Close"].iloc[-1]),
                    reverse_signals=use_reversal,
                    indicators_to_drop=indicators_to_drop,
                )
        except:
            # Fallback to simple strategy
            rsi_oversold = False
            rsi_overbought = False

            if "RSI" in historical_data.columns:
                current_rsi = historical_data["RSI"].iloc[-1]
                rsi_oversold = current_rsi < 30
                rsi_overbought = current_rsi > 70

            # Use simple momentum with RSI
            recent_return = historical_data["Close"].pct_change(5).iloc[-1] * 100

            if rsi_oversold or recent_return < -5:
                decision = "BUY"
            elif rsi_overbought or recent_return > 5:
                decision = "SELL"
            else:
                decision = "HOLD"

            # Apply reversal if needed
            if use_reversal:
                if decision == "BUY":
                    decision = "SELL"
                elif decision == "SELL":
                    decision = "BUY"

        return decision

    return generate_signal


### Off-chain work

# Add to the run_backtest method, replacing your existing position sizing calculation:


def create_signal_generator(predictor, always_reverse=False, autodetect_reversal=False):
    """
    Create a signal generator function that intelligently adapts to market conditions

    Args:
        predictor: An instance of StockPredictor
        always_reverse: If True, always use reversal regardless of autodetection

    Returns:
        function: A function that takes historical data and returns trading signals
    """

    def detect_bottoming_pattern(data):
        # Look for consecutive higher lows after a downtrend
        recent_lows = [min(data["Low"].iloc[i - 5 : i]) for i in range(5, len(data))]
        if len(recent_lows) >= 3:
            # Check for higher lows pattern (bottoming)
            if recent_lows[-3] < recent_lows[-2] < recent_lows[-1]:
                # Confirm with volume expansion
                if (
                    data["Volume"].iloc[-1]
                    > data["Volume"].rolling(20).mean().iloc[-1] * 1.2
                ):
                    return True
        return False

    def generate_signal(historical_data):
        from private_strat import get_entry_signal
        # Update predictor's data with the current slice of historical data
        predictor.data = historical_data.copy()
        volatility = historical_data["Close"].pct_change().rolling(20).std().iloc[-1]
        position_size_factor = min(0.05, max(0.01, 0.03 / (volatility * 10)))

        current_price = (float(historical_data["Close"].iloc[-1]),)
        if (
            detect_bottoming_pattern(historical_data)
            and current_price < historical_data["MA_50"].iloc[-1]
        ):
            # Increase position size at bottoms
            position_size_factor *= 1.5  # Increase allocation at bottoms

        predictor.current_position_size = position_size_factor

        # Generate features that the predictor needs
        if hasattr(predictor, "load_features"):
            predictor.load_features()

        use_reversal = False  # Default
        # Calculate market direction over last 30 days
        market_trend = historical_data["Close"].pct_change(30).mean()
        trend_strength = abs(market_trend)

        # Check if we're in a strong trend
        is_strong_trend = trend_strength > 0.005  # >0.5% daily avg movement

        # Check price relative to moving averages
        has_ma50 = "MA_50" in historical_data.columns
        has_ma200 = "MA_200" in historical_data.columns

        # If always_reverse is True, skip autodetection
        if always_reverse:
            use_reversal = always_reverse

        elif autodetect_reversal:
            # Actually make meaningful reversal decisions based on market conditions

            if has_ma50 and has_ma200:
                price = historical_data["Close"].iloc[-1]
                ma50 = historical_data["MA_50"].iloc[-1]
                ma200 = historical_data["MA_200"].iloc[-1]

                # Logic: In uptrends, normal signals work better; in downtrends, reversed signals work better
                if market_trend > 0 and price > ma200:  # Solid uptrend
                    use_reversal = True  # reverse in strong uptrends
                elif market_trend < 0 and price < ma200:  # Solid downtrend
                    use_reversal = False  # Don't Reverse in downtrends
                elif is_strong_trend:  # Any other strong trend
                    use_reversal = True  # Default to reversal in strong trends
                else:  # Sideways market
                    use_reversal = True  # Default to reversal in uncertain conditions
            else:
                # If we don't have moving averages, use simpler logic
                use_reversal = market_trend < 0  # Reverse in downtrends only

        # Log the decision periodically
        if len(historical_data) % 20 == 0:
            trend_type = "uptrend" if market_trend > 0 else "downtrend"
            strength = "strong" if is_strong_trend else "weak"
            logger.info(
                f"Market analysis: {strength} {trend_type} ({market_trend*100:.2f}% avg daily). Using reversal: {use_reversal}"
            )

        # Get entry signal with the determined reversal setting
        decision, confidence, rationale, levels = get_entry_signal(
            predictor,
            current_price=float(historical_data["Close"].iloc[-1]),
            reverse_signals=use_reversal,
        )
        # Log the decision
        logger.info(f"Whether use reversed decision: {use_reversal}")
        
        return decision

    return generate_signal


def run_live_trading_sim(symbol="APT21794-USD", initial_capital=100):
    """Run a live trading simulation with the Aptos backtester"""
    from datetime import datetime, timedelta

    backtester = AptosBacktester(symbol=symbol, initial_capital=initial_capital)

    # Get current price
    current_price = float(
        yf.download(symbol, period="1d", interval="1m", timeout=10)["Close"].iloc[-1]
    )
    print(f"Current price of {symbol}: ${current_price:.2f}")

    # Simple interactive trading loop
    while True:
        print("\n===== Trading Menu =====")
        print("1. Buy")
        print("2. Sell")
        print("3. Show Portfolio")
        print("4. Update Portfolio Value")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            qty = float(input("Enter quantity to buy: "))
            backtester.execute_buy(current_price, qty)

        elif choice == "2":
            qty = input("Enter quantity to sell (or 'all' for all): ")
            if qty.lower() == "all":
                backtester.execute_sell(current_price)
            else:
                backtester.execute_sell(current_price, float(qty))

        elif choice == "3":
            print("\n===== Portfolio =====")
            print(f"Cash: ${backtester.portfolio['cash']:.2f}")
            for symbol, position in backtester.portfolio["positions"].items():
                print(
                    f"{symbol}: {position['qty']:.6f} @ ${position['entry_price']:.2f}"
                )

            if backtester.portfolio["value_history"]:
                latest = backtester.portfolio["value_history"][-1]
                print(f"Total Value: ${latest['value']:.2f}")

        elif choice == "4":
            value = backtester.update_portfolio_value()
            print(f"Updated Portfolio Value: ${value:.2f}")

        elif choice == "5":
            print("Exiting simulation...")
            break

        else:
            print("Invalid choice. Please enter 1-5.")


async def main():

    # Check entry points for trading
    from predictor import StockPredictor

    from autotrade_aptos import get_alpaca_tradable_cryptos
    alpaca_cryptos, yf_crypto_symbols = get_alpaca_tradable_cryptos()
    symbol = random.choice(yf_crypto_symbols)  # Randomly select a crypto symbol from Alpaca tradable list
    start = "2021-12-01"
    end = "2025-07-17"
    _predictor = StockPredictor(symbol=symbol, start_date=start, end_date=end)
    _predictor.load_data()
    backtester = AptosBacktester(symbols=symbol, initial_capital=100000)
    # Run a simple backtest with default strategy
    print("Running backtest...")

    autodect_reversal = False  # Set to True to enable autodetection of reversal
    history, metrics = backtester.run_backtest(
        start_date=start,
        end_date=end,
        signal_generator=create_signal_generator(
            predictor=_predictor,
            always_reverse=False,
            autodetect_reversal=False,
        ),  # Use the predictor's signal generator, Can only choose one from autodetect_reversal and always_reverse
    )
    logger.info(f"If we autodetect reversal: {autodect_reversal}")

    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Number of BUY orders: {metrics['num_buy_trades']}")
    print(f"Number of SELL orders: {metrics['num_sell_trades']}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Sortino Ratio: {metrics['sortino']:.2f}")

    # Plot results
    backtester.plot_results(history)

    # Initialize backtester
    backtester_on_chain = AptosBacktester(symbols=["BTC-USD"], initial_capital=100000)

    # Run on-chain backtest
    onchain_results = await backtester_on_chain.run_onchain_backtest(
        start_date="2024-01-01",
        end_date="2025-07-25"
    )
    print("\nOn-Chain Backtest Results:")
    logger.info(f"Total Return: {onchain_results['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {onchain_results['sharpe']:.2f}")
    logger.info(f"Max Drawdown: {onchain_results['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {onchain_results['win_rate']:.2%}")
    logger.info(f"Number of Trades: {onchain_results['num_trades']}")
    logger.info(f"Number of BUY orders: {onchain_results['num_buy_trades']}")
    logger.info(f"Number of SELL orders: {onchain_results['num_sell_trades']}")
    logger.info(f"CAGR: {onchain_results['cagr']:.2%}")
    logger.info(f"Sortino Ratio: {onchain_results['sortino']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())