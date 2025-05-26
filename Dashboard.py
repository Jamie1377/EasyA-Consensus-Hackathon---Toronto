import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import yfinance as yf
from matplotlib.figure import Figure
import asyncio
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDateEdit,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QSplitter,
    QMessageBox,
    QProgressBar,
    QCheckBox
    
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt5.QtGui import QFont, QColor

# Import the backtester and Aptos integration components
from aptos_integration_v3_1 import (
    AptosBacktester,
    PortfolioTracker,
    fund_wallet,
    load_or_create_wallet,
    check_balance,
    execute_transfer,
    reconcile_balances,
    check_entry_points,
    RestClient,
    AccountAddress,
)

# Network configuration
NODE_URL = "https://fullnode.devnet.aptoslabs.com/v1"


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

        

# class BacktestWorker(QThread):
#     """Worker thread to run backtests without blocking the UI"""

#     finished = pyqtSignal(object, object)

#     def __init__(self, symbol, start_date, end_date, initial_capital):
#         super().__init__()
#         self.symbol = symbol
#         self.start_date = start_date
#         self.end_date = end_date
#         self.initial_capital = initial_capital

#     def run(self):
#         backtester = AptosBacktester(
#             symbol=self.symbol, initial_capital=self.initial_capital
#         )
#         history, metrics = backtester.run_backtest(
#             start_date=self.start_date, end_date=self.end_date
#         )
#         self.finished.emit(history, metrics)


# Add these imports at the top of your file
import time
import random
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from yfinance.exceptions import YFRateLimitError

# Add this helper function for data fetching with retries
def fetch_yf_data_with_retries(symbol, start_date, end_date, interval="1d", max_retries=5, initial_delay=10):
    """
    Fetch Yahoo Finance data with exponential backoff retry logic to handle rate limiting
    """
    retries = 0
    delay = initial_delay
    
    while retries <= max_retries:
        try:
            # Attempt to download the data
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False, # Disable progress bar
                timeout=10  # Set timeout to 10 seconds
            )
            
            # Check if data is empty
            if data.empty:
                print(f"No data available for {symbol} in the specified date range")
                time.sleep(30)  #delay before trying alternatives
                
                # Try slightly modifying the date range to get some data
                if retries >= 2:
                    # Try extending the date range
                    from datetime import datetime, timedelta
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    start_dt = start_dt - timedelta(days=30)
                    new_start = start_dt.strftime("%Y-%m-%d")
                    print(f"Trying extended date range: {new_start} to {end_date}")
                    data = yf.download(
                        symbol, 
                        start=new_start,
                        end=end_date,
                        interval=interval,
                        progress=False,
                        timeout=20
                    )
            
            if not data.empty:
                if len(data) > 0:
                    return data
            
            print(f"Retrieved empty dataset for {symbol}, retrying ({retries+1}/{max_retries})")
            retries += 1
            
        except YFRateLimitError as e:
            print(f"Rate limit hit: {e}. Retrying in {delay} seconds ({retries+1}/{max_retries})")
            time.sleep(delay)
            delay *= 2
            delay += random.uniform(0, 1)  # Add jitter to avoid synchronized retries
            retries += 1
            
        except (ConnectionError, Timeout, TooManyRedirects) as e:
            print(f"Connection error: {e}. Retrying in {delay} seconds ({retries+1}/{max_retries})")
            time.sleep(delay)
            delay *= 1.5  # Smaller backoff for connection issues
            retries += 1
            
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in {delay} seconds ({retries+1}/{max_retries})")
            time.sleep(delay)
            delay *= 1.5
            retries += 1
    
    # If we exhausted all retries, try one last approach with a minimal dataset
    try:
        print("Trying to fetch minimal recent data as fallback")
        minimal_data = yf.download(symbol, period="1mo", interval=interval, progress=False, timeout=15)
        if not minimal_data.empty and len(minimal_data) > 0:
            return minimal_data
    except Exception as e:
        print(f"Failed to fetch even minimal data: {e}")
    
    # If all attempts fail, raise an exception
    raise ValueError(f"Could not fetch data for {symbol} after multiple attempts")



class BacktestWorker(QThread):
    """Worker thread to run backtests without blocking the UI"""

    finished = pyqtSignal(object, object)

    def __init__(self, symbol, start_date, end_date, initial_capital, reverse_signals=False, auto_detect=False):
        super().__init__()
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.backtester = None  # Add this line to initialize the attribute
        self.reverse_signals = reverse_signals
        self.auto_detect = auto_detect

    # def run(self):
    #     self.backtester = AptosBacktester(  # Save the backtester instance
    #         symbol=self.symbol, initial_capital=self.initial_capital
    #     )
    #     history, metrics = self.backtester.run_backtest(
    #         start_date=self.start_date, end_date=self.end_date
    #     )
    #     self.finished.emit(history, metrics)

        
    def run(self):
        try:
            self.backtester = AptosBacktester(
                symbol=self.symbol, initial_capital=self.initial_capital
            )

            from predictor import StockPredictor
            from aptos_integration_v3_1 import create_signal_generator

            _predictor = StockPredictor(symbol=self.symbol, start_date=self.start_date,
            end_date=self.end_date)
            _predictor.load_data()

            
            # Use the retry function for data loading
            # self.backtester.use_retry_fetch = False
            # self.backtester.fetch_data_function = fetch_yf_data_with_retries
             
            # Create signal generator with the specified reversal settings
            signal_gen = create_signal_generator(
                predictor=_predictor, 
                always_reverse=self.reverse_signals if not self.auto_detect else None,
                autodetect_reversal=self.auto_detect
            )
            
            history, metrics = self.backtester.run_backtest(
                start_date=self.start_date, 
                end_date=self.end_date,
                signal_generator= signal_gen
            )
            self.finished.emit(history, metrics)

        except Exception as e:
            # Handle exceptions in the thread
            import traceback
            traceback.print_exc()
            self.finished.emit({}, {"error": str(e)})

class AsyncWorker(QThread):
    """Worker thread to run async Aptos operations"""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.func(*self.args, **self.kwargs))
            loop.close()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def __del__(self):
        """Ensure the thread is properly cleaned up"""
        self.wait()


class AptosDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aptos Trading Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize UI components
        self.init_ui()

        # Initialize data
        self.wallet_info = None
        self.backtester = None
        self.portfolio_tracker = None
        self.rest_client = None

        # Load wallet and initialize components
        self.load_wallet_info()

    def init_ui(self):
        """Initialize the UI components"""
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs
        self.overview_tab = QWidget()
        self.wallet_tab = QWidget()
        self.backtest_tab = QWidget()
        self.trading_tab = QWidget()
        self.logs_tab = QWidget()

        # Add tabs to widget
        self.tabs.addTab(self.overview_tab, "Overview")
        self.tabs.addTab(self.wallet_tab, "Wallet")
        self.tabs.addTab(self.backtest_tab, "Backtester")
        self.tabs.addTab(self.trading_tab, "Trading")
        self.tabs.addTab(self.logs_tab, "Logs")

        # Set up tabs
        self.setup_overview_tab()
        self.setup_wallet_tab()
        self.setup_backtest_tab()
        self.setup_trading_tab()
        self.setup_logs_tab()

    def setup_overview_tab(self):
        """Setup the overview tab with summary information"""
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Aptos Trading Dashboard")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Status section
        status_box = QGroupBox("System Status")
        status_layout = QFormLayout()

        self.connection_status = QLabel("Not connected")
        self.wallet_status = QLabel("No wallet loaded")
        self.balance_status = QLabel("0 APT")

        status_layout.addRow("Network Connection:", self.connection_status)
        status_layout.addRow("Wallet Status:", self.wallet_status)
        status_layout.addRow("Balance:", self.balance_status)

        status_box.setLayout(status_layout)
        layout.addWidget(status_box)

        # Performance summary
        perf_box = QGroupBox("Performance Summary")
        perf_layout = QVBoxLayout()

        # Create chart for portfolio performance
        self.portfolio_chart = MplCanvas(self, width=5, height=4, dpi=100)
        perf_layout.addWidget(self.portfolio_chart)

        # Summary metrics
        metrics_layout = QHBoxLayout()

        # Total return
        return_group = QGroupBox("Total Return")
        return_layout = QVBoxLayout()
        self.total_return_label = QLabel("0.00%")
        self.total_return_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.total_return_label.setFont(font)
        return_layout.addWidget(self.total_return_label)
        return_group.setLayout(return_layout)
        metrics_layout.addWidget(return_group)

        # Win rate
        win_group = QGroupBox("Win Rate")
        win_layout = QVBoxLayout()
        self.win_rate_label = QLabel("0.00%")
        self.win_rate_label.setAlignment(Qt.AlignCenter)
        self.win_rate_label.setFont(font)
        win_layout.addWidget(self.win_rate_label)
        win_group.setLayout(win_layout)
        metrics_layout.addWidget(win_group)

        # Number of trades
        trades_group = QGroupBox("Total Trades")
        trades_layout = QVBoxLayout()
        self.num_trades_label = QLabel("0")
        self.num_trades_label.setAlignment(Qt.AlignCenter)
        self.num_trades_label.setFont(font)
        trades_layout.addWidget(self.num_trades_label)
        trades_group.setLayout(trades_layout)
        metrics_layout.addWidget(trades_group)

        perf_layout.addLayout(metrics_layout)
        perf_box.setLayout(perf_layout)
        layout.addWidget(perf_box)

        # Refresh button
        refresh_btn = QPushButton("Refresh Dashboard")
        refresh_btn.clicked.connect(self.refresh_overview)
        layout.addWidget(refresh_btn)

        self.overview_tab.setLayout(layout)

    def setup_wallet_tab(self):
        """Setup the wallet management tab"""
        layout = QVBoxLayout()

        # Wallet info section
        wallet_box = QGroupBox("Wallet Information")
        wallet_layout = QFormLayout()

        self.address_label = QLabel("No wallet loaded")
        self.address_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        wallet_layout.addRow("Address:", self.address_label)

        self.wallet_balance_label = QLabel("0 APT")
        wallet_layout.addRow("Balance:", self.wallet_balance_label)

        wallet_box.setLayout(wallet_layout)
        layout.addWidget(wallet_box)

        # Fund wallet section
        fund_box = QGroupBox("Fund Wallet")
        fund_layout = QHBoxLayout()

        self.fund_amount = QSpinBox()
        self.fund_amount.setRange(1, 1000)
        self.fund_amount.setValue(1)
        self.fund_amount.setSuffix(" APT")
        fund_layout.addWidget(QLabel("Amount:"))
        fund_layout.addWidget(self.fund_amount)

        self.fund_type = QComboBox()
        self.fund_type.addItem("APT", "0x1::aptos_coin::AptosCoin")
        self.fund_type.addItem("BTC", "0x1::btc_coin::BtcCoin")
        self.fund_type.addItem("ETH", "0x1::eth_coin::EthCoin")
        self.fund_type.addItem("USDT", "0x1::usdt::Coin")
        fund_layout.addWidget(QLabel("Coin:"))
        fund_layout.addWidget(self.fund_type)

        fund_btn = QPushButton("Fund Wallet")
        fund_btn.clicked.connect(self.fund_wallet_action)
        fund_layout.addWidget(fund_btn)

        fund_box.setLayout(fund_layout)
        layout.addWidget(fund_box)

        # Transfer section
        transfer_box = QGroupBox("Transfer Funds")
        transfer_layout = QFormLayout()

        self.recipient_address = QLineEdit()
        transfer_layout.addRow("Recipient:", self.recipient_address)

        self.transfer_amount = QDoubleSpinBox()
        self.transfer_amount.setRange(0.001, 1000)
        self.transfer_amount.setValue(0.1)
        self.transfer_amount.setSuffix(" APT")
        self.transfer_amount.setDecimals(3)
        transfer_layout.addRow("Amount:", self.transfer_amount)

        transfer_btn = QPushButton("Execute Transfer")
        transfer_btn.clicked.connect(self.execute_transfer_action)
        transfer_layout.addWidget(transfer_btn)

        transfer_box.setLayout(transfer_layout)
        layout.addWidget(transfer_box)

        # Transactions table
        txn_box = QGroupBox("Recent Transactions")
        txn_layout = QVBoxLayout()

        self.txn_table = QTableWidget()
        self.txn_table.setColumnCount(5)
        self.txn_table.setHorizontalHeaderLabels(
            ["Date", "Type", "Amount", "Recipient", "Status"]
        )
        self.txn_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        txn_layout.addWidget(self.txn_table)

        txn_box.setLayout(txn_layout)
        layout.addWidget(txn_box)

        # Refresh button
        refresh_btn = QPushButton("Refresh Wallet")
        refresh_btn.clicked.connect(self.refresh_wallet)
        layout.addWidget(refresh_btn)

        self.wallet_tab.setLayout(layout)

    def setup_backtest_tab(self):
        """Setup the backtesting tab"""
        layout = QVBoxLayout()

        # Backtest parameters
        # 
        param_box = QGroupBox("Backtest Parameters")
        param_layout = QFormLayout()

        self.backtest_symbol = QComboBox()
        self.backtest_symbol.addItems(
            ["APT21794-USD", "BTC-USD", "ETH-USD", "AAPL", "TSLA", "CRWD", "QBTS"]
        )
        self.backtest_symbol.setEditable(True)
        param_layout.addRow("Symbol:", self.backtest_symbol)

        self.backtest_start = QDateEdit()
        self.backtest_start.setDate(QDate.currentDate().addMonths(-3))
        self.backtest_start.setCalendarPopup(True)
        param_layout.addRow("Start Date:", self.backtest_start)

        self.backtest_end = QDateEdit()
        self.backtest_end.setDate(QDate.currentDate())
        self.backtest_end.setCalendarPopup(True)
        param_layout.addRow("End Date:", self.backtest_end)

        self.backtest_capital = QDoubleSpinBox()
        self.backtest_capital.setRange(10, 100000)
        self.backtest_capital.setValue(1000)
        self.backtest_capital.setSuffix(" APT")
        param_layout.addRow("Initial Capital:", self.backtest_capital)

        

        self.backtest_reverse_signals = QCheckBox("Reverse Trading Signals")
        self.backtest_reverse_signals.setChecked(True)  # Default to True since it works better
        self.backtest_reverse_signals.setToolTip("When checked, BUY signals become SELL signals and vice versa")
        param_layout.addRow("Signal Strategy:", self.backtest_reverse_signals)

        # Add an advanced options section (optional)
        advanced_options = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout()

        self.backtest_auto_detect = QCheckBox("Auto-detect Market Regime")
        self.backtest_auto_detect.setChecked(False)  # Default to False for simpler behavior
        self.backtest_auto_detect.setToolTip("When checked, system will attempt to detect market regime and adapt signal reversal")
        self.backtest_auto_detect.stateChanged.connect(self.toggle_auto_detect)
        advanced_layout.addRow("Market Detection:", self.backtest_auto_detect)

        advanced_options.setLayout(advanced_layout)
        param_layout.addWidget(advanced_options)

        run_backtest_btn = QPushButton("Run Backtest")
        run_backtest_btn.clicked.connect(self.run_backtest)
        param_layout.addWidget(run_backtest_btn)

        param_box.setLayout(param_layout)
        layout.addWidget(param_box)

        # Progress bar
        self.backtest_progress = QProgressBar()
        layout.addWidget(self.backtest_progress)

        # Results section
        results_splitter = QSplitter(Qt.Vertical)

        # Chart section
        chart_box = QGroupBox("Performance Chart")
        chart_layout = QVBoxLayout()
        self.backtest_chart = MplCanvas(self, width=6, height=4, dpi=100)
        chart_layout.addWidget(self.backtest_chart)
        chart_box.setLayout(chart_layout)
        results_splitter.addWidget(chart_box)

        # Results table section
        results_box = QGroupBox("Backtest Results")
        results_layout = QVBoxLayout()

        metrics_layout = QHBoxLayout()

        # Return
        metrics_return = QGroupBox("Total Return")
        return_layout = QVBoxLayout()
        self.backtest_return = QLabel("0.00%")
        self.backtest_return.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.backtest_return.setFont(font)
        return_layout.addWidget(self.backtest_return)
        metrics_return.setLayout(return_layout)
        metrics_layout.addWidget(metrics_return)

        # Sharpe
        metrics_sharpe = QGroupBox("Sharpe Ratio")
        sharpe_layout = QVBoxLayout()
        self.backtest_sharpe = QLabel("0.00")
        self.backtest_sharpe.setAlignment(Qt.AlignCenter)
        self.backtest_sharpe.setFont(font)
        sharpe_layout.addWidget(self.backtest_sharpe)
        metrics_sharpe.setLayout(sharpe_layout)
        metrics_layout.addWidget(metrics_sharpe)

        # Max Drawdown
        metrics_drawdown = QGroupBox("Max Drawdown")
        drawdown_layout = QVBoxLayout()
        self.backtest_drawdown = QLabel("0.00%")
        self.backtest_drawdown.setAlignment(Qt.AlignCenter)
        self.backtest_drawdown.setFont(font)
        drawdown_layout.addWidget(self.backtest_drawdown)
        metrics_drawdown.setLayout(drawdown_layout)
        metrics_layout.addWidget(metrics_drawdown)

        # Win Rate
        metrics_win = QGroupBox("Win Rate")
        win_layout = QVBoxLayout()
        self.backtest_win_rate = QLabel("0.00%")
        self.backtest_win_rate.setAlignment(Qt.AlignCenter)
        self.backtest_win_rate.setFont(font)
        win_layout.addWidget(self.backtest_win_rate)
        metrics_win.setLayout(win_layout)
        metrics_layout.addWidget(metrics_win)

        # Trades
        metrics_trades = QGroupBox("Total Trades")
        trades_layout = QVBoxLayout()
        self.backtest_trades = QLabel("0")
        self.backtest_trades.setAlignment(Qt.AlignCenter)
        self.backtest_trades.setFont(font)
        trades_layout.addWidget(self.backtest_trades)
        metrics_trades.setLayout(trades_layout)
        metrics_layout.addWidget(metrics_trades)
        results_layout.addLayout(metrics_layout)


        metrics_buy = QGroupBox("Buy Trades")
        buy_layout = QVBoxLayout()
        self.backtest_buy_trades = QLabel("0")
        self.backtest_buy_trades.setAlignment(Qt.AlignCenter)
        self.backtest_buy_trades.setFont(font)
        buy_layout.addWidget(self.backtest_buy_trades)
        metrics_buy.setLayout(buy_layout)
        metrics_layout.addWidget(metrics_buy)

        metrics_sell = QGroupBox("Sell Trades")
        sell_layout = QVBoxLayout()
        self.backtest_sell_trades = QLabel("0")
        self.backtest_sell_trades.setAlignment(Qt.AlignCenter)
        self.backtest_sell_trades.setFont(font)
        sell_layout.addWidget(self.backtest_sell_trades)
        metrics_sell.setLayout(sell_layout)
        metrics_layout.addWidget(metrics_sell)

        

      

        # Trades table
        self.backtest_trades_table = QTableWidget()
        self.backtest_trades_table.setColumnCount(5)
        self.backtest_trades_table.setHorizontalHeaderLabels(
            ["Date", "Type", "Price", "Quantity", "Value"]
        )
        self.backtest_trades_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        results_layout.addWidget(self.backtest_trades_table)

        results_box.setLayout(results_layout)
        results_splitter.addWidget(results_box)

        layout.addWidget(results_splitter)

        self.backtest_tab.setLayout(layout)

    def setup_trading_tab(self):
        """Setup the trading dashboard tab"""
        layout = QVBoxLayout()

        # Market data section
        market_box = QGroupBox("Market Data")
        market_layout = QHBoxLayout()

        self.trading_symbol = QComboBox()
        self.trading_symbol.addItems(
            ["APT21794-USD", "BTC-USD", "ETH-USD", "AAPL", "TSLA", "CRWD"]
        )
        market_layout.addWidget(QLabel("Symbol:"))
        market_layout.addWidget(self.trading_symbol)

        self.current_price = QLabel("$0.00")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.current_price.setFont(font)
        market_layout.addWidget(QLabel("Current Price:"))
        market_layout.addWidget(self.current_price)

        refresh_price_btn = QPushButton("Refresh Price")
        refresh_price_btn.clicked.connect(self.refresh_price)
        market_layout.addWidget(refresh_price_btn)

        market_box.setLayout(market_layout)
        layout.addWidget(market_box)

        # Trading signals section
        signal_box = QGroupBox("Trading Signals")
        signal_layout = QVBoxLayout()

        # Signal metrics
        signal_metrics = QHBoxLayout()

        # Signal
        signal_type = QGroupBox("Signal")
        signal_type_layout = QVBoxLayout()
        self.signal_label = QLabel("HOLD")
        self.signal_label.setAlignment(Qt.AlignCenter)
        signal_type_layout.addWidget(self.signal_label)
        signal_type.setLayout(signal_type_layout)
        signal_metrics.addWidget(signal_type)

        # Confidence
        signal_conf = QGroupBox("Confidence")
        signal_conf_layout = QVBoxLayout()
        self.confidence_label = QLabel("0%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        signal_conf_layout.addWidget(self.confidence_label)
        signal_conf.setLayout(signal_conf_layout)
        signal_metrics.addWidget(signal_conf)

        # Stop Loss
        signal_stop = QGroupBox("Stop Loss")
        signal_stop_layout = QVBoxLayout()
        self.stop_loss_label = QLabel("$0.00")
        self.stop_loss_label.setAlignment(Qt.AlignCenter)
        signal_stop_layout.addWidget(self.stop_loss_label)
        signal_stop.setLayout(signal_stop_layout)
        signal_metrics.addWidget(signal_stop)

        # Take Profit
        signal_tp = QGroupBox("Take Profit")
        signal_tp_layout = QVBoxLayout()
        self.take_profit_label = QLabel("$0.00")
        self.take_profit_label.setAlignment(Qt.AlignCenter)
        signal_tp_layout.addWidget(self.take_profit_label)
        signal_tp.setLayout(signal_tp_layout)
        signal_metrics.addWidget(signal_tp)

        signal_layout.addLayout(signal_metrics)

        # Signal rationale
        self.signal_rationale = QLabel("No signal rationale available yet.")
        self.signal_rationale.setWordWrap(True)
        signal_layout.addWidget(self.signal_rationale)

        # Check signals button
        check_signals_btn = QPushButton("Check Trading Signals")
        check_signals_btn.clicked.connect(self.check_trading_signals)
        signal_layout.addWidget(check_signals_btn)

        signal_box.setLayout(signal_layout)
        layout.addWidget(signal_box)

        # Trade execution section
        trade_box = QGroupBox("Trade Execution")
        trade_layout = QHBoxLayout()

        # Position size
        self.trade_size = QDoubleSpinBox()
        self.trade_size.setRange(0.001, 100)
        self.trade_size.setValue(1)
        self.trade_size.setSuffix(" APT")
        self.trade_size.setDecimals(3)
        trade_layout.addWidget(QLabel("Size:"))
        trade_layout.addWidget(self.trade_size)

        # Buy/Sell buttons
        buy_btn = QPushButton("Execute Buy")
        buy_btn.clicked.connect(lambda: self.execute_trade("BUY"))
        trade_layout.addWidget(buy_btn)

        sell_btn = QPushButton("Execute Sell")
        sell_btn.clicked.connect(lambda: self.execute_trade("SELL"))
        trade_layout.addWidget(sell_btn)

        trade_box.setLayout(trade_layout)
        layout.addWidget(trade_box)

        # Portfolio section
        portfolio_box = QGroupBox("Portfolio")
        portfolio_layout = QVBoxLayout()

        # Portfolio summary
        portfolio_summary = QHBoxLayout()

        # Cash
        cash_box = QGroupBox("Cash")
        cash_layout = QVBoxLayout()
        self.portfolio_cash = QLabel("0.00 APT")
        self.portfolio_cash.setAlignment(Qt.AlignCenter)
        cash_layout.addWidget(self.portfolio_cash)
        cash_box.setLayout(cash_layout)
        portfolio_summary.addWidget(cash_box)

        # Total Value
        value_box = QGroupBox("Total Value")
        value_layout = QVBoxLayout()
        self.portfolio_value = QLabel("0.00 APT")
        self.portfolio_value.setAlignment(Qt.AlignCenter)
        value_layout.addWidget(self.portfolio_value)
        value_box.setLayout(value_layout)
        portfolio_summary.addWidget(value_box)

        # Unrealized P&L
        pnl_box = QGroupBox("Unrealized P&L")
        pnl_layout = QVBoxLayout()
        self.portfolio_pnl = QLabel("0.00 APT")
        self.portfolio_pnl.setAlignment(Qt.AlignCenter)
        pnl_layout.addWidget(self.portfolio_pnl)
        pnl_box.setLayout(pnl_layout)
        portfolio_summary.addWidget(pnl_box)

        portfolio_layout.addLayout(portfolio_summary)

        # Positions table
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(5)
        self.positions_table.setHorizontalHeaderLabels(
            ["Symbol", "Quantity", "Entry Price", "Current Price", "P&L"]
        )
        self.positions_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        portfolio_layout.addWidget(self.positions_table)

        # Reconcile button
        reconcile_btn = QPushButton("Reconcile with On-Chain Balance")
        reconcile_btn.clicked.connect(self.reconcile_balances_action)
        portfolio_layout.addWidget(reconcile_btn)

        portfolio_box.setLayout(portfolio_layout)
        layout.addWidget(portfolio_box)

        self.trading_tab.setLayout(layout)

    def setup_logs_tab(self):
        """Setup the transaction logs tab"""
        layout = QVBoxLayout()

        # Create tabs for different logs
        log_tabs = QTabWidget()

        # Transactions log
        txn_widget = QWidget()
        txn_layout = QVBoxLayout()

        self.txn_log_table = QTableWidget()
        self.txn_log_table.setColumnCount(6)
        self.txn_log_table.setHorizontalHeaderLabels(
            ["Timestamp", "Type", "Symbol", "Price", "Quantity", "Value"]
        )
        self.txn_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        txn_layout.addWidget(self.txn_log_table)

        txn_widget.setLayout(txn_layout)
        log_tabs.addTab(txn_widget, "Transactions")

        # Portfolio log
        portfolio_widget = QWidget()
        portfolio_layout = QVBoxLayout()

        self.portfolio_log_table = QTableWidget()
        self.portfolio_log_table.setColumnCount(4)
        self.portfolio_log_table.setHorizontalHeaderLabels(
            ["Timestamp", "Total Value", "Cash", "Positions"]
        )
        self.portfolio_log_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        portfolio_layout.addWidget(self.portfolio_log_table)

        portfolio_widget.setLayout(portfolio_layout)
        log_tabs.addTab(portfolio_widget, "Portfolio History")

        # System log
        system_widget = QWidget()
        system_layout = QVBoxLayout()

        self.system_log = QTextEdit()
        self.system_log.setReadOnly(True)
        system_layout.addWidget(self.system_log)

        system_widget.setLayout(system_layout)
        log_tabs.addTab(system_widget, "System Logs")

        layout.addWidget(log_tabs)

        # Refresh button
        refresh_logs_btn = QPushButton("Refresh Logs")
        refresh_logs_btn.clicked.connect(self.refresh_logs)
        layout.addWidget(refresh_logs_btn)

        self.logs_tab.setLayout(layout)

    def toggle_auto_detect(self, state):
        """Handle changes to auto-detect checkbox - disable manual reversal if auto-detect is on"""
        if state == Qt.Checked:
            self.backtest_reverse_signals.setEnabled(False)
            self.backtest_reverse_signals.setToolTip("Disabled when auto-detection is active")
        else:
            self.backtest_reverse_signals.setEnabled(True)
            self.backtest_reverse_signals.setToolTip("When checked, BUY signals become SELL signals and vice versa")

    def load_wallet_info(self):
        """Load wallet information and initialize clients"""
        try:
            private_key, address, public_key = load_or_create_wallet()
            self.wallet_info = {
                "private_key": private_key,
                "address": address,
                "public_key": public_key,
            }

            # Update UI
            self.address_label.setText(address)
            self.wallet_status.setText("Loaded")

            # Initialize REST client
            self.rest_client = RestClient(NODE_URL)
            self.connection_status.setText("Connected to Aptos devnet")

            # Get balance
            self.get_wallet_balance()

            # Initialize portfolio tracker
            self.portfolio_tracker = PortfolioTracker()

            # Load logs
            self.refresh_logs()

            # Log
            self.log_message("Wallet loaded successfully")

        except Exception as e:
            self.log_message(f"Error loading wallet: {str(e)}", error=True)

    def get_wallet_balance(self):
        """Get and update wallet balance"""
        if not self.wallet_info or not self.rest_client:
            return

        worker = AsyncWorker(
            check_balance,
            self.rest_client,
            AccountAddress.from_str(self.wallet_info["address"]),
        )
        worker.finished.connect(self.update_balance_display)
        worker.error.connect(
            lambda e: self.log_message(f"Error getting balance: {e}", error=True)
        )
        worker.start()

    def update_balance_display(self, balance):
        """Update balance display after fetching balance"""
        balance_apt = balance / 1e8
        self.wallet_balance_label.setText(f"{balance_apt:.8f} APT")
        self.balance_status.setText(f"{balance_apt:.8f} APT")

    def refresh_overview(self):
        """Refresh overview tab data"""
        self.get_wallet_balance()

        # Try to update portfolio chart if we have data
        if (
            self.portfolio_tracker
            and len(self.portfolio_tracker.portfolio["value_history"]) > 0
        ):
            self.update_portfolio_chart()

            # Update metrics
            metrics = self.portfolio_tracker.get_pnl_metrics()
            self.total_return_label.setText(f"{metrics['total_return']:.2%}")
            self.win_rate_label.setText(f"{metrics['win_rate']:.2%}")
            self.num_trades_label.setText(f"{metrics['num_trades']}")

        self.log_message("Overview refreshed")

    def update_portfolio_chart(self):
        """Update the portfolio performance chart"""
        if (
            not self.portfolio_tracker
            or len(self.portfolio_tracker.portfolio["value_history"]) <= 1
        ):
            return

        # Get portfolio history
        history = pd.DataFrame(self.portfolio_tracker.portfolio["value_history"])
        history["timestamp"] = pd.to_datetime(history["timestamp"])

        # Clear and plot
        ax = self.portfolio_chart.axes
        ax.clear()
        ax.plot(
            history["timestamp"],
            history["value"],
            label="Portfolio Value",
            color="blue",
        )
        ax.set_title("Portfolio Performance")
        ax.set_ylabel("Value (APT)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

        self.portfolio_chart.fig.tight_layout()
        self.portfolio_chart.draw()

    def fund_wallet_action(self):
        """Fund the wallet with test tokens"""
        if not self.wallet_info:
            self.log_message("No wallet loaded", error=True)
            return

        amount = self.fund_amount.value() * 100_000_000  # Convert to octas
        coin_type = self.fund_type.currentData()

        try:
            result = fund_wallet(self.wallet_info["address"], amount, coin_type)
            if result:
                self.log_message(
                    f"Successfully funded wallet with {self.fund_amount.value()} {self.fund_type.currentText()}"
                )
                # Update balance after funding
                self.get_wallet_balance()
            else:
                self.log_message("Failed to fund wallet", error=True)
        except Exception as e:
            self.log_message(f"Error funding wallet: {str(e)}", error=True)

    def execute_transfer_action(self):
        """Execute a transfer of funds"""
        if not self.wallet_info:
            self.log_message("No wallet loaded", error=True)
            return

        recipient = self.recipient_address.text().strip()
        if not recipient:
            self.log_message("Please enter a recipient address", error=True)
            return

        amount = int(self.transfer_amount.value() * 100_000_000)  # Convert to octas

        # Execute the transfer
        worker = AsyncWorker(
            execute_transfer, self.wallet_info["private_key"], recipient, amount
        )
        worker.finished.connect(self.handle_transfer_result)
        worker.error.connect(
            lambda e: self.log_message(f"Transfer error: {e}", error=True)
        )
        worker.start()

    def handle_transfer_result(self, result):
        """Handle transfer result"""
        if result:
            self.log_message("Transfer executed successfully")
            # Update balance after transfer
            self.get_wallet_balance()
        else:
            self.log_message("Transfer failed", error=True)

    def refresh_wallet(self):
        """Refresh wallet tab information"""
        self.get_wallet_balance()
        self.load_transaction_history()
        self.log_message("Wallet refreshed")

    def load_transaction_history(self):
        """Load transaction history into table"""
        if not self.wallet_info:
            return

        # In a real implementation, this would load on-chain transactions
        # For now, we'll display portfolio transactions if available
        if (
            not self.portfolio_tracker
            or not self.portfolio_tracker.portfolio["transactions"]
        ):
            return

        self.txn_table.setRowCount(0)

        for i, (tx_type, price, qty, timestamp) in enumerate(
            self.portfolio_tracker.portfolio["transactions"]
        ):
            self.txn_table.insertRow(i)
            self.txn_table.setItem(
                i, 0, QTableWidgetItem(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            )
            self.txn_table.setItem(i, 1, QTableWidgetItem(tx_type))
            self.txn_table.setItem(i, 2, QTableWidgetItem(f"{qty:.6f}"))
            # For demo, we use self as recipient
            self.txn_table.setItem(
                i, 3, QTableWidgetItem(self.wallet_info["address"][:10] + "...")
            )
            self.txn_table.setItem(i, 4, QTableWidgetItem("Completed"))

            # Color code based on transaction type
            if tx_type == "BUY":
                self.txn_table.item(i, 1).setBackground(
                    QColor(200, 255, 200)
                )  # Light green
            elif tx_type == "SELL":
                self.txn_table.item(i, 1).setBackground(
                    QColor(255, 200, 200)
                )  # Light red

    def run_backtest(self):
        """Run a backtest with the specified parameters"""
        symbol = self.backtest_symbol.currentText()
        start_date = self.backtest_start.date().toString("yyyy-MM-dd")
        end_date = self.backtest_end.date().toString("yyyy-MM-dd")
        initial_capital = self.backtest_capital.value()

        # Get reversal settings
        reverse_signals = self.backtest_reverse_signals.isChecked()
        auto_detect = self.backtest_auto_detect.isChecked()

        self.log_message(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        self.log_message(f"Signal reversal: {'Auto-detect' if auto_detect else ('Enabled' if reverse_signals else 'Disabled')}")


        # Show progress bar as indeterminate
        self.backtest_progress.setRange(0, 0)

        # Create and run worker thread
        self.backtest_worker = BacktestWorker(
            symbol, 
            start_date, 
            end_date, 
            initial_capital,
            reverse_signals=reverse_signals,
            auto_detect=auto_detect
        )
        self.backtest_worker.finished.connect(self.display_backtest_results)
        self.backtest_worker.start()

    def display_backtest_results(self, history, metrics):
        """Display backtest results"""
        # Reset progress bar
        self.backtest_progress.setRange(0, 100)
        self.backtest_progress.setValue(100)

        if "error" in metrics:
            self.log_message(f"Backtest error: {metrics['error']}", error=True)
            return

        # Update metrics display
        self.backtest_return.setText(f"{metrics['total_return']:.2%}")
        self.backtest_sharpe.setText(f"{metrics.get('sharpe', 0):.2f}")
        self.backtest_drawdown.setText(f"{metrics.get('max_drawdown', 0):.2%}")
        self.backtest_win_rate.setText(f"{metrics.get('win_rate', 0):.2%}")
        self.backtest_trades.setText(f"{metrics.get('num_trades', 0)}")
        self.backtest_buy_trades.setText(f"{metrics.get('num_buy_trades', 0)}")
        self.backtest_sell_trades.setText(f"{metrics.get('num_sell_trades', 0)}")

        
        # Save reference to the backtester
        self.backtester = self.backtest_worker.backtester

        # Plot results
        fig = self.backtester.plot_results(history)
        self.backtest_chart.figure = fig
        self.backtest_chart.draw()

        # Update trades table
        self.update_backtest_trades_table()

        self.log_message(
            f"Backtest completed with {metrics['total_return']:.2%} return"
        )

    def update_backtest_trades_table(self):
        """Update the backtest trades table with transaction data"""
        if not self.backtester or not self.backtester.portfolio["transactions"]:
            return

        self.backtest_trades_table.setRowCount(0)

        for i, (tx_type, price, qty, timestamp) in enumerate(
            self.backtester.portfolio["transactions"]
        ):
            self.backtest_trades_table.insertRow(i)
            self.backtest_trades_table.setItem(
                i, 0, QTableWidgetItem(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            )
            self.backtest_trades_table.setItem(i, 1, QTableWidgetItem(tx_type))
            self.backtest_trades_table.setItem(i, 2, QTableWidgetItem(f"${price:.2f}"))
            self.backtest_trades_table.setItem(i, 3, QTableWidgetItem(f"{qty:.6f}"))
            self.backtest_trades_table.setItem(
                i, 4, QTableWidgetItem(f"${price * qty:.2f}")
            )

            # Color code based on transaction type
            if tx_type == "BUY":
                self.backtest_trades_table.item(i, 1).setBackground(
                    QColor(200, 255, 200)
                )  # Light green
            elif tx_type == "SELL":
                self.backtest_trades_table.item(i, 1).setBackground(
                    QColor(255, 200, 200)
                )  # Light red

    # def refresh_price(self):
    #     """Refresh the current price of the selected symbol"""
    #     symbol = self.trading_symbol.currentText()
    #     self.log_message(f"Fetching current price for {symbol}")

    #     try:
    #         import yfinance as yf

    #         current_price = float(
    #             yf.download(symbol, period="1d", interval="1m")["Close"].iloc[-1]
    #         )
    #         self.current_price.setText(f"${current_price:.2f}")
    #     except Exception as e:
    #         self.log_message(f"Error getting price: {str(e)}", error=True)
    
    def refresh_price(self):
        """Refresh the current price of the selected symbol"""
        symbol = self.trading_symbol.currentText()
        self.log_message(f"Fetching current price for {symbol}")

        try:
            # Use minimal data fetch with retries for price updates
            data = fetch_yf_data_with_retries(symbol, period="1d", interval="1m", max_retries=3)
            current_price = float(data["Close"].iloc[-1])
            self.current_price.setText(f"${current_price:.2f}")
            self.log_message(f"Updated price for {symbol}: ${current_price:.2f}")
        except Exception as e:
            self.log_message(f"Error getting price: {str(e)}", error=True)
            # Provide a fallback UI update
            self.current_price.setText("$-.--")

    def check_trading_signals(self):
        """Check trading signals for the selected symbol"""
        if not self.wallet_info or not self.rest_client:
            self.log_message("Wallet not loaded", error=True)
            return

        symbol = self.trading_symbol.currentText()
        self.log_message(f"Checking trading signals for {symbol}")

        worker = AsyncWorker(check_entry_points, symbol, self.portfolio_tracker)
        worker.finished.connect(self.display_trading_signals)
        worker.error.connect(
            lambda e: self.log_message(f"Signal check error: {e}", error=True)
        )
        worker.start()

    def display_trading_signals(self, result):
        """Display the trading signals"""
        # In a real implementation, result would contain signal info
        # For now, simulate some signal data
        import random

        signals = ["BUY", "HOLD", "SELL"]
        decision = random.choice(signals)
        confidence = random.randint(50, 95)

        # Get current price
        try:
            import yfinance as yf

            current_price = float(
                yf.download(
                    self.trading_symbol.currentText(), period="1d", interval="1m"
                )["Close"].iloc[-1]
            )
        except:
            current_price = 10.0

        # Calculate levels
        stop_loss = current_price * 0.97 if decision == "BUY" else current_price * 1.03
        take_profit = (
            current_price * 1.05 if decision == "BUY" else current_price * 0.95
        )

        # Generate rationale
        if decision == "BUY":
            rationale = "📈 Price above 50MA | 💪 Moderate momentum (RSI 58.6) | 📊 Volume spike detected"
        elif decision == "SELL":
            rationale = "📉 Price below 50MA | 💻 Momentum weakening (RSI 72.3) | ⚠️ Volume declining"
        else:
            rationale = "⏸️ Price consolidating | 📊 Neutral momentum (RSI 53.2) | ⚖️ Average volume"

        # Update UI
        self.signal_label.setText(decision)
        if decision == "BUY":
            self.signal_label.setStyleSheet(
                "background-color: #d4edda; color: #155724; font-weight: bold;"
            )
        elif decision == "SELL":
            self.signal_label.setStyleSheet(
                "background-color: #f8d7da; color: #721c24; font-weight: bold;"
            )
        else:
            self.signal_label.setStyleSheet(
                "background-color: #fff3cd; color: #856404; font-weight: bold;"
            )

        self.confidence_label.setText(f"{confidence}%")
        self.stop_loss_label.setText(f"${stop_loss:.2f}")
        self.take_profit_label.setText(f"${take_profit:.2f}")
        self.signal_rationale.setText(rationale)

        # Update current price
        self.current_price.setText(f"${current_price:.2f}")

        self.log_message(f"Signal generated: {decision} with {confidence}% confidence")

    def execute_trade(self, trade_type):
        """Execute a trade (buy/sell)"""
        if not self.wallet_info or not self.rest_client or not self.portfolio_tracker:
            self.log_message("Wallet not loaded", error=True)
            return

        symbol = self.trading_symbol.currentText()
        size = self.trade_size.value()

        self.log_message(f"Executing {trade_type} for {size} APT of {symbol}")

        worker = AsyncWorker(
            self.portfolio_tracker.update_position,
            symbol,
            float(self.current_price.text().replace("$", "")),
            size,
            trade_type,
        )
        worker.finished.connect(
            lambda: self.handle_trade_result(trade_type, symbol, size)
        )
        worker.error.connect(
            lambda e: self.log_message(f"Trade error: {e}", error=True)
        )
        worker.start()

    def handle_trade_result(self, trade_type, symbol, size):
        """Handle trade execution result"""
        self.log_message(
            f"{trade_type} executed successfully for {size} APT of {symbol}"
        )

        # Update portfolio display
        self.update_portfolio_display()

        # Refresh overview metrics
        self.refresh_overview()

    def update_portfolio_display(self):
        """Update portfolio display in the trading tab"""
        if not self.portfolio_tracker:
            return

        # Update cash and value
        self.portfolio_cash.setText(
            f"{self.portfolio_tracker.portfolio['cash']:.2f} APT"
        )

        # Calculate and update total value
        total_value = self.portfolio_tracker.calculate_current_value()
        self.portfolio_value.setText(f"{total_value:.2f} APT")

        # Calculate and update unrealized P&L
        metrics = self.portfolio_tracker.get_pnl_metrics()
        self.portfolio_pnl.setText(f"{metrics['unrealized_pnl']:.2f} APT")

        # Update positions table
        self.positions_table.setRowCount(0)

        row = 0
        for symbol, position in self.portfolio_tracker.portfolio["positions"].items():
            self.positions_table.insertRow(row)

            # Try to get current price
            try:
                import yfinance as yf

                current_price = float(
                    yf.download(symbol, period="1d", interval="1m")["Close"].iloc[-1]
                )
            except:
                current_price = position["avg_entry_price"]  # Fallback to entry price

            # Calculate P&L
            pnl = (current_price - position["avg_entry_price"]) * position["qty"]
            pnl_pct = (current_price / position["avg_entry_price"] - 1) * 100

            # Set table items
            self.positions_table.setItem(row, 0, QTableWidgetItem(symbol))
            self.positions_table.setItem(
                row, 1, QTableWidgetItem(f"{position['qty']:.6f}")
            )
            self.positions_table.setItem(
                row, 2, QTableWidgetItem(f"${position['avg_entry_price']:.2f}")
            )
            self.positions_table.setItem(
                row, 3, QTableWidgetItem(f"${current_price:.2f}")
            )
            self.positions_table.setItem(
                row, 4, QTableWidgetItem(f"${pnl:.2f} ({pnl_pct:.2f}%)")
            )

            # Color code P&L
            if pnl > 0:
                self.positions_table.item(row, 4).setBackground(
                    QColor(200, 255, 200)
                )  # Light green
            elif pnl < 0:
                self.positions_table.item(row, 4).setBackground(
                    QColor(255, 200, 200)
                )  # Light red

            row += 1

    def reconcile_balances_action(self):
        """Reconcile portfolio tracker with on-chain balance"""
        if not self.wallet_info or not self.rest_client or not self.portfolio_tracker:
            self.log_message("Wallet not loaded", error=True)
            return

        self.log_message("Reconciling on-chain balance with portfolio tracker...")

        worker = AsyncWorker(
            reconcile_balances,
            self.rest_client,
            self.wallet_info["address"],
            self.portfolio_tracker,
        )
        worker.finished.connect(self.handle_reconcile_result)
        worker.error.connect(
            lambda e: self.log_message(f"Reconcile error: {e}", error=True)
        )
        worker.start()

    def handle_reconcile_result(self, result):
        """Handle reconcile balances result"""
        if result:
            self.log_message("Balance reconciliation successful")
            self.update_portfolio_display()
        else:
            self.log_message("Balance reconciliation failed", error=True)

    def refresh_logs(self):
        """Load and refresh log data"""
        self.load_transaction_logs()
        self.load_portfolio_logs()
        self.load_system_logs()
        self.log_message("Logs refreshed")

    def load_transaction_logs(self):
        """Load transaction logs from file"""
        try:
            # Get the log directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tx_log_file = os.path.join(script_dir, "aptos_transactions.csv")

            if os.path.exists(tx_log_file):
                tx_log = pd.read_csv(tx_log_file)

                self.txn_log_table.setRowCount(0)

                for i, row in tx_log.iterrows():
                    self.txn_log_table.insertRow(i)
                    self.txn_log_table.setItem(
                        i, 0, QTableWidgetItem(str(row["timestamp"]))
                    )
                    self.txn_log_table.setItem(i, 1, QTableWidgetItem(str(row["type"])))
                    self.txn_log_table.setItem(
                        i, 2, QTableWidgetItem(str(row["symbol"]))
                    )
                    self.txn_log_table.setItem(
                        i, 3, QTableWidgetItem(f"${row['price']:.2f}")
                    )
                    self.txn_log_table.setItem(
                        i, 4, QTableWidgetItem(f"{row['quantity']:.6f}")
                    )
                    self.txn_log_table.setItem(
                        i, 5, QTableWidgetItem(f"${row['value']:.2f}")
                    )

                    # Color code based on transaction type
                    if str(row["type"]) == "BUY":
                        self.txn_log_table.item(i, 1).setBackground(
                            QColor(200, 255, 200)
                        )  # Light green
                    elif str(row["type"]) == "SELL":
                        self.txn_log_table.item(i, 1).setBackground(
                            QColor(255, 200, 200)
                        )  # Light red

        except Exception as e:
            self.log_message(f"Error loading transaction logs: {str(e)}", error=True)

    def load_portfolio_logs(self):
        """Load portfolio history logs from file"""
        try:
            # Get the log directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            portfolio_log_file = os.path.join(script_dir, "aptos_portfolio.csv")

            if os.path.exists(portfolio_log_file):
                portfolio_log = pd.read_csv(portfolio_log_file)

                self.portfolio_log_table.setRowCount(0)

                for i, row in portfolio_log.iterrows():
                    self.portfolio_log_table.insertRow(i)
                    self.portfolio_log_table.setItem(
                        i, 0, QTableWidgetItem(str(row["timestamp"]))
                    )
                    self.portfolio_log_table.setItem(
                        i, 1, QTableWidgetItem(f"${row['total_value']:.2f}")
                    )
                    self.portfolio_log_table.setItem(
                        i, 2, QTableWidgetItem(f"${row['cash']:.2f}")
                    )
                    self.portfolio_log_table.setItem(
                        i, 3, QTableWidgetItem(str(row["positions"]))
                    )

        except Exception as e:
            self.log_message(f"Error loading portfolio logs: {str(e)}", error=True)

    def load_system_logs(self):
        """Load system logs"""
        try:
            # Get the log directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(
                script_dir, f"aptos_trading_{datetime.now().strftime('%Y%m%d')}.log"
            )

            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    log_contents = f.read()
                    self.system_log.setText(log_contents)

                # Scroll to bottom
                self.system_log.verticalScrollBar().setValue(
                    self.system_log.verticalScrollBar().maximum()
                )

        except Exception as e:
            self.log_message(f"Error loading system logs: {str(e)}", error=True)

    def log_message(self, message, error=False):
        """Add a message to the system log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "ERROR" if error else "INFO"
        log_message = f"{timestamp} - {level} - {message}"

        # Add to system log text area
        current_text = self.system_log.toPlainText()
        self.system_log.setText(
            current_text + "\n" + log_message if current_text else log_message
        )

        # Scroll to bottom
        self.system_log.verticalScrollBar().setValue(
            self.system_log.verticalScrollBar().maximum()
        )

        # Also print to console
        print(log_message)

    
    # Add this method to properly handle application closure
    def closeEvent(self, event):
        """Clean up threads before closing the application"""
        # Wait for any running worker threads to finish
        if hasattr(self, 'backtest_worker') and self.backtest_worker.isRunning():
            self.backtest_worker.wait()
            
        # Find and terminate all AsyncWorker threads
        for child in self.findChildren(AsyncWorker):
            if child.isRunning():
                child.terminate()
                child.wait()
                
        # Accept the close event
        event.accept()


def main():
    app = QApplication(sys.argv)
    dashboard = AptosDashboard()
    dashboard.show()
    
    # Set the Qt.AA_DontCreateNativeWidgetSiblings attribute
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
    
    # Ensure clean shutdown
    app.aboutToQuit.connect(app.deleteLater)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
