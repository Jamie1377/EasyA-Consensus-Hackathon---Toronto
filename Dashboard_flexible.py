import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from aptos_sdk.account_address import AccountAddress
from aptos_sdk.account import Account
from aptos_sdk.cli import RestClient
NODE_URL = "https://fullnode.devnet.aptoslabs.com/v1"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QDateEdit, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QTextEdit, QSplitter, QMessageBox, QProgressBar,
    QSizePolicy, QFrame, QToolButton, QSizeGrip, QScrollArea, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate, QSize, QEvent
from PyQt5.QtGui import QFont, QColor, QIcon

# Import necessary components from your existing modules
from aptos_integration_v3_1 import (
    AptosBacktester, PortfolioTracker, load_or_create_wallet,
    check_balance, execute_transfer, reconcile_balances
)

class CollapsibleSection(QWidget):
    """A section that can be collapsed/expanded"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("collapsible_section")
        self.setStyleSheet("""
            #collapsible_section {
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Header
        self.header = QFrame()
        self.header.setMinimumHeight(30)
        self.header.setMaximumHeight(30)
        self.header.setObjectName("section_header")
        self.header.setStyleSheet("""
            #section_header {
                background-color: #f0f0f0;
                border-bottom: 1px solid #ddd;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
        """)
        
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # Expand/collapse button
        self.toggle_button = QToolButton()
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.setToolTip("Collapse/Expand")
        self.toggle_button.clicked.connect(self.toggle_content)
        header_layout.addWidget(self.toggle_button)
        
        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
        # Size grip for resizing
        self.size_grip = QSizeGrip(self)
        grip_layout = QHBoxLayout()
        grip_layout.setContentsMargins(0, 0, 5, 5)
        grip_layout.addStretch()
        grip_layout.addWidget(self.size_grip)
        
        # Add components to main layout
        self.main_layout.addWidget(self.header)
        self.main_layout.addWidget(self.content)
        self.main_layout.addLayout(grip_layout)
        
        # Set expanded by default
        self.is_expanded = True
    
    def toggle_content(self):
        """Toggle visibility of content"""
        self.is_expanded = not self.is_expanded
        self.content.setVisible(self.is_expanded)
        self.toggle_button.setArrowType(Qt.DownArrow if self.is_expanded else Qt.RightArrow)
    
    def add_widget(self, widget):
        """Add widget to content layout"""
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """Add layout to content layout"""
        self.content_layout.addLayout(layout)


class MplCanvas(FigureCanvasQTAgg):
    """Canvas for matplotlib figures with improved sizing"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Make the canvas respond to window resize
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class BacktestWorker(QThread):
    """Worker thread to run backtests without blocking the UI"""
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(int)
    
    def __init__(self, symbol, start_date, end_date, initial_capital):
        super().__init__()
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.backtester = None
    
    def run(self):
        try:
            self.backtester = AptosBacktester(
                symbol=self.symbol, initial_capital=self.initial_capital
            )
            
            # Run backtest
            history, metrics = self.backtester.run_backtest(
                start_date=self.start_date, 
                end_date=self.end_date
            )
            
            self.finished.emit(history, metrics)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit({}, {"error": str(e)})


class FlexibleDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flexible Trading Dashboard")
        self.setGeometry(100, 100, 1300, 900)
        
        # State tracking
        self.wallet_info = None
        self.backtester = None
        self.portfolio_tracker = None
        
        self.rest_client = RestClient(NODE_URL)
        # Initialize UI
        self.init_ui()
        
        # Load wallet info
        self.load_wallet_info()
    
    def init_ui(self):
        """Initialize the UI"""
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setDocumentMode(True)
        
        # Create tabs
        self.setup_dashboard_tab()
        self.setup_backtest_tab()
        self.setup_trading_tab()
        self.setup_wallet_tab()
        self.setup_logs_tab()
        
        # Add tabs to main layout
        self.main_layout.addWidget(self.tabs)
    
    def setup_dashboard_tab(self):
        """Setup the main dashboard tab"""
        dashboard_tab = QWidget()
        
        # Create scroll area for dashboard
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        dashboard_content = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_content)
        dashboard_layout.setSpacing(10)
        
        # Top row - Portfolio summary and balance
        top_row = QHBoxLayout()
        
        # Portfolio value section
        portfolio_section = CollapsibleSection("Portfolio Value")
        portfolio_widget = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_widget)
        
        # Portfolio chart
        self.portfolio_chart = MplCanvas(width=6, height=4, dpi=100)
        portfolio_layout.addWidget(self.portfolio_chart, 1)
        
        # Portfolio metrics
        metrics_layout = QHBoxLayout()
        
        # Total value
        value_box = self.create_metric_widget("Total Value", "$0.00", "blue")
        metrics_layout.addWidget(value_box)
        
        # Daily change
        change_box = self.create_metric_widget("Daily Change", "+0.00%", "green")
        metrics_layout.addWidget(change_box)
        
        # Returns
        returns_box = self.create_metric_widget("Total Return", "+0.00%", "green")
        metrics_layout.addWidget(returns_box)
        
        portfolio_layout.addLayout(metrics_layout)
        portfolio_section.add_widget(portfolio_widget)
        top_row.addWidget(portfolio_section, 2)
        
        # Balance section
        balance_section = CollapsibleSection("Account Balance")
        balance_widget = QWidget()
        balance_layout = QVBoxLayout(balance_widget)
        
        # Balance info
        balance_form = QFormLayout()
        self.wallet_balance = QLabel("0.00 APT")
        self.wallet_balance.setFont(QFont("Arial", 14, QFont.Bold))
        balance_form.addRow("Available Balance:", self.wallet_balance)
        
        self.wallet_address = QLineEdit()
        self.wallet_address.setReadOnly(True)
        balance_form.addRow("Wallet Address:", self.wallet_address)
        
        # Quick action buttons
        action_layout = QHBoxLayout()
        
        deposit_btn = QPushButton("Deposit")
        deposit_btn.clicked.connect(self.show_deposit_dialog)
        action_layout.addWidget(deposit_btn)
        
        withdraw_btn = QPushButton("Withdraw")
        withdraw_btn.clicked.connect(self.show_withdraw_dialog)
        action_layout.addWidget(withdraw_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_dashboard)
        action_layout.addWidget(refresh_btn)
        
        balance_layout.addLayout(balance_form)
        balance_layout.addLayout(action_layout)
        
        balance_section.add_widget(balance_widget)
        top_row.addWidget(balance_section, 1)
        
        dashboard_layout.addLayout(top_row)
        
        # Middle row - Market watch and recent trades
        middle_row = QHBoxLayout()
        
        # Market watch section
        market_section = CollapsibleSection("Market Watch")
        market_widget = QWidget()
        market_layout = QVBoxLayout(market_widget)
        
        # Market search controls
        search_layout = QHBoxLayout()
        
        self.market_symbol = QComboBox()
        self.market_symbol.addItems(["BTC-USD", "ETH-USD", "APT-USD", "AAPL", "MSFT", "TSLA"])
        self.market_symbol.setEditable(True)
        search_layout.addWidget(self.market_symbol)
        
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_market)
        search_layout.addWidget(search_btn)
        
        add_watchlist_btn = QPushButton("+ Add to Watchlist")
        add_watchlist_btn.clicked.connect(self.add_to_watchlist)
        search_layout.addWidget(add_watchlist_btn)
        
        market_layout.addLayout(search_layout)
        
        # Watchlist table
        self.watchlist = QTableWidget()
        self.watchlist.setColumnCount(5)
        self.watchlist.setHorizontalHeaderLabels(["Symbol", "Price", "Change", "Volume", "Actions"])
        self.watchlist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.watchlist.verticalHeader().setVisible(False)
        
        # Add some sample data
        self.populate_sample_watchlist()
        
        market_layout.addWidget(self.watchlist)
        
        market_section.add_widget(market_widget)
        middle_row.addWidget(market_section)
        
        # Recent trades section
        trades_section = CollapsibleSection("Recent Trades")
        trades_widget = QWidget()
        trades_layout = QVBoxLayout(trades_widget)
        
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(5)
        self.trades_table.setHorizontalHeaderLabels(["Date", "Symbol", "Type", "Price", "Size"])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trades_table.verticalHeader().setVisible(False)
        
        trades_layout.addWidget(self.trades_table)
        
        trades_section.add_widget(trades_widget)
        middle_row.addWidget(trades_section)
        
        dashboard_layout.addLayout(middle_row)
        
        # Bottom row - Performance metrics
        performance_section = CollapsibleSection("Performance Metrics")
        performance_widget = QWidget()
        perf_layout = QHBoxLayout(performance_widget)
        
        # Add various performance metric widgets
        metrics = [
            ("Win Rate", "0%", "green"),
            ("Loss Rate", "0%", "red"),
            ("Profit Factor", "0", "blue"),
            ("Sharpe Ratio", "0", "blue"),
            ("Max Drawdown", "0%", "red"),
            ("Avg Holding Time", "0 days", "gray")
        ]
        
        for title, value, color in metrics:
            metric_widget = self.create_metric_widget(title, value, color)
            perf_layout.addWidget(metric_widget)
        
        performance_section.add_widget(performance_widget)
        dashboard_layout.addWidget(performance_section)
        
        scroll.setWidget(dashboard_content)
        
        # Main layout for dashboard tab
        main_layout = QVBoxLayout(dashboard_tab)
        main_layout.addWidget(scroll)
        
        # Add the tab
        self.tabs.addTab(dashboard_tab, "Dashboard")
    
    def setup_backtest_tab(self):
        """Setup the backtesting tab"""
        backtest_tab = QWidget()
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        backtest_content = QWidget()
        backtest_layout = QVBoxLayout(backtest_content)
        
        # Parameters section
        params_section = CollapsibleSection("Backtest Parameters")
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        
        # Form layout for parameters
        form_layout = QFormLayout()
        
        # Symbol
        self.backtest_symbol = QComboBox()
        self.backtest_symbol.addItems(["BTC-USD", "ETH-USD", "APT-USD", "AAPL", "MSFT", "TSLA", "SPY"])
        self.backtest_symbol.setEditable(True)
        form_layout.addRow("Symbol:", self.backtest_symbol)
        
        # Date range
        date_layout = QHBoxLayout()
        
        self.backtest_start = QDateEdit()
        self.backtest_start.setDate(QDate.currentDate().addMonths(-3))
        self.backtest_start.setCalendarPopup(True)
        date_layout.addWidget(QLabel("Start:"))
        date_layout.addWidget(self.backtest_start)
        
        self.backtest_end = QDateEdit()
        self.backtest_end.setDate(QDate.currentDate())
        self.backtest_end.setCalendarPopup(True)
        date_layout.addWidget(QLabel("End:"))
        date_layout.addWidget(self.backtest_end)
        
        form_layout.addRow("Date Range:", date_layout)
        
        # Initial capital
        self.backtest_capital = QDoubleSpinBox()
        self.backtest_capital.setRange(100, 1000000)
        self.backtest_capital.setValue(10000)
        self.backtest_capital.setPrefix("$")
        # self.backtest_capital.setGroupSeparator(",")
        form_layout.addRow("Initial Capital:", self.backtest_capital)
        
        params_layout.addLayout(form_layout)
        
        # Strategy options section
        strategy_box = QGroupBox("Strategy Options")
        strategy_layout = QVBoxLayout(strategy_box)
        
        # Strategy type
        self.strategy_type = QComboBox()
        self.strategy_type.addItems(["Moving Average Crossover", "RSI", "Bollinger Bands", "Custom"])
        strategy_layout.addWidget(QLabel("Strategy:"))
        strategy_layout.addWidget(self.strategy_type)
        
        # Strategy parameters (example for MA Crossover)
        self.fast_ma = QSpinBox()
        self.fast_ma.setRange(1, 200)
        self.fast_ma.setValue(20)
        strategy_layout.addWidget(QLabel("Fast MA Period:"))
        strategy_layout.addWidget(self.fast_ma)
        
        self.slow_ma = QSpinBox()
        self.slow_ma.setRange(1, 200)
        self.slow_ma.setValue(50)
        strategy_layout.addWidget(QLabel("Slow MA Period:"))
        strategy_layout.addWidget(self.slow_ma)
        
        # Advanced options
        self.use_stop_loss = QCheckBox("Use Stop Loss")
        self.use_stop_loss.setChecked(True)
        strategy_layout.addWidget(self.use_stop_loss)
        
        self.use_trailing_stop = QCheckBox("Use Trailing Stop")
        self.use_trailing_stop.setChecked(True)
        strategy_layout.addWidget(self.use_trailing_stop)
        
        params_layout.addWidget(strategy_box)
        
        # Run button
        run_btn = QPushButton("Run Backtest")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self.run_backtest)
        params_layout.addWidget(run_btn)
        
        # Progress bar
        self.backtest_progress = QProgressBar()
        params_layout.addWidget(self.backtest_progress)
        
        params_section.add_widget(params_widget)
        backtest_layout.addWidget(params_section)
        
        # Results section with splitter
        results_splitter = QSplitter(Qt.Vertical)
        
        # Chart section
        chart_section = CollapsibleSection("Backtest Results Chart")
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        
        self.backtest_chart = MplCanvas(width=8, height=5, dpi=100)
        chart_layout.addWidget(self.backtest_chart)
        
        chart_section.add_widget(chart_widget)
        results_splitter.addWidget(chart_section)
        
        # Metrics section
        metrics_section = CollapsibleSection("Performance Metrics")
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # Key metrics in a row
        key_metrics_layout = QHBoxLayout()
        
        self.backtest_return = self.create_metric_widget("Total Return", "0.00%", "blue")
        key_metrics_layout.addWidget(self.backtest_return)
        
        self.backtest_sharpe = self.create_metric_widget("Sharpe Ratio", "0.00", "blue")
        key_metrics_layout.addWidget(self.backtest_sharpe)
        
        self.backtest_drawdown = self.create_metric_widget("Max Drawdown", "0.00%", "red")
        key_metrics_layout.addWidget(self.backtest_drawdown)
        
        self.backtest_win_rate = self.create_metric_widget("Win Rate", "0.00%", "green")
        key_metrics_layout.addWidget(self.backtest_win_rate)
        
        metrics_layout.addLayout(key_metrics_layout)
        
        # Trades table
        self.backtest_trades = QTableWidget()
        self.backtest_trades.setColumnCount(6)
        self.backtest_trades.setHorizontalHeaderLabels(["Date", "Type", "Price", "Size", "Value", "P&L"])
        self.backtest_trades.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_layout.addWidget(self.backtest_trades)
        
        metrics_section.add_widget(metrics_widget)
        results_splitter.addWidget(metrics_section)
        
        backtest_layout.addWidget(results_splitter, 1)
        
        scroll.setWidget(backtest_content)
        
        # Main layout
        main_layout = QVBoxLayout(backtest_tab)
        main_layout.addWidget(scroll)
        
        # Add tab
        self.tabs.addTab(backtest_tab, "Backtester")
    
    def setup_trading_tab(self):
        """Setup the trading tab"""
        trading_tab = QWidget()
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        trading_content = QWidget()
        trading_layout = QVBoxLayout(trading_content)
        
        # Top row - Market data and order entry
        top_row = QHBoxLayout()
        
        # Market data section
        market_section = CollapsibleSection("Market Data")
        market_widget = QWidget()
        market_layout = QVBoxLayout(market_widget)
        
        # Symbol selection and price
        symbol_layout = QHBoxLayout()
        
        self.trading_symbol = QComboBox()
        self.trading_symbol.addItems(["BTC-USD", "ETH-USD", "APT-USD", "AAPL", "MSFT"])
        self.trading_symbol.setEditable(True)
        symbol_layout.addWidget(self.trading_symbol)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_price)
        symbol_layout.addWidget(refresh_btn)
        
        self.current_price = QLabel("$0.00")
        self.current_price.setFont(QFont("Arial", 16, QFont.Bold))
        self.current_price.setAlignment(Qt.AlignCenter)
        
        market_layout.addLayout(symbol_layout)
        market_layout.addWidget(self.current_price)
        
        # Price chart
        self.price_chart = MplCanvas(width=6, height=4, dpi=100)
        market_layout.addWidget(self.price_chart, 1)
        
        market_section.add_widget(market_widget)
        top_row.addWidget(market_section, 2)
        
        # Order entry section
        order_section = CollapsibleSection("Order Entry")
        order_widget = QWidget()
        order_layout = QVBoxLayout(order_widget)
        
        # Order form
        order_form = QFormLayout()
        
        self.order_type = QComboBox()
        self.order_type.addItems(["Market", "Limit", "Stop", "Stop Limit"])
        order_form.addRow("Order Type:", self.order_type)
        
        self.order_side = QComboBox()
        self.order_side.addItems(["Buy", "Sell"])
        order_form.addRow("Side:", self.order_side)
        
        self.order_quantity = QDoubleSpinBox()
        self.order_quantity.setRange(0.001, 1000000)
        self.order_quantity.setValue(1)
        self.order_quantity.setDecimals(3)
        order_form.addRow("Quantity:", self.order_quantity)
        
        self.order_price = QDoubleSpinBox()
        self.order_price.setRange(0.001, 1000000)
        self.order_price.setValue(0)
        self.order_price.setPrefix("$")
        self.order_price.setDecimals(2)
        order_form.addRow("Price:", self.order_price)
        
        # Advanced order options
        advanced_options = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout(advanced_options)
        
        self.use_stop_loss_trade = QCheckBox()
        self.use_stop_loss_trade.setChecked(True)
        advanced_layout.addRow("Use Stop Loss:", self.use_stop_loss_trade)
        
        self.stop_loss_value = QDoubleSpinBox()
        self.stop_loss_value.setRange(0.1, 20)
        self.stop_loss_value.setValue(5)
        self.stop_loss_value.setSuffix("%")
        advanced_layout.addRow("Stop Loss:", self.stop_loss_value)
        
        self.use_take_profit = QCheckBox()
        self.use_take_profit.setChecked(True)
        advanced_layout.addRow("Use Take Profit:", self.use_take_profit)
        
        self.take_profit_value = QDoubleSpinBox()
        self.take_profit_value.setRange(0.1, 50)
        self.take_profit_value.setValue(10)
        self.take_profit_value.setSuffix("%")
        advanced_layout.addRow("Take Profit:", self.take_profit_value)
        
        order_layout.addLayout(order_form)
        order_layout.addWidget(advanced_options)
        
        # Submit button
        submit_order_btn = QPushButton("Submit Order")
        submit_order_btn.setMinimumHeight(40)
        submit_order_btn.clicked.connect(self.submit_order)
        order_layout.addWidget(submit_order_btn)
        
        order_section.add_widget(order_widget)
        top_row.addWidget(order_section, 1)
        
        trading_layout.addLayout(top_row)
        
        # Bottom row - Open positions and order history
        bottom_row = QSplitter(Qt.Vertical)
        
        # Open positions section
        positions_section = CollapsibleSection("Open Positions")
        positions_widget = QWidget()
        positions_layout = QVBoxLayout(positions_widget)
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)
        self.positions_table.setHorizontalHeaderLabels(
            ["Symbol", "Side", "Size", "Entry Price", "Current Price", "P&L"]
        )
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        positions_layout.addWidget(self.positions_table)
        
        positions_section.add_widget(positions_widget)
        bottom_row.addWidget(positions_section)
        
        # Order history section
        history_section = CollapsibleSection("Order History")
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        
        self.order_history = QTableWidget()
        self.order_history.setColumnCount(7)
        self.order_history.setHorizontalHeaderLabels(
            ["Date", "Symbol", "Type", "Side", "Price", "Size", "Status"]
        )
        self.order_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        history_layout.addWidget(self.order_history)
        
        history_section.add_widget(history_widget)
        bottom_row.addWidget(history_section)
        
        trading_layout.addWidget(bottom_row, 1)
        
        scroll.setWidget(trading_content)
        
        # Main layout
        main_layout = QVBoxLayout(trading_tab)
        main_layout.addWidget(scroll)
        
        # Add tab
        self.tabs.addTab(trading_tab, "Trading")
    
    def setup_wallet_tab(self):
        """Setup the wallet management tab"""
        wallet_tab = QWidget()
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        wallet_content = QWidget()
        wallet_layout = QVBoxLayout(wallet_content)
        
        # Wallet info section
        wallet_info = CollapsibleSection("Wallet Information")
        info_widget = QWidget()
        info_layout = QFormLayout(info_widget)
        
        self.wallet_address_field = QLineEdit()
        self.wallet_address_field.setReadOnly(True)
        info_layout.addRow("Address:", self.wallet_address_field)
        
        self.wallet_balance_field = QLabel("0.00 APT")
        info_layout.addRow("Balance:", self.wallet_balance_field)
        
        wallet_info.add_widget(info_widget)
        wallet_layout.addWidget(wallet_info)
        
        # Transfer section
        transfer_section = CollapsibleSection("Transfer Funds")
        transfer_widget = QWidget()
        transfer_layout = QFormLayout(transfer_widget)
        
        self.transfer_recipient = QLineEdit()
        transfer_layout.addRow("Recipient:", self.transfer_recipient)
        
        self.transfer_amount = QDoubleSpinBox()
        self.transfer_amount.setRange(0.001, 1000)
        self.transfer_amount.setValue(0.1)
        self.transfer_amount.setDecimals(3)
        self.transfer_amount.setSuffix(" APT")
        transfer_layout.addRow("Amount:", self.transfer_amount)
        
        # Transfer button
        transfer_btn = QPushButton("Execute Transfer")
        transfer_btn.clicked.connect(self.execute_transfer_action)
        transfer_layout.addWidget(transfer_btn)
        
        transfer_section.add_widget(transfer_widget)
        wallet_layout.addWidget(transfer_section)
        
        # Transaction history section
        history_section = CollapsibleSection("Transaction History")
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        
        # Filter controls
        filter_layout = QHBoxLayout()
        
        self.tx_filter = QComboBox()
        self.tx_filter.addItems(["All", "Deposits", "Withdrawals", "Trades"])
        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(self.tx_filter)
        
        refresh_tx_btn = QPushButton("Refresh")
        refresh_tx_btn.clicked.connect(self.refresh_transactions)
        filter_layout.addWidget(refresh_tx_btn)
        
        history_layout.addLayout(filter_layout)
        
        # Transactions table
        self.tx_table = QTableWidget()
        self.tx_table.setColumnCount(5)
        self.tx_table.setHorizontalHeaderLabels(
            ["Date", "Type", "Amount", "Recipient", "Status"]
        )
        self.tx_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        history_layout.addWidget(self.tx_table)
        
        history_section.add_widget(history_widget)
        wallet_layout.addWidget(history_section)
        
        scroll.setWidget(wallet_content)
        
        # Main layout
        main_layout = QVBoxLayout(wallet_tab)
        main_layout.addWidget(scroll)
        
        # Add tab
        self.tabs.addTab(wallet_tab, "Wallet")
    
    def setup_logs_tab(self):
        """Setup the logs tab"""
        logs_tab = QWidget()
        
        layout = QVBoxLayout(logs_tab)
        
        # Create tabs for different logs
        log_tabs = QTabWidget()
        
        # Trading log
        trading_log_widget = QWidget()
        trading_log_layout = QVBoxLayout(trading_log_widget)
        
        self.trading_log = QTextEdit()
        self.trading_log.setReadOnly(True)
        trading_log_layout.addWidget(self.trading_log)
        
        log_tabs.addTab(trading_log_widget, "Trading Log")
        
        # System log
        system_log_widget = QWidget()
        system_log_layout = QVBoxLayout(system_log_widget)
        
        self.system_log = QTextEdit()
        self.system_log.setReadOnly(True)
        system_log_layout.addWidget(self.system_log)
        
        log_tabs.addTab(system_log_widget, "System Log")
        
        # Error log
        error_log_widget = QWidget()
        error_log_layout = QVBoxLayout(error_log_widget)
        
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        error_log_layout.addWidget(self.error_log)
        
        log_tabs.addTab(error_log_widget, "Error Log")
        
        layout.addWidget(log_tabs)
        
        # Clear logs button
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.clear_logs)
        layout.addWidget(clear_btn)
        
        # Add tab
        self.tabs.addTab(logs_tab, "Logs")
    
    def create_metric_widget(self, title, value, color="blue"):
        """Create a widget to display a metric with title and value"""
        widget = QGroupBox(title)
        layout = QVBoxLayout(widget)
        
        label = QLabel(value)
        label.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 14, QFont.Bold)
        label.setFont(font)
        
        # Set color based on parameter
        if color == "green":
            label.setStyleSheet("color: #28a745;")
        elif color == "red":
            label.setStyleSheet("color: #dc3545;")
        elif color == "blue":
            label.setStyleSheet("color: #007bff;")
        elif color == "gray":
            label.setStyleSheet("color: #6c757d;")
        
        layout.addWidget(label)
        
        return widget
    
    def populate_sample_watchlist(self):
        """Populate sample watchlist data"""
        symbols = ["BTC-USD", "ETH-USD", "APT-USD", "AAPL", "MSFT"]
        prices = ["$50,431.20", "$2,890.15", "$8.72", "$187.45", "$398.25"]
        changes = ["+2.3%", "-0.8%", "+5.2%", "+0.5%", "-1.2%"]
        volumes = ["28.5B", "12.8B", "124M", "54.2M", "22.1M"]
        
        self.watchlist.setRowCount(len(symbols))
        
        for i, (symbol, price, change, volume) in enumerate(zip(symbols, prices, changes, volumes)):
            self.watchlist.setItem(i, 0, QTableWidgetItem(symbol))
            self.watchlist.setItem(i, 1, QTableWidgetItem(price))
            self.watchlist.setItem(i, 2, QTableWidgetItem(change))
            self.watchlist.setItem(i, 3, QTableWidgetItem(volume))
            
            # Add action button
            trade_btn = QPushButton("Trade")
            trade_btn.clicked.connect(lambda _, s=symbol: self.quick_trade(s))
            self.watchlist.setCellWidget(i, 4, trade_btn)
            
            # Color the change cell based on value
            if "+" in change:
                self.watchlist.item(i, 2).setBackground(QColor(200, 255, 200))
            else:
                self.watchlist.item(i, 2).setBackground(QColor(255, 200, 200))
    
    def load_wallet_info(self):
        """Load wallet information"""
        try:
            private_key, address, public_key = load_or_create_wallet()
            self.wallet_info = {
                "private_key": private_key,
                "address": address,
                "public_key": public_key
            }
            
            # Update UI
            self.wallet_address.setText(address)
            self.wallet_address_field.setText(address)
            
            # Initialize portfolio tracker
            self.portfolio_tracker = PortfolioTracker()
            
            # Get balance
            self.get_wallet_balance()
            
            # Log success
            self.log("Wallet loaded successfully", "system")
            
        except Exception as e:
            self.log(f"Error loading wallet: {str(e)}", "error")
    
    # def get_wallet_balance(self):
    #     """Get wallet balance"""
    #     if not self.wallet_info:
    #         return
        
    #     try:
    #         # For demo, simulate balance
    #         import random
    #         balance = random.uniform(100, 500)
            
    #         self.wallet_balance.setText(f"{balance:.2f} APT")
    #         self.wallet_balance_field.setText(f"{balance:.2f} APT")
            
    #     except Exception as e:
    #         self.log(f"Error fetching balance: {str(e)}", "error")
    
    def get_wallet_balance(self):
        """Get and update wallet balance"""
        if not self.wallet_info or not self.rest_client:
            return

        worker = BacktestWorker(
            check_balance,
            self.rest_client,
            AccountAddress.from_str(self.wallet_info["address"]),
        )
        worker.finished.connect(self.update_balance_display)
        worker.error.connect(
            lambda e: self.log_message(f"Error getting balance: {e}", error=True)
        )
        worker.start()
    def refresh_dashboard(self):
        """Refresh dashboard data"""
        self.get_wallet_balance()
        self.update_portfolio_chart()
        self.log("Dashboard refreshed", "system")
    
    def update_portfolio_chart(self):
        """Update the portfolio chart with demo data"""
        # Generate sample portfolio value history
        import numpy as np
        import pandas as pd
        import matplotlib.dates as mdates
        
        # Generate random portfolio values with an upward trend
        np.random.seed(42)  # For reproducibility
        days = 90
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        
        base = 10000
        random_walk = np.random.normal(0.001, 0.02, days).cumsum()
        trend = np.linspace(0, 0.2, days)  # Add slight upward trend
        values = base * (1 + random_walk + trend)
        
        # Clear chart and plot
        self.portfolio_chart.axes.clear()
        self.portfolio_chart.axes.plot(dates, values, 'b-', linewidth=2)
        
        # Format axes
        self.portfolio_chart.axes.set_title('Portfolio Value History')
        self.portfolio_chart.axes.set_xlabel('Date')
        self.portfolio_chart.axes.set_ylabel('Value ($)')
        
        # Format dates nicely
        self.portfolio_chart.axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.portfolio_chart.axes.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        
        # Rotate date labels
        plt = self.portfolio_chart.figure.canvas
        plt.draw_idle()
    
    def show_deposit_dialog(self):
        """Show deposit dialog"""
        QMessageBox.information(self, "Deposit Funds", "This feature is not implemented in the demo.")
    
    def show_withdraw_dialog(self):
        """Show withdrawal dialog"""
        QMessageBox.information(self, "Withdraw Funds", "This feature is not implemented in the demo.")
    
    def search_market(self):
        """Search for a market symbol"""
        symbol = self.market_symbol.currentText()
        self.log(f"Searching for {symbol}", "system")
        
        # In a real implementation, this would fetch actual market data
        self.refresh_price()
    
    def add_to_watchlist(self):
        """Add current symbol to watchlist"""
        symbol = self.market_symbol.currentText()
        
        # Check if symbol already exists
        for row in range(self.watchlist.rowCount()):
            if self.watchlist.item(row, 0).text() == symbol:
                self.log(f"{symbol} already in watchlist", "system")
                return
        
        # Add new row
        row = self.watchlist.rowCount()
        self.watchlist.insertRow(row)
        
        # Set sample data
        import random
        price = random.uniform(10, 1000)
        change = random.uniform(-5, 5)
        volume = random.uniform(1, 100)
        
        self.watchlist.setItem(row, 0, QTableWidgetItem(symbol))
        self.watchlist.setItem(row, 1, QTableWidgetItem(f"${price:.2f}"))
        self.watchlist.setItem(row, 2, QTableWidgetItem(f"{change:.2f}%"))
        self.watchlist.setItem(row, 3, QTableWidgetItem(f"{volume:.1f}M"))
        
        # Add action button
        trade_btn = QPushButton("Trade")
        trade_btn.clicked.connect(lambda _, s=symbol: self.quick_trade(s))
        self.watchlist.setCellWidget(row, 4, trade_btn)
        
        # Color the change cell
        if change > 0:
            self.watchlist.item(row, 2).setBackground(QColor(200, 255, 200))
        else:
            self.watchlist.item(row, 2).setBackground(QColor(255, 200, 200))
        
        self.log(f"Added {symbol} to watchlist", "system")
    
    def quick_trade(self, symbol):
        """Quick access to trading for a symbol"""
        # Switch to trading tab
        self.tabs.setCurrentIndex(2)  # Trading tab
        
        # Set symbol
        index = self.trading_symbol.findText(symbol)
        if index >= 0:
            self.trading_symbol.setCurrentIndex(index)
        else:
            self.trading_symbol.setEditText(symbol)
        
        # Refresh price
        self.refresh_price()
    
    def run_backtest(self):
        """Run backtest with specified parameters"""
        symbol = self.backtest_symbol.currentText()
        start_date = self.backtest_start.date().toString("yyyy-MM-dd")
        end_date = self.backtest_end.date().toString("yyyy-MM-dd")
        initial_capital = self.backtest_capital.value()
        
        self.log(f"Starting backtest for {symbol} from {start_date} to {end_date}", "system")
        
        # Show indeterminate progress
        self.backtest_progress.setRange(0, 0)
        
        # Create and run worker thread
        self.backtest_worker = BacktestWorker(symbol, start_date, end_date, initial_capital)
        self.backtest_worker.finished.connect(self.display_backtest_results)
        self.backtest_worker.start()
    
    def display_backtest_results(self, history, metrics):
        """Display backtest results"""
        # Reset progress bar
        self.backtest_progress.setRange(0, 100)
        self.backtest_progress.setValue(100)
        
        # Check for errors
        if "error" in metrics:
            self.log(f"Backtest error: {metrics['error']}", "error")
            return
        
        # Save reference to backtester
        self.backtester = self.backtest_worker.backtester
        
        # Update metrics display
        self.backtest_return.findChild(QLabel).setText(f"{metrics.get('total_return', 0):.2%}")
        self.backtest_sharpe.findChild(QLabel).setText(f"{metrics.get('sharpe', 0):.2f}")
        self.backtest_drawdown.findChild(QLabel).setText(f"{metrics.get('max_drawdown', 0):.2%}")
        self.backtest_win_rate.findChild(QLabel).setText(f"{metrics.get('win_rate', 0):.2%}")
        
        # Plot results
        if self.backtester:
            fig = self.backtester.plot_results(history)
            self.backtest_chart.figure = fig
            self.backtest_chart.draw()
        
        # Update trades table
        self.update_backtest_trades()
        
        self.log(f"Backtest completed with {metrics.get('total_return', 0):.2%} return", "system")
    
    def update_backtest_trades(self):
        """Update the backtest trades table"""
        if not self.backtester or not self.backtester.portfolio.get("transactions"):
            return
        
        transactions = self.backtester.portfolio["transactions"]
        self.backtest_trades.setRowCount(len(transactions))
        
        for i, (tx_type, price, qty, timestamp) in enumerate(transactions):
            self.backtest_trades.setItem(i, 0, QTableWidgetItem(timestamp.strftime("%Y-%m-%d")))
            self.backtest_trades.setItem(i, 1, QTableWidgetItem(tx_type))
            self.backtest_trades.setItem(i, 2, QTableWidgetItem(f"${price:.2f}"))
            self.backtest_trades.setItem(i, 3, QTableWidgetItem(f"{qty:.6f}"))
            self.backtest_trades.setItem(i, 4, QTableWidgetItem(f"${price * qty:.2f}"))
            
            # Calculate P&L (simple approach)
            if i > 0 and tx_type == "SELL" and transactions[i-1][0] == "BUY":
                prev_price = transactions[i-1][1]
                pnl = (price - prev_price) * qty
                self.backtest_trades.setItem(i, 5, QTableWidgetItem(f"${pnl:.2f}"))
                
                # Color code P&L
                if pnl > 0:
                    self.backtest_trades.item(i, 5).setBackground(QColor(200, 255, 200))
                else:
                    self.backtest_trades.item(i, 5).setBackground(QColor(255, 200, 200))
    
    def refresh_price(self):
        """Refresh current price of selected symbol"""
        symbol = self.trading_symbol.currentText()
        self.log(f"Fetching price for {symbol}", "system")
        
        try:
            # For demo purposes, generate a random price
            import random
            price = random.uniform(10, 5000)
            self.current_price.setText(f"${price:.2f}")
            
            # Update price chart with mock data
            self.update_price_chart(symbol)
            
        except Exception as e:
            self.log(f"Error fetching price: {str(e)}", "error")
    
    def update_price_chart(self, symbol):
        """Update price chart with mock data"""
        import numpy as np
        import pandas as pd
        import matplotlib.dates as mdates
        
        # Generate random price history
        np.random.seed(hash(symbol) % 100)  # Seed based on symbol for consistent results
        days = 30
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        
        # Base price varies by symbol
        if "BTC" in symbol:
            base = 50000
            volatility = 0.03
        elif "ETH" in symbol:
            base = 3000
            volatility = 0.025
        elif "APT" in symbol:
            base = 10
            volatility = 0.04
        else:
            base = 100
            volatility = 0.015
        
        # Generate price movement
        random_walk = np.random.normal(0, volatility, days).cumsum()
        values = base * (1 + random_walk)
        
        # Clear chart and plot
        self.price_chart.axes.clear()
        self.price_chart.axes.plot(dates, values, 'b-', linewidth=2)
        
        # Format axes
        self.price_chart.axes.set_title(f'{symbol} Price History')
        self.price_chart.axes.set_xlabel('Date')
        self.price_chart.axes.set_ylabel('Price ($)')
        
        # Format dates
        self.price_chart.axes.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        self.price_chart.axes.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        
        # Draw
        self.price_chart.figure.tight_layout()
        self.price_chart.draw()
    
    def submit_order(self):
        """Submit trade order"""
        symbol = self.trading_symbol.currentText()
        order_type = self.order_type.currentText()
        side = self.order_side.currentText()
        quantity = self.order_quantity.value()
        price = self.order_price.value() if order_type != "Market" else float(self.current_price.text().replace("$", ""))
        
        self.log(f"Submitting {side} order for {quantity} {symbol} at ${price:.2f}", "trading")
        
        # Add to order history
        row = self.order_history.rowCount()
        self.order_history.insertRow(row)
        
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.order_history.setItem(row, 0, QTableWidgetItem(now))
        self.order_history.setItem(row, 1, QTableWidgetItem(symbol))
        self.order_history.setItem(row, 2, QTableWidgetItem(order_type))
        self.order_history.setItem(row, 3, QTableWidgetItem(side))
        self.order_history.setItem(row, 4, QTableWidgetItem(f"${price:.2f}"))
        self.order_history.setItem(row, 5, QTableWidgetItem(f"{quantity:.6f}"))
        self.order_history.setItem(row, 6, QTableWidgetItem("Filled"))
        
        # Update positions
        self.update_positions(symbol, side, quantity, price)
        
        # Show confirmation
        QMessageBox.information(self, "Order Submitted", f"Your {side} order for {quantity} {symbol} has been filled at ${price:.2f}")
    
    def update_positions(self, symbol, side, quantity, price):
        """Update positions table after trade"""
        # Check if position already exists
        position_exists = False
        for row in range(self.positions_table.rowCount()):
            if self.positions_table.item(row, 0).text() == symbol:
                position_exists = True
                
                # Update position
                current_size = float(self.positions_table.item(row, 2).text())
                entry_price = float(self.positions_table.item(row, 3).text().replace("$", ""))
                
                if side == "Buy":
                    new_size = current_size + quantity
                    # Update average entry price
                    new_entry = ((current_size * entry_price) + (quantity * price)) / new_size
                    self.positions_table.setItem(row, 3, QTableWidgetItem(f"${new_entry:.2f}"))
                else:  # Sell
                    new_size = current_size - quantity
                    # Keep entry price the same
                
                if new_size <= 0:
                    self.positions_table.removeRow(row)
                else:
                    self.positions_table.setItem(row, 2, QTableWidgetItem(f"{new_size:.6f}"))
                    
                    # Update P&L
                    current_price = price  # Use execution price
                    pnl = (current_price - new_entry) * new_size
                    pnl_pct = (current_price / new_entry - 1) * 100
                    
                    self.positions_table.setItem(row, 4, QTableWidgetItem(f"${current_price:.2f}"))
                    self.positions_table.setItem(row, 5, QTableWidgetItem(f"${pnl:.2f} ({pnl_pct:.2f}%)"))
                    
                    # Color code P&L
                    if pnl > 0:
                        self.positions_table.item(row, 5).setBackground(QColor(200, 255, 200))
                    else:
                        self.positions_table.item(row, 5).setBackground(QColor(255, 200, 200))
                
                break
        
        # If position doesn't exist and buying, add new position
        if not position_exists and side == "Buy":
            row = self.positions_table.rowCount()
            self.positions_table.insertRow(row)
            
            self.positions_table.setItem(row, 0, QTableWidgetItem(symbol))
            self.positions_table.setItem(row, 1, QTableWidgetItem("Long"))
            self.positions_table.setItem(row, 2, QTableWidgetItem(f"{quantity:.6f}"))
            self.positions_table.setItem(row, 3, QTableWidgetItem(f"${price:.2f}"))
            self.positions_table.setItem(row, 4, QTableWidgetItem(f"${price:.2f}"))
            self.positions_table.setItem(row, 5, QTableWidgetItem("$0.00 (0.00%)"))
    
    def execute_transfer_action(self):
        """Execute a transfer of funds"""
        if not self.wallet_info:
            self.log("No wallet loaded", "error")
            return
        
        recipient = self.transfer_recipient.text().strip()
        if not recipient:
            self.log("Please enter a recipient address", "error")
            return
        
        amount = self.transfer_amount.value()
        
        # For demo, simulate transfer
        self.log(f"Transferring {amount:.3f} APT to {recipient}", "trading")
        
        # Add to transaction history
        self.update_transfer_history(recipient, amount)
        
        QMessageBox.information(self, "Transfer Complete", f"Successfully transferred {amount:.3f} APT to {recipient}")
    
    def update_transfer_history(self, recipient, amount):
        """Update transfer history in the tx_table"""
        row = self.tx_table.rowCount()
        self.tx_table.insertRow(row)
        
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.tx_table.setItem(row, 0, QTableWidgetItem(now))
        self.tx_table.setItem(row, 1, QTableWidgetItem("Transfer"))
        self.tx_table.setItem(row, 2, QTableWidgetItem(f"{amount:.3f} APT"))
        self.tx_table.setItem(row, 3, QTableWidgetItem(recipient[:10] + "..."))
        self.tx_table.setItem(row, 4, QTableWidgetItem("Completed"))
    
    def refresh_transactions(self):
        """Refresh transaction history"""
        self.log("Refreshing transaction history", "system")
        
        # In a real implementation, this would fetch transactions from the blockchain
        filter_type = self.tx_filter.currentText()
        # Add implementation here
    
    def log(self, message, log_type="system"):
        """Log a message to the appropriate log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | {message}"
        
        if log_type == "trading":
            self.trading_log.append(log_entry)
            # Scroll to bottom
            self.trading_log.verticalScrollBar().setValue(
                self.trading_log.verticalScrollBar().maximum()
            )
        elif log_type == "error":
            self.error_log.append(log_entry)
            # Highlight errors
            self.error_log.append('<span style="color: red;">Error occurred</span>')
            # Scroll to bottom
            self.error_log.verticalScrollBar().setValue(
                self.error_log.verticalScrollBar().maximum()
            )
        else:  # system logs
            self.system_log.append(log_entry)
            # Scroll to bottom
            self.system_log.verticalScrollBar().setValue(
                self.system_log.verticalScrollBar().maximum()
            )
    
    def clear_logs(self):
        """Clear all logs"""
        self.trading_log.clear()
        self.system_log.clear()
        self.error_log.clear()
        self.log("Logs cleared", "system")
    
    def closeEvent(self, event):
        """Clean up before closing"""
        # Wait for any running threads to finish
        if hasattr(self, 'backtest_worker') and self.backtest_worker.isRunning():
            self.backtest_worker.terminate()
            self.backtest_worker.wait()
        
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Apply stylesheet for a modern look
    app.setStyle("Fusion")
    
    # Set app-wide stylesheet
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            margin-top: 0.5em;
            padding-top: 0.5em;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QPushButton {
            background-color: #f8f9fa;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: #e9ecef;
        }
        QPushButton:pressed {
            background-color: #dee2e6;
        }
        QTableWidget {
            border: 1px solid #d0d0d0;
            gridline-color: #f0f0f0;
        }
        QHeaderView::section {
            background-color: #f8f9fa;
            border: 1px solid #d0d0d0;
            padding: 4px;
        }
    """)
    
    window = FlexibleDashboard()
    window.show()
    
    sys.exit(app.exec_())