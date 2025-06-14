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
from dotenv import dotenv_values
from aptos_sdk_wrapper import get_balance
from agents import get_balance_in_apt_sync
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


# Network configuration
NODE_URL = "https://fullnode.devnet.aptoslabs.com/v1"
FAUCET_URL = "https://faucet.devnet.aptoslabs.com"


def get_wallet_path():
    """Return the path to the wallet file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "aptos_wallet.json")


def load_or_create_wallet():
    """Load existing wallet or create a new one"""
    wallet_file = get_wallet_path()

    if os.path.exists(wallet_file):
        try:
            with open(wallet_file, "r") as f:
                wallet = json.load(f)
                private_key = wallet["private_key"]
                address = wallet["address"]
                public_key = wallet["public_key"]
                print(
                    f"Existing Account:\nPrivate Key: {private_key}\nAddress: {address}"
                )
                return private_key, address, public_key
        except json.JSONDecodeError:
            return create_new_wallet(wallet_file)
    else:
        return create_new_wallet(wallet_file)


def create_new_wallet(wallet_file):
    """Create a new wallet and save it to file"""
    os.makedirs(os.path.dirname(wallet_file), exist_ok=True)

    account = Account.generate()
    private_key = account.private_key
    address = account.address()
    public_key = account.public_key

    print(f"New Account:\nPrivate Key: {private_key}\nAddress: {address}")

    with open(wallet_file, "w") as f:
        json.dump(
            {"private_key": private_key, "address": address, "public_key": public_key},
            f,
            indent=4,
        )

    print(f"New account generated and saved to {wallet_file}")
    return private_key, address, public_key


def fund_wallet(address, amount=100_000_000, coin_type="0x1::aptos_coin::AptosCoin"):
    """Fund a wallet using the faucet"""
    url = f"https://faucet.devnet.aptoslabs.com/mint?address={address}&amount={amount}"
    headers = {"Content-Type": "application/json"}
    data = {"amount": amount, "coin_type": coin_type}
    response = requests.post(url, headers=headers, json=data)
    if coin_type == "0x1::aptos_coin::AptosCoin":
        print(f"Funding {address} with {amount/1e8} APT...")
    else:
        print(f"Funding {address} with {amount} token...")
    print("Funded!" if response.status_code == 200 else "Failed")
    return response.status_code == 200


async def build_transaction(rest_client, sender_address, recipient_address, amount):
    """Build a transaction to transfer APT"""
    print("\n=== 1. Building the transaction ===")

    # Create the entry function payload
    entry_function = EntryFunction.natural(
        "0x1::aptos_account",  # Module address and name
        "transfer",  # Function name
        [],  # Type arguments
        [
            # Function arguments
            TransactionArgument(
                AccountAddress.from_str(recipient_address), Serializer.struct
            ),
            TransactionArgument(amount, Serializer.u64),
        ],
    )

    # Get the chain ID and sequence number
    chain_id = await rest_client.chain_id()
    account_data = await rest_client.account(sender_address)
    sequence_number = int(account_data["sequence_number"])

    # Create the raw transaction
    raw_transaction = RawTransaction(
        sender=sender_address,
        sequence_number=sequence_number,
        payload=TransactionPayload(entry_function),
        max_gas_amount=2000,
        gas_unit_price=100,
        expiration_timestamps_secs=int(time.time()) + 600,
        chain_id=chain_id,
    )

    print("Transaction built successfully")
    print(f"Sender: {raw_transaction.sender}")
    print(f"Sequence Number: {raw_transaction.sequence_number}")
    print(f"Max Gas Amount: {raw_transaction.max_gas_amount}")
    print(f"Gas Unit Price: {raw_transaction.gas_unit_price}")
    print(
        f"Expiration Timestamp: {time.ctime(raw_transaction.expiration_timestamps_secs)}"
    )

    return entry_function, sequence_number


async def simulate_transaction(rest_client, account, entry_function):
    """Simulate a transaction to estimate costs"""
    print("\n=== 2. Simulating the transaction ===")

    # Create a BCS transaction for simulation
    simulation_transaction = await rest_client.create_bcs_transaction(
        account, TransactionPayload(entry_function)
    )

    # Simulate the transaction
    simulation_result = await rest_client.simulate_transaction(
        simulation_transaction, account
    )

    # Extract results
    gas_used = int(simulation_result[0]["gas_used"])
    gas_unit_price = int(simulation_result[0]["gas_unit_price"])
    success = simulation_result[0]["success"]

    print(f"Estimated gas units: {gas_used}")
    print(f"Estimated gas cost: {gas_used * gas_unit_price} octas")
    print(f"Transaction would {'succeed' if success else 'fail'}")

    return success, gas_used, gas_unit_price


async def sign_and_submit_transaction(
    rest_client, account, entry_function, sequence_number
):
    """Sign and submit a transaction"""
    print("\n=== 3. Signing the transaction ===")

    # Sign the transaction
    signed_transaction = await rest_client.create_bcs_signed_transaction(
        account, TransactionPayload(entry_function), sequence_number=sequence_number
    )

    print("Transaction signed successfully")

    # Submit the transaction
    print("\n=== 4. Submitting the transaction ===")
    tx_hash = await rest_client.submit_bcs_transaction(signed_transaction)
    print(f"Transaction submitted with hash: {tx_hash}")

    return tx_hash


async def wait_for_transaction(rest_client, tx_hash):
    """Wait for a transaction to complete and get its status"""
    print("\n=== 5. Waiting for transaction completion ===")

    # Wait for the transaction to be processed
    await rest_client.wait_for_transaction(tx_hash)

    # Get transaction details
    transaction_details = await rest_client.transaction_by_hash(tx_hash)
    success = transaction_details["success"]
    vm_status = transaction_details["vm_status"]
    gas_used = transaction_details["gas_used"]

    print(f"Transaction completed with status: {'SUCCESS' if success else 'FAILURE'}")
    print(f"VM Status: {vm_status}")
    print(f"Gas used: {gas_used}")

    return success, vm_status, gas_used


async def check_balance(rest_client, address):
    """Check the balance of an address"""
    balance = await rest_client.account_balance(address)
    return balance


async def execute_transfer(sender_private_key, recipient_address, amount):
    """Execute a complete transfer transaction"""
    # Initialize the clients
    rest_client = RestClient(NODE_URL)

    # Load account from private key
    account = Account.load_key(sender_private_key)
    sender_address = account.address()

    # Initial balance
    initial_balance = await check_balance(rest_client, sender_address)
    print(f"Initial balance: {initial_balance} octas")

    # Build the transaction
    entry_function, sequence_number = await build_transaction(
        rest_client, sender_address, recipient_address, amount
    )

    # Simulate the transaction
    success, gas_used, gas_unit_price = await simulate_transaction(
        rest_client, account, entry_function
    )

    if not success:
        print("Transaction simulation failed. Aborting.")
        return False

    # Sign and submit the transaction
    tx_hash = await sign_and_submit_transaction(
        rest_client, account, entry_function, sequence_number
    )

    # Wait for the transaction to complete
    tx_success, vm_status, final_gas_used = await wait_for_transaction(
        rest_client, tx_hash
    )

    # Check final balance
    final_balance = await check_balance(rest_client, sender_address)
    print("\n=== Final Balances ===")
    print(
        f"Balance: {final_balance} octas (spent {initial_balance - final_balance} octas on transfer and gas)"
    )

    return tx_success


# Attempt to import private strategy for entry signals
try:
    from private_strat import get_entry_signal


except ImportError:
    # Fallback to a basic public strategy if private strategy is not available
    def get_entry_signal(
        predictor, symbol=None, current_price=None, reverse_signals=False
    ):
        # Basic placeholder strategy for public repository
        return "HOLD", 50, "Strategy code hidden in public repository", {}


# Configure logging
log_directory = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    log_directory, f"aptos_trading_{datetime.now().strftime('%Y%m%d')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Track Aptos positions and calculate profit/loss"""

    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.portfolio = {
            "cash": initial_capital,
            "positions": {},  # symbol -> {qty, avg_entry_price}
            "transactions": [],  # [type, price, qty, timestamp]
            "value_history": [],  # [{timestamp, value, cash}]
        }
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

    def record_transaction(
        self, transaction_type, symbol, price, quantity, timestamp=None
    ):
        """Record a buy or sell transaction"""
        if timestamp is None:
            timestamp = datetime.now()
        value = price * quantity

        # Add to in-memory transaction list
        self.portfolio["transactions"].append(
            (transaction_type, price, quantity, timestamp)
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
        logger.info(f"{transaction_type} executed: {quantity} {symbol} at ${price:.4f}")

        return True

    def update_position(self, symbol, price, quantity, transaction_type):
        """Update portfolio positions after a transaction"""
        if transaction_type == "BUY":
            # Add to position
            if symbol in self.portfolio["positions"]:
                current_position = self.portfolio["positions"][symbol]

                # Calculate new average entry price
                total_shares = current_position["qty"] + quantity
                new_avg_price = (
                    (current_position["qty"] * current_position["avg_entry_price"])
                    + (quantity * price)
                ) / total_shares

                self.portfolio["positions"][symbol] = {
                    "qty": total_shares,
                    "avg_entry_price": new_avg_price,
                }
            else:
                # New position
                self.portfolio["positions"][symbol] = {
                    "qty": quantity,
                    "avg_entry_price": price,
                }

            # Deduct from cash
            self.portfolio["cash"] -= price * quantity

        elif transaction_type == "SELL":
            if symbol in self.portfolio["positions"]:
                current_position = self.portfolio["positions"][symbol]

                # Reduce position
                if quantity >= current_position["qty"]:
                    # Selling entire position
                    quantity = current_position["qty"]
                    del self.portfolio["positions"][symbol]
                else:
                    # Partial sale - keep same average price
                    self.portfolio["positions"][symbol]["qty"] -= quantity

                # Add to cash
                self.portfolio["cash"] += price * quantity
            else:
                # Shorting - not supported in this basic implementation
                logger.warning(f"Short selling not supported: {quantity} {symbol}")
                return False

        # Record the transaction
        return self.record_transaction(transaction_type, symbol, price, quantity)

    def calculate_current_value(self, symbol_prices=None):
        """Calculate current portfolio value"""
        if symbol_prices is None:
            symbol_prices = {}

        position_value = 0
        for symbol, position in self.portfolio["positions"].items():
            # Use provided price or current position's entry price as fallback
            price = symbol_prices.get(symbol, position["avg_entry_price"])
            position_value += position["qty"] * price

        total_value = self.portfolio["cash"] + position_value

        # Record in history
        # timestamp is the time of the backtest not
        timestamp = datetime.now()
        # timestamp = self.portfolio["transactions"][-1][3] if self.portfolio['transactions'] else datetime.now()
        self.portfolio["value_history"].append(
            {
                "timestamp": timestamp,
                "value": total_value,
                "cash": self.portfolio["cash"],
            }
        )

        # Log portfolio value
        portfolio_record = {
            "timestamp": timestamp,
            "total_value": total_value,
            "cash": self.portfolio["cash"],
            "positions": str(
                self.portfolio["positions"]
            ),  # Convert positions dict to string
        }

        pd.DataFrame([portfolio_record]).to_csv(
            self.portfolio_log_file, mode="a", header=False, index=False
        )

        return total_value

    def get_pnl_metrics(self):
        """Calculate performance metrics"""
        if not self.portfolio["value_history"]:
            return {
                "total_return": 0,
                "unrealized_pnl": 0,
                "realized_pnl": 0,
                "win_rate": 0,
                "num_trades": 0,
            }

        # Calculate total return
        current_value = self.portfolio["value_history"][-1]["value"]
        total_return = (current_value / self.initial_capital) - 1

        # Calculate realized P&L from completed trades
        realized_pnl = 0
        buy_positions = {}
        for transaction_type, price, qty, timestamp in self.portfolio["transactions"]:
            if transaction_type == "BUY":
                # Add to open positions
                if "APT" not in buy_positions:
                    buy_positions["APT"] = []
                buy_positions["APT"].append((price, qty))
            elif transaction_type == "SELL":
                # Calculate profit for matched positions (FIFO)
                remaining_qty = qty
                while (
                    remaining_qty > 0
                    and "APT" in buy_positions
                    and buy_positions["APT"]
                ):
                    buy_price, buy_qty = buy_positions["APT"][0]

                    if buy_qty <= remaining_qty:
                        # Fully realize this buy position
                        realized_pnl += (price - buy_price) * buy_qty
                        remaining_qty -= buy_qty
                        buy_positions["APT"].pop(0)
                    else:
                        # Partially realize this buy position
                        realized_pnl += (price - buy_price) * remaining_qty
                        buy_positions["APT"][0] = (buy_price, buy_qty - remaining_qty)
                        remaining_qty = 0

        # Calculate unrealized P&L for current positions
        unrealized_pnl = 0
        for symbol, position in self.portfolio["positions"].items():
            # For simplicity, we use the last transaction price
            last_price = (
                self.portfolio["transactions"][-1][1]
                if self.portfolio["transactions"]
                else 0
            )
            unrealized_pnl += (last_price - position["avg_entry_price"]) * position[
                "qty"
            ]

        # Calculate win rate
        num_trades = len([t for t in self.portfolio["transactions"] if t[0] == "SELL"])
        winning_trades = 0
        for i in range(len(self.portfolio["transactions"])):
            if self.portfolio["transactions"][i][0] == "SELL":
                sell_price = self.portfolio["transactions"][i][1]

                # Look for matching buy transaction
                for j in range(i):
                    if self.portfolio["transactions"][j][0] == "BUY":
                        buy_price = self.portfolio["transactions"][j][1]
                        if sell_price > buy_price:
                            winning_trades += 1
                        break

        win_rate = winning_trades / num_trades if num_trades > 0 else 0

        return {
            "total_return": total_return,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "win_rate": win_rate,
            "num_trades": num_trades,
        }


# Integration with Aptos DEX (example for your Python code)
async def execute_on_chain_trade(private_key, signal, symbol, size):
    """Execute trade using on-chain DEX"""
    # Initialize account from private key
    account = Account.load_key(private_key)
    rest_client = RestClient(NODE_URL)

    # Convert symbol to token addresses
    # This mapping would need to come from a config or lookup
    token_mapping = {
        "APT-USD": {
            "base": "0x1::aptos_coin::AptosCoin",
            "quote": "0x1::usdc_coin::USDCoin",
        }
    }

    # Get token addresses
    if symbol not in token_mapping:
        logger.error(f"No token mapping for {symbol}")
        return False

    base_token = token_mapping[symbol]["base"]
    quote_token = token_mapping[symbol]["quote"]

    balance = await check_balance(rest_client, account.address)

    # Calculate amount based on size
    amount = int(balance * size)

    # Create transaction payload
    if signal == "BUY":
        direction = 1
    elif signal == "SELL":
        direction = 2
    else:
        return False

    # Create function call to your trading contract
    payload = EntryFunction.natural(
        "trading_strategy::auto_trader",  # Module address and name
        "execute_trade",  # Function name
        [],  # Type arguments
        [
            # Function arguments
            TransactionArgument(direction, Serializer.u8),
            TransactionArgument(amount, Serializer.u64),
            TransactionArgument(0, Serializer.u64),  # price (placeholder)
            TransactionArgument(AccountAddress.from_hex(base_token), Serializer.struct),
            TransactionArgument(
                AccountAddress.from_hex(quote_token), Serializer.struct
            ),
        ],
    )

    # Execute the transaction
    tx_hash = await sign_and_submit_transaction(rest_client, account, payload)

    # Wait for transaction and return result
    return await wait_for_transaction(rest_client, tx_hash)


async def execute_trade_with_tracking(signal, symbol, size, tracker):
    """Execute a trade and update portfolio tracking"""
    # Load or create wallet
    private_key, address, _ = load_or_create_wallet()

    # Connect to Aptos network
    rest_client = RestClient(NODE_URL)
    account_address = AccountAddress.from_str(address)

    # Check current balance
    balance = await check_balance(rest_client, account_address)
    logger.info(f"Current balance: {balance/1e8} APT")

    # Get current token price from market data
    try:
        import yfinance as yf

        current_price = float(
            yf.download(symbol, period="1d", interval="1m", timeout=10)["Close"].iloc[
                -1
            ]
        )
        logger.info(f"Current market price: ${current_price}")
    except Exception as e:
        logger.error(f"Error getting price: {str(e)}")
        current_price = 1.0  # Default value

    # Simulate a trading contract address (in production this would be a real trading contract)
    trading_contract = (
        "0x5ae6789dd2fec1a9ec9cccfb3acaf12e93d432f0a3a42c92fe1a9d490b7bbc06"
    )

    # Execute the trade based on signal
    if signal == "BUY":
        # Calculate APT amount based on USD size and current price
        # Example: If we want to buy $10 worth of the asset at current price
        usd_amount = balance / 1e8 * size  # Use a percentage of our balance in USD
        apt_amount = int((usd_amount / current_price) * 1e8)  # Convert to APT octas

        if apt_amount > 0:
            logger.info(
                f"Executing BUY: ${usd_amount:.2f} worth ({apt_amount/1e8} APT) at ${current_price}"
            )

            # In a real system, you'd transfer to a trading contract
            # For simulation, we'll transfer a small amount to the "trading contract" address
            # This simulates sending funds to exchange/protocol
            simulation_amount = min(
                apt_amount, 10000
            )  # Limit to small amount for testing
            await execute_transfer(private_key, trading_contract, simulation_amount)

            # Record the trade in our portfolio tracker
            # Use the full calculated amount for portfolio tracking, even though we only transfer a small simulation
            tracker.update_position(
                symbol, current_price, usd_amount / current_price, "BUY"
            )
            logger.info(
                f"Bought {usd_amount/current_price:.6f} units at ${current_price}"
            )

    elif signal == "SELL":
        # Check if we have a position to sell
        if symbol in tracker.portfolio["positions"]:
            position = tracker.portfolio["positions"][symbol]

            # Calculate how much to sell (percentage of our position)
            sell_quantity = position["qty"] * size
            usd_value = sell_quantity * current_price

            logger.info(
                f"Executing SELL: {sell_quantity} units (${usd_value:.2f}) at ${current_price}"
            )

            # In a real system, this would execute on the trading protocol
            # For simulation, we'll transfer from the "trading contract" to our wallet
            simulation_amount = (
                10000  # Simulate receiving funds back (small fixed amount)
            )

            # We don't actually need to execute a transfer here since in devnet
            # we're not really getting funds back from a trading contract
            # But we'll record it in our tracker
            tracker.update_position(symbol, current_price, sell_quantity, "SELL")
            logger.info(f"Sold {sell_quantity:.6f} units at ${current_price}")
        else:
            logger.warning(f"No position to sell for {symbol}")

    # Calculate current portfolio value with updated market prices
    current_value = tracker.calculate_current_value({symbol: current_price})
    metrics = tracker.get_pnl_metrics()

    logger.info(
        f"Portfolio value: ${current_value:.2f}, Realized PnL: ${metrics['realized_pnl']:.2f}"
    )
    return True


async def check_entry_points(symbol="APT21794-USD", tracker=None):
    """
    Check entry points for Aptos token and execute trades when appropriate

    Args:
        symbol: The token symbol to check for trading opportunities
        tracker: PortfolioTracker object to track positions and PnL
    """
    # Create tracker if not provided
    if tracker is None:
        tracker = PortfolioTracker(initial_capital=100)

    # Load or create wallet
    private_key, address, _ = load_or_create_wallet()

    # Initialize REST client
    rest_client = RestClient(NODE_URL)
    logger.info(f"Connected to Aptos network at {NODE_URL}")

    # Convert address string to AccountAddress
    account_address = AccountAddress.from_str(address)

    # Check current balance
    balance = await check_balance(rest_client, account_address)
    logger.info(f"Current balance: {balance/1e8} APT")

    try:
        # Import necessary functions from stock_prediction
        from predictor import StockPredictor
        import pandas as pd
        from datetime import date, timedelta, datetime

        # Create a predictor instance
        predictor = StockPredictor(
            symbol=symbol,
            start_date=date.today() - pd.Timedelta(days=500),
            end_date=date.today() + pd.Timedelta(days=1),
            interval="1d",
        )

        # Load data and prepare for analysis
        predictor.load_data()

        # Get trading signal with confidence levels using get_entry_signal
        decision, confidence, rationale, levels = get_entry_signal(predictor, symbol)

        logger.info(f"\n🔍 {symbol} Entry Check:")
        logger.info(f"  Decision: {decision} ({confidence}% confidence)")
        logger.info(f"  Rationale: {rationale}")
        logger.info(f"  Key Levels:")
        logger.info(f"    Current: ${levels['current_price'][0]:.2f}")
        logger.info(f"    Stop Loss: ${levels['stop_loss'][0]:.2f}")
        logger.info(f"    Take Profit: ${levels['take_profit'][0]:.2f}")

        # Execute trade if decision is BUY or SELL and we have enough confidence
        if decision != "HOLD" and confidence > 65:
            await execute_trade_with_tracking(
                signal=decision,
                symbol=symbol,
                size=0.01,  # Use 1% of available balance
                tracker=tracker,
            )
        else:
            logger.info(f"No trade executed - {decision} with {confidence}% confidence")

        # Calculate current metrics
        metrics = tracker.get_pnl_metrics()
        logger.info(f"Current metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error in check_entry_points: {str(e)}")

    return tracker


# class AptosBacktester:
#     """
#     Portfolio tracker for Aptos transactions with functionality similar to
#     the Backtester in stock_prediction/core/predictor.py
#     """

#     def __init__(self, symbol="APT21794-USD", initial_capital=100):
#         """
#         Initialize the backtester

#         Args:
#             symbol (str): Token symbol to track
#             initial_capital (float): Initial capital in USD
#         """
#         self.symbol = symbol
#         self.initial_capital = initial_capital
#         self.portfolio = {
#             "cash": initial_capital,
#             "positions": {},  # symbol -> {qty, entry_price}
#             "value_history": [],  # [{timestamp, value, cash}]
#             "transactions": [],  # [type, price, qty, timestamp]
#         }

#         # Trade parameters
#         self.slippage = 0.002  # 10 basis points
#         self.commission = 0.001  # 0.1% per transaction and usually fixed

#         # Add a reference index for the symbol
#         self.reference_ticker = "QQQ"
#         self.reference_data = None

#         # Configure logging
#         log_directory = os.path.dirname(os.path.abspath(__file__))
#         log_file = os.path.join(
#             log_directory, f"aptos_backtest_{date.today().strftime('%Y%m%d')}.log"
#         )

#         self.logger = logging.getLogger("aptos_backtest")
#         self.logger.setLevel(logging.INFO)

#         if not self.logger.handlers:
#             # File handler
#             file_handler = logging.FileHandler(log_file)
#             file_handler.setFormatter(
#                 logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#             )
#             self.logger.addHandler(file_handler)

#             # Console handler
#             console_handler = logging.StreamHandler()
#             console_handler.setFormatter(
#                 logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#             )
#             self.logger.addHandler(console_handler)

#         # Setup transaction log files
#         self.transaction_log_file = os.path.join(
#             log_directory, "aptos_transactions.csv"
#         )
#         self.portfolio_log_file = os.path.join(log_directory, "aptos_portfolio.csv")

#         # Initialize transaction log if it doesn't exist
#         if not os.path.exists(self.transaction_log_file):
#             pd.DataFrame(
#                 columns=["timestamp", "type", "symbol", "price", "quantity", "value"]
#             ).to_csv(self.transaction_log_file, index=False)

#         # Initialize portfolio log if it doesn't exist
#         if not os.path.exists(self.portfolio_log_file):
#             pd.DataFrame(
#                 columns=["timestamp", "total_value", "cash", "positions"]
#             ).to_csv(self.portfolio_log_file, index=False)

#         self.logger.info(
#             f"AptosBacktester initialized with {initial_capital} USD for {symbol}"
#         )

#     def record_transaction(
#         self, transaction_type, symbol, price, quantity, timestamp=None
#     ):
#         """Record a buy or sell transaction"""
#         if timestamp is None:
#             timestamp = datetime.now()
#         value = price * quantity

#         # Add to in-memory transaction list
#         self.portfolio["transactions"].append(
#             (transaction_type, price, quantity, timestamp)
#         )

#         # Log transaction to file
#         transaction_df = pd.DataFrame(
#             [
#                 {
#                     "timestamp": timestamp,
#                     "type": transaction_type,
#                     "symbol": symbol,
#                     "price": price,
#                     "quantity": quantity,
#                     "value": value,
#                 }
#             ]
#         )

#         transaction_df.to_csv(
#             self.transaction_log_file, mode="a", header=False, index=False
#         )
#         self.logger.info(
#             f"{transaction_type} executed: {quantity:.6f} {symbol} at ${price:.4f}"
#         )

#         return True

#     def execute_buy(self, price, quantity, timestamp=None):
#         """Execute a buy order"""
#         if timestamp is None:
#             timestamp = datetime.now()

#         # Apply slippage to buy price
#         executed_price = price * (1 + self.slippage)

#         # Calculate transaction cost
#         cost = executed_price * quantity
#         commission = cost * self.commission
#         total_cost = cost + commission

#         # Check if enough cash
#         if total_cost > self.portfolio["cash"]:
#             self.logger.warning(
#                 f"Insufficient funds for purchase. Required: ${total_cost:.2f}, Available: ${self.portfolio['cash']:.2f}"
#             )
#             # Adjust quantity to available cash
#             adjusted_quantity = (
#                 self.portfolio["cash"] * 0.025
#             ) / executed_price  # Leave 90% buffer
#             if adjusted_quantity <= 0:
#                 return False

#             quantity = adjusted_quantity
#             cost = executed_price * quantity
#             commission = cost * self.commission
#             total_cost = cost + commission
#             self.logger.info(
#                 f"Adjusted buy quantity to {quantity:.6f} based on available funds"
#             )

#         # Update portfolio
#         self.portfolio["cash"] -= total_cost

#         if self.symbol in self.portfolio["positions"]:
#             # Update existing position
#             position = self.portfolio["positions"][self.symbol]
#             total_quantity = position["qty"] + quantity
#             avg_price = (
#                 (position["qty"] * position["entry_price"])
#                 + (quantity * executed_price)
#             ) / total_quantity

#             self.portfolio["positions"][self.symbol] = {
#                 "qty": total_quantity,
#                 "entry_price": avg_price,
#             }
#         else:
#             # Create new position
#             self.portfolio["positions"][self.symbol] = {
#                 "qty": quantity,
#                 "entry_price": executed_price,
#             }

#         # Record the transaction
#         self.record_transaction("BUY", self.symbol, executed_price, quantity, timestamp)

#         return True

#     def execute_sell(self, price, quantity=None, timestamp=None):
#         """Execute a sell order with support for naked short selling"""
#         if timestamp is None:
#             timestamp = datetime.now()

#         # Apply slippage to sell price (negative for sells)
#         executed_price = price * (1 - self.slippage)

#         # If no quantity specified, sell all holdings or create standard short position
#         if quantity is None:
#             if self.symbol in self.portfolio["positions"]:
#                 quantity = (
#                     self.portfolio["positions"][self.symbol]["qty"] * 0.025
#                 )  # 0.3 % of current position
#             else:
#                 # Default short position size (10% of cash value)
#                 quantity = (self.portfolio["cash"] * 0.025) / executed_price
#                 self.logger.info(
#                     f"No position to sell, creating short position of {quantity:.6f} shares"
#                 )

#         # Calculate transaction value and fees
#         value = executed_price * quantity
#         commission = value * self.commission
#         net_proceeds = value - commission

#         if self.symbol in self.portfolio["positions"]:
#             # We have an existing position
#             position = self.portfolio["positions"][self.symbol]

#             if position["qty"] >= quantity:
#                 # Normal sell - we have enough shares
#                 position["qty"] -= quantity
#                 if position["qty"] <= 0:
#                     # Remove position if sold out completely
#                     self.portfolio["positions"].pop(self.symbol)

#                 # Add proceeds to cash
#                 self.portfolio["cash"] += net_proceeds

#             else:
#                 # Selling more than we own - partial short
#                 # First sell all existing shares
#                 existing_qty = position["qty"]
#                 remaining_qty = quantity - existing_qty

#                 # Add proceeds from existing shares
#                 self.portfolio["cash"] += (executed_price * existing_qty) - (
#                     commission * existing_qty / quantity
#                 )

#                 # Create short position with remaining quantity
#                 self.portfolio["positions"][self.symbol] = {
#                     "qty": -remaining_qty,  # Negative quantity indicates short
#                     "entry_price": executed_price,
#                 }

#                 # Add proceeds from short sale (these are held as cash but may be restricted in real trading)
#                 self.portfolio["cash"] += (executed_price * remaining_qty) - (
#                     commission * remaining_qty / quantity
#                 )

#                 self.logger.info(
#                     f"Partial short created: sold {existing_qty:.6f} owned shares and shorted {remaining_qty:.6f} additional shares"
#                 )

#         else:
#             # No existing position - creating a pure short position
#             self.portfolio["positions"][self.symbol] = {
#                 "qty": -quantity,  # Negative quantity indicates short
#                 "entry_price": executed_price,
#             }

#             # Add proceeds to cash (in real trading this might be held as margin)
#             self.portfolio["cash"] += net_proceeds

#             self.logger.info(
#                 f"Created new short position of {quantity:.6f} shares at ${executed_price:.2f}"
#             )

#         # Record the transaction
#         self.record_transaction(
#             "SELL", self.symbol, executed_price, quantity, timestamp
#         )

#         return True

#     def update_portfolio_value(self, current_price=None, timestamp=None):
#         """Calculate current portfolio value and record to history, supporting short positions"""
#         if timestamp is None:
#             timestamp = datetime.now()

#         # If no price provided, try to get current price
#         if current_price is None:
#             try:
#                 current_price = float(
#                     yf.download(self.symbol, period="1d", interval="1m", timeout=10)[
#                         "Close"
#                     ].iloc[-1]
#                 )
#             except Exception as e:
#                 self.logger.error(f"Failed to get current price: {e}")
#                 # Use last known price or position entry price as fallback
#                 if self.portfolio["value_history"]:
#                     # Use last valuation price
#                     current_price = self.portfolio["value_history"][-1].get(
#                         "price", 1.0
#                     )
#                 elif self.symbol in self.portfolio["positions"]:
#                     # Use position entry price
#                     current_price = self.portfolio["positions"][self.symbol][
#                         "entry_price"
#                     ]
#                 else:
#                     # Default to 1.0 if all else fails
#                     current_price = 1.0

#         # Calculate position value, handling both long and short positions
#         position_value = 0
#         for symbol, position in self.portfolio["positions"].items():
#             # For short positions (negative qty), value increases when price falls
#             # Possibly the wrong way to calculate this
#             # if position["qty"] < 0:  # Short position
#             #     # Value of short is based on entry price vs current price
#             #     # If current price is lower than entry, we're profitable
#             #     short_value = -position["qty"] * (
#             #         position["entry_price"] - current_price
#             #     )
#             #     position_value += (
#             #         position["entry_price"] * -position["qty"]
#             #     )  # Original proceeds
#             #     position_value += short_value  # Profit/loss
#             if position["qty"] < 0:  # Short position
#                 # For short positions, we owe shares that we need to buy back
#                 # Our liability is the current value of those shares
#                 # Our profit/loss is the difference between what we sold them for and what we'd pay now
#                 liability = -position["qty"] * current_price
#                 # original_proceeds = -position["qty"] * position["entry_price"]
#                 # Add the net value: what we received minus what we owe
#                 position_value -= liability
#             else:  # Long position
#                 position_value += position["qty"] * current_price

#         total_value = self.portfolio["cash"] + position_value

#         # Record to history
#         value_entry = {
#             "timestamp": timestamp,
#             "value": total_value,
#             "cash": self.portfolio["cash"],
#             "price": current_price,
#         }
#         self.portfolio["value_history"].append(value_entry)

#         # Log to portfolio file
#         portfolio_record = {
#             "timestamp": timestamp,
#             "total_value": total_value,
#             "cash": self.portfolio["cash"],
#             "positions": str(self.portfolio["positions"]),
#             "price": current_price,
#         }

#         pd.DataFrame([portfolio_record]).to_csv(
#             self.portfolio_log_file, mode="a", header=False, index=False
#         )

#         return total_value

#     def run_backtest(
#         self, start_date, end_date, price_data=None, signal_generator=None
#     ):
#         """
#         Run a backtest over a date range

#         Args:
#             start_date (str): Start date for backtest (YYYY-MM-DD)
#             end_date (str): End date for backtest (YYYY-MM-DD)
#             price_data (pd.DataFrame): Historical price data (if None, will be downloaded)
#             signal_generator (callable): Function that generates trading signals
#                                         Should return "BUY", "SELL", or "HOLD"

#         Returns:
#             tuple: (history_df, performance_metrics)
#         """

#         self.logger.info(f"Starting backtest from {start_date} to {end_date}")
#         from predictor import StockPredictor

#         _predictor_ = StockPredictor(
#             symbol=self.symbol, start_date=start_date, end_date=end_date, interval="1d"
#         )
#         _predictor_.load_data()

#         self.reference_data = yf.download(
#             self.reference_ticker, start=start_date, end=end_date, interval="1d"
#         )
#         self.reference_data = self.reference_data.loc[_predictor_.data.index]

#         # Get price data if not provided
#         if price_data is None:
#             try:
#                 # Check if we should use the retry function
#                 if (
#                     hasattr(self, "use_retry_fetch")
#                     and self.use_retry_fetch
#                     and hasattr(self, "fetch_data_function")
#                 ):
#                     self.logger.info("Using retry fetch function for data download")
#                     # price_data = self.fetch_data_function(
#                     #     self.symbol,
#                     #     start_date,
#                     #     end_date,
#                     #     interval="1d"
#                     # )
#                     price_data = _predictor_.data

#                 else:
#                     # Default download method
#                     self.logger.info("Using standard yfinance download")
#                     # price_data = yf.download(self.symbol, start=start_date, end=end_date, interval='1d', progress=False, timeout=20)
#                     price_data = _predictor_.data
#                 if price_data.empty:
#                     self.logger.error(
#                         f"No data available for {self.symbol} in the specified date range"
#                     )
#                     return pd.DataFrame(), {"error": "No data available"}

#                 # Log successful data retrieval
#                 self.logger.info(
#                     f"Successfully downloaded {len(price_data)} data points"
#                 )

#             except Exception as e:
#                 self.logger.error(f"Failed to download price data: {e}")
#                 return pd.DataFrame(), {"error": f"Failed to download data: {str(e)}"}

#         # Reset portfolio for backtest
#         self.portfolio = {
#             "cash": self.initial_capital,
#             "positions": {},
#             "value_history": [],
#             "transactions": [],
#         }

#         # Verify data structure
#         if "Close" not in price_data.columns:
#             self.logger.error("Price data does not contain 'Close' column")
#             return pd.DataFrame(), {
#                 "error": "Invalid data format - missing 'Close' column"
#             }

#         # Daily loop for backtest
#         dates = price_data.index

#         self.logger.info(f"Running backtest over {len(dates)} trading days")
#         min_portfolio_value = self.initial_capital * 0.1
#         for i, date in enumerate(
#             dates
#         ):  # so in this loop, if autodect_reversal is true, it would decide whether to reverse before each day.
#             # Skip first day (need previous data for signals)
#             if i <= 5:
#                 # Just record initial portfolio value (n = 5)
#                 self.update_portfolio_value(
#                     float(price_data["Close"].iloc[i]), timestamp=date
#                 )
#                 continue

#             # Calculate recent trend direction and strength
#             if i > 20:
#                 recent_trend = (
#                     price_data["Close"].iloc[i - 20 : i].pct_change().mean() * 100
#                 )  # Trend as percentage
#             else:
#                 recent_trend = 0

#             # Convert Series to float to avoid comparison issues
#             current_price = float(
#                 price_data["Open"].iloc[i]
#             )  # execution price is open price of the day i (when the trade is executed)

#             # Check if portfolio value is below minimum threshold
#             if (
#                 self.portfolio["value_history"]
#                 and self.portfolio["value_history"][-1]["value"] <= min_portfolio_value
#             ):
#                 self.logger.warning(
#                     f"Portfolio value fell below minimum threshold ({min_portfolio_value:.2f}). "
#                     f"Stopping backtest at {date} with final value: "
#                     f"{self.portfolio['value_history'][-1]['value']:.2f}"
#                 )
#                 break

#             # Check if we're completely out of cash and have no positions
#             if self.portfolio["cash"] <= 0 and not self.portfolio["positions"]:
#                 self.logger.warning(
#                     f"Portfolio is bankrupt! Stopping backtest at {date}"
#                 )
#                 break

#             # Generate trading signal
#             if signal_generator:
#                 # Use provided signal generator with historical data up to this point
#                 signal = signal_generator(
#                     price_data.iloc[:i]
#                 )  # at date i u can use the data up to i-1 (no close price for i so no other indicators)

#             else:
#                 # Simple momentum strategy as default
#                 # Convert Series to float to avoid comparison issues
#                 prev_price = float(price_data["Close"].iloc[i - 1])

#                 if current_price > prev_price * 1.01:  # 1% increase
#                     signal = "BUY"
#                 elif current_price < prev_price * 0.99:  # 1% decrease
#                     signal = "SELL"
#                 else:
#                     signal = "HOLD"

#             # Execute trades based on signal

#             if signal == "BUY" and self.portfolio["cash"] > 0:
#                 # Calculate position size (use 90% of available cash)
#                 # position_size_factor = (
#                 #     min(
#                 #         0.035, max(0.02, 0.03 / price_data["Volatility"].iloc[-1] * 10)
#                 #     )
#                 #     * 2
#                 # )  ## leverage
#                 position_size_factor = self.calculate_dynamic_position_size(
#                     price_data=price_data,
#                     current_price=current_price,
#                     i=i,
#                     recent_trend=recent_trend,
#                 )

#                 print(f"position_size_factor: {position_size_factor}")
#                 # Increase position size in uptrends
#                 if recent_trend > 0:
#                     position_size_factor *= 1 + min(
#                         recent_trend * 0.5, 0.5
#                     )  # Up to 50% larger

#                 position_size = (
#                     self.portfolio["cash"] * position_size_factor / current_price
#                 )

#                 # position_size = self.portfolio["cash"] * 0.025 / current_price

#                 if position_size <= 1e-3:
#                     self.logger.warning(
#                         f"Insufficient funds for purchase. Required: ${current_price:.2f}, Available: ${self.portfolio['cash']:.2f}"
#                     )
#                     self.update_portfolio_value(current_price, timestamp=date)
#                     continue
#                 if self.portfolio["cash"] >= self.initial_capital * 0.3:
#                     self.execute_buy(current_price, position_size, timestamp=date)
#                 else:
#                     print("Want to keep 40% of the initial capital in cash")
#                 self.logger.info(
#                     f"Day {i}: BUY signal at ${current_price:.2f}, bought {position_size:.4f} units"
#                 )

#             elif signal == "SELL":
#                 if self.symbol in self.portfolio["positions"]:
#                     # Sell all holdings
#                     position_qty = self.portfolio["positions"][self.symbol]["qty"]
#                     # position_size = self.portfolio["cash"] * 0.025 / current_price
#                     # position_size = position_qty * 0.25  # Sell 25% of position
#                     # position_size_factor = (
#                     #     min(
#                     #         0.0325,
#                     #         max(0.02, 0.035 / price_data["Volatility"].iloc[-1] * 10),
#                     #     )
#                     #     * 1.5
#                     # )  ## leverage
#                     position_size_factor = self.calculate_dynamic_position_size(
#                         price_data=price_data,
#                         current_price=current_price,
#                         i=i,
#                         recent_trend=recent_trend,
#                         is_sell=True,
#                     )

#                     position_size = position_qty * position_size_factor
#                     min_quantity = 1e-2  # Minimum tradeable quantity
#                     if position_size < min_quantity:
#                         position_size = (
#                             min_quantity  # Sell entire position if it's too small
#                         )
#                         self.logger.info(
#                             f"Position size too small, selling entire position of {position_qty:.8f} units"
#                         )

#                     self.execute_sell(current_price, position_size, timestamp=date)
#                     self.logger.info(
#                         f"Day {i}: SELL signal at ${current_price:.2f}, sold {position_qty:.4f} units from existing position"
#                     )
#                 else:  # Create a naked short position (if allowed)
#                     # position_size_factor = (
#                     #     min(
#                     #         0.0325,
#                     #         max(0.02, 0.035 / price_data["Volatility"].iloc[-1] * 10),
#                     #     )
#                     #     * 3
#                     # )  ## leverage
#                     position_size_factor = self.calculate_dynamic_position_size(
#                         price_data=price_data,
#                         current_price=current_price,
#                         i=i,
#                         recent_trend=recent_trend,
#                         is_sell=True,
#                     )

#                     position_size = (
#                         self.portfolio["cash"] * position_size_factor / current_price
#                     )
#                     if position_size > 0:
#                         min_quantity = 1e-2
#                         if position_size < min_quantity:
#                             position_size = (
#                                 min_quantity  # Sell entire position if it's too small
#                             )
#                             self.logger.info(
#                                 f"Position size too small, selling entire position of {min_quantity:.8f} units"
#                             )

#                         self.execute_sell(current_price, position_size, timestamp=date)
#                         self.logger.info(
#                             f"Day {i}: SELL signal - created short position of {position_size:.4f} units"
#                         )

#             # # Version 2
#             # if signal == "BUY" and self.portfolio["cash"] > 0:
#             #     # Calculate position size using our dynamic function
#             #     position_size_factor = self.calculate_dynamic_position_size(
#             #         price_data, current_price, i, recent_trend
#             #     )

#             #     # Calculate position size based on cash
#             #     position_size = self.portfolio["cash"] * position_size_factor / current_price

#             #     # Implement minimum position size check
#             #     if position_size <= 1e-3:
#             #         self.logger.warning(
#             #             f"Insufficient funds for purchase. Required: ${current_price:.2f}, Available: ${self.portfolio['cash']:.2f}"
#             #         )
#             #         self.update_portfolio_value(current_price, timestamp=date)
#             #         continue

#             #     # Cash reserve strategy - keep at least 30% of initial capital
#             #     if self.portfolio["cash"] >= self.initial_capital * 0.3:
#             #         # Check for price level relative to moving averages if available
#             #         if "MA_50" in price_data.columns and "MA_200" in price_data.columns and i > 200:
#             #             ma50 = float(price_data["MA_50"].iloc[i-1])
#             #             ma200 = float(price_data["MA_200"].iloc[i-1])

#             #             # Increase position size when price is between MAs (potential support)
#             #             if current_price < ma50 and current_price > ma200:
#             #                 position_size *= 1.2
#             #                 self.logger.info(f"Increased position size due to price between MAs")

#             #             # Decrease position size when price is extended above MAs
#             #             elif current_price > ma50 * 1.1 and current_price > ma200 * 1.2:
#             #                 position_size *= 0.8
#             #                 self.logger.info(f"Reduced position size due to extended price")

#             #         # Execute the buy with optimized size
#             #         self.execute_buy(current_price, position_size, timestamp=date)
#             #         self.logger.info(f"Day {i}: BUY signal at ${current_price:.2f}, bought {position_size:.4f} units")
#             #     else:
#             #         self.logger.info("Want to keep 30% of the initial capital in cash")

#             # elif signal == "SELL":
#             #     if self.symbol in self.portfolio["positions"]:
#             #         # Implement graduated selling based on profit levels
#             #         position = self.portfolio["positions"][self.symbol]
#             #         position_qty = position["qty"]
#             #         entry_price = position["entry_price"]
#             #         profit_pct = (current_price / entry_price) - 1

#             #         # Adjust sell size based on profit percentage
#             #         if profit_pct > 0.25:
#             #             # Higher profit = sell more
#             #             position_size_factor = min(0.04, max(0.025, 0.035 / price_data["Volatility"].iloc[-1] * 10)) * 2.0
#             #         elif profit_pct > 0.15:
#             #             # Moderate profit = moderate sell
#             #             position_size_factor = min(0.035, max(0.02, 0.03 / price_data["Volatility"].iloc[-1] * 10)) * 1.5
#             #         elif profit_pct > 0.05:
#             #             # Small profit = smaller sell
#             #             position_size_factor = min(0.03, max(0.015, 0.025 / price_data["Volatility"].iloc[-1] * 10)) * 1.2
#             #         else:
#             #             # No profit or loss = standard sell
#             #             position_size_factor = min(0.0325, max(0.02, 0.035 / price_data["Volatility"].iloc[-1] * 10)) * 1.5

#             #         # Calculate sell size
#             #         position_size = position_qty * position_size_factor

#             #         # Ensure minimum tradeable quantity
#             #         min_quantity = 1e-2
#             #         if position_size < min_quantity:
#             #             position_size = min(min_quantity, position_qty)
#             #             self.logger.info(f"Position size too small, adjusting to {position_size:.8f} units")

#             #         # Execute the sell
#             #         self.execute_sell(current_price, position_size, timestamp=date)
#             #         self.logger.info(
#             #             f"Day {i}: SELL signal at ${current_price:.2f}, sold {position_size:.4f} units from existing position"
#             #         )
#             #     else:
#             #         # Create a naked short position (if allowed)
#             #         position_size_factor = (
#             #             min(0.0325, max(0.02, 0.035 / price_data["Volatility"].iloc[-1] * 10)) * 3
#             #         )
#             #         position_size = self.portfolio["cash"] * position_size_factor / current_price

#             #         # Enhanced short position logic
#             #         if "MA_200" in price_data.columns and i > 200:
#             #             ma200 = float(price_data["MA_200"].iloc[i-1])

#             #             # Only take shorts when price is below 200MA (confirmed downtrend)
#             #             if current_price < ma200:
#             #                 if position_size > 0:
#             #                     min_quantity = 1e-2
#             #                     if position_size < min_quantity:
#             #                         position_size = min_quantity

#             #                     self.execute_sell(current_price, position_size, timestamp=date)
#             #                     self.logger.info(f"Day {i}: SELL signal - created short position of {position_size:.4f} units")
#             #             else:
#             #                 self.logger.info(f"Skipped short position as price is above 200MA")
#             #         else:
#             #             # If no MA data, proceed with regular short logic
#             #             if position_size > 0:
#             #                 min_quantity = 1e-2
#             #                 if position_size < min_quantity:
#             #                     position_size = min_quantity

#             #                 self.execute_sell(current_price, position_size, timestamp=date)
#             #                 self.logger.info(f"Day {i}: SELL signal - created short position of {position_size:.4f} units")

#             # Update portfolio value using today's close for accurate end-of-day valuation
#             self.update_portfolio_value(
#                 float(price_data["Close"].iloc[i]), timestamp=date
#             )

#             # original risk management logic
#             # for symbol, position in list(self.portfolio["positions"].items()):
#             #     entry_price = position["entry_price"]
#             #     current_price = float(price_data["Open"].iloc[i])

#             #     # Trailing stop - tightens as profit increases
#             #     if position["qty"] > 0:
#             #         profit_pct = (current_price - entry_price) / entry_price

#             #         # Aggressive take-profit (15% gain)
#             #         if profit_pct > 0.3:
#             #             # Take 10% off the table at 30% gain
#             #             self.execute_sell(
#             #                 current_price, position["qty"] * 0.1, timestamp=date
#             #             )
#             #             self.logger.info(
#             #                 f"Take-profit triggered at {current_price:.2f}"
#             #             )

#             #         elif profit_pct > 0.25:
#             #             # Take 50% off the table at 15% gain
#             #             self.execute_sell(
#             #                 current_price, position["qty"] * 0.1, timestamp=date
#             #             )
#             #             self.logger.info(
#             #                 f"Take-profit triggered at {current_price:.2f}"
#             #             )

#             #         # Trailing stop gets tighter as profit increases
#             #         elif profit_pct > 0.15:
#             #             # If price drops more than 3% from peak, exit remaining position
#             #             recent_high = price_data["Close"].iloc[i - 10 : i].max()
#             #             if current_price < recent_high * 0.9:
#             #                 self.execute_sell(
#             #                     current_price, position["qty"] * 0.4, timestamp=date
#             #                 )
#             #                 self.logger.info(
#             #                     f"Trailing stop triggered at {current_price:.2f}"
#             #                 )

#             #         # Wider stop-loss for new positions (12%)
#             #         elif profit_pct < -0.25:
#             #             # Cut losses at 12%
#             #             self.execute_sell(
#             #                 current_price, position["qty"] * 0.4, timestamp=date
#             #             )
#             #             self.logger.info(f"Stop-loss triggered at {current_price:.2f}")

#             # Risk management - trailing stops and take-profit levels
#             for symbol, position in list(self.portfolio["positions"].items()):
#                 entry_price = position["entry_price"]
#                 current_price = float(price_data["Open"].iloc[i])
#                 position_qty = position["qty"]

#                 if position_qty > 0:  # Long position
#                     profit_pct = (current_price - entry_price) / entry_price

#                     # Define different profit-taking and stop-loss levels
#                     if profit_pct > 0.4:  # >40% profit - lock in gains aggressively
#                         # Take 20% off the table at 40% gain
#                         sell_qty = position_qty * 0.2
#                         if sell_qty > 1e-3:  # Minimum quantity check
#                             self.execute_sell(current_price, sell_qty, timestamp=date)
#                             self.logger.info(
#                                 f"40% Take-profit triggered at ${current_price:.2f}, selling {sell_qty:.4f} units"
#                             )

#                     elif profit_pct > 0.3:  # >30% profit - take some profits
#                         # Take 15% off the table at 30% gain
#                         sell_qty = position_qty * 0.15
#                         if sell_qty > 1e-3:  # Minimum quantity check
#                             self.execute_sell(current_price, sell_qty, timestamp=date)
#                             self.logger.info(
#                                 f"30% Take-profit triggered at ${current_price:.2f}, selling {sell_qty:.4f} units"
#                             )

#                     elif profit_pct > 0.2:  # >20% profit - take smaller profits
#                         # Take 10% off the table at 20% gain
#                         sell_qty = position_qty * 0.1
#                         if sell_qty > 1e-3:  # Minimum quantity check
#                             self.execute_sell(current_price, sell_qty, timestamp=date)
#                             self.logger.info(
#                                 f"20% Take-profit triggered at ${current_price:.2f}, selling {sell_qty:.4f} units"
#                             )

#                     # Dynamic trailing stop based on profit level
#                     if profit_pct > 0.25:
#                         # Tighter trailing stop when in large profit (7%)
#                         recent_high = price_data["Close"].iloc[i - 10 : i].max()
#                         if current_price < recent_high * 0.93:
#                             sell_qty = (
#                                 position_qty * 0.5
#                             )  # Sell half on initial trigger
#                             if sell_qty > 1e-3:
#                                 self.execute_sell(
#                                     current_price, sell_qty, timestamp=date
#                                 )
#                                 self.logger.info(
#                                     f"Tight trailing stop triggered at ${current_price:.2f} (7% below recent high of ${recent_high:.2f})"
#                                 )

#                     elif profit_pct > 0.15:
#                         # Moderate trailing stop when in decent profit (10%)
#                         recent_high = price_data["Close"].iloc[i - 15 : i].max()
#                         if current_price < recent_high * 0.9:
#                             sell_qty = position_qty * 0.4
#                             if sell_qty > 1e-3:
#                                 self.execute_sell(
#                                     current_price, sell_qty, timestamp=date
#                                 )
#                                 self.logger.info(
#                                     f"Medium trailing stop triggered at ${current_price:.2f} (10% below recent high of ${recent_high:.2f})"
#                                 )

#                     # Hard stop loss - prevent catastrophic losses
#                     elif profit_pct < -0.20:
#                         # Cut losses at 20% drawdown - emergency exit
#                         sell_qty = position_qty * 0.75  # Sell most of position
#                         if sell_qty > 1e-3:
#                             self.execute_sell(current_price, sell_qty, timestamp=date)
#                             self.logger.info(
#                                 f"Hard stop-loss triggered at ${current_price:.2f} (-20% from entry)"
#                             )

#                 elif position_qty < 0:  # Short position
#                     # Handle short position risk management
#                     short_profit_pct = (entry_price - current_price) / entry_price

#                     # Take profits on shorts when price declines significantly
#                     if short_profit_pct > 0.2:  # >20% profit on short
#                         cover_qty = abs(position_qty) * 0.4  # Cover 40% of short
#                         if cover_qty > 1e-3:
#                             self.execute_buy(current_price, cover_qty, timestamp=date)
#                             self.logger.info(
#                                 f"Short take-profit triggered at ${current_price:.2f}"
#                             )

#                     # Stop loss for shorts - limit losses if price rises
#                     elif short_profit_pct < -0.15:  # 15% loss on short
#                         cover_qty = abs(position_qty) * 0.6  # Cover most of short
#                         if cover_qty > 1e-3:
#                             self.execute_buy(current_price, cover_qty, timestamp=date)
#                             self.logger.info(
#                                 f"Short stop-loss triggered at ${current_price:.2f}"
#                             )

#         # Generate report
#         self.logger.info("Backtest completed, generating report...")
#         return self.generate_report()

#     def generate_report(self):
#         """Generate performance report"""
#         # Convert history to DataFrame
#         if not self.portfolio["value_history"]:
#             self.logger.warning("No history data to generate report")
#             return pd.DataFrame(), {"error": "No history data"}

#         history_df = pd.DataFrame(self.portfolio["value_history"])
#         history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
#         history_df = history_df.set_index("timestamp")

#         # Calculate metrics
#         metrics = {}

#         # Total return
#         initial_value = self.initial_capital
#         final_value = history_df["value"].iloc[-1]
#         total_return = (final_value / initial_value) - 1
#         metrics["total_return"] = total_return

#         # Get daily returns
#         history_df["daily_return"] = history_df["value"].pct_change()

#         # Sharpe ratio (annualized, assuming risk-free rate of 0)
#         if len(history_df) > 1:
#             daily_returns = history_df["daily_return"].dropna()
#             if not daily_returns.empty and daily_returns.std() != 0:
#                 sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
#             else:
#                 sharpe = 0
#             metrics["sharpe"] = sharpe

#         # Max drawdown
#         if len(history_df) > 1:
#             history_df["peak"] = history_df["value"].cummax()
#             history_df["drawdown"] = (history_df["value"] / history_df["peak"]) - 1
#             max_drawdown = history_df["drawdown"].min()
#             metrics["max_drawdown"] = max_drawdown

#         # # Win rate
#         # buys = [t for t in self.portfolio["transactions"] if t[0] == "BUY"]
#         # sells = [t for t in self.portfolio["transactions"] if t[0] == "SELL"]

#         # winning_trades = 0
#         # for i, sell in enumerate(sells):
#         #     if i < len(buys):
#         #         buy_price = buys[i][1]
#         #         sell_price = sell[1]
#         #         if sell_price > buy_price:
#         #             winning_trades += 1

#         # win_rate = winning_trades / len(sells) if sells else 0
#         # metrics["win_rate"] = win_rate
#         # metrics["num_trades"] = len(buys) + len(sells)

#         # Improved win rate calculation that only counts liquidated trades
#         buys = [t for t in self.portfolio["transactions"] if t[0] == "BUY"]
#         sells = [t for t in self.portfolio["transactions"] if t[0] == "SELL"]

#         # Use FIFO (First-In-First-Out) to match buys and sells
#         buy_queue = []  # Store (price, quantity) tuples
#         winning_trades = 0
#         losing_trades = 0

#         for transaction_type, price, qty, timestamp in self.portfolio["transactions"]:
#             if transaction_type == "BUY":
#                 # Add to buy queue
#                 buy_queue.append((price, qty))

#             elif transaction_type == "SELL" and buy_queue:
#                 # Process sell against available buys
#                 remaining_sell_qty = qty

#                 while remaining_sell_qty > 0 and buy_queue:
#                     buy_price, buy_qty = buy_queue[0]

#                     # Determine how much of this buy is being sold
#                     match_qty = min(remaining_sell_qty, buy_qty)

#                     # Count this as one trade (or partial trade)
#                     if price > buy_price:
#                         # Profitable trade
#                         winning_trades += 1
#                     else:
#                         # Unprofitable trade
#                         losing_trades += 1

#                     # Update quantities
#                     remaining_sell_qty -= match_qty

#                     if match_qty >= buy_qty:
#                         # Consumed entire buy
#                         buy_queue.pop(0)
#                     else:
#                         # Partially consumed buy
#                         buy_queue[0] = (buy_price, buy_qty - match_qty)
#                         break

#         # Calculate win rate based only on completed trades
#         total_closed_trades = winning_trades + losing_trades
#         win_rate = (
#             winning_trades / total_closed_trades if total_closed_trades > 0 else 0
#         )

#         days = (history_df.index[-1] - history_df.index[0]).days
#         years = days / 365.0
#         cagr = (final_value / initial_value) ** (1 / years) - 1
#         metrics["cagr"] = cagr
#         # Sortino ratio
#         downside_returns = daily_returns.copy()
#         downside_returns[downside_returns > 0] = 0
#         sortino = daily_returns.mean() / downside_returns.std() * np.sqrt(252)
#         metrics["sortino"] = sortino

#         metrics["win_rate"] = win_rate
#         metrics["total_closed_trades"] = total_closed_trades
#         metrics["num_trades"] = len(buys) + len(sells)
#         metrics["num_buy_trades"] = len(buys)
#         metrics["num_sell_trades"] = len(sells)

#         # Log report summary
#         self.logger.info(
#             f"Backtest Report: Total Return: {total_return:.2%}, Sharpe: {metrics.get('sharpe', 0):.2f}, Max Drawdown: {metrics.get('max_drawdown', 0):.2%}, Win Rate: {metrics.get('win_rate', 0):.2%}, Number of Trades: {metrics.get('num_trades', 0)}, Buy Trades: {metrics.get('num_buy_trades', 0)}, Sell Trades: {metrics.get('num_sell_trades', 0)},\
#             CAGR: {metrics.get('cagr', 0):.2%}, Sortino: {metrics.get('sortino', 0):.2f}"
#         )

#         return history_df, metrics

#     def plot_results(self, history_df=None):
#         """Plot backtest results with fixed stackplot rendering"""
#         if history_df is None:
#             history_df = pd.DataFrame(self.portfolio["value_history"])
#             history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
#             history_df = history_df.set_index("timestamp")

#         if history_df.empty:
#             self.logger.warning("No data to plot")
#             return

#         # Create figure with two subplots
#         fig, (ax1, ax2) = plt.subplots(
#             2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]}
#         )

#         # Plot portfolio value
#         ax1.plot(
#             history_df.index,
#             history_df["value"] / history_df["value"].iloc[0],
#             label="Portfolio Value",
#             color="blue",
#         )

#         # Plot reference portfolio if available
#         if self.reference_data is not None and not self.reference_data.empty:
#             ax1.plot(
#                 self.reference_data.index,
#                 self.reference_data["Close"] / self.reference_data["Close"].iloc[0],
#                 label="Reference Portfolio",
#                 color="orange",
#             )

#         # Plot price if available
#         if "price" in history_df.columns:
#             ax1_twin = ax1.twinx()
#             ax1_twin.plot(
#                 history_df.index,
#                 history_df["price"],
#                 label="Price",
#                 color="gray",
#                 alpha=0.6,
#             )
#             ax1_twin.set_ylabel("Price ($)")

#         # Mark buy and sell points (Green for buy, Red for sell)
#         for transaction in self.portfolio["transactions"]:
#             tx_type, price, qty, timestamp = transaction
#             if isinstance(timestamp, str):
#                 timestamp = pd.to_datetime(timestamp)
#             if tx_type == "BUY":
#                 ax1.scatter(
#                     timestamp,
#                     (
#                         history_df.loc[history_df.index == timestamp, "value"].iloc[0]
#                         / history_df["value"].iloc[0]
#                         if len(history_df.loc[history_df.index == timestamp]) > 0
#                         else 0
#                     ),
#                     color="green",
#                     marker="^",
#                     s=100,
#                 )
#             elif tx_type == "SELL":
#                 ax1.scatter(
#                     timestamp,
#                     (
#                         history_df.loc[history_df.index == timestamp, "value"].iloc[0]
#                         / history_df["value"].iloc[0]
#                         if len(history_df.loc[history_df.index == timestamp]) > 0
#                         else 0
#                     ),
#                     color="red",
#                     marker="v",
#                     s=100,
#                 )

#         # Manually set legend for buy/sell markers
#         # handles, labels = plt.gca().get_legend_handles_labels()
#         handles, labels = ax1.get_legend_handles_labels()

#         # Create manual symbols for legend
#         # patch = mpatches.Patch(color="grey", label="manual patch")
#         buy = Line2D(
#             [0],
#             [0],
#             label="Buy",
#             marker="^",
#             markersize=10,
#             color="green",
#             markerfacecolor="green",
#             linestyle="",
#         )
#         sell = Line2D(
#             [0],
#             [0],
#             label="Sell",
#             marker="v",
#             markersize=10,
#             color="red",
#             markerfacecolor="red",
#             linestyle="",
#         )

#         # Add manual symbols to auto legend
#         handles.extend([buy, sell])

#         # Add cash vs. position allocation - FIX STACKED PLOT
#         if len(history_df) > 0:
#             # Calculate position value by subtracting cash from total value
#             history_df["position_value"] = history_df["value"] - history_df["cash"]

#             # Split into long and short components
#             history_df["long_positions"] = history_df["position_value"].apply(
#                 lambda x: max(0, x)
#             )
#             history_df["short_positions"] = (
#                 history_df["position_value"].apply(lambda x: min(0, x)).abs()
#             )

#             # Prepare data for stackplot (cash and position components)
#             x = history_df.index
#             y1 = (
#                 history_df["cash"] - history_df["short_positions"]
#             )  # Cash minus short liability
#             y2 = history_df["long_positions"]  # Long positions
#             y3 = history_df[
#                 "short_positions"
#             ]  # Short positions (as positive values for plotting)

#             # Create stackplot with correct data
#             ax2.stackplot(
#                 x,
#                 y1,
#                 y2,
#                 y3,
#                 labels=["Cash", "Long Positions", "Short Positions"],
#                 colors=["#86c7f3", "#ffe29a", "#ffb3b3"],
#             )

#         # Configure plots
#         ax1.set_title(f"Portfolio Performance: {self.symbol}")
#         ax1.set_ylabel("Portfolio Value ($)")
#         ax1.legend(loc="upper left", handles=handles)

#         ax2.set_title("Portfolio Composition")
#         ax2.set_ylabel("Value ($)")
#         ax2.set_xlabel("Date")
#         ax2.legend(loc="upper left")

#         # Improve date formatting
#         import matplotlib.dates as mdates

#         date_format = mdates.DateFormatter("%Y-%m-%d")
#         ax1.xaxis.set_major_formatter(date_format)
#         ax2.xaxis.set_major_formatter(date_format)

#         # Set major tick locations
#         ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
#         ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

#         # Rotate date labels
#         plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
#         plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

#         plt.tight_layout()
#         plt.show()

#         return fig

#     def calculate_dynamic_position_size(
#         self, price_data, current_price, i, recent_trend, is_sell=False
#     ):
#         """Calculate optimal position size based on market conditions"""

#         # Base volatility adjustment
#         volatility = (
#             price_data["Volatility"].iloc[-1]
#             if "Volatility" in price_data.columns
#             else 0.02
#         )
#         volatility = max(0.01, min(0.05, volatility))  # Bound volatility between 1-5%

#         # Detect bottoming pattern
#         bottoming_pattern = False
#         if i > 20:
#             # Look for higher lows after a downtrend
#             recent_lows = [
#                 min(price_data["Low"].iloc[j - 5 : j]) for j in range(i - 15, i)
#             ]
#             if (
#                 len(recent_lows) >= 3
#                 and recent_lows[-3] < recent_lows[-2] < recent_lows[-1]
#             ):
#                 # Check for volume expansion
#                 if (
#                     price_data["Volume"].iloc[-1]
#                     > price_data["Volume"].rolling(20).mean().iloc[-1] * 1.2
#                 ):
#                     bottoming_pattern = True
#                     self.logger.info(f"Bottoming pattern detected at day {i}")

#         # Calculate RSI if available
#         oversold_condition = False
#         if "RSI" in price_data.columns and len(price_data) > i and i > 0:
#             current_rsi = price_data["RSI"].iloc[i - 1]  # Use previous day's RSI
#             if current_rsi < 30:
#                 oversold_condition = True
#                 self.logger.info(f"Oversold condition detected: RSI={current_rsi:.1f}")

#         # Base position size factor (inversely related to volatility)
#         position_size_factor = min(0.04, max(0.015, 0.03 / (volatility * 10)))

#         # Adjust for market conditions
#         if bottoming_pattern:
#             position_size_factor *= 1.5  # 50% larger positions on bottoming patterns
#             self.logger.info(
#                 f"Increased position size due to bottoming pattern: {position_size_factor:.4f}"
#             )

#         if oversold_condition:
#             position_size_factor *= 1.3  # 30% larger positions on oversold conditions
#             self.logger.info(
#                 f"Increased position size due to oversold condition: {position_size_factor:.4f}"
#             )

#         # Trend-based adjustment (increase in uptrends)
#         if recent_trend > 0:
#             trend_boost = min(recent_trend * 0.5, 0.5)  # Cap at 50% increase
#             position_size_factor *= 1 + trend_boost

#         # Apply leverage multiplier but cap at reasonable level
#         position_size_factor *= 2.0  # Base leverage multiplier
#         if is_sell:
#             position_size_factor *= 2.0
#             position_size_factor = min(
#                 0.5, position_size_factor
#             )  # Cap at 50% for sells
#         else:
#             position_size_factor = min(0.2, position_size_factor)  # Cap at 20% for buys

#         self.logger.info(f"Final position size factor: {position_size_factor:.4f}")
#         return position_size_factor


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
        self.reference_ticker = "SPGI"
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

        # Record the transaction
        self.record_transaction("BUY", symbol, executed_price, quantity, timestamp)

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

        # Record the transaction
        self.record_transaction("SELL", symbol, executed_price, quantity, timestamp)

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
                        if self.portfolio["cash"] >= self.initial_capital * 0.3:
                            self.execute_buy(
                                symbol, current_price, position_size, timestamp=date
                            )
                            self.logger.info(
                                f"{date} - {symbol}: BUY signal at ${current_price:.2f}, bought {position_size:.4f} units"
                            )

                elif signal == "SELL":
                    if symbol in self.portfolio["positions"]:
                        position_qty = self.portfolio["positions"][symbol]["qty"]

                        # Calculate sell size
                        position_size_factor = self.calculate_dynamic_position_size(
                            price_data=symbol_data,
                            current_price=current_price,
                            recent_trend=recent_trend,
                            is_sell=True,
                        )

                        position_size = position_qty * position_size_factor
                        min_quantity = 1e-2
                        if position_size < min_quantity:
                            position_size = min_quantity

                        self.execute_sell(
                            symbol, current_price, position_size, timestamp=date
                        )
                        self.logger.info(
                            f"{date} - {symbol}: SELL signal at ${current_price:.2f}, sold {position_size:.4f} units"
                        )
                    else:
                        # Create a short position if market conditions are good
                        position_size_factor = self.calculate_dynamic_position_size(
                            price_data=symbol_data,
                            current_price=current_price,
                            recent_trend=recent_trend,
                            is_sell=True,
                        )

                        # Only short in a downtrend
                        if recent_trend < -0.5:
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
                                    f"{date} - {symbol}: Created short position of {position_size:.4f} units"
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
        """Run risk management checks across the portfolio"""
        for symbol, position in list(self.portfolio["positions"].items()):
            # Skip if we don't have data for this symbol/date
            if symbol not in price_data or current_date not in price_data[symbol].index:
                continue

            symbol_data = price_data[symbol]
            entry_price = position["entry_price"]
            current_price = float(symbol_data.loc[current_date, "Open"])
            position_qty = position["qty"]

            if position_qty > 0:  # Long position
                profit_pct = (current_price - entry_price) / entry_price

                # Take profit at different levels
                if profit_pct > 0.4:  # 40% profit
                    sell_qty = position_qty * 0.2
                    if sell_qty > 1e-3:
                        self.execute_sell(
                            symbol, current_price, sell_qty, timestamp=current_date
                        )
                        self.logger.info(
                            f"{current_date} - {symbol}: 40% Take-profit triggered at ${current_price:.2f}"
                        )

                elif profit_pct > 0.3:  # 25% profit
                    sell_qty = position_qty * 0.15
                    if sell_qty > 1e-3:
                        self.execute_sell(
                            symbol, current_price, sell_qty, timestamp=current_date
                        )
                        self.logger.info(
                            f"{current_date} - {symbol}: 25% Take-profit triggered at ${current_price:.2f}"
                        )

                # Dynamic trailing stop based on profit level
                if profit_pct > 0.25:
                    # Tighter stop when in good profit (7% from recent high)
                    recent_prices = symbol_data.loc[:current_date]
                    if len(recent_prices) >= 10:
                        recent_high = recent_prices["Close"].iloc[-10:].max()
                        if current_price < recent_high * 0.93:
                            sell_qty = position_qty * 0.5
                            if sell_qty > 1e-3:
                                self.execute_sell(
                                    symbol,
                                    current_price,
                                    sell_qty,
                                    timestamp=current_date,
                                )
                                self.logger.info(
                                    f"{current_date} - {symbol}: Trailing stop triggered at ${current_price:.2f}"
                                )
                elif profit_pct > 0.15:
                    # Moderate trailing stop when in decent profit (10%)
                    recent_high = symbol_data["Close"].iloc[- 15 : ].max()
                    if current_price < recent_high * 0.9:
                        sell_qty = position_qty * 0.4
                        if sell_qty > 1e-3:
                            self.execute_sell(
                                current_price, sell_qty, timestamp=date
                            )
                            self.logger.info(
                                f"Medium trailing stop triggered at ${current_price:.2f} (10% below recent high of ${recent_high:.2f})"
                            )

                # Hard stop loss
                elif profit_pct < -0.20:
                    sell_qty = position_qty * 0.75
                    if sell_qty > 1e-3:
                        self.execute_sell(
                            symbol, current_price, sell_qty, timestamp=current_date
                        )
                        self.logger.info(
                            f"{current_date} - {symbol}: Stop-loss triggered at ${current_price:.2f}"
                        )

            elif position_qty < 0:  # Short position
                short_profit_pct = (entry_price - current_price) / entry_price

                # Take profit on shorts
                if short_profit_pct > 0.2:
                    cover_qty = abs(position_qty) * 0.4
                    if cover_qty > 1e-3:
                        self.execute_buy(
                            symbol, current_price, cover_qty, timestamp=current_date
                        )
                        self.logger.info(
                            f"{current_date} - {symbol}: Short take-profit triggered at ${current_price:.2f}"
                        )

                # Stop loss for shorts
                elif short_profit_pct < -0.15:
                    cover_qty = abs(position_qty) * 0.6
                    if cover_qty > 1e-3:
                        self.execute_buy(
                            symbol, current_price, cover_qty, timestamp=current_date
                        )
                        self.logger.info(
                            f"{current_date} - {symbol}: Short stop-loss triggered at ${current_price:.2f}"
                        )

    def calculate_dynamic_position_size(
        self, price_data, current_price, recent_trend=0, is_sell=False
    ):
        """Calculate optimal position size based on market conditions"""

        # Base volatility adjustment
        volatility = (
            price_data["Volatility"].iloc[-1]
            if "Volatility" in price_data.columns and len(price_data) > 0
            else 0.02
        )
        volatility = max(0.01, min(0.05, volatility))  # Bound volatility between 1-5%

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

        # Base position size factor (inversely related to volatility)
        position_size_factor = min(0.04, max(0.015, 0.03 / (volatility * 10)))

        # Adjust for market conditions
        if bottoming_pattern and not is_sell:
            position_size_factor *= 1.5  # 50% larger positions on bottoming patterns
            self.logger.info(
                f"Increased position size due to bottoming pattern: {position_size_factor:.4f}"
            )

        # Calculate RSI if available
        if "RSI" in price_data.columns and len(price_data) > 0:
            try:
                # Use RSI to adjust position size
                current_rsi = price_data["RSI"].iloc[-1]

                if not is_sell and current_rsi < 30:  # Oversold
                    position_size_factor *= 1.3
                    self.logger.info(
                        f"Increased buy size due to oversold RSI: {current_rsi:.1f}"
                    )

                elif is_sell and current_rsi > 70:  # Overbought
                    position_size_factor *= 1.3
                    self.logger.info(
                        f"Increased sell size due to overbought RSI: {current_rsi:.1f}"
                    )
            except:
                pass

        # Trend-based adjustment
        if recent_trend > 0 and not is_sell:
            trend_boost = min(recent_trend * 0.5, 0.5)  # Cap at 50% increase
            position_size_factor *= 1 + trend_boost
        elif recent_trend < 0 and is_sell:
            trend_boost = min(abs(recent_trend) * 0.5, 0.5)
            position_size_factor *= 1 + trend_boost

        # Apply leverage multiplier but cap at reasonable level
        position_size_factor *= 1.8  # Base leverage multiplier

        if is_sell:
            position_size_factor = min(
                0.4, position_size_factor
            )  # Cap at 40% for sells
        else:
            position_size_factor = min(
                0.15, position_size_factor
            )  # Cap at 15% for buys

        # Adjust position size based on portfolio diversification
        num_symbols = len(self.symbols)
        if num_symbols > 1:
            # Scale down as we add more symbols to maintain proper diversification
            position_size_factor *= 0.8  # Base reduction for multi-stock portfolio

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

        # Log report summary
        self.logger.info(
            f"Backtest Report: Total Return: {total_return:.2%}, Sharpe: {metrics.get('sharpe', 0):.2f}, "
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}, Win Rate: {metrics.get('win_rate', 0):.2%}, "
            f"Number of Trades: {metrics.get('num_trades', 0)}, CAGR: {metrics.get('cagr', 0):.2%}, "
            f"Sortino: {metrics.get('sortino', 0):.2f}"
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

        # Plot reference portfolio if available
        if self.reference_data is not None and not self.reference_data.empty:
            # Make sure reference data aligns with our history dates
            aligned_ref = pd.DataFrame(index=history_df.index)
            aligned_ref = aligned_ref.join(self.reference_data["Close"], how="left")
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
        stock_sum = np.zeros(len(history_df))
        for col in symbol_price_columns:
            symbol = col.replace("_price", "")
            prices = history_df[col].dropna()
            

            if len(prices) > 0:
                # Normalize to starting value
                normalized = prices / prices.iloc[0]
                stock_sum += normalized
                ax3.plot(prices.index, normalized, label=symbol, alpha=0.8)
        ax3.plot(stock_sum.index, stock_sum/stock_num, label = 'Average', c = 'black' )
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

def select_stocks(num_stocks=5, universe="S&P 100", lookback_days=90, strategy="momentum", end_date=None):
    """
    Select promising stocks from a universe based on fundamental and technical metrics
    
    Args:
        num_stocks (int): Number of stocks to select
        universe (str): Stock universe to select from ("S&P 100", "NASDAQ 100", etc.)
        lookback_days (int): Days to look back for performance metrics
        strategy (str): Selection strategy - "momentum", "value", "quality", or "blend"
        
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
        
        if universe == "S&P 100":
            stocks = ticker_data.get_stocks_by_index("S&P 100")
        elif universe == "NASDAQ 100":
            stocks = ticker_data.get_stocks_by_index("NASDAQ 100")
        elif universe == "DOW JONES":
            stocks = ticker_data.get_stocks_by_index("DOW JONES")
        elif universe == 'S&P 500':
            stocks = ticker_data.get_stocks_by_index("S&P 500")
        else:
            stocks = ticker_data.get_stocks_by_index("S&P 100")  # Default
        
        # Extract Yahoo Finance symbols
        for stock in stocks:
            for symbol in stock["symbols"]:
                if symbol["currency"] == "USD":
                    symbols.append(symbol["yahoo"])
        
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
            batch_symbols = symbols[i:i+batch_size]
            try:
                batch_data = yf.download(
                    batch_symbols,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                # Handle single stock case where yfinance doesn't return MultiIndex
                if isinstance(batch_data.columns, pd.MultiIndex):
                    for symbol in batch_symbols:
                        if ('Close', symbol) in batch_data.columns:
                            # Extract data for this symbol
                            symbol_data = batch_data.xs(symbol, axis=1, level=1)
                            if not symbol_data.empty and len(symbol_data) > lookback_days//2:
                                stock_data[symbol] = symbol_data
                else:
                    # Single stock case
                    if len(batch_data) > lookback_days//2 and not batch_data.empty:
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
                closes = data['Close']
                volumes = data['Volume'] if 'Volume' in data else None
                
                # 1. Momentum metrics
                returns_1m = (closes.iloc[-1] / closes.iloc[-min(21, len(closes))]) - 1
                returns_3m = (closes.iloc[-1] / closes.iloc[-min(63, len(closes))]) - 1
                
                # 2. Volatility metrics
                returns = closes.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                
                # 3. Technical indicators
                # a. Moving averages
                ma50 = closes.rolling(window=min(50, len(closes))).mean().iloc[-1]
                ma200 = closes.rolling(window=min(200, len(closes))).mean().iloc[-1]
                price_to_ma50 = closes.iloc[-1] / ma50 - 1  # % diff from 50d MA
                price_to_ma200 = closes.iloc[-1] / ma200 - 1  # % diff from 200d MA
                
                # b. RSI (14-day)
                delta = closes.diff().dropna()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                avg_loss = loss.rolling(window=14, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs)).fillna(50)
                current_rsi = rsi.iloc[-1]
                
                # c. Volume trends
                if volumes is not None and not volumes.empty:
                    recent_volume = volumes.iloc[-5:].mean()
                    past_volume = volumes.iloc[-20:-5].mean() if len(volumes) >= 20 else volumes.mean()
                    volume_change = (recent_volume / past_volume) - 1 if past_volume > 0 else 0
                else:
                    volume_change = 0
                
                # d. Rate of change (ROC)
                roc_5 = (closes.iloc[-1] / closes.iloc[-min(6, len(closes))]) - 1
                roc_10 = (closes.iloc[-1] / closes.iloc[-min(11, len(closes))]) - 1
                
                # Calculate a composite score based on strategy
                if strategy == "momentum":
                    # Momentum strategy prioritizes recent performance and uptrends
                    score = (
                        0.3 * returns_1m +
                        0.3 * returns_3m +
                        0.2 * price_to_ma50 +
                        0.1 * (1 - volatility) +  # Lower volatility is better
                        0.1 * (volume_change if volume_change > 0 else 0)  # Increasing volume is good
                    )
                    
                elif strategy == "value":
                    # Value strategy looks for undervalued stocks (lower relative to MAs, oversold)
                    # Better if using P/E ratios etc., but using technical proxies here
                    score = (
                        0.3 * (0.3 - rsi/100) +  # Lower RSI is better (oversold)
                        0.3 * (-price_to_ma200) +  # Lower price vs 200MA
                        0.2 * (-price_to_ma50) +  # Lower price vs 50MA
                        0.2 * (1 - volatility)  # Lower volatility is better
                    )
                    
                elif strategy == "quality":
                    # Quality focuses on stable trends and lower volatility
                    score = (
                        0.3 * (1 - volatility) +
                        0.2 * (50 - abs(rsi - 50))/50 +  # Closer to RSI 50 is better (stable)
                        0.2 * (0.05 - abs(price_to_ma50)) +  # Near but above 50MA
                        0.2 * (0.05 - abs(price_to_ma200)) +  # Near but above 200MA
                        0.1 * ((returns_3m > 0) * returns_3m)  # Positive returns only
                    )
                    
                else:  # "blend" or default
                    # Balanced approach
                    momentum_score = 0.5 * returns_1m + 0.5 * returns_3m
                    trend_score = 0.7 * (price_to_ma50 > 0) + 0.3 * (price_to_ma200 > 0)
                    volatility_score = 1 - min(1, volatility)
                    rsi_score = 0
                    
                    if current_rsi < 30:  # Oversold
                        rsi_score = 0.8  # Good buying opportunity
                    elif current_rsi > 70:  # Overbought
                        rsi_score = 0.2  # Caution
                    else:
                        rsi_score = 0.5  # Neutral
                    
                    score = (
                        0.4 * momentum_score +
                        0.3 * trend_score +
                        0.2 * volatility_score +
                        0.1 * rsi_score
                    )
                
                # Store metrics
                metrics.append({
                    'symbol': symbol,
                    'score': score,
                    'returns_1m': returns_1m,
                    'returns_3m': returns_3m,
                    'volatility': volatility,
                    'rsi': current_rsi,
                    'price_to_ma50': price_to_ma50,
                    'price_to_ma200': price_to_ma200,
                    'volume_change': volume_change
                })
                
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
        selected_symbols = metrics_df.sort_values('score', ascending=False).head(num_stocks)['symbol'].tolist()
        
        # Print selected stocks with metrics
        print(f"\nSelected {len(selected_symbols)} stocks based on {strategy} strategy:")
        summary = metrics_df[metrics_df['symbol'].isin(selected_symbols)].set_index('symbol')
        
        # Format the summary for better display
        formatted_summary = summary.copy()
        for col in ['returns_1m', 'returns_3m', 'price_to_ma50', 'price_to_ma200', 'volume_change']:
            if col in formatted_summary.columns:
                formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{x:.2%}")
        
        formatted_summary['volatility'] = formatted_summary['volatility'].apply(lambda x: f"{x:.2%}")
        formatted_summary['rsi'] = formatted_summary['rsi'].apply(lambda x: f"{x:.1f}")
        formatted_summary['score'] = formatted_summary['score'].apply(lambda x: f"{x:.3f}")
        
        print(formatted_summary)
        return selected_symbols
        
    except Exception as e:
        print(f"Error in stock selection: {e}")
        # Fallback to default stocks
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"][:num_stocks]

def create_multi_stock_signal_generator(
    predictor_class, always_reverse=False, autodetect_reversal=False
):
    """
    Create a signal generator that works with multiple stocks

    Args:
        predictor_class: The StockPredictor class to use for each symbol
        always_reverse: If True, always reverse signals
        autodetect_reversal: If True, dynamically decide when to reverse signals

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

            decision, confidence, rationale, levels = get_entry_signal(
                predictor,
                symbol=symbol,
                current_price=float(historical_data["Close"].iloc[-1]),
                reverse_signals=use_reversal,
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
        # Log every 10 days
        # if len(historical_data) % 10 == 0:
        #     logger.info(f"Date: {historical_data.index[-1]}, Signal: {decision}, Confidence: {confidence}%")
        #     logger.info(f"Reversal: {use_reversal}, Rationale: {rationale}")

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


async def reconcile_balances(rest_client, address, tracker):
    """Reconcile on-chain balance with portfolio tracker balance"""
    print("Reconciling balances...")

    # Get actual on-chain balance
    account_address = AccountAddress.from_str(address)
    actual_balance = await check_balance(rest_client, account_address)
    actual_apt = actual_balance / 1e8  # Convert from octas to APT

    print(f"On-chain balance: {actual_apt} APT")
    print(f"Tracker cash balance: {tracker.portfolio['cash']} APT")

    # Update the tracker's cash amount to match the on-chain balance
    # This assumes all funds are in cash (not in positions)
    tracker.portfolio["cash"] = actual_apt

    # Record a new portfolio value with the updated balance
    tracker.calculate_current_value()

    print(f"Balances reconciled. New tracker balance: {tracker.portfolio['cash']} APT")
    return True


# async def main():

#     # Load or create wallet
#     # private_key, address, _ = load_or_create_wallet()

#     # # Initialize the clients
#     # rest_client = RestClient(NODE_URL)
#     # print("Connected to Aptos devnet")

#     # # Convert address string to AccountAddress
#     # account_address = AccountAddress.from_str(address)

#     # # Print current balance
#     # balance = await check_balance(rest_client, account_address)
#     # print(f"Current balance: {balance} octas ({balance/1e8} APT)")

#     # tracker = PortfolioTracker(
#     #     initial_capital=balance / 1e8
#     # )  # Initialize tracker with current balance

#     # # If balance is low, fund the wallet
#     # if balance < 100_000_000:  # Less than 1 APT
#     #     print("Balance is low, funding wallet...")
#     #     fund_wallet(address)

#     # fund_wallet(address, amount=100_000_000, coin_type="0x1::btc_coin_coin::BtcCoin")  # Fund with 1 APT for testing)
#     # Check balance again

#     # # Coin transfer example
#     # await reconcile_balances(rest_client, address, tracker)
#     # # Example: transfer 10,000 octas to yourself (for testing)
#     # recipient = address  # Replace with another address for real transfer
#     # transfer_amount = 10000
#     # success = await execute_transfer(private_key, recipient, transfer_amount)
#     # print(f"Transfer {'successful' if success else 'failed'}")

#     # Check entry points for trading
#     from predictor import StockPredictor

#     # symbol = "QBTS"
#     # symbol = "DOCU"
#     symbol = "JPM"

#     full_list_tickers = []
#     from pytickersymbols import PyTickerSymbols

#     for item in list(PyTickerSymbols().get_stocks_by_index("S&P 100")):
#         for diff_item in item["symbols"]:
#             if diff_item["currency"] == "USD":
#                 full_list_tickers.append(diff_item["yahoo"])
#     import random

#     # symbol = random.choice(full_list_tickers)
#     from autotrade_aptos import get_alpaca_tradable_cryptos

#     alpaca_cryptos, yf_crypto_symbols = get_alpaca_tradable_cryptos()
#     # symbol = random.choice(yf_crypto_symbols)  # Randomly select a crypto symbol from Alpaca tradable list
#     # symbol = 'CHTR'  # Example symbol, replace with your choice
#     # start = "2010-03-01"
#     start = "2020-12-01"
#     end = "2025-05-17"
#     # end = date.today()
#     _predictor = StockPredictor(symbol=symbol, start_date=start, end_date=end)
#     _predictor.load_data()
#     # print(_predictor.data.columns)
#     if symbol == "APT21794-USD":
#         backtester = AptosBacktester(symbol=symbol, initial_capital=balance / 1e8)
#     else:
#         backtester = AptosBacktester(symbol=symbol, initial_capital=100000)
#     # Run a simple backtest with default strategy
#     print("Running backtest...")

#     autodect_reversal = True  # Set to True to enable autodetection of reversal
#     history, metrics = backtester.run_backtest(
#         start_date=start,
#         end_date=end,
#         signal_generator=create_signal_generator(
#             predictor=_predictor,
#             always_reverse=False,
#             autodetect_reversal=autodect_reversal,
#         ),  # Use the predictor's signal generator, Can only choose one from autodetect_reversal and always_reverse
#     )
#     logger.info(f"If we autodetect reversal: {autodect_reversal}")

#     print("\nBacktest Results:")
#     print(f"Total Return: {metrics['total_return']:.2%}")
#     print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
#     print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
#     print(f"Win Rate: {metrics['win_rate']:.2%}")
#     print(f"Number of Trades: {metrics['num_trades']}")
#     print(f"Number of BUY orders: {metrics['num_buy_trades']}")
#     print(f"Number of SELL orders: {metrics['num_sell_trades']}")
#     print(f"CAGR: {metrics['cagr']:.2%}")
#     print(f"Sortino Ratio: {metrics['sortino']:.2f}")

#     # Plot results
#     backtester.plot_results(history)


# if __name__ == "__main__":
#     asyncio.run(main())


async def run_multi_stock_backtest():
    """Run a backtest with multiple stocks"""
    # Import necessary classes
    from predictor import StockPredictor
    start_date = "2021-12-01"
    end_date = "2025-05-31"
   



    try:
        selected_stocks = select_stocks(num_stocks=10,  universe="S&P 100", 
            lookback_days= 150,
            strategy="blend",  # You can choose: "momentum", "value", "quality", or "blend"
            end_date=start_date
        )
    except:
        # Fallback if import fails
        selected_stocks = ["AAPL", "MSFT", "GOOGL", ""]

    

    print(f"Selected stocks: {selected_stocks}")

    # Create backtester with multiple stocks
    backtester = AptosBacktester(symbols=selected_stocks, initial_capital=100000)

    # Define time period
    
    

    # Run backtest using multi-stock signal generator
    history, metrics = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        signal_generator=create_multi_stock_signal_generator(
            predictor_class=StockPredictor,
            always_reverse=False,
            autodetect_reversal=True,
        ),
    )

    # Print results
    print("\nMulti-Stock Portfolio Backtest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['num_trades']}")
    print(f"Number of BUY trades: {metrics['num_buy_trades']}")
    print(f"Number of SELL trades: {metrics['num_sell_trades']}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Sortino Ratio: {metrics['sortino']:.2f}")
    

    # Plot results
    backtester.plot_results(history)


# Add this to your main function
if __name__ == "__main__":
    # asyncio.run(main())
    # Uncomment the line below to run the multi-stock backtest
    asyncio.run(run_multi_stock_backtest())
