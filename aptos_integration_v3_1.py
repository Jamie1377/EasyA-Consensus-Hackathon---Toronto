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
import random


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
        predictor,
        symbol=None,
        current_price=None,
        reverse_signals=False,
        indicators_to_drop=["theta_extreme_high", "theta_extreme_low"],
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