from aptos_sdk.account import Account
from aptos_sdk.account_address import AccountAddress
from aptos_sdk.async_client import FaucetClient, RestClient
from aptos_sdk.transactions import EntryFunction, TransactionPayload, TransactionArgument, RawTransaction
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
                print(f"Existing Account:\nPrivate Key: {private_key}\nAddress: {address}")
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
        json.dump({
            "private_key": private_key,
            "address": address,
            "public_key": public_key
        }, f, indent=4)
    
    print(f"New account generated and saved to {wallet_file}")
    return private_key, address, public_key


def fund_wallet(address, amount=100_000_000):
    """Fund a wallet using the faucet"""
    url = f"https://faucet.devnet.aptoslabs.com/mint?address={address}&amount={amount}"
    headers = {"Content-Type": "application/json"}
    data = {"amount": amount, "coin_type": "0x1::aptos_coin::AptosCoin"}
    response = requests.post(url, headers=headers, json=data)
    print(f"Funding {address} with {amount/1e8} APT...")
    print("Funded!" if response.status_code == 200 else "Failed")
    return response.status_code == 200


async def build_transaction(rest_client, sender_address, recipient_address, amount):
    """Build a transaction to transfer APT"""
    print("\n=== 1. Building the transaction ===")
    
    # Create the entry function payload
    entry_function = EntryFunction.natural(
        "0x1::aptos_account",  # Module address and name
        "transfer",            # Function name
        [],                    # Type arguments
        [
            # Function arguments
            TransactionArgument(AccountAddress.from_str(recipient_address), Serializer.struct),
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
    print(f"Expiration Timestamp: {time.ctime(raw_transaction.expiration_timestamps_secs)}")
    
    return entry_function, sequence_number


async def simulate_transaction(rest_client, account, entry_function):
    """Simulate a transaction to estimate costs"""
    print("\n=== 2. Simulating the transaction ===")
    
    # Create a BCS transaction for simulation
    simulation_transaction = await rest_client.create_bcs_transaction(
        account, 
        TransactionPayload(entry_function)
    )
    
    # Simulate the transaction
    simulation_result = await rest_client.simulate_transaction(simulation_transaction, account)
    
    # Extract results
    gas_used = int(simulation_result[0]['gas_used'])
    gas_unit_price = int(simulation_result[0]['gas_unit_price'])
    success = simulation_result[0]['success']
    
    print(f"Estimated gas units: {gas_used}")
    print(f"Estimated gas cost: {gas_used * gas_unit_price} octas")
    print(f"Transaction would {'succeed' if success else 'fail'}")
    
    return success, gas_used, gas_unit_price


async def sign_and_submit_transaction(rest_client, account, entry_function, sequence_number):
    """Sign and submit a transaction"""
    print("\n=== 3. Signing the transaction ===")
    
    # Sign the transaction
    signed_transaction = await rest_client.create_bcs_signed_transaction(
        account,
        TransactionPayload(entry_function),
        sequence_number=sequence_number
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
        rest_client, 
        sender_address, 
        recipient_address, 
        amount
    )
    
    # Simulate the transaction
    success, gas_used, gas_unit_price = await simulate_transaction(
        rest_client, 
        account, 
        entry_function
    )
    
    if not success:
        print("Transaction simulation failed. Aborting.")
        return False
    
    # Sign and submit the transaction
    tx_hash = await sign_and_submit_transaction(
        rest_client, 
        account, 
        entry_function, 
        sequence_number
    )
    
    # Wait for the transaction to complete
    tx_success, vm_status, final_gas_used = await wait_for_transaction(
        rest_client, 
        tx_hash
    )
    
    # Check final balance
    final_balance = await check_balance(rest_client, sender_address)
    print("\n=== Final Balances ===")
    print(f"Balance: {final_balance} octas (spent {initial_balance - final_balance} octas on transfer and gas)")
    
    return tx_success


# async def check_entry_points(symbol="APT21794-USD"):
#     """
#     Check entry points for Aptos token and execute trades when appropriate
    
#     Args:
#         symbol: The token symbol to check for trading opportunities
#     """
#     # Load or create wallet
#     private_key, address, _ = load_or_create_wallet()
    
#     # Initialize REST client
#     rest_client = RestClient(NODE_URL)
#     print(f"Connected to Aptos network at {NODE_URL}")
    
#     # Convert address string to AccountAddress
#     account_address = AccountAddress.from_str(address)
    
#     # Check current balance
#     balance = await check_balance(rest_client, account_address)
#     print(f"Current balance: {balance} octas ({balance/1e8} APT)")
    
#     try:
#         # Import necessary functions from stock_prediction
#         from stock_prediction.core import StockPredictor
#         import pandas as pd
#         from datetime import date, timedelta, datetime
#         import yfinance as yf
        
#         # Create a predictor instance
#         if date.today().weekday() == 0:  # If Monday, get more history
#             predictor = StockPredictor(
#                 symbol=symbol,
#                 start_date=date.today() - pd.Timedelta(days=500),
#                 end_date=date.today() + pd.Timedelta(days=1),
#                 interval="1d"  # Using hourly data for crypto
#             )
#         else:
#             predictor = StockPredictor(
#                 symbol=symbol,
#                 start_date=date.today() - pd.Timedelta(days=500),
#                 end_date=date.today() + pd.Timedelta(days=1),
#                 interval="1d"  # Using hourly data for crypto
#             )
        
#         # Load data and prepare for analysis
#         predictor.load_data()
#         print(predictor.data.head())
        
#         # Get trading signal with confidence levels
#         decision, confidence, rationale, levels = get_entry_signal(predictor, symbol)
        
#         print(f"\n🔍 {symbol} Entry Check:")
#         print(f"  Decision: {decision} ({confidence}% confidence)")
#         print(f"  Rationale: {rationale}")
#         print(f"  Key Levels:")
#         print(f"    Current: ${levels['current_price'][0]:.2f}")
#         print(f"    Stop Loss: ${levels['stop_loss'][0]:.2f}")
#         print(f"    Take Profit: ${levels['take_profit'][0]:.2f}")
        
#         # Execute trade if decision is BUY or SELL and we have enough confidence
#         if decision != "HOLD" and confidence > 65:
#             amount = 0
#             if decision == "BUY":
#                 # Calculate position size (1% of available balance)
#                 amount = int(balance * 0.01)
#                 print(f"Executing BUY: {amount/1e8} APT")
#                 # Execute transfer to a trading smart contract or another wallet
#                 await execute_transfer(private_key, address, amount)
                
#             elif decision == "SELL" and balance > 0:
#                 # Calculate position size (1% of available balance)
#                 amount = int(balance * 0.01)
#                 print(f"Executing SELL: {amount/1e8} APT")
#                 # For sell, we would execute a different type of transaction
#                 # This is a placeholder - in a real system you'd have a proper sell function
#                 recipient = "0x5ae6789dd2fec1a9ec9cccfb3acaf12e93d432f0a3a42c92fe1a9d490b7bbc06"  # Example trading contract
#                 await execute_transfer(private_key, recipient, amount)
            
#             print(f"Transaction completed with amount: {amount/1e8} APT")
    
#     except Exception as e:
#         print(f"Error in check_entry_points: {str(e)}")


# def get_entry_signal(predictor, symbol=None, current_price: int =None):
#     """Generate real-time entry signal with confidence scoring
#     Returns: (decision, confidence, rationale, levels)
#     """
#     # Get real-time price if available
#     if current_price is None:
#         if symbol is None:
#             symbol = predictor.symbol
            
#         try:
#             current_price = yf.download(symbol, period='1d', interval='1m')['Close'].iloc[-1]
#             current_market = yf.download('^GSPC', period='1d', interval='1m')['Close'].iloc[-1]
#         except:
#             # Fallback if real-time data is unavailable
#             current_price = predictor.data['Close'].iloc[-1]
#             current_market = 0
            
#     last_row = predictor.data.iloc[-1]
#     second_last_row = predictor.data.iloc[-2] if len(predictor.data) > 1 else last_row
    
#     # Calculate signal components
#     signals = {
#         'trend': {
#             'value': current_price > predictor.data['Close'].rolling(50).mean().iloc[-1] if 'MA_50' not in predictor.data.columns else current_price > second_last_row['MA_50'],
#             'weight': 0.5
#         },
#         'momentum': {
#             'value': last_row['RSI'] < 65 if 'RSI' in predictor.data.columns else True,
#             'weight': 0.5
#         },
#         'volume': {
#             'value': last_row['Volume'] > predictor.data['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in predictor.data.columns else True,
#             'weight': 0.5
#         },
#         'volatility': {
#             'value': last_row['ATR'] > predictor.data['ATR'].rolling(14).mean().iloc[-1] if 'ATR' in predictor.data.columns else True,
#             'weight': 1
#         }
#     }
    
#     # Calculate score (0-5 scale)
#     score = sum(condition['weight'] for name, condition in signals.items() if condition['value'] == True)
#     max_score = sum(condition['weight'] for name, condition in signals.items())
#     confidence = min(100, max(0, int((score / max_score) * 100)))
    
#     # Generate rationale
#     rationales = []
#     if signals['trend']['value'] == True:
#         rationales.append(f"📈 Price {current_price:.2f} above 50MA")
#     else:
#         rationales.append(f"📉 Price {current_price:.2f} below 50MA")
        
#     if 'RSI' in predictor.data.columns and signals['momentum']['value'] == True:
#         rationales.append(f"💪 Moderate momentum (RSI {last_row['RSI']:.1f})")
    
#     # Make decision
#     decision = "BUY" if score >= max_score * 0.6 else "SELL" if score <= max_score * 0.3 else "HOLD"
    
#     # Add risk parameters
#     risk_params = {
#         'stop_loss_pct': 0.03,  # 3% stop loss
#         'take_profit_pct': 0.05,  # 5% take profit
#     }
    
#     return (
#         decision,
#         confidence,
#         " | ".join(rationales),
#         {
#             'current_price': [current_price],
#             'stop_loss': [current_price * (1 - risk_params['stop_loss_pct'])] if decision == "BUY" 
#                         else [current_price * (1 + risk_params['stop_loss_pct'])],
#             'take_profit': [current_price * (1 + risk_params['take_profit_pct'])] if decision == "BUY" 
#                           else [current_price * (1 - risk_params['take_profit_pct'])],
#         }
#     )

# def get_entry_signal(predictor, symbol=None, current_price:int =None):
#     """Generate real-time entry signal with confidence scoring
#     Returns: (decision, confidence, rationale, levels)
#     """
#     # Get real-time price if available
#     if current_price is None:
#         if symbol is None:
#             symbol = predictor.symbol
            
#         try:
#             current_price = yf.download(symbol, period='1d', interval='1m')['Close'].iloc[-1]
#             current_market = yf.download('^GSPC', period='1d', interval='1m')['Close'].iloc[-1]
#         except:
#             # Fallback if real-time data is unavailable
#             current_price = predictor.data['Close'].iloc[-1]
#             current_market = 0
            
#     last_row = predictor.data.iloc[-1]
#     second_last_row = predictor.data.iloc[-2] if len(predictor.data) > 1 else last_row
    
#     # Calculate signal components - FIX: Use .iloc[-1] to get scalar values or .bool() for Series
#     signals = {
#         'trend': {
#             'value': float(current_price) > float(predictor.data['Close'].rolling(50).mean().iloc[-1]) if 'MA_50' not in predictor.data.columns else float(current_price) > float(second_last_row['MA_50']),
#             'weight': 0.5
#         },
#         'momentum': {
#             'value': float(last_row['RSI']) < 65 if 'RSI' in predictor.data.columns else True,
#             'weight': 0.5
#         },
#         'volume': {
#             'value': float(last_row['Volume']) > float(predictor.data['Volume'].rolling(20).mean().iloc[-1]) if 'Volume' in predictor.data.columns else True,
#             'weight': 0.5
#         },
#         'volatility': {
#             'value': float(last_row['ATR']) > float(predictor.data['ATR'].rolling(14).mean().iloc[-1]) if 'ATR' in predictor.data.columns else True,
#             'weight': 1
#         }
#     }
    
#     # Calculate score (0-5 scale)
#     score = sum(condition['weight'] for name, condition in signals.items() if condition['value'] == True)
#     max_score = sum(condition['weight'] for name, condition in signals.items())
#     confidence = min(100, max(0, int((score / max_score) * 100)))
    
#     # Generate rationale
#     rationales = []
#     if signals['trend']['value'] == True:
#         rationales.append(f"📈 Price {current_price:.2f} above 50MA")
#     else:
#         rationales.append(f"📉 Price {current_price:.2f} below 50MA")
        
#     if 'RSI' in predictor.data.columns and signals['momentum']['value'] == True:
#         rationales.append(f"💪 Moderate momentum (RSI {float(last_row['RSI']):.1f})")
    
#     # Make decision
#     decision = "BUY" if score >= max_score * 0.6 else "SELL" if score <= max_score * 0.3 else "HOLD"
    
#     # Add risk parameters
#     risk_params = {
#         'stop_loss_pct': 0.03,  # 3% stop loss
#         'take_profit_pct': 0.05,  # 5% take profit
#     }
    
#     return (
#         decision,
#         confidence,
#         " | ".join(rationales),
#         {
#             'current_price': [float(current_price)],
#             'stop_loss': [float(current_price) * (1 - risk_params['stop_loss_pct'])] if decision == "BUY" 
#                         else [float(current_price) * (1 + risk_params['stop_loss_pct'])],
#             'take_profit': [float(current_price) * (1 + risk_params['take_profit_pct'])] if decision == "BUY" 
#                           else [float(current_price) * (1 - risk_params['take_profit_pct'])],
#         }
#     )

def get_entry_signal(predictor, symbol=None, current_price=None):
    """Generate real-time entry signal with confidence scoring
    Returns: (decision, confidence, rationale, levels)
    """
    # Get real-time price if available
    if current_price is None:
        if symbol is None:
            symbol = predictor.symbol
            
        try:
            current_price = float(yf.download(symbol, period='1d', interval='1m')['Close'].iloc[-1])
            current_market = float(yf.download('^GSPC', period='1d', interval='1m')['Close'].iloc[-1])
        except Exception as e:
            print(f"Error getting real-time price: {e}")
            # Fallback if real-time data is unavailable
            current_price = float(predictor.data['Close'].iloc[-1])
            current_market = 0
            
    last_row = predictor.data.iloc[-1]
    second_last_row = predictor.data.iloc[-2] if len(predictor.data) > 1 else last_row
    
    # Calculate signal components - Make sure we're using scalar values
    signals = {
        'trend': {
            'value': float(current_price) > float(predictor.data['Close'].rolling(50).mean().iloc[-1]) if 'MA_50' not in predictor.data.columns else float(current_price) > float(second_last_row['MA_50']),
            'weight': 0.5
        },
        'momentum': {
            'value': float(last_row.get('RSI', 0)) < 65 if 'RSI' in predictor.data.columns else True,
            'weight': 0.5
        },
        'volume': {
            'value': float(last_row.get('Volume', 0)) > float(predictor.data['Volume'].rolling(20).mean().iloc[-1]) if 'Volume' in predictor.data.columns else True,
            'weight': 0.5
        },
        'volatility': {
            'value': float(last_row.get('ATR', 0)) > float(predictor.data['ATR'].rolling(14).mean().iloc[-1]) if 'ATR' in predictor.data.columns else True,
            'weight': 1
        }
    }
    
    # Calculate score (0-5 scale)
    score = sum(condition['weight'] for name, condition in signals.items() if condition['value'] == True)
    max_score = sum(condition['weight'] for name, condition in signals.items())
    confidence = min(100, max(0, int((score / max_score) * 100)))
    
    # Generate rationale with proper formatting of scalar values
    rationales = []
    if signals['trend']['value'] == True:
        rationales.append(f"📈 Price {float(current_price):.2f} above 50MA")
    else:
        rationales.append(f"📉 Price {float(current_price):.2f} below 50MA")
        
    if 'RSI' in predictor.data.columns and signals['momentum']['value'] == True:
        # Make sure we're using a scalar value for RSI
        rsi_value = float(last_row.get('RSI', 0)) 
        rationales.append(f"💪 Moderate momentum (RSI {rsi_value:.1f})")
    
    # Make decision
    decision = "BUY" if score >= max_score * 0.6 else "SELL" if score <= max_score * 0.3 else "HOLD"
    
    # Add risk parameters
    risk_params = {
        'stop_loss_pct': 0.03,  # 3% stop loss
        'take_profit_pct': 0.05,  # 5% take profit
    }
    
    current_price_float = float(current_price)
    
    return (
        decision,
        confidence,
        " | ".join(rationales),
        {
            'current_price': [current_price_float],
            'stop_loss': [current_price_float * (1 - risk_params['stop_loss_pct'])] if decision == "BUY" 
                        else [current_price_float * (1 + risk_params['stop_loss_pct'])],
            'take_profit': [current_price_float * (1 + risk_params['take_profit_pct'])] if decision == "BUY" 
                          else [current_price_float * (1 - risk_params['take_profit_pct'])],
        }
    )



# Configure logging
log_directory = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_directory, f"aptos_trading_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PortfolioTracker:
    """Track Aptos positions and calculate profit/loss"""
    
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},  # symbol -> {qty, avg_entry_price}
            'transactions': [],  # [type, price, qty, timestamp]
            'value_history': []  # [{timestamp, value, cash}]
        }
        self.transaction_log_file = os.path.join(log_directory, "aptos_transactions.csv")
        self.portfolio_log_file = os.path.join(log_directory, "aptos_portfolio.csv")
        
        # Initialize transaction log if it doesn't exist
        if not os.path.exists(self.transaction_log_file):
            pd.DataFrame(columns=['timestamp', 'type', 'symbol', 'price', 'quantity', 'value']).to_csv(
                self.transaction_log_file, index=False
            )
        
        # Initialize portfolio log if it doesn't exist
        if not os.path.exists(self.portfolio_log_file):
            pd.DataFrame(columns=['timestamp', 'total_value', 'cash', 'positions']).to_csv(
                self.portfolio_log_file, index=False
            )
    
    def record_transaction(self, transaction_type, symbol, price, quantity):
        """Record a buy or sell transaction"""
        timestamp = datetime.now()
        value = price * quantity
        
        # Add to in-memory transaction list
        self.portfolio['transactions'].append((transaction_type, price, quantity, timestamp))
        
        # Log transaction to file
        transaction_df = pd.DataFrame([{
            'timestamp': timestamp,
            'type': transaction_type,
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'value': value
        }])
        
        transaction_df.to_csv(self.transaction_log_file, mode='a', header=False, index=False)
        logger.info(f"{transaction_type} executed: {quantity} {symbol} at ${price:.4f}")
        
        return True
    
    def update_position(self, symbol, price, quantity, transaction_type):
        """Update portfolio positions after a transaction"""
        if transaction_type == "BUY":
            # Add to position
            if symbol in self.portfolio['positions']:
                current_position = self.portfolio['positions'][symbol]
                
                # Calculate new average entry price
                total_shares = current_position['qty'] + quantity
                new_avg_price = (
                    (current_position['qty'] * current_position['avg_entry_price']) + 
                    (quantity * price)
                ) / total_shares
                
                self.portfolio['positions'][symbol] = {
                    'qty': total_shares,
                    'avg_entry_price': new_avg_price
                }
            else:
                # New position
                self.portfolio['positions'][symbol] = {
                    'qty': quantity,
                    'avg_entry_price': price
                }
                
            # Deduct from cash
            self.portfolio['cash'] -= (price * quantity)
            
        elif transaction_type == "SELL":
            if symbol in self.portfolio['positions']:
                current_position = self.portfolio['positions'][symbol]
                
                # Reduce position
                if quantity >= current_position['qty']:
                    # Selling entire position
                    quantity = current_position['qty']
                    del self.portfolio['positions'][symbol]
                else:
                    # Partial sale - keep same average price
                    self.portfolio['positions'][symbol]['qty'] -= quantity
                
                # Add to cash
                self.portfolio['cash'] += (price * quantity)
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
        for symbol, position in self.portfolio['positions'].items():
            # Use provided price or current position's entry price as fallback
            price = symbol_prices.get(symbol, position['avg_entry_price'])
            position_value += position['qty'] * price
            
        total_value = self.portfolio['cash'] + position_value
        
        # Record in history
        timestamp = datetime.now()
        self.portfolio['value_history'].append({
            'timestamp': timestamp,
            'value': total_value,
            'cash': self.portfolio['cash']
        })
        
        # Log portfolio value
        portfolio_record = {
            'timestamp': timestamp,
            'total_value': total_value,
            'cash': self.portfolio['cash'],
            'positions': str(self.portfolio['positions'])  # Convert positions dict to string
        }
        
        pd.DataFrame([portfolio_record]).to_csv(
            self.portfolio_log_file, mode='a', header=False, index=False
        )
        
        return total_value
    
    def get_pnl_metrics(self):
        """Calculate performance metrics"""
        if not self.portfolio['value_history']:
            return {
                'total_return': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0,
                'win_rate': 0,
                'num_trades': 0
            }
        
        # Calculate total return
        current_value = self.portfolio['value_history'][-1]['value']
        total_return = (current_value / self.initial_capital) - 1
        
        # Calculate realized P&L from completed trades
        realized_pnl = 0
        buy_positions = {}
        for transaction_type, price, qty, timestamp in self.portfolio['transactions']:
            if transaction_type == "BUY":
                # Add to open positions
                if 'APT' not in buy_positions:
                    buy_positions['APT'] = []
                buy_positions['APT'].append((price, qty))
            elif transaction_type == "SELL":
                # Calculate profit for matched positions (FIFO)
                remaining_qty = qty
                while remaining_qty > 0 and 'APT' in buy_positions and buy_positions['APT']:
                    buy_price, buy_qty = buy_positions['APT'][0]
                    
                    if buy_qty <= remaining_qty:
                        # Fully realize this buy position
                        realized_pnl += (price - buy_price) * buy_qty
                        remaining_qty -= buy_qty
                        buy_positions['APT'].pop(0)
                    else:
                        # Partially realize this buy position
                        realized_pnl += (price - buy_price) * remaining_qty
                        buy_positions['APT'][0] = (buy_price, buy_qty - remaining_qty)
                        remaining_qty = 0
        
        # Calculate unrealized P&L for current positions
        unrealized_pnl = 0
        for symbol, position in self.portfolio['positions'].items():
            # For simplicity, we use the last transaction price
            last_price = self.portfolio['transactions'][-1][1] if self.portfolio['transactions'] else 0
            unrealized_pnl += (last_price - position['avg_entry_price']) * position['qty']
        
        # Calculate win rate
        num_trades = len([t for t in self.portfolio['transactions'] if t[0] == "SELL"])
        winning_trades = 0
        for i in range(len(self.portfolio['transactions'])):
            if self.portfolio['transactions'][i][0] == "SELL":
                sell_price = self.portfolio['transactions'][i][1]
                
                # Look for matching buy transaction
                for j in range(i):
                    if self.portfolio['transactions'][j][0] == "BUY":
                        buy_price = self.portfolio['transactions'][j][1]
                        if sell_price > buy_price:
                            winning_trades += 1
                        break
        
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'win_rate': win_rate,
            'num_trades': num_trades
        }


# async def execute_trade_with_tracking(signal, symbol, size, tracker):
#     """Execute a trade and update portfolio tracking"""
#     # Load or create wallet
#     private_key, address, _ = load_or_create_wallet()
    
#     # Connect to Aptos network
#     rest_client = RestClient(NODE_URL)
#     account_address = AccountAddress.from_str(address)
    
#     # Check current balance
#     balance = await check_balance(rest_client, account_address)
#     logger.info(f"Current balance: {balance/1e8} APT")
    
#     # Get current token price
#     try:
#         import yfinance as yf
#         current_price = float(yf.download(symbol, period='1d', interval='1m')['Close'].iloc[-1])
#     except Exception as e:
#         logger.error(f"Error getting price: {str(e)}")
#         current_price = 1.0  # Default value
    
#     # Calculate position size
#     if signal == 'BUY':
#         # Calculate position size in APT (use a percentage of balance)
#         position_size = int(balance * size)
        
#         if position_size > 0:
#             # Execute transfer to our own wallet (in a real system, you'd transfer to a trading contract)
#             logger.info(f"Executing BUY: {position_size/1e8} APT at ${current_price}")
#             await execute_transfer(private_key, address, position_size)
            
#             # Update portfolio tracker
#             tracker.update_position(symbol, current_price, position_size/1e8, "BUY")
            
#     elif signal == 'SELL' and balance > 0:
#         # Calculate position size (sell a percentage of balance)
#         position_size = int(balance * size)
        
#         if position_size > 0:
#             logger.info(f"Executing SELL: {position_size/1e8} APT at ${current_price}")
            
#             # In a real system, you would execute a SELL transaction
#             # Here we just transfer back to ourselves
#             await execute_transfer(private_key, address, position_size)
            
#             # Update portfolio tracker
#             tracker.update_position(symbol, current_price, position_size/1e8, "SELL")
    
#     # Calculate current portfolio value
#     current_value = tracker.calculate_current_value({symbol: current_price})
#     metrics = tracker.get_pnl_metrics()
    
#     logger.info(f"Portfolio value: ${current_value:.2f}, PnL: ${metrics['realized_pnl']:.2f}")
#     return True

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
        current_price = float(yf.download(symbol, period='1d', interval='1m')['Close'].iloc[-1])
        logger.info(f"Current market price: ${current_price}")
    except Exception as e:
        logger.error(f"Error getting price: {str(e)}")
        current_price = 1.0  # Default value
    
    # Simulate a trading contract address (in production this would be a real trading contract)
    trading_contract = "0x5ae6789dd2fec1a9ec9cccfb3acaf12e93d432f0a3a42c92fe1a9d490b7bbc06"
    
    # Execute the trade based on signal
    if signal == 'BUY':
        # Calculate APT amount based on USD size and current price
        # Example: If we want to buy $10 worth of the asset at current price
        usd_amount = balance/1e8 * size  # Use a percentage of our balance in USD
        apt_amount = int((usd_amount / current_price) * 1e8)  # Convert to APT octas
        
        if apt_amount > 0:
            logger.info(f"Executing BUY: ${usd_amount:.2f} worth ({apt_amount/1e8} APT) at ${current_price}")
            
            # In a real system, you'd transfer to a trading contract
            # For simulation, we'll transfer a small amount to the "trading contract" address
            # This simulates sending funds to exchange/protocol
            simulation_amount = min(apt_amount, 10000)  # Limit to small amount for testing
            await execute_transfer(private_key, trading_contract, simulation_amount)
            
            # Record the trade in our portfolio tracker
            # Use the full calculated amount for portfolio tracking, even though we only transfer a small simulation
            tracker.update_position(symbol, current_price, usd_amount/current_price, "BUY")
            logger.info(f"Bought {usd_amount/current_price:.6f} units at ${current_price}")
            
    elif signal == 'SELL':
        # Check if we have a position to sell
        if symbol in tracker.portfolio['positions']:
            position = tracker.portfolio['positions'][symbol]
            
            # Calculate how much to sell (percentage of our position)
            sell_quantity = position['qty'] * size
            usd_value = sell_quantity * current_price
            
            logger.info(f"Executing SELL: {sell_quantity} units (${usd_value:.2f}) at ${current_price}")
            
            # In a real system, this would execute on the trading protocol
            # For simulation, we'll transfer from the "trading contract" to our wallet
            simulation_amount = 10000  # Simulate receiving funds back (small fixed amount)
            
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
    
    logger.info(f"Portfolio value: ${current_value:.2f}, Realized PnL: ${metrics['realized_pnl']:.2f}")
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
            interval="1d"
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
                tracker=tracker
            )
        else:
            logger.info(f"No trade executed - {decision} with {confidence}% confidence")
            
        # Calculate current metrics
        metrics = tracker.get_pnl_metrics()
        logger.info(f"Current metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error in check_entry_points: {str(e)}")
        
    return tracker




class AptosBacktester:
    """
    Portfolio tracker for Aptos transactions with functionality similar to 
    the Backtester in stock_prediction/core/predictor.py
    """
    
    def __init__(self, symbol="APT21794-USD", initial_capital=100):
        """
        Initialize the backtester
        
        Args:
            symbol (str): Token symbol to track
            initial_capital (float): Initial capital in USD
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},  # symbol -> {qty, entry_price}
            'value_history': [],  # [{timestamp, value, cash}]
            'transactions': []  # [type, price, qty, timestamp]
        }
        
        # Trade parameters
        self.slippage = 0.0005  # 5 basis points
        self.commission = 0.001  # 0.1% per transaction
        
        # Configure logging
        log_directory = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(log_directory, f"aptos_backtest_{date.today().strftime('%Y%m%d')}.log")
        
        self.logger = logging.getLogger('aptos_backtest')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)

        # Setup transaction log files
        self.transaction_log_file = os.path.join(log_directory, "aptos_transactions.csv")
        self.portfolio_log_file = os.path.join(log_directory, "aptos_portfolio.csv")
        
        # Initialize transaction log if it doesn't exist
        if not os.path.exists(self.transaction_log_file):
            pd.DataFrame(columns=['timestamp', 'type', 'symbol', 'price', 'quantity', 'value']).to_csv(
                self.transaction_log_file, index=False
            )
        
        # Initialize portfolio log if it doesn't exist
        if not os.path.exists(self.portfolio_log_file):
            pd.DataFrame(columns=['timestamp', 'total_value', 'cash', 'positions']).to_csv(
                self.portfolio_log_file, index=False
            )
        
        self.logger.info(f"AptosBacktester initialized with {initial_capital} USD for {symbol}")

    def record_transaction(self, transaction_type, symbol, price, quantity):
        """Record a buy or sell transaction"""
        timestamp = datetime.now()
        value = price * quantity
        
        # Add to in-memory transaction list
        self.portfolio['transactions'].append((transaction_type, price, quantity, timestamp))
        
        # Log transaction to file
        transaction_df = pd.DataFrame([{
            'timestamp': timestamp,
            'type': transaction_type,
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'value': value
        }])
        
        transaction_df.to_csv(self.transaction_log_file, mode='a', header=False, index=False)
        self.logger.info(f"{transaction_type} executed: {quantity:.6f} {symbol} at ${price:.4f}")
        
        return True
    
    def execute_buy(self, price, quantity, timestamp=None):
        """Execute a buy order"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Apply slippage to buy price
        executed_price = price * (1 + self.slippage)
        
        # Calculate transaction cost
        cost = executed_price * quantity
        commission = cost * self.commission
        total_cost = cost + commission
        
        # Check if enough cash
        if total_cost > self.portfolio['cash']:
            self.logger.warning(f"Insufficient funds for purchase. Required: ${total_cost:.2f}, Available: ${self.portfolio['cash']:.2f}")
            # Adjust quantity to available cash
            adjusted_quantity = (self.portfolio['cash'] * 0.98) / executed_price  # Leave 2% buffer
            if adjusted_quantity <= 0:
                return False
                
            quantity = adjusted_quantity
            cost = executed_price * quantity
            commission = cost * self.commission
            total_cost = cost + commission
            self.logger.info(f"Adjusted buy quantity to {quantity:.6f} based on available funds")
        
        # Update portfolio
        self.portfolio['cash'] -= total_cost
        
        if self.symbol in self.portfolio['positions']:
            # Update existing position
            position = self.portfolio['positions'][self.symbol]
            total_quantity = position['qty'] + quantity
            avg_price = ((position['qty'] * position['entry_price']) + (quantity * executed_price)) / total_quantity
            
            self.portfolio['positions'][self.symbol] = {
                'qty': total_quantity,
                'entry_price': avg_price
            }
        else:
            # Create new position
            self.portfolio['positions'][self.symbol] = {
                'qty': quantity,
                'entry_price': executed_price
            }
        
        # Record the transaction
        self.record_transaction('BUY', self.symbol, executed_price, quantity)
        
        return True
    
    def execute_sell(self, price, quantity=None, timestamp=None):
        """Execute a sell order"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Apply slippage to sell price (negative for sells)
        executed_price = price * (1 - self.slippage)
        
        # If no quantity specified, sell all holdings
        if quantity is None:
            if self.symbol in self.portfolio['positions']:
                quantity = self.portfolio['positions'][self.symbol]['qty']
            else:
                self.logger.warning(f"No position in {self.symbol} to sell")
                return False
        
        # Check if trying to sell more than owned
        if self.symbol in self.portfolio['positions']:
            position = self.portfolio['positions'][self.symbol]
            if quantity > position['qty']:
                self.logger.warning(f"Attempted to sell {quantity} but only have {position['qty']}")
                quantity = position['qty']
        else:
            self.logger.warning(f"No position in {self.symbol} to sell")
            return False
            
        # Calculate proceeds
        proceeds = executed_price * quantity
        commission = proceeds * self.commission
        net_proceeds = proceeds - commission
        
        # Update portfolio
        self.portfolio['cash'] += net_proceeds
        
        # Update or remove position
        position = self.portfolio['positions'][self.symbol]
        if quantity >= position['qty']:
            # Selling entire position
            self.portfolio['positions'].pop(self.symbol)
        else:
            # Partial sale
            position['qty'] -= quantity
            
        # Record the transaction
        self.record_transaction('SELL', self.symbol, executed_price, quantity)
        
        return True
        
    def update_portfolio_value(self, current_price=None):
        """Calculate current portfolio value and record to history"""
        timestamp = datetime.now()
        
        # If no price provided, try to get current price
        if current_price is None:
            try:
                current_price = float(yf.download(self.symbol, period='1d', interval='1m')['Close'].iloc[-1])
            except Exception as e:
                self.logger.error(f"Failed to get current price: {e}")
                # Use last known price or position entry price as fallback
                if self.portfolio['value_history']:
                    # Use last valuation price
                    current_price = self.portfolio['value_history'][-1].get('price', 1.0)
                elif self.symbol in self.portfolio['positions']:
                    # Use position entry price
                    current_price = self.portfolio['positions'][self.symbol]['entry_price']
                else:
                    # Default to 1.0 if all else fails
                    current_price = 1.0
                    
        # Calculate position value
        position_value = 0
        for symbol, position in self.portfolio['positions'].items():
            position_value += position['qty'] * current_price
            
        total_value = self.portfolio['cash'] + position_value
        
        # Record to history
        value_entry = {
            'timestamp': timestamp,
            'value': total_value,
            'cash': self.portfolio['cash'],
            'price': current_price
        }
        self.portfolio['value_history'].append(value_entry)
        
        # Log to portfolio file
        portfolio_record = {
            'timestamp': timestamp,
            'total_value': total_value,
            'cash': self.portfolio['cash'],
            'positions': str(self.portfolio['positions']),
            'price': current_price
        }
        
        pd.DataFrame([portfolio_record]).to_csv(
            self.portfolio_log_file, mode='a', header=False, index=False
        )
        
        return total_value
        
    # def run_backtest(self, start_date, end_date, price_data=None, signal_generator=None):
    #     """
    #     Run a backtest over a date range
        
    #     Args:
    #         start_date (str): Start date for backtest (YYYY-MM-DD)
    #         end_date (str): End date for backtest (YYYY-MM-DD)
    #         price_data (pd.DataFrame): Historical price data (if None, will be downloaded)
    #         signal_generator (callable): Function that generates trading signals
    #                                     Should return "BUY", "SELL", or "HOLD"
        
    #     Returns:
    #         tuple: (history_df, performance_metrics)
    #     """
    #     self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
    #     # Get price data if not provided
    #     if price_data is None:
    #         try:
    #             price_data = yf.download(self.symbol, start=start_date, end=end_date, interval='1d')
    #             if price_data.empty:
    #                 self.logger.error(f"No data available for {self.symbol} in the specified date range")
    #                 return pd.DataFrame(), {"error": "No data available"}
    #         except Exception as e:
    #             self.logger.error(f"Failed to download price data: {e}")
    #             return pd.DataFrame(), {"error": f"Failed to download data: {str(e)}"}
        
    #     # Reset portfolio for backtest
    #     self.portfolio = {
    #         'cash': self.initial_capital,
    #         'positions': {},
    #         'value_history': [],
    #         'transactions': []
    #     }
        
    #     # Daily loop for backtest
    #     dates = price_data.index
        
    #     for i, date in enumerate(dates):
    #         # Skip first day (need previous data for signals)
    #         if i == 0:
    #             # Just record initial portfolio value
    #             self.update_portfolio_value(price_data['Close'].iloc[i])
    #             continue
                
    #         current_price = price_data['Close'].iloc[i]
            
    #         # Generate trading signal
    #         if signal_generator:
    #             # Use provided signal generator with historical data up to this point
    #             signal = signal_generator(price_data.iloc[:i+1])
    #         else:
    #             # Simple momentum strategy as default
    #             prev_price = price_data['Close'].iloc[i-1]
    #             if current_price > prev_price * 1.01:  # 1% increase
    #                 signal = "BUY"
    #             elif current_price < prev_price * 0.99:  # 1% decrease
    #                 signal = "SELL"
    #             else:
    #                 signal = "HOLD"
            
    #         # Execute trades based on signal
    #         if signal == "BUY" and self.portfolio['cash'] > 0:
    #             # Calculate position size (use 90% of available cash)
    #             position_size = self.portfolio['cash'] * 0.9 / current_price
    #             self.execute_buy(current_price, position_size, timestamp=date)
                
    #         elif signal == "SELL" and self.symbol in self.portfolio['positions']:
    #             # Sell all holdings
    #             self.execute_sell(current_price, timestamp=date)
            
    #         # Update portfolio value
    #         self.update_portfolio_value(current_price)
            
    #     # Generate report
    #     return self.generate_report()

    def run_backtest(self, start_date, end_date, price_data=None, signal_generator=None):
        """
        Run a backtest over a date range
        
        Args:
            start_date (str): Start date for backtest (YYYY-MM-DD)
            end_date (str): End date for backtest (YYYY-MM-DD)
            price_data (pd.DataFrame): Historical price data (if None, will be downloaded)
            signal_generator (callable): Function that generates trading signals
                                        Should return "BUY", "SELL", or "HOLD"
        
        Returns:
            tuple: (history_df, performance_metrics)
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Get price data if not provided
        if price_data is None:
            try:
                price_data = yf.download(self.symbol, start=start_date, end=end_date, interval='1d')
                if price_data.empty:
                    self.logger.error(f"No data available for {self.symbol} in the specified date range")
                    return pd.DataFrame(), {"error": "No data available"}
            except Exception as e:
                self.logger.error(f"Failed to download price data: {e}")
                return pd.DataFrame(), {"error": f"Failed to download data: {str(e)}"}
        
        # Reset portfolio for backtest
        self.portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'value_history': [],
            'transactions': []
        }
        
        # Daily loop for backtest
        dates = price_data.index
        
        for i, date in enumerate(dates):
            # Skip first day (need previous data for signals)
            if i == 0:
                # Just record initial portfolio value
                self.update_portfolio_value(float(price_data['Close'].iloc[i]))
                continue
                
            # FIX: Convert Series to float to avoid comparison issues
            current_price = float(price_data['Close'].iloc[i])
            
            # Generate trading signal
            if signal_generator:
                # Use provided signal generator with historical data up to this point
                signal = signal_generator(price_data.iloc[:i+1])
            else:
                # Simple momentum strategy as default
                # FIX: Convert Series to float to avoid comparison issues
                prev_price = float(price_data['Close'].iloc[i-1])
                
                if current_price > prev_price * 1.01:  # 1% increase
                    signal = "BUY"
                elif current_price < prev_price * 0.99:  # 1% decrease
                    signal = "SELL"
                else:
                    signal = "HOLD"
            
            # Execute trades based on signal
            if signal == "BUY" and self.portfolio['cash'] > 0:
                # Calculate position size (use 90% of available cash)
                position_size = self.portfolio['cash'] * 0.9 / current_price
                self.execute_buy(current_price, position_size, timestamp=date)
                
            elif signal == "SELL" and self.symbol in self.portfolio['positions']:
                # Sell all holdings
                self.execute_sell(current_price, timestamp=date)
            
            # Update portfolio value
            self.update_portfolio_value(current_price)
            
        # Generate report
        return self.generate_report()

    def generate_report(self):
        """Generate performance report"""
        # Convert history to DataFrame
        if not self.portfolio['value_history']:
            self.logger.warning("No history data to generate report")
            return pd.DataFrame(), {"error": "No history data"}
            
        history_df = pd.DataFrame(self.portfolio['value_history'])
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.set_index('timestamp')
        
        # Calculate metrics
        metrics = {}
        
        # Total return
        initial_value = self.initial_capital
        final_value = history_df['value'].iloc[-1]
        total_return = (final_value / initial_value) - 1
        metrics['total_return'] = total_return
        
        # Get daily returns
        history_df['daily_return'] = history_df['value'].pct_change()
        
        # Sharpe ratio (annualized, assuming risk-free rate of 0)
        if len(history_df) > 1:
            daily_returns = history_df['daily_return'].dropna()
            if not daily_returns.empty and daily_returns.std() != 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            metrics['sharpe'] = sharpe
        
        # Max drawdown
        if len(history_df) > 1:
            history_df['peak'] = history_df['value'].cummax()
            history_df['drawdown'] = (history_df['value'] / history_df['peak']) - 1
            max_drawdown = history_df['drawdown'].min()
            metrics['max_drawdown'] = max_drawdown
        
        # Win rate
        buys = [t for t in self.portfolio['transactions'] if t[0] == 'BUY']
        sells = [t for t in self.portfolio['transactions'] if t[0] == 'SELL']
        
        winning_trades = 0
        for i, sell in enumerate(sells):
            if i < len(buys):
                buy_price = buys[i][1]
                sell_price = sell[1]
                if sell_price > buy_price:
                    winning_trades += 1
        
        win_rate = winning_trades / len(sells) if sells else 0
        metrics['win_rate'] = win_rate
        metrics['num_trades'] = len(buys) + len(sells)
        
        # Log report summary
        self.logger.info(f"Backtest Report: Total Return: {total_return:.2%}, Sharpe: {metrics.get('sharpe', 0):.2f}, Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        return history_df, metrics
    
    def plot_results(self, history_df=None):
        """Plot backtest results"""
        if history_df is None:
            history_df = pd.DataFrame(self.portfolio['value_history'])
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.set_index('timestamp')
            
        if history_df.empty:
            self.logger.warning("No data to plot")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot portfolio value
        ax1.plot(history_df.index, history_df['value'], label='Portfolio Value', color='blue')
        
        # Plot price if available
        if 'price' in history_df.columns:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(history_df.index, history_df['price'], label='Price', color='gray', alpha=0.6)
            ax1_twin.set_ylabel('Price ($)')
            
        # Mark buy and sell points
        for transaction in self.portfolio['transactions']:
            tx_type, price, qty, timestamp = transaction
            if tx_type == 'BUY':
                ax1.scatter(timestamp, history_df.loc[history_df.index == timestamp, 'value'].iloc[0] if len(history_df.loc[history_df.index == timestamp]) > 0 else 0, 
                           color='green', marker='^', s=100)
            elif tx_type == 'SELL':
                ax1.scatter(timestamp, history_df.loc[history_df.index == timestamp, 'value'].iloc[0] if len(history_df.loc[history_df.index == timestamp]) > 0 else 0,
                           color='red', marker='v', s=100)
        
        # Add cash vs. position allocation
        if len(history_df) > 0:
            position_values = history_df['value'] - history_df['cash']
            ax2.stackplot(history_df.index, history_df['cash'], position_values, 
                         labels=['Cash', 'Positions'], colors=['#86c7f3', '#ffe29a'])
            
        # Configure plots
        ax1.set_title('Portfolio Performance')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        
        ax2.set_title('Portfolio Composition')
        ax2.set_ylabel('Value ($)')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def create_signal_generator(predictor):
    """
    Create a signal generator function that uses the StockPredictor
    
    Args:
        predictor: An instance of StockPredictor
        
    Returns:
        function: A function that takes historical data and returns trading signals
    """
    def generate_signal(historical_data):
        # Update predictor's data
        predictor.data = historical_data.copy()
        
        # Generate signal using predictor's method
        signal = predictor.generate_trading_signal(predictor.symbol, horizon=5)
        
        return signal
    
    return generate_signal

def run_live_trading_sim(symbol="APT21794-USD", initial_capital=100):
    """Run a live trading simulation with the Aptos backtester"""
    from datetime import datetime, timedelta
    
    backtester = AptosBacktester(symbol=symbol, initial_capital=initial_capital)
    
    # Get current price
    current_price = float(yf.download(symbol, period='1d', interval='1m')['Close'].iloc[-1])
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
        
        if choice == '1':
            qty = float(input("Enter quantity to buy: "))
            backtester.execute_buy(current_price, qty)
            
        elif choice == '2':
            qty = input("Enter quantity to sell (or 'all' for all): ")
            if qty.lower() == 'all':
                backtester.execute_sell(current_price)
            else:
                backtester.execute_sell(current_price, float(qty))
                
        elif choice == '3':
            print("\n===== Portfolio =====")
            print(f"Cash: ${backtester.portfolio['cash']:.2f}")
            for symbol, position in backtester.portfolio['positions'].items():
                print(f"{symbol}: {position['qty']:.6f} @ ${position['entry_price']:.2f}")
                
            if backtester.portfolio['value_history']:
                latest = backtester.portfolio['value_history'][-1]
                print(f"Total Value: ${latest['value']:.2f}")
                
        elif choice == '4':
            value = backtester.update_portfolio_value()
            print(f"Updated Portfolio Value: ${value:.2f}")
            
        elif choice == '5':
            print("Exiting simulation...")
            break
            
        else:
            print("Invalid choice. Please enter 1-5.")










async def main():

   
    # Load or create wallet
    private_key, address, _ = load_or_create_wallet()
    
    # Initialize the clients
    rest_client = RestClient(NODE_URL)
    print("Connected to Aptos devnet")
    
    # Convert address string to AccountAddress
    account_address = AccountAddress.from_str(address)
    
    # Print current balance
    balance = await check_balance(rest_client, account_address)
    print(f"Current balance: {balance} octas ({balance/1e8} APT)")

    tracker = PortfolioTracker(initial_capital=balance/1e8) # Initialize tracker with current balance
   
    # If balance is low, fund the wallet
    if balance < 100_000_000:  # Less than 1 APT
        print("Balance is low, funding wallet...")
        fund_wallet(address)
    
    # Example: transfer 10,000 octas to yourself (for testing)
    recipient = address  # Replace with another address for real transfer
    transfer_amount = 10000
    success = await execute_transfer(private_key, recipient, transfer_amount)
    
    print(f"Transfer {'successful' if success else 'failed'}")

    # # Check entry points for trading
    # await check_entry_points(symbol="APT21794-USD", tracker=tracker)
    # metrics = tracker.get_pnl_metrics()
    # print(f"== Performance Metrics ==")
    # print(f"Total Return: {metrics['total_return']*100:.2f}%")
    # print(f"Realized P&L: ${metrics['realized_pnl']:.2f}")
    # print(f"Unrealized P&L: ${metrics['unrealized_pnl']:.2f}")
    # print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    # print(f"Number of Trades: {metrics['num_trades']}")


    backtester = AptosBacktester(symbol="APT21794-USD", initial_capital=balance/1e8)
    # Run a simple backtest with default strategy
    print("Running backtest...")
    history, metrics = backtester.run_backtest(
        start_date="2024-01-01",
        end_date="2025-05-15"
    )
    
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    
    # Plot results
    backtester.plot_results(history)



if __name__ == "__main__":
    asyncio.run(main())