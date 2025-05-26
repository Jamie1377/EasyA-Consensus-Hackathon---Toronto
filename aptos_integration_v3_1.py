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

# def get_entry_signal(predictor, symbol=None, current_price=None):
#     """Generate real-time entry signal with confidence scoring
#     Returns: (decision, confidence, rationale, levels)
#     """
#     # Get real-time price if available
#     if current_price is None:
#         if symbol is None:
#             symbol = predictor.symbol

#         try:
#             current_price = float(yf.download(symbol, period='1d', interval='1m', timeout=10)['Close'].iloc[-1])
#             current_market = float(yf.download('^GSPC', period='1d', interval='1m', timeout=10)['Close'].iloc[-1])
#         except Exception as e:
#             print(f"Error getting real-time price: {e}")
#             # Fallback if real-time data is unavailable
#             current_price = float(predictor.data['Close'].iloc[-1])
#             current_market = 0

#     last_row = predictor.data.iloc[-1]
#     second_last_row = predictor.data.iloc[-2] if len(predictor.data) > 1 else last_row

#     # Calculate signal components - Make sure we're using scalar values
#     signals = {
#         'trend': {
#             'value': float(current_price) > float(predictor.data['Close'].rolling(50).mean().iloc[-1]) if 'MA_50' not in predictor.data.columns else float(current_price) > float(second_last_row['MA_50']),
#             'weight': 0.5
#         },
#         'momentum': {
#             'value': float(last_row.get('RSI', 0)) < 65 if 'RSI' in predictor.data.columns else True,
#             'weight': 0.5
#         },
#         'volume': {
#             'value': float(last_row.get('Volume', 0)) > float(predictor.data['Volume'].rolling(20).mean().iloc[-1]) if 'Volume' in predictor.data.columns else True,
#             'weight': 0.5
#         },
#         'volatility': {
#             'value': float(last_row.get('ATR', 0)) > float(predictor.data['ATR'].rolling(14).mean().iloc[-1]) if 'ATR' in predictor.data.columns else True,
#             'weight': 1
#         },
#         'mean_reversion': {
#             'value': current_price < last_row['Lower_Bollinger'] + 0.2*last_row['ATR'],
#             'weight': 0.5
#         },
#         'market_picture_increase': {
#             'value': (second_last_row['SP500'] < last_row['SP500']),
#             'weight': 1
#         },
#         "sector_picture_increase": {
#             'value': (second_last_row['Tech'] < last_row['Tech']),
#             'weight': 1
#         },

#     }

#     # Calculate score (0-5 scale)
#     score = sum(condition['weight'] for name, condition in signals.items() if condition['value'] == True)
#     max_score = sum(condition['weight'] for name, condition in signals.items())
#     confidence = min(100, max(0, int((score / max_score) * 100)))

#     # Generate rationale with proper formatting of scalar values
#     rationales = []
#     if signals['trend']['value'] == True:
#         rationales.append(f"📈 Price {float(current_price):.2f} above 50MA")
#     else:
#         rationales.append(f"📉 Price {float(current_price):.2f} below 50MA")

#     if 'RSI' in predictor.data.columns and signals['momentum']['value'] == True:
#         # Make sure we're using a scalar value for RSI
#         rsi_value = float(last_row.get('RSI', 0))
#         rationales.append(f"💪 Moderate momentum (RSI {rsi_value:.1f})")

#     # Make decision
#     if score >= max_score * 0.6:
#         decision = "BUY"
#     elif score <= max_score * 0.3:
#         decision = "SELL"
#     else:
#         decision = "HOLD"

#     # Add risk parameters
#     risk_params = {
#         'stop_loss_pct': 0.03,  # 3% stop loss
#         'take_profit_pct': 0.05,  # 5% take profit
#     }

#     current_price_float = float(current_price)

#     return (
#         decision,
#         confidence,
#         " | ".join(rationales),
#         {
#             'current_price': [current_price_float],
#             'stop_loss': [current_price_float * (1 - risk_params['stop_loss_pct'])] if decision == "BUY"
#                         else [current_price_float * (1 + risk_params['stop_loss_pct'])],
#             'take_profit': [current_price_float * (1 + risk_params['take_profit_pct'])] if decision == "BUY"
#                           else [current_price_float * (1 - risk_params['take_profit_pct'])],
#         }
#     )


# def get_entry_signal(predictor, symbol=None, current_price=None):
#     """Generate real-time entry signal with confidence scoring
#     Returns: (decision, confidence, rationale, levels)
#     """
#     # Get real-time price if available
#     if current_price is None:
#         if symbol is None:
#             symbol = predictor.symbol

#         # try:
#         #     current_price = float(yf.download(symbol, period='1d', interval='1m', timeout=10)['Close'].iloc[-1])
#         #     current_market = float(yf.download('^GSPC', period='1d', interval='1m', timeout=10)['Close'].iloc[-1])
#         # except Exception as e:
#         # print(f"Error getting real-time price: {e}")
#         # Fallback if real-time data is unavailable
#         current_price = float(predictor.data["Close"].iloc[-1])
#         current_market = 0

#     last_row = predictor.data.iloc[-1]
#     second_last_row = predictor.data.iloc[-2] if len(predictor.data) > 1 else last_row

#     # Calculate signal components - Make sure we're using scalar values
#     signals = {
#         "trend": {
#             "value": (
#                 float(current_price)
#                 > float(predictor.data["Close"].rolling(50).mean().iloc[-1])
#                 if "MA_50" not in predictor.data.columns
#                 else float(current_price) > float(second_last_row["MA_50"])
#             ),
#             "weight": 0.1,
#         },
#         "mean_reversion": {
#             "value": predictor.data["RSI"].iloc[-1] < 30,  # Oversold condition
#             "weight": 1.0,
#         },
#         "volatility_breakout": {
#             "value": predictor.data["ATR"].iloc[-1]
#             > predictor.data["ATR"].rolling(20).mean().iloc[-1] * 1.5,
#             "weight": 0.8,
#         },
#         "volume_spike": {
#             "value": predictor.data["Volume"].iloc[-1]
#             > predictor.data["Volume"].rolling(20).mean().iloc[-1] * 2,
#             "weight": 0.7,
#         },
#         "support_resistance": {
#             "value": abs(current_price - predictor.data["MA_50"].iloc[-1])
#             / current_price
#             < 0.01,  # Price near MA50
#             "weight": 1.2,
#         },
#         "momentum": {
#             "value": (
#                 float(last_row.get("RSI", 0)) < 65
#                 if "RSI" in predictor.data.columns
#                 else True
#             ),
#             "weight": 0.5,
#         },
#         "volume": {
#             "value": (
#                 float(last_row.get("Volume", 0))
#                 > float(predictor.data["Volume"].rolling(20).mean().iloc[-1])
#                 if "Volume" in predictor.data.columns
#                 else True
#             ),
#             "weight": 0.5,
#         },
#         "volatility": {
#             "value": (
#                 float(last_row.get("ATR", 0))
#                 > float(predictor.data["ATR"].rolling(14).mean().iloc[-1])
#                 if "ATR" in predictor.data.columns
#                 else True
#             ),
#             "weight": 1,
#         },
#     }

#     # Add optional signals only if the columns exist
#     if "Lower_Bollinger" in predictor.data.columns and "ATR" in predictor.data.columns:
#         signals["mean_reversion"] = {
#             "value": current_price
#             < last_row["Lower_Bollinger"] + 0.2 * last_row["ATR"],
#             "weight": 0.5,
#         }

#     if "SP500" in predictor.data.columns:
#         signals["market_picture_increase"] = {
#             "value": (second_last_row["SP500"] < last_row["SP500"]),
#             "weight": 1,
#         }

#     if "Tech" in predictor.data.columns:
#         signals["sector_picture_increase"] = {
#             "value": (second_last_row["Tech"] < last_row["Tech"]),
#             "weight": 1,
#         }

#     # Encourage Sell
#     signals["overbought"] = {
#         "value": predictor.data["RSI"].iloc[-1] > 70,  # Overbought condition
#         "weight": 1.2,  # Higher weight to encourage selling
#     }

#     signals["price_resistance"] = {
#         "value": abs(current_price - predictor.data["Upper_Bollinger"].iloc[-1])
#         / current_price
#         < 0.01,
#         "weight": 1.0,
#     }

#     signals["profit_taking"] = {
#         "value": current_price
#         > predictor.data["MA_50"].iloc[-1] * 1.1,  # 10% above MA50
#         "weight": 0.8,
#     }

#     # Calculate score (0-5 scale)
#     score = sum(
#         condition["weight"]
#         for name, condition in signals.items()
#         if condition["value"] == True
#     )
#     max_score = sum(condition["weight"] for name, condition in signals.items())
#     confidence = min(100, max(0, int((score / max_score) * 100)))
#     print(f"Score: {score} / {max_score} ({confidence}%)")

#     # Generate rationale with proper formatting of scalar values
#     rationales = []
#     if signals["trend"]["value"] == True:
#         rationales.append(f"📈 Price {float(current_price):.2f} above 50MA")
#     else:
#         rationales.append(f"📉 Price {float(current_price):.2f} below 50MA")

#     if "RSI" in predictor.data.columns and signals["momentum"]["value"] == True:
#         # Make sure we're using a scalar value for RSI
#         rsi_value = float(last_row.get("RSI", 0))
#         rationales.append(f"💪 Moderate momentum (RSI {rsi_value:.1f})")

#     # Make decision
#     if score >= max_score * 0.6:
#         decision = "BUY"
#     elif score <= max_score * 0.25:
#         decision = "SELL"
#     else:
#         decision = "HOLD"

#     # Add risk parameters
#     risk_params = {
#         "stop_loss_pct": 0.03,  # 3% stop loss
#         "take_profit_pct": 0.05,  # 5% take profit
#     }

#     current_price_float = float(current_price)

#     return (
#         decision,
#         confidence,
#         " | ".join(rationales),
#         {
#             "current_price": [current_price_float],
#             "stop_loss": (
#                 [current_price_float * (1 - risk_params["stop_loss_pct"])]
#                 if decision == "BUY"
#                 else [current_price_float * (1 + risk_params["stop_loss_pct"])]
#             ),
#             "take_profit": (
#                 [current_price_float * (1 + risk_params["take_profit_pct"])]
#                 if decision == "BUY"
#                 else [current_price_float * (1 - risk_params["take_profit_pct"])]
#             ),
#         },
#     )

# def get_entry_signal(predictor, symbol=None, current_price=None):
#     """Generate real-time entry signal with improved dip-buying and top-selling logic"""
#     # Get real-time price if available
#     if current_price is None:
#         if symbol is None:
#             symbol = predictor.symbol
#         current_price = float(predictor.data["Close"].iloc[-1])
    
#     last_row = predictor.data.iloc[-1]
#     second_last_row = predictor.data.iloc[-2] if len(predictor.data) > 1 else last_row
    
#     # Calculate key technical indicators if not already present
#     if 'RSI' not in predictor.data.columns and len(predictor.data) >= 14:
#         predictor.data['RSI'] = calculate_rsi(predictor.data['Close'], window=14)
    
#     if 'MA_50' not in predictor.data.columns:
#         predictor.data['MA_50'] = predictor.data['Close'].rolling(50).mean()
    
#     if 'Upper_Bollinger' not in predictor.data.columns or 'Lower_Bollinger' not in predictor.data.columns:
#         # Calculate Bollinger Bands with 20-period SMA and 2 standard deviations
#         ma20 = predictor.data['Close'].rolling(20).mean()
#         std20 = predictor.data['Close'].rolling(20).std()
#         predictor.data['Upper_Bollinger'] = ma20 + (std20 * 2)
#         predictor.data['Lower_Bollinger'] = ma20 - (std20 * 2)
    
#     # Advanced peak/valley detection
#     # Look for local maxima/minima over the past N periods
#     lookback = 10
#     if len(predictor.data) > lookback:
#         recent_prices = predictor.data['Close'].iloc[-lookback:]
#         is_local_max = (current_price == recent_prices.max()) and (current_price > recent_prices.mean() * 1.02)
#         is_local_min = (current_price == recent_prices.min()) and (current_price < recent_prices.mean() * 0.98)
#     else:
#         is_local_max = is_local_min = False
    
#     # Calculate price distance from recent highs/lows
#     if len(predictor.data) > 20:
#         highest_high = predictor.data['High'].iloc[-20:].max()
#         lowest_low = predictor.data['Low'].iloc[-20:].min()
#         distance_from_high = (highest_high - current_price) / highest_high
#         distance_from_low = (current_price - lowest_low) / current_price
#     else:
#         distance_from_high = distance_from_low = 0
    
#     # --- SELL SIGNALS (weigh these more heavily near tops) ---
#     sell_signals = {
#         # Traditional overbought conditions
#         "overbought": {
#             "value": 'RSI' in predictor.data.columns and predictor.data['RSI'].iloc[-1] > 70,
#             "weight": 1.0
#         },
#         # Price near upper Bollinger Band
#         "upper_band": {
#             "value": 'Upper_Bollinger' in predictor.data.columns and 
#                     current_price > predictor.data['Upper_Bollinger'].iloc[-1] * 0.98,
#             "weight": 1.2
#         },
#         # Local price maximum - strong sell signal
#         "local_max": {
#             "value": is_local_max,
#             "weight": 2.0
#         },
#         # Extended run-up (price far from recent lows)
#         "extended_rally": {
#             "value": distance_from_low > 0.15,  # Price at least 15% above recent lows
#             "weight": 1.0
#         },
#         # Price volume divergence (rising price, falling volume)
#         "volume_divergence": {
#             "value": (predictor.data['Close'].iloc[-3:].pct_change().sum() > 0) and 
#                      (predictor.data['Volume'].iloc[-3:].pct_change().sum() < 0),
#             "weight": 0.7
#         },
#         # Momentum deceleration
#         "momentum_slowing": {
#             "value": 'RSI' in predictor.data.columns and 
#                    (predictor.data['RSI'].iloc[-1] < predictor.data['RSI'].iloc[-2]) and 
#                    (predictor.data['RSI'].iloc[-1] > 65),
#             "weight": 0.8
#         }
#     }
    
#     # --- BUY SIGNALS (weigh these more heavily near bottoms) ---
#     buy_signals = {
#         # Traditional oversold conditions
#         "oversold": {
#             "value": 'RSI' in predictor.data.columns and predictor.data['RSI'].iloc[-1] < 30,
#             "weight": 1.0
#         },
#         # Price near lower Bollinger Band
#         "lower_band": {
#             "value": 'Lower_Bollinger' in predictor.data.columns and 
#                     current_price < predictor.data['Lower_Bollinger'].iloc[-1] * 1.02,
#             "weight": 1.2
#         },
#         # Local price minimum - strong buy signal
#         "local_min": {
#             "value": is_local_min,
#             "weight": 2.0
#         },
#         # Extended sell-off (price far from recent highs)
#         "extended_selloff": {
#             "value": distance_from_high > 0.15,  # Price at least 15% below recent highs
#             "weight": 1.0
#         },
#         # Bullish volume spike
#         "volume_spike": {
#             "value": (predictor.data['Volume'].iloc[-1] > predictor.data['Volume'].iloc[-20:].mean() * 1.5) and 
#                     (predictor.data['Close'].iloc[-1] > predictor.data['Close'].iloc[-2]),
#             "weight": 0.7
#         },
#         # Momentum turning up from low levels
#         "momentum_bottoming": {
#             "value": 'RSI' in predictor.data.columns and 
#                    (predictor.data['RSI'].iloc[-1] > predictor.data['RSI'].iloc[-2]) and 
#                    (predictor.data['RSI'].iloc[-1] < 35),
#             "weight": 0.8
#         }
#     }
    
#     # Calculate sell score (0-100%)
#     sell_score = sum(signal["weight"] for name, signal in sell_signals.items() if signal["value"] == True)
#     max_sell_score = sum(signal["weight"] for name, signal in sell_signals.items())
#     sell_confidence = min(100, max(0, int((sell_score / max_sell_score) * 100)))
    
#     # Calculate buy score (0-100%)
#     buy_score = sum(signal["weight"] for name, signal in buy_signals.items() if signal["value"] == True)
#     max_buy_score = sum(signal["weight"] for name, signal in buy_signals.items())
#     buy_confidence = min(100, max(0, int((buy_score / max_buy_score) * 100)))
    
#     # Generate rationale
#     rationales = []
    
#     # Add sell signal rationales
#     for name, signal in sell_signals.items():
#         if signal["value"]:
#             if name == "overbought":
#                 rationales.append(f"⚠️ Overbought (RSI: {predictor.data['RSI'].iloc[-1]:.1f})")
#             elif name == "upper_band":
#                 rationales.append(f"📈 Price near upper Bollinger Band")
#             elif name == "local_max":
#                 rationales.append(f"🔺 Local price maximum detected")
#             elif name == "extended_rally":
#                 rationales.append(f"⚠️ Extended rally ({distance_from_low*100:.1f}% from lows)")
#             elif name == "volume_divergence":
#                 rationales.append(f"⚠️ Price-volume divergence")
#             elif name == "momentum_slowing":
#                 rationales.append(f"🔻 Momentum slowing down")
    
#     # Add buy signal rationales
#     for name, signal in buy_signals.items():
#         if signal["value"]:
#             if name == "oversold":
#                 rationales.append(f"✅ Oversold (RSI: {predictor.data['RSI'].iloc[-1]:.1f})")
#             elif name == "lower_band":
#                 rationales.append(f"📉 Price near lower Bollinger Band")
#             elif name == "local_min":
#                 rationales.append(f"🔻 Local price minimum detected")
#             elif name == "extended_selloff":
#                 rationales.append(f"✅ Extended selloff ({distance_from_high*100:.1f}% from highs)")
#             elif name == "volume_spike":
#                 rationales.append(f"📊 Bullish volume spike")
#             elif name == "momentum_bottoming":
#                 rationales.append(f"🔼 Momentum turning up from low levels")
    
#     # Determine decision based on the stronger signal
#     if buy_confidence > sell_confidence + 10:  # Buy signal must be clearly stronger than sell
#         decision = "BUY"
#         confidence = buy_confidence
#     elif sell_confidence > buy_confidence + 10:  # Sell signal must be clearly stronger than buy
#         decision = "SELL"
#         confidence = sell_confidence
#     else:  # When signals are close, hold
#         decision = "HOLD"
#         confidence = 50
    
#     # Add risk parameters
#     risk_params = {
#         'stop_loss_pct': 0.05,  # 5% stop loss
#         'take_profit_pct': 0.10,  # 10% take profit
#     }
    
#     # If we're near a key level, tighten stops
#     if is_local_max or is_local_min:
#         risk_params['stop_loss_pct'] = 0.03  # Tighter stop near key levels
    
#     # Generate levels
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
#         },
#     )

try:
    from private_strat import get_entry_signal
    
 
except ImportError:
    def get_entry_signal(predictor, symbol=None, current_price=None, reverse_signals=False):
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
            "cash": initial_capital,
            "positions": {},  # symbol -> {qty, entry_price}
            "value_history": [],  # [{timestamp, value, cash}]
            "transactions": [],  # [type, price, qty, timestamp]
        }

        # Trade parameters
        self.slippage = 0.002  # 10 basis points
        self.commission = 0.001  # 0.1% per transaction and usually fixed

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
            f"AptosBacktester initialized with {initial_capital} USD for {symbol}"
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
        self.logger.info(
            f"{transaction_type} executed: {quantity:.6f} {symbol} at ${price:.4f}"
        )

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

        if self.symbol in self.portfolio["positions"]:
            # Update existing position
            position = self.portfolio["positions"][self.symbol]
            total_quantity = position["qty"] + quantity
            avg_price = (
                (position["qty"] * position["entry_price"])
                + (quantity * executed_price)
            ) / total_quantity

            self.portfolio["positions"][self.symbol] = {
                "qty": total_quantity,
                "entry_price": avg_price,
            }
        else:
            # Create new position
            self.portfolio["positions"][self.symbol] = {
                "qty": quantity,
                "entry_price": executed_price,
            }

        # Record the transaction
        self.record_transaction("BUY", self.symbol, executed_price, quantity, timestamp)

        return True

    def execute_sell(self, price, quantity=None, timestamp=None):
        """Execute a sell order with support for naked short selling"""
        if timestamp is None:
            timestamp = datetime.now()

        # Apply slippage to sell price (negative for sells)
        executed_price = price * (1 - self.slippage)

        # If no quantity specified, sell all holdings or create standard short position
        if quantity is None:
            if self.symbol in self.portfolio["positions"]:
                quantity = (
                    self.portfolio["positions"][self.symbol]["qty"] * 0.025
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

        if self.symbol in self.portfolio["positions"]:
            # We have an existing position
            position = self.portfolio["positions"][self.symbol]

            if position["qty"] >= quantity:
                # Normal sell - we have enough shares
                position["qty"] -= quantity
                if position["qty"] <= 0:
                    # Remove position if sold out completely
                    self.portfolio["positions"].pop(self.symbol)

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
                self.portfolio["positions"][self.symbol] = {
                    "qty": -remaining_qty,  # Negative quantity indicates short
                    "entry_price": executed_price,
                }

                # Add proceeds from short sale (these are held as cash but may be restricted in real trading)
                self.portfolio["cash"] += (executed_price * remaining_qty) - (
                    commission * remaining_qty / quantity
                )

                self.logger.info(
                    f"Partial short created: sold {existing_qty:.6f} owned shares and shorted {remaining_qty:.6f} additional shares"
                )

        else:
            # No existing position - creating a pure short position
            self.portfolio["positions"][self.symbol] = {
                "qty": -quantity,  # Negative quantity indicates short
                "entry_price": executed_price,
            }

            # Add proceeds to cash (in real trading this might be held as margin)
            self.portfolio["cash"] += net_proceeds

            self.logger.info(
                f"Created new short position of {quantity:.6f} shares at ${executed_price:.2f}"
            )

        # Record the transaction
        self.record_transaction(
            "SELL", self.symbol, executed_price, quantity, timestamp
        )

        return True

    def update_portfolio_value(self, current_price=None, timestamp=None):
        """Calculate current portfolio value and record to history, supporting short positions"""
        if timestamp is None:
            timestamp = datetime.now()

        # If no price provided, try to get current price
        if current_price is None:
            try:
                current_price = float(
                    yf.download(self.symbol, period="1d", interval="1m", timeout=10)[
                        "Close"
                    ].iloc[-1]
                )
            except Exception as e:
                self.logger.error(f"Failed to get current price: {e}")
                # Use last known price or position entry price as fallback
                if self.portfolio["value_history"]:
                    # Use last valuation price
                    current_price = self.portfolio["value_history"][-1].get(
                        "price", 1.0
                    )
                elif self.symbol in self.portfolio["positions"]:
                    # Use position entry price
                    current_price = self.portfolio["positions"][self.symbol][
                        "entry_price"
                    ]
                else:
                    # Default to 1.0 if all else fails
                    current_price = 1.0

        # Calculate position value, handling both long and short positions
        position_value = 0
        for symbol, position in self.portfolio["positions"].items():
            # For short positions (negative qty), value increases when price falls
            # Possibly the wrong way to calculate this
            # if position["qty"] < 0:  # Short position
            #     # Value of short is based on entry price vs current price
            #     # If current price is lower than entry, we're profitable
            #     short_value = -position["qty"] * (
            #         position["entry_price"] - current_price
            #     )
            #     position_value += (
            #         position["entry_price"] * -position["qty"]
            #     )  # Original proceeds
            #     position_value += short_value  # Profit/loss
            if position["qty"] < 0:  # Short position
                # For short positions, we owe shares that we need to buy back
                # Our liability is the current value of those shares
                # Our profit/loss is the difference between what we sold them for and what we'd pay now
                liability = -position["qty"] * current_price
                # original_proceeds = -position["qty"] * position["entry_price"]
                # Add the net value: what we received minus what we owe
                position_value -= liability
            else:  # Long position
                position_value += position["qty"] * current_price

        total_value = self.portfolio["cash"] + position_value

        # Record to history
        value_entry = {
            "timestamp": timestamp,
            "value": total_value,
            "cash": self.portfolio["cash"],
            "price": current_price,
        }
        self.portfolio["value_history"].append(value_entry)

        # Log to portfolio file
        portfolio_record = {
            "timestamp": timestamp,
            "total_value": total_value,
            "cash": self.portfolio["cash"],
            "positions": str(self.portfolio["positions"]),
            "price": current_price,
        }

        pd.DataFrame([portfolio_record]).to_csv(
            self.portfolio_log_file, mode="a", header=False, index=False
        )

        return total_value

    # def execute_sell(self, price, quantity=None, timestamp=None):
    #     """Execute a sell order"""
    #     if timestamp is None:
    #         timestamp = datetime.now()

    #     # Apply slippage to sell price (negative for sells)
    #     executed_price = price * (1 - self.slippage)

    #     # If no quantity specified, sell all holdings
    #     if quantity is None:
    #         if self.symbol in self.portfolio['positions']:
    #             quantity = self.portfolio['positions'][self.symbol]['qty']
    #         else:
    #             # default sell quantity to 10%
    #             self.logger.info(f"No position in {self.symbol} to sell; naked shorting defaulting to 10% of cash")
    #             quantity = self.portfolio['cash'] * 0.1 / executed_price  # 10% of cash in APT

    #      # Calculate transaction value and fees
    #     value = executed_price * quantity
    #     commission = value * self.commission
    #     net_proceeds = value - commission

    #     # Check if trying to sell more than owned
    #     if self.symbol in self.portfolio['positions']:
    #         position = self.portfolio['positions'][self.symbol]

    #         if quantity > position['qty']:
    #             self.logger.info(f"Attempted to sell {quantity} but only have {position['qty']} so selling all")
    #             quantity = position['qty']
    #         elif quantity <= position['qty']:
    #             # Normal sell case
    #             position['qty'] -= quantity
    #             if position['qty'] == 0:
    #                 # Remove position if sold out
    #                 self.portfolio['positions'].pop(self.symbol)
    #             self.portfolio['cash'] += net_proceeds

    #     else: # not holding any position
    #         self.logger.info(f"No position in {self.symbol} to sell but trying to short {quantity}")

    #         return False

    #     # Update portfolio
    #     self.portfolio['cash'] += net_proceeds

    #     # Update or remove position
    #     position = self.portfolio['positions'][self.symbol]
    #     if quantity >= position['qty']:
    #         # Selling entire position
    #         self.portfolio['positions'].pop(self.symbol)
    #     else:
    #         # Partial sale
    #         position['qty'] -= quantity

    #     # Record the transaction
    #     self.record_transaction('SELL', self.symbol, executed_price, quantity)

    #     return True

    # def update_portfolio_value(self, current_price=None):
    #     """Calculate current portfolio value and record to history"""
    #     timestamp = datetime.now()

    #     # If no price provided, try to get current price
    #     if current_price is None:
    #         try:
    #             current_price = float(yf.download(self.symbol, period='1d', interval='1m', timeout=10)['Close'].iloc[-1])
    #         except Exception as e:
    #             self.logger.error(f"Failed to get current price: {e}")
    #             # Use last known price or position entry price as fallback
    #             if self.portfolio['value_history']:
    #                 # Use last valuation price
    #                 current_price = self.portfolio['value_history'][-1].get('price', 1.0)
    #             elif self.symbol in self.portfolio['positions']:
    #                 # Use position entry price
    #                 current_price = self.portfolio['positions'][self.symbol]['entry_price']
    #             else:
    #                 # Default to 1.0 if all else fails
    #                 current_price = 1.0

    #     # Calculate position value
    #     position_value = 0
    #     for symbol, position in self.portfolio['positions'].items():
    #         position_value += position['qty'] * current_price

    #     total_value = self.portfolio['cash'] + position_value

    #     # Record to history
    #     value_entry = {
    #         'timestamp': timestamp,
    #         'value': total_value,
    #         'cash': self.portfolio['cash'],
    #         'price': current_price
    #     }
    #     self.portfolio['value_history'].append(value_entry)

    #     # Log to portfolio file
    #     portfolio_record = {
    #         'timestamp': timestamp,
    #         'total_value': total_value,
    #         'cash': self.portfolio['cash'],
    #         'positions': str(self.portfolio['positions']),
    #         'price': current_price
    #     }

    #     pd.DataFrame([portfolio_record]).to_csv(
    #         self.portfolio_log_file, mode='a', header=False, index=False
    #     )

    #     return total_value

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
    #             self.update_portfolio_value(float(price_data['Close'].iloc[i]))
    #             continue

    #         # FIX: Convert Series to float to avoid comparison issues
    #         current_price = float(price_data['Close'].iloc[i])

    #         # Generate trading signal
    #         if signal_generator:
    #             # Use provided signal generator with historical data up to this point
    #             signal = signal_generator(price_data.iloc[:i+1])
    #         else:
    #             # Simple momentum strategy as default
    #             # FIX: Convert Series to float to avoid comparison issues
    #             prev_price = float(price_data['Close'].iloc[i-1])

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

    def run_backtest(
        self, start_date, end_date, price_data=None, signal_generator=None
    ):
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
        from predictor import StockPredictor

        _predictor_ = StockPredictor(
            symbol=self.symbol, start_date=start_date, end_date=end_date, interval="1d"
        )
        _predictor_.load_data()

        # Get price data if not provided
        if price_data is None:
            try:
                # Check if we should use the retry function
                if (
                    hasattr(self, "use_retry_fetch")
                    and self.use_retry_fetch
                    and hasattr(self, "fetch_data_function")
                ):
                    self.logger.info("Using retry fetch function for data download")
                    # price_data = self.fetch_data_function(
                    #     self.symbol,
                    #     start_date,
                    #     end_date,
                    #     interval="1d"
                    # )
                    price_data = _predictor_.data

                else:
                    # Default download method
                    self.logger.info("Using standard yfinance download")
                    # price_data = yf.download(self.symbol, start=start_date, end=end_date, interval='1d', progress=False, timeout=20)
                    price_data = _predictor_.data
                if price_data.empty:
                    self.logger.error(
                        f"No data available for {self.symbol} in the specified date range"
                    )
                    return pd.DataFrame(), {"error": "No data available"}

                # Log successful data retrieval
                self.logger.info(
                    f"Successfully downloaded {len(price_data)} data points"
                )

            except Exception as e:
                self.logger.error(f"Failed to download price data: {e}")
                return pd.DataFrame(), {"error": f"Failed to download data: {str(e)}"}

        # Reset portfolio for backtest
        self.portfolio = {
            "cash": self.initial_capital,
            "positions": {},
            "value_history": [],
            "transactions": [],
        }

        # Verify data structure
        if "Close" not in price_data.columns:
            self.logger.error("Price data does not contain 'Close' column")
            return pd.DataFrame(), {
                "error": "Invalid data format - missing 'Close' column"
            }

        # Daily loop for backtest
        dates = price_data.index

        self.logger.info(f"Running backtest over {len(dates)} trading days")
        min_portfolio_value = self.initial_capital * 0.1
        for i, date in enumerate(dates): # so in this loop, if autodect_reversal is true, it would decide whether to reverse before each day.
            # Skip first day (need previous data for signals)
            if i <= 1:
                # Just record initial portfolio value
                self.update_portfolio_value(
                    float(price_data["Close"].iloc[i]), timestamp=date
                )
                continue

            # Calculate recent trend direction and strength
            if i > 20:
                recent_trend = (
                    price_data["Close"].iloc[i - 20 : i].pct_change().mean() * 100
                )  # Trend as percentage
            else:
                recent_trend = 0

            # Convert Series to float to avoid comparison issues
            current_price = float(
                price_data["Open"].iloc[i]
            )  # execution price is open price of the day i (when the trade is executed)

            # Check if portfolio value is below minimum threshold
            if (
                self.portfolio["value_history"]
                and self.portfolio["value_history"][-1]["value"] <= min_portfolio_value
            ):
                self.logger.warning(
                    f"Portfolio value fell below minimum threshold ({min_portfolio_value:.2f}). "
                    f"Stopping backtest at {date} with final value: "
                    f"{self.portfolio['value_history'][-1]['value']:.2f}"
                )
                break

            # Check if we're completely out of cash and have no positions
            if self.portfolio["cash"] <= 0 and not self.portfolio["positions"]:
                self.logger.warning(
                    f"Portfolio is bankrupt! Stopping backtest at {date}"
                )
                break

            # Generate trading signal
            if signal_generator:
                # Use provided signal generator with historical data up to this point
                signal = signal_generator(
                    price_data.iloc[:i]
                )  # at date i u can use the data up to i-1 (no close price for i so no other indicators)

            else:
                # Simple momentum strategy as default
                # Convert Series to float to avoid comparison issues
                prev_price = float(price_data["Close"].iloc[i - 1])

                if current_price > prev_price * 1.01:  # 1% increase
                    signal = "BUY"
                elif current_price < prev_price * 0.99:  # 1% decrease
                    signal = "SELL"
                else:
                    signal = "HOLD"

            # Execute trades based on signal

            if signal == "BUY" and self.portfolio["cash"] > 0:
                # Calculate position size (use 90% of available cash)
                position_size_factor = (
                    min(
                        0.035, max(0.02, 0.03 / price_data["Volatility"].iloc[-1] * 10)
                    )
                    * 2
                )  ## leverage
                print(f"position_size_factor: {position_size_factor}")
                # Increase position size in uptrends
                if recent_trend > 0:
                    position_size_factor *= 1 + min(
                        recent_trend * 0.5, 0.5
                    )  # Up to 50% larger

                position_size = (
                    self.portfolio["cash"] * position_size_factor / current_price
                )

                # position_size = self.portfolio["cash"] * 0.025 / current_price

                if position_size <= 1e-3:
                    self.logger.warning(
                        f"Insufficient funds for purchase. Required: ${current_price:.2f}, Available: ${self.portfolio['cash']:.2f}"
                    )
                    self.update_portfolio_value(current_price, timestamp=date)
                    continue
                if self.portfolio["cash"] >= self.initial_capital * 0.3:
                    self.execute_buy(current_price, position_size, timestamp=date)
                else:
                    print("Want to keep 40% of the initial capital in cash")
                self.logger.info(
                    f"Day {i}: BUY signal at ${current_price:.2f}, bought {position_size:.4f} units"
                )

            elif signal == "SELL":
                if self.symbol in self.portfolio["positions"]:
                    # Sell all holdings
                    position_qty = self.portfolio["positions"][self.symbol]["qty"]
                    # position_size = self.portfolio["cash"] * 0.025 / current_price
                    # position_size = position_qty * 0.25  # Sell 25% of position
                    position_size_factor = (
                        min(
                            0.0325,
                            max(0.02, 0.035 / price_data["Volatility"].iloc[-1] * 10),
                        )
                        * 1.5
                    )  ## leverage
                    position_size = position_qty * position_size_factor
                    min_quantity = 1e-2  # Minimum tradeable quantity
                    if position_size < min_quantity:
                        position_size = (
                            min_quantity  # Sell entire position if it's too small
                        )
                        self.logger.info(
                            f"Position size too small, selling entire position of {position_qty:.8f} units"
                        )

                    self.execute_sell(current_price, position_size, timestamp=date)
                    self.logger.info(
                        f"Day {i}: SELL signal at ${current_price:.2f}, sold {position_qty:.4f} units from existing position"
                    )
                else:  # Create a naked short position (if allowed)
                    position_size_factor = (
                        min(
                            0.0325,
                            max(0.02, 0.035 / price_data["Volatility"].iloc[-1] * 10),
                        )
                        * 3
                    )  ## leverage
                    position_size = (
                        self.portfolio["cash"] * position_size_factor / current_price
                    )
                    if position_size > 0:
                        min_quantity = 1e-2
                        if position_size < min_quantity:
                            position_size = (
                                min_quantity  # Sell entire position if it's too small
                            )
                            self.logger.info(
                                f"Position size too small, selling entire position of {min_quantity:.8f} units"
                            )

                        self.execute_sell(current_price, position_size, timestamp=date)
                        self.logger.info(
                            f"Day {i}: SELL signal - created short position of {position_size:.4f} units"
                        )

            # Update portfolio value using today's close for accurate end-of-day valuation
            self.update_portfolio_value(
                float(price_data["Close"].iloc[i]), timestamp=date
            )

            # # Apply stop-loss and take-profit before regular signals
            # for symbol, position in list(self.portfolio["positions"].items()):
            #     entry_price = position["entry_price"]
            #     current_price = float(price_data["Open"].iloc[i])

            #     # Stop-loss (7% loss)

            #     if position["qty"] > 0 and current_price < entry_price * 0.88:
            #         self.execute_sell(
            #             current_price, position["qty"] * 0.99, timestamp=date
            #         )
            #         self.logger.info(f"Stop-loss triggered at {current_price:.2f}")
            #         continue
            #     elif position["qty"] > 0 and current_price < entry_price * 0.92:
            #         self.execute_sell(
            #             current_price, position["qty"] * 0.7, timestamp=date
            #         )
            #         self.logger.info(f"Stop-loss triggered at {current_price:.2f}")
            #         continue

            #     # Take-profit (20% gain)
            #     if position["qty"] > 0 and current_price > entry_price * 1.20:
            #         self.execute_sell(
            #             current_price, position["qty"] * 0.05, timestamp=date
            #         )
            #         self.logger.info(f"Take-profit triggered at {current_price:.2f}")
            #         continue
            #     elif position["qty"] > 0 and current_price > entry_price * 1.15:
            #         self.execute_sell(
            #             current_price, position["qty"] * 0.025, timestamp=date
            #         )
            #         self.logger.info(f"Take-profit triggered at {current_price:.2f}")
            #         continue

            # In your run_backtest method
            for symbol, position in list(self.portfolio["positions"].items()):
                entry_price = position["entry_price"]
                current_price = float(price_data["Open"].iloc[i])

                # Trailing stop - tightens as profit increases
                if position["qty"] > 0:
                    profit_pct = (current_price - entry_price) / entry_price

                    # Aggressive take-profit (15% gain)
                    if profit_pct > 0.3:
                        # Take 10% off the table at 30% gain
                        self.execute_sell(
                            current_price, position["qty"] * 0.1, timestamp=date
                        )
                        self.logger.info(
                            f"Take-profit triggered at {current_price:.2f}"
                        )

                    elif profit_pct > 0.25:
                        # Take 50% off the table at 15% gain
                        self.execute_sell(
                            current_price, position["qty"] * 0.1, timestamp=date
                        )
                        self.logger.info(
                            f"Take-profit triggered at {current_price:.2f}"
                        )

                    # Trailing stop gets tighter as profit increases
                    elif profit_pct > 0.15:
                        # If price drops more than 3% from peak, exit remaining position
                        recent_high = price_data["Close"].iloc[i - 10 : i].max()
                        if current_price < recent_high * 0.9:
                            self.execute_sell(
                                current_price, position["qty"] * 0.4, timestamp=date
                            )
                            self.logger.info(
                                f"Trailing stop triggered at {current_price:.2f}"
                            )

                    # Wider stop-loss for new positions (12%)
                    elif profit_pct < -0.25:
                        # Cut losses at 12%
                        self.execute_sell(
                            current_price, position["qty"] * 0.4, timestamp=date
                        )
                        self.logger.info(f"Stop-loss triggered at {current_price:.2f}")

            # # Add to your run_backtest function
            # def should_hedge(price_data, i):
            #     # Check if we're in a downtrend
            #     if i < 50:
            #         return False

            #     ma50 = price_data["Close"].iloc[i - 50 : i].mean()
            #     ma20 = price_data["Close"].iloc[i - 20 : i].mean()
            #     price = price_data["Close"].iloc[i]

            #     # Downtrend conditions
            #     in_downtrend = price < ma50 and ma20 < ma50
            #     # Volatility spike
            #     vol_spike = (
            #         price_data["Close"].iloc[i - 10 : i].std()
            #         > price_data["Close"].iloc[i - 30 : i - 10].std() * 1.5
            #     )

            #     return in_downtrend and vol_spike

            # # In daily loop
            # if (
            #     should_hedge(price_data, i)
            #     and self.symbol in self.portfolio["positions"]
            # ):
            #     # Add hedge by selling 30% of position
            #     position = self.portfolio["positions"][self.symbol]
            #     self.execute_sell(current_price, position["qty"] * 0.25, timestamp=date)
            #     self.logger.info(f"Added hedge during downtrend at {current_price:.2f}")

            # # Calculate local swing points every day
            # # Add these variables near the beginning of the function
            # lookback_period = 20  # For detecting local extremes
            # trailing_stop_pct = 0.05  # 5% trailing stop
            # last_swing_high = 0
            # last_swing_low = float("inf")

            # if i >= lookback_period:
            #     price_window = price_data["Close"].iloc[i - lookback_period : i + 1]
            #     current_price = float(price_data["Open"].iloc[i])

            #     # Update swing high (highest price in last lookback_period days)
            #     recent_high = price_window.max()
            #     if recent_high > last_swing_high:
            #         last_swing_high = recent_high

            #     # Update swing low (lowest price in last lookback_period days)
            #     recent_low = price_window.min()
            #     if recent_low < last_swing_low:
            #         last_swing_low = recent_low

            #     # Check for positions that need stop loss management
            #     for symbol, position in list(self.portfolio["positions"].items()):
            #         # For long positions - use trailing stop based on recent high
            #         if position["qty"] > 0:
            #             # If price falls below trailing stop from swing high
            #             trailing_stop_level = last_swing_high * (1 - trailing_stop_pct)

            #             if current_price < trailing_stop_level:
            #                 # Execute trailing stop - sell entire position
            #                 self.execute_sell(
            #                     current_price, position["qty"] * 0.01, timestamp=date
            #                 )
            #                 self.logger.info(
            #                     f"Trailing stop triggered at {current_price:.2f}, selling {position['qty'] * 0.01:.4f} units "
            #                     f"(fell {(1-(current_price/last_swing_high)):.2%} from high of {last_swing_high:.2f})"
            #                 )

            #         # For short positions - use trailing stop based on recent low
            #         elif position["qty"] < 0:
            #             # If price rises above trailing stop from swing low
            #             trailing_stop_level = last_swing_low * (1 + trailing_stop_pct)

            #             if current_price > trailing_stop_level:
            #                 # Execute trailing stop - buy to cover entire position
            #                 self.execute_buy(
            #                     current_price, -position["qty"]*0.01, timestamp=date
            #                 )
            #                 self.logger.info(
            #                     f"Short cover stop triggered at {current_price:.2f}, buying to cover {-position['qty']*0.01:.4f} units "
            #                     f"(rose {((current_price/last_swing_low)-1):.2%} from low of {last_swing_low:.2f})"
            #                 )

        # Generate report
        self.logger.info("Backtest completed, generating report...")
        return self.generate_report()

    def generate_report(self):
        """Generate performance report"""
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

        # # Win rate
        # buys = [t for t in self.portfolio["transactions"] if t[0] == "BUY"]
        # sells = [t for t in self.portfolio["transactions"] if t[0] == "SELL"]

        # winning_trades = 0
        # for i, sell in enumerate(sells):
        #     if i < len(buys):
        #         buy_price = buys[i][1]
        #         sell_price = sell[1]
        #         if sell_price > buy_price:
        #             winning_trades += 1

        # win_rate = winning_trades / len(sells) if sells else 0
        # metrics["win_rate"] = win_rate
        # metrics["num_trades"] = len(buys) + len(sells)

        # Improved win rate calculation that only counts liquidated trades
        buys = [t for t in self.portfolio["transactions"] if t[0] == "BUY"]
        sells = [t for t in self.portfolio["transactions"] if t[0] == "SELL"]

        # Use FIFO (First-In-First-Out) to match buys and sells
        buy_queue = []  # Store (price, quantity) tuples
        winning_trades = 0
        losing_trades = 0

        for transaction_type, price, qty, timestamp in self.portfolio["transactions"]:
            if transaction_type == "BUY":
                # Add to buy queue
                buy_queue.append((price, qty))

            elif transaction_type == "SELL" and buy_queue:
                # Process sell against available buys
                remaining_sell_qty = qty

                while remaining_sell_qty > 0 and buy_queue:
                    buy_price, buy_qty = buy_queue[0]

                    # Determine how much of this buy is being sold
                    match_qty = min(remaining_sell_qty, buy_qty)

                    # Count this as one trade (or partial trade)
                    if price > buy_price:
                        # Profitable trade
                        winning_trades += 1
                    else:
                        # Unprofitable trade
                        losing_trades += 1

                    # Update quantities
                    remaining_sell_qty -= match_qty

                    if match_qty >= buy_qty:
                        # Consumed entire buy
                        buy_queue.pop(0)
                    else:
                        # Partially consumed buy
                        buy_queue[0] = (buy_price, buy_qty - match_qty)
                        break

        # Calculate win rate based only on completed trades
        total_closed_trades = winning_trades + losing_trades
        win_rate = (
            winning_trades / total_closed_trades if total_closed_trades > 0 else 0
        )
        metrics["win_rate"] = win_rate
        metrics["total_closed_trades"] = total_closed_trades
        metrics["num_trades"] = len(buys) + len(sells)
        metrics["num_buy_trades"] = len(buys)
        metrics["num_sell_trades"] = len(sells)

        # Log report summary
        self.logger.info(
            f"Backtest Report: Total Return: {total_return:.2%}, Sharpe: {metrics.get('sharpe', 0):.2f}, Max Drawdown: {metrics.get('max_drawdown', 0):.2%}, Win Rate: {metrics.get('win_rate', 0):.2%}, Number of Trades: {metrics.get('num_trades', 0)}, Buy Trades: {metrics.get('num_buy_trades', 0)}, Sell Trades: {metrics.get('num_sell_trades', 0)}"
        )

        return history_df, metrics

    # def plot_results(self, history_df=None):
    #     """Plot backtest results"""
    #     if history_df is None:
    #         history_df = pd.DataFrame(self.portfolio["value_history"])
    #         history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    #         history_df = history_df.set_index("timestamp")

    #     if history_df.empty:
    #         self.logger.warning("No data to plot")
    #         return

    #     # Create figure with two subplots
    #     fig, (ax1, ax2) = plt.subplots(
    #         2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]}
    #     )

    #     # Plot portfolio value
    #     ax1.plot(
    #         history_df.index, history_df["value"], label="Portfolio Value", color="blue"
    #     )

    #     # Plot price if available
    #     if "price" in history_df.columns:
    #         ax1_twin = ax1.twinx()
    #         ax1_twin.plot(
    #             history_df.index,
    #             history_df["price"],
    #             label="Price",
    #             color="gray",
    #             alpha=0.6,
    #         )
    #         ax1_twin.set_ylabel("Price ($)")

    #     # Mark buy and sell points (Green for buy, Red for sell)
    #     for transaction in self.portfolio["transactions"]:
    #         tx_type, price, qty, timestamp = transaction
    #         if isinstance(timestamp, str):
    #             timestamp = pd.to_datetime(timestamp)
    #         if tx_type == "BUY":
    #             ax1.scatter(
    #                 timestamp,
    #                 (
    #                     history_df.loc[history_df.index == timestamp, "value"].iloc[0]
    #                     if len(history_df.loc[history_df.index == timestamp]) > 0
    #                     else 0
    #                 ),
    #                 color="green",
    #                 marker="^",
    #                 s=100,
    #             )
    #         elif tx_type == "SELL":
    #             ax1.scatter(
    #                 timestamp,
    #                 (
    #                     history_df.loc[history_df.index == timestamp, "value"].iloc[0]
    #                     if len(history_df.loc[history_df.index == timestamp]) > 0
    #                     else 0
    #                 ),
    #                 color="red",
    #                 marker="v",
    #                 s=100,
    #             )

    #     # Manually set legend for buy/sell markers
    #     handles, labels = plt.gca().get_legend_handles_labels()

    #     # create manual symbols for legend
    #     patch = mpatches.Patch(color="grey", label="manual patch")
    #     buy = Line2D(
    #         [0],
    #         [0],
    #         label="Buy",
    #         marker="^",
    #         markersize=10,
    #         color="green",
    #         markerfacecolor="green",
    #         linestyle="",
    #     )
    #     sell = Line2D(
    #         [0],
    #         [0],
    #         label="Sell",
    #         marker="v",
    #         markersize=10,
    #         color="red",
    #         markerfacecolor="red",
    #         linestyle="",
    #     )

    #     # add manual symbols to auto legend
    #     handles.extend([patch, buy, sell])

    #     # Add cash vs. position allocation
    #     if len(history_df) > 0:
    #         position_values = history_df["value"] - history_df["cash"]

    #         cash_values = history_df["cash"].copy()
    #         pos_position_values = np.maximum(0, position_values)
    #         neg_position_values = np.minimum(0, position_values)
    #         cash_to_plot = cash_values - neg_position_values
    #         ax2.stackplot(
    #             history_df.index,
    #             cash_to_plot,
    #             pos_position_values,
    #             neg_position_values,
    #             labels=["Cash", "Long Positions", "Short Positions"],
    #             colors=["#86c7f3", "#ffe29a", "#ffb3b3"],
    #         )

    #     # Configure plots
    #     ax1.set_title(f"Portfolio Performance {self.symbol}")
    #     ax1.set_ylabel("Portfolio Value ($)")
    #     ax1.legend(loc="upper left", handles=handles)

    #     ax2.set_title("Portfolio Composition")
    #     ax2.set_ylabel("Value ($)")
    #     ax2.set_xlabel("Date")
    #     ax2.legend(loc="upper left")

    #     plt.tight_layout()
    #     plt.show()

    #     return fig

    def plot_results(self, history_df=None):
        """Plot backtest results with fixed stackplot rendering"""
        if history_df is None:
            history_df = pd.DataFrame(self.portfolio["value_history"])
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
            history_df = history_df.set_index("timestamp")

        if history_df.empty:
            self.logger.warning("No data to plot")
            return

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]}
        )

        # Plot portfolio value
        ax1.plot(
            history_df.index, history_df["value"], label="Portfolio Value", color="blue"
        )

        # Plot price if available
        if "price" in history_df.columns:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(
                history_df.index,
                history_df["price"],
                label="Price",
                color="gray",
                alpha=0.6,
            )
            ax1_twin.set_ylabel("Price ($)")

        # Mark buy and sell points (Green for buy, Red for sell)
        for transaction in self.portfolio["transactions"]:
            tx_type, price, qty, timestamp = transaction
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            if tx_type == "BUY":
                ax1.scatter(
                    timestamp,
                    (
                        history_df.loc[history_df.index == timestamp, "value"].iloc[0]
                        if len(history_df.loc[history_df.index == timestamp]) > 0
                        else 0
                    ),
                    color="green",
                    marker="^",
                    s=100,
                )
            elif tx_type == "SELL":
                ax1.scatter(
                    timestamp,
                    (
                        history_df.loc[history_df.index == timestamp, "value"].iloc[0]
                        if len(history_df.loc[history_df.index == timestamp]) > 0
                        else 0
                    ),
                    color="red",
                    marker="v",
                    s=100,
                )

        # Manually set legend for buy/sell markers
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create manual symbols for legend
        # patch = mpatches.Patch(color="grey", label="manual patch")
        buy = Line2D(
            [0],
            [0],
            label="Buy",
            marker="^",
            markersize=10,
            color="green",
            markerfacecolor="green",
            linestyle="",
        )
        sell = Line2D(
            [0],
            [0],
            label="Sell",
            marker="v",
            markersize=10,
            color="red",
            markerfacecolor="red",
            linestyle="",
        )

        # Add manual symbols to auto legend
        handles.extend([buy, sell])

        # Add cash vs. position allocation - FIX STACKED PLOT
        if len(history_df) > 0:
            # Calculate position value by subtracting cash from total value
            history_df["position_value"] = history_df["value"] - history_df["cash"]

            # Split into long and short components
            history_df["long_positions"] = history_df["position_value"].apply(
                lambda x: max(0, x)
            )
            history_df["short_positions"] = (
                history_df["position_value"].apply(lambda x: min(0, x)).abs()
            )

            # Prepare data for stackplot (cash and position components)
            x = history_df.index
            y1 = (
                history_df["cash"] - history_df["short_positions"]
            )  # Cash minus short liability
            y2 = history_df["long_positions"]  # Long positions
            y3 = history_df[
                "short_positions"
            ]  # Short positions (as positive values for plotting)

            # Create stackplot with correct data
            ax2.stackplot(
                x,
                y1,
                y2,
                y3,
                labels=["Cash", "Long Positions", "Short Positions"],
                colors=["#86c7f3", "#ffe29a", "#ffb3b3"],
            )

        # Configure plots
        ax1.set_title(f"Portfolio Performance: {self.symbol}")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="upper left", handles=handles)

        ax2.set_title("Portfolio Composition")
        ax2.set_ylabel("Value ($)")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper left")

        # Improve date formatting
        import matplotlib.dates as mdates

        date_format = mdates.DateFormatter("%Y-%m-%d")
        ax1.xaxis.set_major_formatter(date_format)
        ax2.xaxis.set_major_formatter(date_format)

        # Set major tick locations
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # Rotate date labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

        return fig

### Off-chain work



# # def create_signal_generator(predictor):
# #     """
# #     Create a signal generator function that uses the StockPredictor
# #     Args:
# #         predictor: An instance of StockPredictor
# #     Returns:
# #         function: A function that takes historical data and returns trading signals
# #     """
# #     def generate_signal(historical_data):
# #         # Update predictor's data
# #         predictor.data = historical_data.copy()
# #         # Generate signal using predictor's method
# #         # signal = predictor.generate_trading_signal(predictor.symbol, horizon=5)
# #         signal =  get_entry_signal(predictor, predictor.symbol)[0]
# #         return signal
# #     return generate_signal


# def create_signal_generator(predictor):
#     """
#     Create a signal generator function that uses StockPredictor with get_entry_signal

#     Args:
#         predictor: An instance of StockPredictor

#     Returns:
#         function: A function that takes historical data and returns trading signals
#     """

#     def generate_signal(historical_data):
#         # Update predictor's data with the current slice of historical data
#         predictor.data = historical_data.copy()
#         volatility = historical_data["Close"].pct_change().rolling(20).std().iloc[-1]
#         position_size_factor = min(0.05, max(0.01, 0.03 / (volatility * 10)))
#         predictor.current_position_size = position_size_factor

#         # Generate features that the predictor needs
#         # This might be needed depending on your predictor implementation
#         if hasattr(predictor, "load_features"):
#             predictor.load_features()

#         # Use get_entry_signal to generate a trading signal
#         # The function returns (decision, confidence, rationale, levels)
#         decision, confidence, rationale, levels = get_entry_signal(
#             predictor, current_price=float(historical_data["Close"].iloc[-1]),
#             reverse_signals=True
#         )


        
#         if len(historical_data) % 10 == 0:  # Log every 10 days
#             print(
#                 f"Date: {historical_data.index[-1]}, Signal: {decision}, Confidence: {confidence}%"
#             )
#             print(f"Rationale: {rationale}")

#         return decision

#         # return decision

#     return generate_signal

def create_signal_generator(predictor, always_reverse=False, autodetect_reversal=False):
    """
    Create a signal generator function that intelligently adapts to market conditions
    
    Args:
        predictor: An instance of StockPredictor
        always_reverse: If True, always use reversal regardless of autodetection
        
    Returns:
        function: A function that takes historical data and returns trading signals
    """
   
    def generate_signal(historical_data):
        # Update predictor's data with the current slice of historical data
        predictor.data = historical_data.copy()
        volatility = historical_data["Close"].pct_change().rolling(20).std().iloc[-1]
        position_size_factor = min(0.05, max(0.01, 0.03 / (volatility * 10)))
        predictor.current_position_size = position_size_factor

        # Generate features that the predictor needs
        if hasattr(predictor, "load_features"):
            predictor.load_features()
        

        use_reversal = False  # Default
        # Calculate market direction over last 30 days
        market_trend = historical_data['Close'].pct_change(30).mean() 
        trend_strength = abs(market_trend)
        
        # Check if we're in a strong trend
        is_strong_trend = trend_strength > 0.005  # >0.5% daily avg movement
        
        # Check price relative to moving averages
        has_ma50 = 'MA_50' in historical_data.columns
        has_ma200 = 'MA_200' in historical_data.columns
            
        # If always_reverse is True, skip autodetection
        if always_reverse:
            use_reversal = always_reverse
        
        
        elif autodetect_reversal:
            # Actually make meaningful reversal decisions based on market conditions
            
           
            if has_ma50 and has_ma200:
                price = historical_data['Close'].iloc[-1]
                ma50 = historical_data['MA_50'].iloc[-1] 
                ma200 = historical_data['MA_200'].iloc[-1]
                
                # Logic: In uptrends, normal signals work better; in downtrends, reversed signals work better
                if market_trend > 0 and price > ma200:  # Solid uptrend
                    use_reversal = True  # reverse in strong uptrends
                elif market_trend < 0 and price < ma200:  # Solid downtrend
                    use_reversal = False   # Don't Reverse in downtrends
                elif is_strong_trend:     # Any other strong trend
                    use_reversal = True   # Default to reversal in strong trends
                else:                     # Sideways market
                    use_reversal = True   # Default to reversal in uncertain conditions
            else:
                # If we don't have moving averages, use simpler logic
                use_reversal = market_trend < 0  # Reverse in downtrends only
            
        # Log the decision periodically
        if len(historical_data) % 20 == 0:
            trend_type = "uptrend" if market_trend > 0 else "downtrend"
            strength = "strong" if is_strong_trend else "weak"
            logger.info(f"Market analysis: {strength} {trend_type} ({market_trend*100:.2f}% avg daily). Using reversal: {use_reversal}")
        
        # Get entry signal with the determined reversal setting
        decision, confidence, rationale, levels = get_entry_signal(
            predictor, 
            current_price=float(historical_data["Close"].iloc[-1]),
            reverse_signals=use_reversal
        )
        # Log the decision
        logger.info(f'Whether use reversed decision: {use_reversal}')
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


async def main():

    # Load or create wallet
    # private_key, address, _ = load_or_create_wallet()

    # # Initialize the clients
    # rest_client = RestClient(NODE_URL)
    # print("Connected to Aptos devnet")

    # # Convert address string to AccountAddress
    # account_address = AccountAddress.from_str(address)

    # # Print current balance
    # balance = await check_balance(rest_client, account_address)
    # print(f"Current balance: {balance} octas ({balance/1e8} APT)")

    # tracker = PortfolioTracker(
    #     initial_capital=balance / 1e8
    # )  # Initialize tracker with current balance

    # # If balance is low, fund the wallet
    # if balance < 100_000_000:  # Less than 1 APT
    #     print("Balance is low, funding wallet...")
    #     fund_wallet(address)

    # fund_wallet(address, amount=100_000_000, coin_type="0x1::btc_coin_coin::BtcCoin")  # Fund with 1 APT for testing)
    # Check balance again

    # # Coin transfer example
    # await reconcile_balances(rest_client, address, tracker)
    # # Example: transfer 10,000 octas to yourself (for testing)
    # recipient = address  # Replace with another address for real transfer
    # transfer_amount = 10000
    # success = await execute_transfer(private_key, recipient, transfer_amount)
    # print(f"Transfer {'successful' if success else 'failed'}")

    # Check entry points for trading
    from predictor import StockPredictor

    symbol = "QBTS"
    # symbol = "CRWD" 
    start = "2010-03-01"
    end = "2025-05-17"
    # end = date.today()
    _predictor = StockPredictor(symbol=symbol, start_date=start, end_date=end)
    _predictor.load_data()
    # print(_predictor.data.columns)
    if symbol == "APT21794-USD":
        backtester = AptosBacktester(symbol=symbol, initial_capital=balance / 1e8)
    else:
        backtester = AptosBacktester(symbol=symbol, initial_capital=100000)
    # Run a simple backtest with default strategy
    print("Running backtest...")

    autodect_reversal = True  # Set to True to enable autodetection of reversal
    history, metrics = backtester.run_backtest(
        start_date=start,
        end_date=end,
        signal_generator=create_signal_generator(
            predictor=_predictor, always_reverse=False, 
            autodetect_reversal=autodect_reversal
        ),  # Use the predictor's signal generator
    
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

    # Plot results
    backtester.plot_results(history)


if __name__ == "__main__":
    asyncio.run(main())
