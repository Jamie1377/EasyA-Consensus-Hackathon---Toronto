from aptos_sdk.account import Account
from aptos_sdk.account_address import AccountAddress
from dotenv import dotenv_values
from aptos_sdk_wrapper import get_balance
from agents import get_balance_in_apt_sync
import os
import json

# wallet_file = "aptos_wallet.json"
script_dir = os.path.dirname(os.path.abspath(__file__))
wallet_file = os.path.join(script_dir, "aptos_wallet.json")
if os.path.exists(wallet_file):
    try:
        with open(wallet_file, "r") as f:
            wallet = json.load(f)
            private_key = wallet["private_key"]
            address = wallet["address"]
            public_key = wallet["public_key"]
            print(f"Existing Account:\nPrivate Key: {private_key}\nAddress: {address}")

    except json.JSONDecodeError:
        account = Account.generate()  # Generate a new account if the file is corrupted
        private_key = str(account.private_key)
        address = str(account.address())
        public_key = str(account.public_key())
        print(f"New Account:\nPrivate Key: {private_key}\nAddress: {address}")
        if os.path.exists(wallet_file):
            with open(wallet_file, "w") as f:
                json.dump({
                    "private_key": private_key,
                    "address": address,
                    "public_key": public_key
                }, f, indent=4)
            print(f"New account generated and saved to {wallet_file}")
    
# print(f"New Account:\nPrivate Key: {private_key}\nAddress: {address}")
else:
    # create a directory to the file
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

# l
import requests

# fund wallet is good for devnet
def fund_wallet(address):
    url = f"https://faucet.devnet.aptoslabs.com/mint?address={address}&amount={100_000_000}"
    headers = {"Content-Type": "application/json"}
    data = {"amount": 100_000_000, "coin_type": "0x1::aptos_coin::AptosCoin"}  # 100 APT
    response = requests.post(url, headers=headers, json=data)
    print(f"Funding {address} with 1 APT...")
    print("Funded!" if response.status_code == 200 else "Failed")

# fund_wallet(address)  # Replace with your address


# from wallet get the address

print(f"Balance: {get_balance_in_apt_sync()} Octas.")

print(f"Address: {AccountAddress.from_str(address)}")

# Start trading



from aptos_sdk.transactions import EntryFunction, TransactionArgument, TransactionPayload
import asyncio
from aptos_sdk.account import Account
from aptos_sdk.async_client import FaucetClient, RestClient
from aptos_sdk.transactions import EntryFunction, TransactionPayload, TransactionArgument, RawTransaction
from aptos_sdk.bcs import Serializer
import time


# Network configuration
NODE_URL = "https://fullnode.devnet.aptoslabs.com/v1"
# NODE_URL = "https://api.devnet.aptoslabs.com/v1"
FAUCET_URL = "https://faucet.devnet.aptoslabs.com"
 
async def main():
    # Initialize the clients
    rest_client = RestClient(NODE_URL)
    faucet_client = FaucetClient(FAUCET_URL, rest_client)
    print("Connected to Aptos devnet")
    


    # Check initial balances
    balance = await rest_client.account_balance(AccountAddress.from_str(address))
    
    print("\n=== Initial Balances ===")
    print(f"Balance: {balance} octas")
 
    # 1. Build the transaction
    


    print("\n=== 1. Building the transaction ===")
    
    # Create the entry function payload
    # This specifies which function to call and with what arguments
    entry_function = EntryFunction.natural(
        "0x1::aptos_account",  # Module address and name
        "transfer",            # Function name
        [],                    # Type arguments (empty for this function)
        [
            # Function arguments with their serialization type
            TransactionArgument(AccountAddress.from_str(address), Serializer.struct),  # Recipient address
            TransactionArgument(10000, Serializer.u64),              # Amount to transfer (1000 octas)
        ],
    )

    # Get the chain ID for the transaction
    chain_id =  await rest_client.chain_id()
    
    # Get the sender's current sequence number
    account_data =  await rest_client.account(address)
    sequence_number = int(account_data["sequence_number"])
    
    # Create the raw transaction with all required fields
    raw_transaction = RawTransaction(
        sender=address,                                    # Sender's address
        sequence_number=sequence_number,                           # Sequence number to prevent replay attacks
        payload=TransactionPayload(entry_function),                # The function to call
        max_gas_amount=2000,                                       # Maximum gas units to use
        gas_unit_price=100,                                        # Price per gas unit in octas
        expiration_timestamps_secs=int(time.time()) + 600,         # Expires in 10 minutes
        chain_id=chain_id,                                         # Chain ID to ensure correct network
    )
    
    print("Transaction built successfully")
    print(f"Sender: {raw_transaction.sender}")
    print(f"Sequence Number: {raw_transaction.sequence_number}")
    print(f"Max Gas Amount: {raw_transaction.max_gas_amount}")
    print(f"Gas Unit Price: {raw_transaction.gas_unit_price}")
    print(f"Expiration Timestamp: {time.ctime(raw_transaction.expiration_timestamps_secs)}")


     # 2. Simulate the transaction
    print("\n=== 2. Simulating the transaction ===")
    
    # Create a BCS transaction for simulation
    # This doesn't actually submit the transaction to the blockchain
    account = Account.load_key(private_key)
    simulation_transaction = await rest_client.create_bcs_transaction(account, TransactionPayload(entry_function))
    
    # Simulate the transaction to estimate gas costs and check for errors
    simulation_result = await rest_client.simulate_transaction(simulation_transaction, account)
    
    # Extract and display the simulation results
    gas_used = int(simulation_result[0]['gas_used'])
    gas_unit_price = int(simulation_result[0]['gas_unit_price'])
    success = simulation_result[0]['success']
    
    print(f"Estimated gas units: {gas_used}")
    print(f"Estimated gas cost: {gas_used * gas_unit_price} octas")
    print(f"Transaction would {'succeed' if success else 'fail'}")
 
    # 3. Sign the transaction
    print("\n=== 3. Signing the transaction ===")
    
    # Sign the raw transaction with the sender's private key
    # This creates a cryptographic signature that proves the sender authorized this transaction
    signed_transaction = await rest_client.create_bcs_signed_transaction(
        account,                                  # Account with the private key
        TransactionPayload(entry_function),     # The payload from our transaction
        sequence_number=sequence_number         # Use the same sequence number as before
    )
    
    print("Transaction signed successfully")
    # We can't easily extract the signature from the signed transaction object,
    # but we can confirm it was created
 
    # 4. Submit the transaction
    print("\n=== 4. Submitting the transaction ===")
    
    # Submit the signed transaction to the blockchain
    # This broadcasts the transaction to the network for processing
    tx_hash = await rest_client.submit_bcs_transaction(signed_transaction)
    
    print(f"Transaction submitted with hash: {tx_hash}")
 
    # 5. Wait for the transaction to complete
    print("\n=== 5. Waiting for transaction completion ===")
    
    # Wait for the transaction to be processed by the blockchain
    # This polls the blockchain until the transaction is confirmed
    await rest_client.wait_for_transaction(tx_hash)
    
    # Get the transaction details to check its status
    transaction_details = await rest_client.transaction_by_hash(tx_hash)
    success = transaction_details["success"]
    vm_status = transaction_details["vm_status"]
    gas_used = transaction_details["gas_used"]
    
    print(f"Transaction completed with status: {'SUCCESS' if success else 'FAILURE'}")
    print(f"VM Status: {vm_status}")
    print(f"Gas used: {gas_used}")
 
    # Check final balances
    account_final_balance = await rest_client.account_balance(account.address())
   
    print("\n=== Final Balances ===")
    print(f"Alice: {account_final_balance} octas (spent {balance - account_final_balance} octas on transfer and gas)")
    






if __name__ == "__main__":
    asyncio.run(main())

