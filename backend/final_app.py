from flask import Flask, request, jsonify
import numpy as np
from web3 import Web3
from web3.datastructures import AttributeDict
from hexbytes import HexBytes
import json

# Import your AI modules
from agent import DQNAgent
from generator import build_generator

app = Flask(__name__)

def convert_obj(obj):
    """Recursively convert HexBytes and AttributeDict objects to serializable formats."""
    if isinstance(obj, HexBytes):
        return obj.hex()
    elif isinstance(obj, AttributeDict):
        return {k: convert_obj(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: convert_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_obj(item) for item in obj]
    else:
        return obj

# --- AI Initialization ---
state_size = 5
action_size = 3
agent = DQNAgent(state_size, action_size)

latent_dim = 10
market_generator = build_generator(latent_dim)

# --- Blockchain Setup ---
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

# Load the contract artifact using the correct path
with open("../blockchain/build/contracts/MarketOptimizer.json", "r") as f:
    contract_artifact = json.load(f)

contract_address = contract_artifact["networks"]["5777"]["address"]
contract_abi = contract_artifact["abi"]
market_optimizer_contract = web3.eth.contract(address=contract_address, abi=contract_abi)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        state = data.get("state")
        print("Received state:", state)
        
        action = agent.act(state)
        
        noise = np.random.normal(0, 1, (1, latent_dim))
        simulated_data = market_generator.predict(noise, verbose=0)[0].tolist()
        
        account = web3.eth.accounts[0]
        tx_hash = market_optimizer_contract.functions.recordDecision("update pricing", action).transact({'from': account})
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert the transaction receipt to a JSON-serializable format
        converted_receipt = convert_obj(tx_receipt)
        
        return jsonify({
            "action": action,
            "simulated_market": simulated_data,
            "tx_receipt": converted_receipt
        })
    except Exception as e:
        print("Error in /predict endpoint:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

