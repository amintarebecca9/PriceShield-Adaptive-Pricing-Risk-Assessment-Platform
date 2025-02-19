from flask import Flask, request, jsonify
import numpy as np
from web3 import Web3
from web3.datastructures import AttributeDict
from hexbytes import HexBytes
import json

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

def risk_assessment(state):
    """
    A simple risk assessment function that computes a risk level
    based on the sum of the state values.
    Returns:
      0 for low risk,
      1 for medium risk,
      2 for high risk.
    (Adjust the thresholds as needed.)
    """
    total = sum(state)
    if total < 1.0:
        return 0  # Low risk
    elif total < 2.0:
        return 1  # Medium risk
    else:
        return 2  # High risk

# --- AI Initialization ---
# For demonstration, assume state size = 5 and action size = 3 for pricing adjustment
state_size = 5
action_size = 3
agent = DQNAgent(state_size, action_size)

# Initialize GAN generator (for market simulation)
latent_dim = 10
market_generator = build_generator(latent_dim)

# --- Blockchain Setup ---
# Connect to your local Ganache node (ensure Ganache is running on port 7545)
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

# Load the contract artifact (adjust the path as necessary)
with open("../blockchain/build/contracts/MarketOptimizer.json", "r") as f:
    contract_artifact = json.load(f)

# Use the network id that matches your Ganache instance (commonly "5777")
contract_address = contract_artifact["networks"]["5777"]["address"]
contract_abi = contract_artifact["abi"]
market_optimizer_contract = web3.eth.contract(address=contract_address, abi=contract_abi)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        state = data.get("state")
        print("Received state:", state)
        
        # 1. Compute Pricing Adjustment Decision using the AI agent
        pricing_decision = agent.act(state)
        
        # 2. Compute Risk Assessment Decision using our simple function
        risk_decision = risk_assessment(state)
        
        # 3. Generate simulated market data using the GAN generator
        noise = np.random.normal(0, 1, (1, latent_dim))  # shape (1, latent_dim)
        simulated_data = market_generator.predict(noise, verbose=0)[0].tolist()
        
        # 4. Record the pricing adjustment on the blockchain
        account = web3.eth.accounts[0]  # Use the first Ganache account
        tx_hash_pricing = market_optimizer_contract.functions.recordDecision("pricing_adjustment", pricing_decision).transact({'from': account})
        tx_receipt_pricing = web3.eth.wait_for_transaction_receipt(tx_hash_pricing)
        
        # 5. Record the risk assessment on the blockchain (record risk level as uint)
        tx_hash_risk = market_optimizer_contract.functions.recordDecision("risk_assessment", risk_decision).transact({'from': account})
        tx_receipt_risk = web3.eth.wait_for_transaction_receipt(tx_hash_risk)
        
        # Convert transaction receipts for JSON serialization
        converted_receipt_pricing = convert_obj(tx_receipt_pricing)
        converted_receipt_risk = convert_obj(tx_receipt_risk)
        
        # 6. Return a combined response
        return jsonify({
            "pricing_decision": pricing_decision,
            "risk_decision": risk_decision,
            "simulated_market": simulated_data,
            "tx_receipt_pricing": converted_receipt_pricing,
            "tx_receipt_risk": converted_receipt_risk
        })
    except Exception as e:
        print("Error in /predict endpoint:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

