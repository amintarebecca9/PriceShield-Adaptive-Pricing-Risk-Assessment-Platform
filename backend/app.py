from flask import Flask, request, jsonify
import numpy as np
import json
from web3 import Web3
from web3.datastructures import AttributeDict
from hexbytes import HexBytes
from tensorflow.keras.models import load_model
from agent import DQNAgent

app = Flask(__name__)

# Load GAN model for market simulation
gan_model = load_model("GAN/gan.h5")  # Update path as necessary
latent_dim = 10  # Ensure this matches your trained GAN

def convert_obj(obj):
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

def generate_market_factor():
    noise = np.random.normal(0, 1, (1, latent_dim))
    market_factor = gan_model.predict(noise, verbose=0)[0, 0]
    return market_factor

def risk_assessment(state, market_factor):
    avg_risk_score = state[0]  # Extract user's risk score from the 5D vector
    adjusted_risk = avg_risk_score + (market_factor * 0.1)
    if adjusted_risk < 0.5:
        return 0  # Low risk
    elif adjusted_risk < 1.0:
        return 1  # Medium risk
    else:
        return 2  # High risk

# AI Initialization
state_size = 6  # Updated to include market factor (5D vector + market factor)
action_size = 3
agent = DQNAgent(state_size, action_size)

# Blockchain Setup
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
with open("../blockchain/build/contracts/MarketOptimizer.json", "r") as f:
    contract_artifact = json.load(f)
contract_address = contract_artifact["networks"]["5777"]["address"]
contract_abi = contract_artifact["abi"]
market_optimizer_contract = web3.eth.contract(address=contract_address, abi=contract_abi)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "5D_vector" not in data:
            return jsonify({"error": "Missing '5D_vector' in request"}), 400

        # Convert the provided 5D vector to a NumPy array
        state = np.array(data["5D_vector"], dtype=np.float64)
        print("Debug: Received 5D vector:", state)
        
        # Generate market factor using the GAN model
        market_factor = generate_market_factor()
        print("Debug: Generated market factor:", market_factor)
        
        # Compute risk assessment with market influence
        risk_decision = risk_assessment(state, market_factor)
        print("Debug: Computed risk decision:", risk_decision)
        
        # Combine the 5D vector with the market factor to form a 6D vector
        state_with_market = np.append(state, market_factor)
        print("Debug: Combined 6D vector:", state_with_market)
        
        # Compute the interest rate using the DQN agent with the 6D vector
        interest_rate = agent.act(state_with_market)
        print("Debug: Agent returned interest rate:", interest_rate)
        
        # Record the decision on the blockchain
        account = web3.eth.accounts[0]
        tx_hash_interest = market_optimizer_contract.functions.recordDecision("interest_rate", interest_rate)\
                                                            .transact({'from': account})
        print("Debug: Transaction hash:", tx_hash_interest)
        
        tx_receipt_interest = web3.eth.wait_for_transaction_receipt(tx_hash_interest)
        print("Debug: Transaction receipt:", tx_receipt_interest)
        
        converted_receipt_interest = convert_obj(tx_receipt_interest)
        
        return jsonify({
            "risk_decision": risk_decision,
            "market_factor": market_factor,
            "interest_rate": interest_rate,
            "tx_receipt_interest": converted_receipt_interest
        })
    except Exception as e:
        print("Error in /predict endpoint:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
