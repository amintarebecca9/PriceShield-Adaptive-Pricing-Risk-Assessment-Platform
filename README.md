# PriceShield-Adaptive-Pricing-Risk-Assessment-Platform
PriceShield is a cutting-edge e-commerce and fintech solution that harnesses the power of artificial intelligence and blockchain to dynamically adjust prices and assess risk in real time. By combining advanced machine learning models, generative simulations, and immutable blockchain logging, PriceShield provides transparency, accountability, and adaptability in market decision-making.

Overview
PriceShield integrates three core components:

Adaptive Pricing:
A Deep Q-Network (DQN) agent that analyzes market conditions—represented as a feature vector ("state")—to decide on optimal pricing adjustments. In production, these decisions could translate to percentage-based price changes or more nuanced adjustments based on market dynamics.

Risk Assessment:
A risk evaluation module processes the same state data to determine a risk level (e.g., low, medium, high) using defined thresholds. This helps flag potential issues, such as increased volatility or credit risk.

Blockchain Logging & Transparency:
Every decision—both pricing adjustments and risk assessments—is recorded on a blockchain through smart contracts. This not only ensures an immutable audit trail but also sets the stage for potential tokenized governance in the future.

Additionally, PriceShield uses a Generative Adversarial Network (GAN) to simulate market scenarios, giving insights into potential future market conditions that further inform pricing and risk decisions.

Technologies Used
Backend: Python, Flask
AI & Machine Learning: TensorFlow/Keras, NumPy
Blockchain: Solidity, Truffle, Ganache, Web3.py
Deployment: Local development environment with Ganache (CLI or GUI)

Project Structure

PriceShield: Adaptive Pricing & Risk Assessment Platform
├── backend
│   ├── app.py                   # Flask application integrating AI & blockchain calls
│   ├── agent.py                 # DQNAgent for pricing adjustment decisions
│   ├── generator.py             # GAN generator for simulated market data
│   ├── requirements.txt         # Python dependencies
│   └── venv/                    # Python virtual environment
├── blockchain
│   ├── contracts
│   │   └── MarketOptimizer.sol  # Solidity smart contract for recording decisions
│   ├── migrations
│   │   └── 2_deploy_contracts.js # Deployment script
│   ├── build/                   # Compiled contract artifacts (generated by Truffle)
│   ├── truffle-config.js        # Truffle configuration file
│   └── package.json             # Node dependencies for Truffle
└── README.md                    # Project documentation


Getting Started
Prerequisites
Python 3.11+ and pip
Node.js (with npm)
Truffle (install via npm install -g truffle)
Ganache CLI or Ganache GUI
(Optional) nvm for managing Node versions

Installation
Clone the Repository:
git clone https://github.com/yourusername/PriceShield.git
cd PriceShield

Set Up the Python Environment:

cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Install Node Dependencies (for Truffle):

cd ../blockchain
npm install
Deployment
Start Ganache:

Incase of using Ganache CLI:

npx ganache-cli -p 7545 -i 5777

Incase of using the Ganache GUI, ensure it’s configured to run on port 7545 with network id 5777.

Compile & Deploy the Smart Contract:

truffle compile --all
truffle migrate --reset --network development

Verify Deployment:

In the Truffle console:

truffle console --network development
let instance = await MarketOptimizer.deployed();
let events = await instance.getPastEvents("DecisionRecorded", { fromBlock: 0, toBlock: 'latest' });
console.log(events);

Running the Flask Application
Start the Flask Server:

cd ../backend
source venv/bin/activate
python app.py
Test the /predict Endpoint:

Use curl or Postman:

curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"state": [0.1, 0.2, 0.3, 0.4, 0.5]}'
The response will include:

pricing_decision: The AI agent’s pricing adjustment.
risk_decision: The risk level (e.g., 0 for low, 1 for medium, 2 for high).
simulated_market: Simulated market data from the GAN.
tx_receipt_pricing & tx_receipt_risk: Transaction receipts confirming that decisions were recorded on-chain.

How It Works
Input & State Representation:
The client sends a state vector (e.g., [0.1, 0.2, 0.3, 0.4, 0.5]) representing market conditions, customer behavior, or other relevant features.

Decision Making:
The DQN agent processes the state and outputs a pricing adjustment decision.
A simple risk assessment function evaluates the state to assign a risk level.

Market Simulation:
A GAN generator produces simulated market data, offering insights into potential future conditions.

Blockchain Logging:
Both the pricing decision and risk assessment are recorded on-chain by invoking the recordDecision function of the smart contract. Transaction receipts provide verifiable evidence of these updates.

Response:
The system responds with the AI’s decisions, simulated market data, and blockchain transaction details, ensuring transparency and accountability.
