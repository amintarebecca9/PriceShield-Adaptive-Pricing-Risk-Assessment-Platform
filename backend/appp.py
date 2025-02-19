from flask import Flask, request, jsonify
from agent import DQNAgent
from generator import build_generator
import numpy as np
app = Flask(__name__)

# For demonstration, assume state size = 5 and action size = 3
state_size = 5
action_size = 3
agent = DQNAgent(state_size, action_size)

# Initialize GAN generator (for market simulation)
latent_dim = 10
market_generator = build_generator(latent_dim)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        state = data.get("state")
        print("Received state:", state)
        action = agent.act(state)
        # Generate simulated market data
        noise = np.random.normal(0, 1, (1, latent_dim))
        simulated_data = market_generator.predict(noise, verbose=0)[0].tolist()
        return jsonify({"action": action, "simulated_market": simulated_data})
    except Exception as e:
        # Print error details to the console (or log them as needed)
        print("Error in /predict endpoint:", str(e))
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

