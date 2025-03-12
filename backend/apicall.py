import requests
import pandas as pd
import random
import ast
import re

def parse_vector(val):
    # Replace np.float64(some_number) with some_number
    # For example: "np.float64(0.5738791623171935)" becomes "0.5738791623171935"
    cleaned = re.sub(r'np\.float64\(([^)]+)\)', r'\1', val)
    return ast.literal_eval(cleaned)

# Load the CSV file containing user 5D vectors
df = pd.read_csv("dataset/user_5D_vectors.csv")

# Convert the "5D_vector" column from its string representation to an actual list
df["5D_vector"] = df["5D_vector"].apply(parse_vector)

# Randomly select a user row
random_user = df.sample(n=1).iloc[0]
vector_5d = random_user["5D_vector"]

print("Selected 5D vector:", vector_5d)

# Define the endpoint URL
url = "http://127.0.0.1:5000/predict"

# Create payload with the selected 5D vector
payload = {"5D_vector": vector_5d}

# Make the POST request
response = requests.post(url, json=payload)

print("Response from server:", response.json())
