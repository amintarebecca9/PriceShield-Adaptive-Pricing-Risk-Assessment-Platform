import pandas as pd
import numpy as np
import random

# Load Lending Club Loan dataset (Ensure you have the CSV file)
df = pd.read_csv("dataset/lending_club_loan_data.csv", low_memory=False)  # Update with actual dataset file name

# Select relevant columns (Modify based on dataset structure)
selected_columns = ["loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "term", "grade", "sub_grade", "loan_status", "purpose", "issue_d", "emp_length", "total_acc", "revol_bal", "delinq_2yrs", "earliest_cr_line"]
df = df[selected_columns]

# Ensure 'int_rate' is treated as a string before replacing characters
df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "", regex=False)
df["int_rate"] = pd.to_numeric(df["int_rate"], errors='coerce')

# Convert earliest credit line to numeric credit age (years)
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors='coerce')
df["credit_age"] = (pd.to_datetime("today") - df["earliest_cr_line"]).dt.days / 365.25

# Calculate Debt-to-Income (DTI) Ratio
df["DTI"] = df["loan_amnt"] / df["annual_inc"]
df["DTI"].replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
df["DTI"].fillna(df["DTI"].mean(), inplace=True)  # Replace NaN with mean DTI


# Define risk score calculation based on loan status and credit history
def calculate_risk_score(loan_status, delinquencies, credit_age, total_acc, revol_bal, dti):
    risk = 0.5  # Base risk score
    if isinstance(loan_status, str):
        if "Default" in loan_status or "Charged Off" in loan_status:
            risk += 0.4  # High risk
        elif "Late" in loan_status:
            risk += 0.2  # Medium risk
        elif "Fully Paid" in loan_status:
            risk -= 0.3  # Very Low risk
        else:
            risk -= 0.1  # Low risk
    
    # Adjustments based on financial history
    if delinquencies > 2:
        risk += 0.2  # More delinquencies = Higher risk
    if credit_age < 5:
        risk += 0.1  # Shorter credit history = Higher risk
    if total_acc < 5:
        risk += 0.1  # Fewer accounts = Higher risk
    if revol_bal > 50000:
        risk += 0.2  # High revolving balance = Higher risk
    if dti > 0.4:
        risk += 0.2  # High debt-to-income ratio = Higher risk
    
    return round(max(0, min(risk, 1)), 2)  # Ensure risk is between 0 and 1

# Generate Transaction Histories
def generate_transaction_history(df, num_users=1000):
    transaction_data = []
    
    for user_id in range(1, num_users + 1):
        num_transactions = random.randint(5, 50)  # Each user has 5-50 transactions
        user_row = df.sample(n=1).iloc[0]  # Randomly pick a user's loan data
        risk_score = calculate_risk_score(user_row["loan_status"], user_row["delinq_2yrs"], user_row["credit_age"], user_row["total_acc"], user_row["revol_bal"], user_row["DTI"])
        
        for _ in range(num_transactions):
            transaction = {
                "user_id": user_id,
                "transaction_id": random.randint(100000, 999999),
                "transaction_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(random.randint(1, 365), unit='D'),
                "amount": round(random.uniform(50, 5000), 2),
                "category": random.choice(["Loan Payment", "Groceries", "Shopping", "Bills", "Entertainment", "Healthcare"]),
                "payment_method": random.choice(["Credit Card", "Debit Card", "Bank Transfer", "Crypto", "Mobile Wallet"]),
                "loan_status": user_row["loan_status"],
                "risk_score": risk_score,
            }
            transaction_data.append(transaction)
    
    return pd.DataFrame(transaction_data)

# Generate transaction history
df_transactions = generate_transaction_history(df)

# Generate Wallet Usage Metrics (Capturing Spending Trends Over Time)
def generate_wallet_metrics(df_transactions):
    df_transactions["month"] = df_transactions["transaction_date"].dt.to_period("M")
    wallet_usage = df_transactions.groupby("user_id").agg(
        total_spent=("amount", "sum"),
        avg_transaction_value=("amount", "mean"),
        transaction_count=("transaction_id", "count"),
        unique_categories=("category", "nunique"),
        most_used_payment_method=("payment_method", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        avg_risk_score=("risk_score", "mean"),
        spending_trend=("amount", lambda x: x.pct_change().mean(skipna=True))  # Monthly spending trend
    ).reset_index()
    
    return wallet_usage

# Generate spending patterns and wallet metrics
df_wallet_metrics = generate_wallet_metrics(df_transactions)

# Create 5D Feature Vector for DeepQ IR Network
def generate_5d_feature_vector(df_wallet_metrics):
    df_wallet_metrics["5D_vector"] = df_wallet_metrics.apply(
        lambda row: [
            row["avg_risk_score"],  # Risk Score
            df.loc[df["loan_amnt"].notna(), "DTI"].mean(),  # Average DTI from dataset
            row["spending_trend"],  # Spending Trend
            row["total_spent"],  # Total Spending
            row["transaction_count"]  # Transaction Frequency
        ], axis=1
    )
    return df_wallet_metrics[["user_id", "5D_vector"]]

# Generate 5D feature vectors
df_5d_vectors = generate_5d_feature_vector(df_wallet_metrics)

# Save outputs
df_transactions.to_csv("dataset/synthetic__transaction_history.csv", index=False)
df_wallet_metrics.to_csv("dataset/wallet__usage_metrics.csv", index=False)
df_5d_vectors.to_csv("dataset/user_5D_vectors.csv", index=False)

print("Synthetic transaction history, wallet usage metrics, and 5D feature vectors saved successfully!")
