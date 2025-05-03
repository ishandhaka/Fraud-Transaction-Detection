import pandas as pd
import numpy as np
import joblib
import os
from fraud_detection import FraudDetector
import matplotlib.pyplot as plt
import seaborn as sns

def get_float_input(prompt, default=None, min_value=None, max_value=None):
    """Get float input from user with validation"""
    default_text = f" (default: {default})" if default is not None else ""
    while True:
        try:
            value_str = input(f"{prompt}{default_text}: ")
            if value_str == "" and default is not None:
                return default
            value = float(value_str)
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be no more than {max_value}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")

def get_int_input(prompt, default=None, min_value=None, max_value=None):
    """Get integer input from user with validation"""
    default_text = f" (default: {default})" if default is not None else ""
    while True:
        try:
            value_str = input(f"{prompt}{default_text}: ")
            if value_str == "" and default is not None:
                return default
            value = int(value_str)
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be no more than {max_value}")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer")

def predict_transaction():
    print("\n===== Fraud Detection System =====")
    print("Please enter the transaction details")
    print("(Press Enter to use default values)\n")
    
    # Get all inputs from user
    amount = get_float_input("Transaction amount ($)", min_value=0)
    hour = get_int_input("Hour of the transaction (0-23)", default=12, min_value=0, max_value=23)
    day = get_int_input("Day of the week (0=Monday, 6=Sunday)", default=3, min_value=0, max_value=6)
    
    print("\nCustomer Information:")
    customer_mean = get_float_input("Mean transaction amount for this customer", default=150, min_value=0)
    customer_std = get_float_input("Standard deviation of transaction amount for this customer", default=50, min_value=0)
    customer_count = get_int_input("Number of transactions for this customer", default=20, min_value=1)
    customer_fraud_rate = get_float_input("Historical fraud rate for this customer (0-1)", default=0.05, min_value=0, max_value=1)
    
    print("\nTerminal Information:")
    terminal_mean = get_float_input("Mean transaction amount for this terminal", default=150, min_value=0)
    terminal_std = get_float_input("Standard deviation of transaction amount for this terminal", default=50, min_value=0)
    terminal_count = get_int_input("Number of transactions for this terminal", default=100, min_value=1)
    terminal_fraud_rate = get_float_input("Historical fraud rate for this terminal (0-1)", default=0.05, min_value=0, max_value=1)
    
    # Load or train model
    model_path = 'models/fraud_detector.joblib'
    if os.path.exists(model_path):
        detector = joblib.load(model_path)
    else:
        print("\nModel not found. Training a new model...")
        detector = FraudDetector()
        # Load and preprocess data
        df = detector.load_data('dataset/data')
        X, y = detector.preprocess_data(df)
        detector.train(X, y)
        # Save the model
        os.makedirs('models', exist_ok=True)
        joblib.dump(detector, model_path)
        print(f"Model saved to {model_path}")
    
    # Create a dataframe with the input data
    transaction_data = pd.DataFrame({
        'TX_AMOUNT': [amount],
        'hour': [hour],
        'day_of_week': [day],
        'TX_AMOUNT_mean_customer': [customer_mean],
        'TX_AMOUNT_std_customer': [customer_std],
        'TX_AMOUNT_count_customer': [customer_count],
        'TX_FRAUD_mean_customer': [customer_fraud_rate],
        'TX_AMOUNT_mean_terminal': [terminal_mean],
        'TX_AMOUNT_std_terminal': [terminal_std],
        'TX_AMOUNT_count_terminal': [terminal_count],
        'TX_FRAUD_mean_terminal': [terminal_fraud_rate]
    })
    
    print("\nAnalyzing transaction...")
    
    # Make prediction
    prediction = detector.predict(transaction_data)[0]
    
    # If the model supports probability estimation
    probability = None
    if hasattr(detector.model, 'predict_proba'):
        probability = detector.model.predict_proba(detector.scaler.transform(transaction_data))[0, 1]
    
    # Display the result
    print("\n===== Transaction Fraud Analysis =====")
    print(f"Transaction Amount: ${amount:.2f}")
    print(f"Hour of Day: {hour}")
    print(f"Day of Week: {day} (0=Monday, 6=Sunday)")
    print("\nPrediction Result:")
    
    if prediction == 1:
        print("⚠️  FRAUD DETECTED ⚠️")
        print("This transaction is classified as fraudulent.")
    else:
        print("✅ LEGITIMATE TRANSACTION")
        print("This transaction is classified as legitimate.")
    
    if probability is not None:
        print(f"\nFraud Probability: {probability:.2%}")
    
    print("\nContributing Factors:")
    
    # Explain the prediction based on known rules
    if amount > 220:
        print(f"- Transaction amount (${amount:.2f}) exceeds the fraud threshold ($220.00)")
    
    if customer_fraud_rate > 0.1:
        print(f"- Customer has a high historical fraud rate ({customer_fraud_rate:.2%})")
    
    if terminal_fraud_rate > 0.1:
        print(f"- Terminal has a high historical fraud rate ({terminal_fraud_rate:.2%})")
    
    if abs(amount - customer_mean) > 2 * customer_std:
        print("- Transaction amount deviates significantly from customer's typical behavior")
    
    print("\n===================================")
    
    # Ask if user wants to analyze another transaction
    another = input("\nWould you like to analyze another transaction? (y/n): ").lower()
    if another.startswith('y'):
        predict_transaction()

if __name__ == "__main__":
    predict_transaction() 