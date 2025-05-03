# Fraud Transaction Detection System

This system is designed to detect fraudulent transactions using machine learning. It processes transaction data and identifies potential fraudulent activities based on various features and patterns.

## Dataset Description

The dataset contains simulated transaction data with the following columns:
- TRANSACTION_ID: Unique identifier for each transaction
- TX_DATETIME: Date and time of the transaction
- CUSTOMER_ID: Unique identifier for the customer
- TERMINAL_ID: Unique identifier for the merchant terminal
- TX_AMOUNT: Transaction amount
- TX_FRAUD: Binary label (0 for legitimate, 1 for fraudulent)

## Fraud Scenarios

The system is designed to detect three types of fraud scenarios:
1. Transactions with amount > 220
2. Transactions on terminals that have been compromised for 28 days
3. Transactions from customers whose credentials have been leaked (1/3 of their transactions have amounts multiplied by 5)

## Features

The system includes:
- Data preprocessing and feature engineering
- Random Forest classifier for fraud detection
- Feature importance analysis
- Model evaluation metrics
- Visualization of results

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python fraud_detection.py
```

The script will:
1. Load and combine all transaction data
2. Preprocess the data and create features
3. Train the model
4. Evaluate the model's performance
5. Display feature importance analysis

## Model Features

The model uses the following features:
- Transaction amount
- Time-based features (hour, day of week)
- Customer behavior features (mean amount, standard deviation, transaction count)
- Terminal behavior features (mean amount, standard deviation, transaction count)
- Historical fraud rates for customers and terminals

## Output

The system provides:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- Feature importance plot
- Top 5 most important features for fraud detection 