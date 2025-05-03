import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime, timedelta

class FraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, data_dir):
        """Load and combine all pickle files from the data directory"""
        all_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        df_list = []
        
        for file in all_files:
            df = pd.read_pickle(file)
            df_list.append(df)
            
        return pd.concat(df_list, ignore_index=True)
    
    def preprocess_data(self, df):
        """Preprocess the data and create features"""
        # Convert datetime to features
        df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        df['hour'] = df['TX_DATETIME'].dt.hour
        df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
        
        # Create customer and terminal features
        # Use separate aggregations to avoid MultiIndex
        customer_amount_mean = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].mean().reset_index().rename(columns={'TX_AMOUNT': 'TX_AMOUNT_mean_customer'})
        customer_amount_std = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].std().reset_index().rename(columns={'TX_AMOUNT': 'TX_AMOUNT_std_customer'})
        customer_amount_count = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].count().reset_index().rename(columns={'TX_AMOUNT': 'TX_AMOUNT_count_customer'})
        customer_fraud_mean = df.groupby('CUSTOMER_ID')['TX_FRAUD'].mean().reset_index().rename(columns={'TX_FRAUD': 'TX_FRAUD_mean_customer'})
        
        terminal_amount_mean = df.groupby('TERMINAL_ID')['TX_AMOUNT'].mean().reset_index().rename(columns={'TX_AMOUNT': 'TX_AMOUNT_mean_terminal'})
        terminal_amount_std = df.groupby('TERMINAL_ID')['TX_AMOUNT'].std().reset_index().rename(columns={'TX_AMOUNT': 'TX_AMOUNT_std_terminal'})
        terminal_amount_count = df.groupby('TERMINAL_ID')['TX_AMOUNT'].count().reset_index().rename(columns={'TX_AMOUNT': 'TX_AMOUNT_count_terminal'})
        terminal_fraud_mean = df.groupby('TERMINAL_ID')['TX_FRAUD'].mean().reset_index().rename(columns={'TX_FRAUD': 'TX_FRAUD_mean_terminal'})
        
        # Merge all features back to main dataframe
        df = df.merge(customer_amount_mean, on='CUSTOMER_ID', how='left')
        df = df.merge(customer_amount_std, on='CUSTOMER_ID', how='left')
        df = df.merge(customer_amount_count, on='CUSTOMER_ID', how='left')
        df = df.merge(customer_fraud_mean, on='CUSTOMER_ID', how='left')
        
        df = df.merge(terminal_amount_mean, on='TERMINAL_ID', how='left')
        df = df.merge(terminal_amount_std, on='TERMINAL_ID', how='left')
        df = df.merge(terminal_amount_count, on='TERMINAL_ID', how='left')
        df = df.merge(terminal_fraud_mean, on='TERMINAL_ID', how='left')
        
        # Fill NaN values with 0 for std
        df.fillna(0, inplace=True)
        
        # Select features for training
        features = [
            'TX_AMOUNT', 'hour', 'day_of_week',
            'TX_AMOUNT_mean_customer', 'TX_AMOUNT_std_customer', 'TX_AMOUNT_count_customer',
            'TX_FRAUD_mean_customer',
            'TX_AMOUNT_mean_terminal', 'TX_AMOUNT_std_terminal', 'TX_AMOUNT_count_terminal',
            'TX_FRAUD_mean_terminal'
        ]
        
        X = df[features]
        y = df['TX_FRAUD']
        
        # Store feature names for later use
        self.feature_names = features
        
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return X_test_scaled, y_test
    
    def predict(self, X):
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get and plot feature importance"""
        # Use the stored feature names instead of relying on feature_names_in_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.show()
        
        return feature_importance

def main():
    # Initialize the fraud detector
    detector = FraudDetector()
    
    # Load and preprocess data
    print("Loading data...")
    df = detector.load_data('dataset/data')
    print(f"Loaded {len(df)} transactions")
    
    print("\nPreprocessing data...")
    X, y = detector.preprocess_data(df)
    
    print("\nTraining model...")
    X_test, y_test = detector.train(X, y)
    
    print("\nAnalyzing feature importance...")
    feature_importance = detector.get_feature_importance()
    print("\nTop 5 most important features:")
    print(feature_importance.head())

if __name__ == "__main__":
    main() 