import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from fraud_detection import FraudDetector
import joblib
import os

def test_model_on_file(model_path, test_file):
    """Test a saved model on a specific file"""
    # Load the model
    detector = joblib.load(model_path)
    
    # Load test data
    df = pd.read_pickle(test_file)
    
    # Preprocess
    print(f"Testing on {test_file}")
    X, y = detector.preprocess_data(df)
    
    # Make predictions
    y_pred = detector.predict(X)
    
    # Evaluate
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC curve
    if hasattr(detector.model, 'predict_proba'):
        y_proba = detector.model.predict_proba(detector.scaler.transform(X))[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    return y, y_pred

def train_and_save_model():
    """Train the model and save it"""
    detector = FraudDetector()
    
    # Load and preprocess data
    print("Loading data...")
    df = detector.load_data('dataset/data')
    print(f"Loaded {len(df)} transactions")
    
    print("\nPreprocessing data...")
    X, y = detector.preprocess_data(df)
    
    print("\nTraining model...")
    X_test, y_test = detector.train(X, y)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(detector, 'models/fraud_detector.joblib')
    print("Model saved to models/fraud_detector.joblib")
    
    return detector

def test_model_on_amount_threshold():
    """Test if the model correctly captures the rule: amounts > 220 are fraudulent"""
    detector = FraudDetector()
    
    # Create test data: transactions with amounts around the threshold
    amounts = np.linspace(200, 240, 100)
    test_data = pd.DataFrame({
        'TX_AMOUNT': amounts,
        'hour': [12] * 100,
        'day_of_week': [3] * 100,
        'TX_AMOUNT_mean_customer': [150] * 100,
        'TX_AMOUNT_std_customer': [50] * 100,
        'TX_AMOUNT_count_customer': [20] * 100,
        'TX_FRAUD_mean_customer': [0.1] * 100,
        'TX_AMOUNT_mean_terminal': [150] * 100,
        'TX_AMOUNT_std_terminal': [50] * 100,
        'TX_AMOUNT_count_terminal': [100] * 100,
        'TX_FRAUD_mean_terminal': [0.1] * 100
    })
    
    # Load model if exists, otherwise train
    if os.path.exists('models/fraud_detector.joblib'):
        detector = joblib.load('models/fraud_detector.joblib')
    else:
        detector = train_and_save_model()
    
    # Make predictions
    y_pred = detector.predict(test_data)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(amounts, y_pred, c=y_pred, cmap='coolwarm', alpha=0.8)
    plt.axvline(x=220, color='r', linestyle='--', label='Fraud Threshold (220)')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Prediction (0=Normal, 1=Fraud)')
    plt.title('Model Predictions by Transaction Amount')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Option 1: Train and save model
    if not os.path.exists('models/fraud_detector.joblib'):
        train_and_save_model()
    
    # Option 2: Test on a specific date
    test_model_on_file('models/fraud_detector.joblib', 'dataset/data/2018-09-30.pkl')
    
    # Option 3: Test the amount rule (transactions > 220)
    test_model_on_amount_threshold() 