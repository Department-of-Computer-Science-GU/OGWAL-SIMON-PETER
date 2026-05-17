import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

def train():
    print("Loading dataset...")
    # Read the dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'phishing 2.csv')
    df = pd.read_csv(data_path)
    
    # Select features
    features = ['isIp', 'urlLen', 'is@', 'isredirect', 'haveDash', 'domainLen', 'nosOfSubdomain']
    X = df[features]
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")
    
    # Save the model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train()