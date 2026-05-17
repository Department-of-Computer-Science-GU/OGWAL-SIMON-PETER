import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from custom_ensembles import CustomRandomForest, CustomAdaBoost, CustomVotingClassifier
import pickle
import os

def train_ensemble():
    print("Loading dataset...")
    # The dataset is located in the logistic_regression folder
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logistic_regression', 'data', 'phishing 2.csv'))
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        return

    # Select the same features we used before for consistency
    features = ['isIp', 'urlLen', 'is@', 'isredirect', 'haveDash', 'domainLen', 'nosOfSubdomain']
    X = df[features]
    y = df['label']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Initializing custom ensemble models...")
    # Initialize custom ensemble methods
    rf = CustomRandomForest(n_estimators=100, random_state=42)
    ada = CustomAdaBoost(n_estimators=50, random_state=42)
    
    # Combine them into a single Custom Voting Classifier
    ensemble_model = CustomVotingClassifier(
        estimators=[
            ('rf', rf), 
            ('ada', ada)
        ],
        voting='soft'
    )
    
    print("Training the ensemble model. This might take a moment...")
    ensemble_model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = ensemble_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Ensemble Model Accuracy: {acc:.4f}")
    
    # Save the trained model
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'ensemble_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    print(f"Ensemble model successfully saved to {model_path}")

if __name__ == '__main__':
    train_ensemble()
