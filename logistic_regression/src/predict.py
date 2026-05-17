import pickle
import os
import pandas as pd
from src.feature_extractor import extract_features

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = None

def predict_url(url):
    global model
    if model is None:
        model = load_model()
    
    features = extract_features(url)
    feature_names = ['isIp', 'urlLen', 'is@', 'isredirect', 'haveDash', 'domainLen', 'nosOfSubdomain']
    
    # Create DataFrame to match the feature names used during training
    df = pd.DataFrame([features])[feature_names]
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1] if prediction == 1 else model.predict_proba(df)[0][0]
    
    return {
        'is_scam': bool(prediction == 1),
        'probability': float(probability),
        'features': features
    }
