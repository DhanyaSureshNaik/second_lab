import joblib
import os

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "../model/cancer_model.pkl")
model = joblib.load(model_path)

# Class names
CLASS_NAMES = {0: "malignant", 1: "benign"}

def predict_cancer(features: list):
    """Predict cancer class and probability"""
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][prediction]
    return {"class": CLASS_NAMES[prediction], "probability": float(probability)}
