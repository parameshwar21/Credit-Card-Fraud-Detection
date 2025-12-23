import joblib
import numpy as np
from src.config import MODEL_PATH, SCALER_PATH

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_transaction(features):
    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]
    return "Fraud" if pred == 1 else "Normal"
