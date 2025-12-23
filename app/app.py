import sys
import os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, request

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import MODEL_PATH, SCALER_PATH

# Initialize app
app = Flask(__name__)

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return render_template("index.html", prediction="No file uploaded.")
        
        file = request.files['file']
        if file.filename == "":
            return render_template("index.html", prediction="No file selected.")

        # Read CSV
        df = pd.read_csv(file)

        # Check required columns
        required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        for col in required_cols:
            if col not in df.columns:
                return render_template("index.html", prediction=f"Missing column: {col}")

        # Scale V1–V28
        v_features = df[[f'V{i}' for i in range(1, 29)]].values
        scaled_v = scaler.transform(v_features)

        # Combine scaled features + Amount
        amount = df['Amount'].values.reshape(-1, 1)
        final_input = np.hstack([scaled_v, amount])

        # Predict
        predictions = model.predict(final_input)

        # Prepare results
        df['Prediction'] = ['Fraudulent' if p==1 else 'Legitimate' for p in predictions]

        # Convert to HTML table
        result_table = df.to_html(classes='result-table', index=False)

        return render_template("index.html", prediction=result_table)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
