import sys
import os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, request
from pathlib import Path

# --- ROBUST PATH SETUP ---
# Get the absolute path of the directory where credit.py is located (the 'app' folder)
BASE_DIR = Path(__file__).resolve().parent

# Go up one level to the Project Root (where 'models' and 'app' live side-by-side)
PROJECT_ROOT = BASE_DIR.parent

# Define the absolute paths to your files
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

# Add project root to sys.path so 'src' imports work correctly
sys.path.insert(0, str(PROJECT_ROOT))

# WARNING: If src.config also defines MODEL_PATH as a relative string like "../models",
# importing it will break the absolute paths we just created. 
# Keep this commented out unless you update src/config.py to use absolute paths.
# from src.config import MODEL_PATH, SCALER_PATH 

# --- INITIALIZE APP ---
app = Flask(__name__)

# --- LOAD MODEL AND SCALER ---
try:
    # We use str() because joblib expects a string path
    model = joblib.load(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    print(f"✅ Success: Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: File not found at {MODEL_PATH}")
    print("Check if your 'models' folder is in the root directory.")
    sys.exit(1)

# --- ROUTES ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

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

        # Check required columns (V1-V28 and Amount)
        required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        for col in required_cols:
            if col not in df.columns:
                return render_template("index.html", prediction=f"Missing column: {col}")

        # Scale V1–V28
        # We assume the scaler was trained on V1-V28 only
        v_features = df[[f'V{i}' for i in range(1, 29)]].values
        scaled_v = scaler.transform(v_features)

        # Combine scaled features + Amount
        # Note: If your model expects Amount to be scaled too, 
        # you'll need to include it in the scaler.transform step.
        amount = df['Amount'].values.reshape(-1, 1)
        final_input = np.hstack([scaled_v, amount])

        # Predict (0 = Legitimate, 1 = Fraudulent)
        predictions = model.predict(final_input)

        # Prepare results for the UI
        df['Prediction'] = ['Fraudulent' if p == 1 else 'Legitimate' for p in predictions]

        # Convert the top 10 rows to an HTML table for display
        result_table = df.head(10).to_html(classes='result-table', index=False)

        return render_template("index.html", prediction=result_table)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    # Run the Flask server
    app.run(debug=True)
