import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from src.config import DATA_PATH, SCALER_PATH

def preprocess():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved.")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    return X_res, y_res
