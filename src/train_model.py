from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.data_preprocessing import preprocess
from src.config import MODEL_PATH

def train():
    X, y = preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=120)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print("Model saved:", MODEL_PATH)

    return model, X_test, y_test
