from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.train_model import train
from src.config import MODEL_PATH, CM_IMAGE_PATH, ROC_IMAGE_PATH, REPORT_PATH

def evaluate():
    model, X_test, y_test = train()

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(report)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(CM_IMAGE_PATH)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.savefig(ROC_IMAGE_PATH)
    plt.close()

    print("Evaluation completed. Reports saved.")
