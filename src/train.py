import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "clean_data.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)

    drop_cols = ['EarlyLabel', 'Patient_ID', 'Unnamed: 0']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['EarlyLabel']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss"
    )

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nAUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print("Model saved →", MODEL_PATH)


if __name__ == "__main__":
    main()
