import pandas as pd
import os
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "clean_data.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH)

    # EXACT SAME drops as training
    drop_cols = ['EarlyLabel', 'Patient_ID', 'Unnamed: 0']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    model = joblib.load(MODEL_PATH)

    print("Computing SHAP values...")

    sample = X.sample(1000, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, show=False)
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/shap_summary.png")
    print("Saved SHAP plot → outputs/plots/shap_summary.png")


if __name__ == "__main__":
    main()
