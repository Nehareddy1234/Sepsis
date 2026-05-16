import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text

DATA_PATH = "data/clean_data.csv"
MODEL_PATH = "models/xgb_model.pkl"


def main():
    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    drop_cols = ['EarlyLabel', 'Patient_ID', 'Unnamed: 0']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['EarlyLabel']

    model = joblib.load(MODEL_PATH)

    print("Getting predictions from black-box model...")
    preds = model.predict(X)

    print("Training interpretable surrogate tree...")

    # 🔥 IMPROVED SETTINGS
    tree = DecisionTreeClassifier(
        max_depth=4,              # slightly deeper
        min_samples_leaf=50,      # allow smaller leaves
        class_weight="balanced",  # VERY IMPORTANT
        random_state=42
    )

    tree.fit(X, preds)

    print("\n=== Extracted Clinical Rules ===\n")

    rules = export_text(tree, feature_names=list(X.columns))
    print(rules)

    import os
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/rules.txt", "w") as f:
        f.write(rules)

    print("\nRules saved → outputs/rules.txt")


if __name__ == "__main__":
    main()
