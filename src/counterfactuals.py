import pandas as pd
import joblib

DATA_PATH = "data/clean_data.csv"
MODEL_PATH = "models/xgb_model.pkl"


def main():
    print("Loading data...")

    df = pd.read_csv(DATA_PATH)

    # --------------------------------
    # Use SAME features as training
    # --------------------------------
    drop_cols = ['EarlyLabel', 'Patient_ID', 'Unnamed: 0']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    model = joblib.load(MODEL_PATH)

    # --------------------------------
    # Find highest-risk patient
    # --------------------------------
    print("Finding highest-risk patient...")

    probs = model.predict_proba(X)[:, 1]
    idx = probs.argmax()

    sample = X.iloc[[idx]].copy()
    original_prob = probs[idx]

    print(f"\nSelected patient index: {idx}")
    print(f"Original predicted sepsis risk: {original_prob:.3f}")

    # --------------------------------
    # SMART Counterfactual (NEW LOGIC)
    # --------------------------------
    print("\nApplying model-guided improvements...")

    cf = sample.copy()
    rows = []

    normal_values = {
        'HR': 75,
        'Temp': 36.8,
        'MAP': 85,
        'SBP': 120,
        'Resp': 16,
        'Lactate': 1.2,
        'O2Sat': 98
    }

    current_prob = original_prob

    for col, val in normal_values.items():

        if col not in cf.columns:
            continue

        original_val = cf.iloc[0][col]

        # Try modification
        temp_cf = cf.copy()
        temp_cf[col] = val

        new_prob = model.predict_proba(temp_cf)[0][1]

        # Keep only if improvement
        if new_prob < current_prob:
            print(f"✓ {col}: {original_val} → {val}  | risk {current_prob:.3f} → {new_prob:.3f}")
            cf[col] = val
            current_prob = new_prob
            rows.append((col, original_val, val, new_prob))
        else:
            print(f"✗ {col}: change rejected (risk would increase)")

    # --------------------------------
    # Results
    # --------------------------------
    print("\nFinal counterfactual patient values:")
    print(cf)

    print("\nNew predicted risk:", round(current_prob, 3))
    print("Total risk reduction:", round(original_prob - current_prob, 3))


if __name__ == "__main__":
    main()
