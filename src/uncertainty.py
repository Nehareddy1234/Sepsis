import numpy as np
import pandas as pd

def risk_confidence(model, sample, n=50):

    preds = []

    x = sample.iloc[0].values.astype(float)

    for _ in range(n):
        noise = np.random.normal(0, 0.02, size=len(x))
        x_noisy = x + noise
        p = model.predict_proba(pd.DataFrame([x_noisy], columns=sample.columns))[:,1][0]
        preds.append(p)

    preds = np.array(preds)

    mean = preds.mean()
    std = preds.std()

    confidence = max(0, 100 - std*1000)

    return mean*100, confidence
