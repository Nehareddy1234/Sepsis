import numpy as np
import pandas as pd
import joblib

class TemporalCounterfactualPlanner:

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

        # safe clinical bounds per step
        self.step_bounds = {
            'HR': (-20, 20),
            'Temp': (-1.5, 0),
            'MAP': (-10, 15),
            'SBP': (-15, 20),
            'Resp': (-6, 6),
            'Lactate': (-1.0, 0),
            'O2Sat': (0, 5)
        }

    def predict(self, x):
        return self.model.predict_proba(pd.DataFrame([x]))[:,1][0]

    def plan(self, x0, horizon=3, target=0.1):

        x = x0.copy()
        steps = []

        risk = self.predict(x)

        for t in range(horizon):

            # always try to improve risk (even if already low)
            if t == horizon-1:
                break

            best_reduction = 0
            best_feature = None
            best_new_val = None
            best_new_risk = risk

            for f, (lo, hi) in self.step_bounds.items():

                if f not in x:
                    continue

                new_val = x[f] + hi
                x_temp = x.copy()
                x_temp[f] = new_val

                r = self.predict(x_temp)

                reduction = risk - r

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = f
                    best_new_val = new_val
                    best_new_risk = r

            if best_feature is None:
                break

            steps.append({
                "step": t+1,
                "feature": best_feature,
                "old": x[best_feature],
                "new": best_new_val,
                "risk_after": best_new_risk
            })

            x[best_feature] = best_new_val
            risk = best_new_risk

        return steps
