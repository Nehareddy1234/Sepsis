import pandas as pd

def simulate_timeline(model, sample, hours=6, treatment=False):

    temp = sample.copy()
    risks = []

    for i in range(hours):

        # simulate physiological change
        if not treatment:
            if 'HR' in temp.columns:
                temp['HR'] += 2
            if 'Temp' in temp.columns:
                temp['Temp'] += 0.2
            if 'MAP' in temp.columns:
                temp['MAP'] -= 1
        else:
            if 'HR' in temp.columns:
                temp['HR'] -= 1
            if 'Temp' in temp.columns:
                temp['Temp'] -= 0.1
            if 'MAP' in temp.columns:
                temp['MAP'] += 1

        prob = model.predict_proba(temp)[0][1]
        risks.append(prob * 100)

    return risks