import pandas as pd

INPUT_PATH = "data/sepsis_dataset.csv"
OUTPUT_PATH = "data/clean_data.csv"

PRED_HOURS = 6

def create_early_label(df):
    df['EarlyLabel'] = 0

    if 'PatientID' not in df.columns:
        print("No PatientID found → skipping early window labeling")
        df['EarlyLabel'] = df['SepsisLabel']
        return df

    for pid, group in df.groupby('PatientID'):
        idx = group[group['SepsisLabel'] == 1].index

        if len(idx) > 0:
            first = idx[0]
            start = first - PRED_HOURS

            if start > group.index.min():
                df.loc[start:first-1, 'EarlyLabel'] = 1

    return df


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Shape:", df.shape)

    # forward fill time-series
    if 'PatientID' in df.columns:
        df = df.groupby('PatientID').apply(lambda x: x.ffill())

    # fill missing
    df = df.fillna(df.median(numeric_only=True))

    # create early label
    df = create_early_label(df)

    # drop leakage columns
    drop_cols = ['SepsisLabel', 'Hour']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved cleaned data →", OUTPUT_PATH)


if __name__ == "__main__":
    main()
