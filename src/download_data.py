"""Download the UCI Heart Disease (Cleveland) dataset and save a clean CSV."""
from pathlib import Path
import pandas as pd

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

RAW_PATH = Path("data/heart_raw.csv")
CLEAN_PATH = Path("data/heart_clean.csv")


def download() -> pd.DataFrame:
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    RAW_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)   # 0 = no disease, 1 = disease
    # TODO (yours): decide how to handle missing 'ca' and 'thal'
    # (drop rows, median/mode fill, ...). Note your choice for the report.
    return df


if __name__ == "__main__":
    raw = download()
    print("Raw shape:", raw.shape)
    print("Missing values:\n", raw.isna().sum()[raw.isna().sum() > 0])
    cleaned = clean(raw)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print("Saved", CLEAN_PATH, cleaned.shape)
