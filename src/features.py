"""Feature definitions and preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/heart_clean.csv"
TARGET = "target"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["cp", "restecg", "slope", "thal", "ca"]   # TODO (yours): is `ca` better as numeric?
BINARY = ["sex", "fbs", "exang"]
FEATURES = NUMERIC + CATEGORICAL + BINARY


def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].astype(float)
    return df[FEATURES], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    binary = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
        ("bin", binary, BINARY),
    ])