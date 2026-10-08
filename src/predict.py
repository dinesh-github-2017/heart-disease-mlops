"""Load the saved pipeline and make predictions from raw feature dicts."""
import joblib
import pandas as pd

from src.features import FEATURES

MODEL_PATH = "models/final_model.joblib"
THRESHOLD = 0.5          # TODO (yours, optional): test a lower value to favour recall

_model = None


def load_model(path: str = MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def predict(records: list[dict]) -> list[dict]:
    df = pd.DataFrame(records)[FEATURES].astype(float)
    proba = load_model().predict_proba(df)[:, 1]
    out = []
    for p in proba:
        label = int(p >= THRESHOLD)
        out.append({"prediction": label,
                    "confidence": round(float(p if label else 1 - p), 4)})
    return out


if __name__ == "__main__":
    sample = {"age": 54, "sex": 1, "cp": 3, "trestbps": 130, "chol": 246,
              "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
              "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3}
    print(predict([sample]))