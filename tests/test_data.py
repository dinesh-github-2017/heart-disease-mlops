"""Tests for data download/cleaning and data loading."""
from pathlib import Path

import pandas as pd
import pytest

from src.download_data import COLUMNS, clean
from src.features import DATA_PATH, FEATURES, TARGET, load_data


def make_raw() -> pd.DataFrame:
    base = dict(age=55, sex=1, cp=3, trestbps=130, chol=240, fbs=0, restecg=0,
                thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0.0, thal=3.0)
    rows = [{**base, "target": t} for t in [0, 1, 2, 3, 4, 0]]
    rows[0]["ca"] = float("nan")       # missing value on a healthy row
    rows[5]["thal"] = float("nan")     # missing value on another healthy row
    return pd.DataFrame(rows, columns=COLUMNS)


def test_clean_makes_target_binary():
    cleaned = clean(make_raw())
    assert set(cleaned["target"].unique()) <= {0, 1}


def test_clean_maps_any_disease_level_to_one():
    cleaned = clean(make_raw())
    # raw targets 1, 2, 3, 4 are all "disease present"
    assert cleaned["target"].sum() == 4


def test_clean_keeps_required_columns():
    cleaned = clean(make_raw())
    assert set(FEATURES + [TARGET]) <= set(cleaned.columns)


def test_clean_does_not_modify_input():
    raw = make_raw()
    clean(raw)
    assert raw["target"].max() == 4


def test_load_data_on_real_file():
    if not Path(DATA_PATH).exists():
        pytest.skip("data/heart_clean.csv not found - run src/download_data.py first")
    X, y = load_data()
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) > 100
    assert set(y.unique()) <= {0, 1}
