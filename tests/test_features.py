"""Tests for the preprocessing pipeline."""
import numpy as np

from src.features import CATEGORICAL, FEATURES, NUMERIC, TARGET, build_preprocessor


def dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


def test_feature_lists_are_consistent():
    assert len(set(FEATURES)) == len(FEATURES)       # no duplicates
    assert TARGET not in FEATURES                      # no label leakage


def test_preprocessor_keeps_row_count_and_has_no_nan(xy):
    X, _ = xy
    out = dense(build_preprocessor().fit_transform(X))
    assert out.shape[0] == len(X)
    assert not np.isnan(out).any()


def test_preprocessor_handles_missing_values(xy):
    X, _ = xy
    X.loc[X.index[0], "chol"] = np.nan
    X.loc[X.index[1], "thal"] = np.nan
    out = dense(build_preprocessor().fit_transform(X))
    assert not np.isnan(out).any()


def test_preprocessor_ignores_unseen_category(xy):
    X, _ = xy
    pre = build_preprocessor().fit(X)
    new = X.head(3).copy()
    new["thal"] = 99.0                                 # category never seen in training
    out = dense(pre.transform(new))
    assert out.shape[0] == 3


def test_numeric_columns_are_scaled(xy):
    X, _ = xy
    out = dense(build_preprocessor().fit_transform(X))
    numeric_block = out[:, :len(NUMERIC)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-8)


def test_onehot_expands_categorical_columns(xy):
    X, _ = xy
    out = dense(build_preprocessor().fit_transform(X))
    assert out.shape[1] > len(FEATURES)                # categoricals became several columns
    assert len(CATEGORICAL) > 0
