"""Tests for `src.data_processing`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    build_preprocessor,
    load_data,
    train_test_split_data,
)


def test_feature_lists_consistent():
    """The exposed feature constants must be aligned with each other."""
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert FEATURE_NAMES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_load_data_returns_expected_shapes(synthetic_csv):
    X, y = load_data(synthetic_csv)
    assert list(X.columns) == FEATURE_NAMES
    assert X.shape[0] == y.shape[0] > 0
    assert set(y.unique()).issubset({0, 1})


def test_load_data_collapses_target_to_binary(tmp_path):
    df = pd.DataFrame({col: [1, 2, 3] for col in FEATURE_NAMES} | {"target": [0, 2, 4]})
    csv = tmp_path / "heart.csv"
    df.to_csv(csv, index=False)
    _, y = load_data(csv)
    assert list(y) == [0, 1, 1]


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


def test_train_test_split_is_stratified(synthetic_csv):
    X, y = load_data(synthetic_csv)
    X_tr, X_te, y_tr, y_te = train_test_split_data(X, y, test_size=0.25)
    assert len(X_tr) + len(X_te) == len(X)
    # within stratified split, class proportions stay close to original
    overall_ratio = y.mean()
    assert abs(y_tr.mean() - overall_ratio) < 0.15
    assert abs(y_te.mean() - overall_ratio) < 0.15


def test_train_test_split_is_deterministic(synthetic_csv):
    X, y = load_data(synthetic_csv)
    a1, a2, _, _ = train_test_split_data(X, y, test_size=0.25, seed=123)
    b1, b2, _, _ = train_test_split_data(X, y, test_size=0.25, seed=123)
    pd.testing.assert_frame_equal(a1, b1)
    pd.testing.assert_frame_equal(a2, b2)


def test_preprocessor_handles_missing_values(synthetic_dataset):
    """Preprocessor must impute NaNs and emit a 2-D float matrix."""
    df = synthetic_dataset.copy()
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    df.loc[2, "trestbps"] = np.nan
    X = df[FEATURE_NAMES]
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert out.ndim == 2
    assert not np.isnan(out).any()
    # Numeric features should be standardised: mean ~ 0
    numeric_idx = list(range(len(NUMERIC_FEATURES)))
    np.testing.assert_allclose(out[:, numeric_idx].mean(axis=0), 0.0, atol=1e-7)


def test_preprocessor_handles_unknown_categories(synthetic_dataset):
    """OneHotEncoder must ignore categories unseen in fit."""
    train = synthetic_dataset.iloc[:50].copy()
    test = synthetic_dataset.iloc[50:].copy()
    test.loc[test.index[0], "thal"] = 99  # unseen category
    pre = build_preprocessor()
    pre.fit(train[FEATURE_NAMES])
    out = pre.transform(test[FEATURE_NAMES])
    assert out.shape[0] == len(test)
