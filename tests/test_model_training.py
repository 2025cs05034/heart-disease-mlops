"""Smoke-tests for the training pipeline (no MLflow server required)."""

from __future__ import annotations

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.data_processing import FEATURE_NAMES, build_preprocessor, load_data, train_test_split_data


def _build_simple_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def test_pipeline_fits_and_predicts(synthetic_csv):
    X, y = load_data(synthetic_csv)
    X_tr, X_te, y_tr, y_te = train_test_split_data(X, y, test_size=0.3)

    pipe = _build_simple_pipeline()
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    preds = pipe.predict(X_te)

    assert proba.shape == (len(X_te),)
    assert set(preds.tolist()).issubset({0, 1})
    # AUC should not be perfectly random on this synthetic but learnable target
    auc = roc_auc_score(y_te, proba)
    assert auc >= 0.5


def test_pipeline_serialises_round_trip(synthetic_csv, tmp_path):
    X, y = load_data(synthetic_csv)
    pipe = _build_simple_pipeline()
    pipe.fit(X, y)
    out = tmp_path / "model.pkl"
    joblib.dump(pipe, out)
    restored = joblib.load(out)

    sample = X.head(3)
    assert (pipe.predict(sample) == restored.predict(sample)).all()


def test_input_dataframe_shape_matches_feature_names(synthetic_csv):
    X, y = load_data(synthetic_csv)
    assert list(X.columns) == FEATURE_NAMES


@pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
def test_split_size_respected(synthetic_csv, test_size):
    X, y = load_data(synthetic_csv)
    _, X_te, _, _ = train_test_split_data(X, y, test_size=test_size)
    assert abs(len(X_te) / len(X) - test_size) < 0.1
