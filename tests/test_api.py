"""Tests for the FastAPI service."""

from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_processing import build_preprocessor, load_data

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


@pytest.fixture(autouse=True)
def trained_model(tmp_path, synthetic_csv, monkeypatch):
    """Train a tiny model and point the API module at it for the duration of the test.

    Note: we do **not** ``importlib.reload`` the API module because the
    ``prometheus_fastapi_instrumentator`` and our own Counter/Histogram objects
    register themselves with a process-wide registry on import - reloading
    triggers a "Duplicated timeseries" error.  Instead we patch the
    module-level ``MODEL_PATH`` and ``model_store`` attributes directly.
    """
    X, y = load_data(synthetic_csv)
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    model_path = tmp_path / "best_model.pkl"
    joblib.dump(pipe, model_path)

    import src.api as api_module

    monkeypatch.setattr(api_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(api_module, "model_store", api_module.ModelStore(model_path))
    yield api_module


def _client(api_module):
    return TestClient(api_module.app)


def test_health(trained_model):
    res = _client(trained_model).get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ready_when_model_present(trained_model):
    res = _client(trained_model).get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_root_lists_endpoints(trained_model):
    res = _client(trained_model).get("/")
    assert res.status_code == 200
    body = res.json()
    assert "endpoints" in body
    assert "/predict" in body["endpoints"]


def test_predict_returns_well_formed_response(trained_model):
    res = _client(trained_model).post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("heart_disease", "no_heart_disease")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_of_disease"] <= 1.0
    assert body["request_id"]


def test_predict_invalid_payload(trained_model):
    bad = dict(VALID_PAYLOAD, age=-5)
    res = _client(trained_model).post("/predict", json=bad)
    assert res.status_code == 422


def test_predict_missing_field(trained_model):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    res = _client(trained_model).post("/predict", json=bad)
    assert res.status_code == 422


def test_predict_batch(trained_model):
    res = _client(trained_model).post(
        "/predict/batch",
        json={"instances": [VALID_PAYLOAD, VALID_PAYLOAD]},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["predictions"]) == 2


def test_metrics_endpoint_exposes_prometheus(trained_model):
    # Hit /predict first so a counter is incremented
    _client(trained_model).post("/predict", json=VALID_PAYLOAD)
    res = _client(trained_model).get("/metrics")
    assert res.status_code == 200
    assert "heart_disease_predictions_total" in res.text


def test_request_id_header_round_trip(trained_model):
    given = "test-correlation-id-1234"
    res = _client(trained_model).post(
        "/predict",
        json=VALID_PAYLOAD,
        headers={"X-Request-Id": given},
    )
    assert res.status_code == 200
    assert res.headers["X-Request-Id"] == given
    assert res.json()["request_id"] == given
