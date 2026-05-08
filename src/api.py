"""FastAPI service exposing the heart-disease classifier.

Endpoints
---------
GET  /            -> service info
GET  /health      -> liveness probe
GET  /ready       -> readiness probe (model loaded?)
POST /predict     -> single prediction
POST /predict/batch -> batch predictions
GET  /metrics     -> Prometheus metrics (also exposed for scraping)

The model is loaded lazily on first request (or eagerly via env flag) from
the path stored in `MODEL_PATH` (default: `models/best_model.pkl`).

Logging is JSON-formatted (`python-json-logger`) so log aggregators can
ingest it directly. Each request gets a correlation id (`X-Request-Id`).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator
from pythonjsonlogger import jsonlogger

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("heart_disease_api")
logger.setLevel(LOG_LEVEL)
_handler = logging.StreamHandler()
_handler.setFormatter(
    jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
)
if not logger.handlers:
    logger.addHandler(_handler)
logger.propagate = False

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pkl"))
SERVICE_NAME = os.getenv("SERVICE_NAME", "heart-disease-api")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")

INPUT_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# --------------------------------------------------------------------------- #
# Prometheus custom metrics                                                   #
# --------------------------------------------------------------------------- #
PREDICTION_COUNTER = Counter(
    "heart_disease_predictions_total",
    "Total number of predictions made.",
    ["prediction"],
)
PREDICTION_LATENCY = Histogram(
    "heart_disease_prediction_latency_seconds",
    "Latency of /predict in seconds.",
)
PREDICTION_ERRORS = Counter(
    "heart_disease_prediction_errors_total",
    "Total prediction errors.",
    ["error_type"],
)


# --------------------------------------------------------------------------- #
# Pydantic schemas                                                            #
# --------------------------------------------------------------------------- #
class HeartDiseaseInput(BaseModel):
    """Input schema validated against the training feature set."""

    age: float = Field(..., ge=0, le=120, description="Age in years.")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female).")
    cp: int = Field(..., ge=0, le=3, description="Chest-pain type (0-3).")
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure (mm Hg).")
    chol: float = Field(..., ge=100, le=700, description="Serum cholesterol (mg/dl).")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true).")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2).")
    thalach: float = Field(..., ge=50, le=260, description="Maximum heart rate achieved.")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1=yes).")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise.")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2).")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy.")
    thal: int = Field(..., ge=0, le=7, description="Thalassemia indicator.")

    @field_validator("oldpeak")
    @classmethod
    def _round_oldpeak(cls, v: float) -> float:
        # Tolerate float input but clip negatives.
        return max(float(v), 0.0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionOutput(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., description="0 = no heart disease, 1 = heart disease present.")
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probability_of_disease: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    request_id: str


class BatchInput(BaseModel):
    instances: List[HeartDiseaseInput]


class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #
class ModelStore:
    """Lazy loader for the trained sklearn pipeline."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.path}. " "Train it via `python -m src.train` first."
                )
            logger.info(
                "Loading model",
                extra={"model_path": str(self.path)},
            )
            self._model = joblib.load(self.path)
        return self._model

    def is_loaded(self) -> bool:
        return self._model is not None or self.path.exists()


model_store = ModelStore(MODEL_PATH)


# --------------------------------------------------------------------------- #
# FastAPI app                                                                 #
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the model at startup if available."""
    try:
        if model_store.is_loaded():
            _ = model_store.model  # trigger load
            logger.info("Model preloaded at startup.")
        else:
            logger.warning(
                "Model file missing at startup; will fail on /predict until trained.",
                extra={"model_path": str(MODEL_PATH)},
            )
    except Exception as exc:
        logger.error("Failed to preload model: %s", exc)
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Production-ready ML service that predicts the risk of heart disease "
        "from patient health features. Built for the BITS Pilani MLOps assignment."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# Prometheus instrumentation - exposes /metrics
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
).instrument(
    app
).expose(app, endpoint="/metrics", include_in_schema=False)


# --------------------------------------------------------------------------- #
# Middleware: correlation id + structured access logging                      #
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "elapsed_ms": round(elapsed_ms, 2),
                "error": str(exc),
            },
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-Id"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    return response


# --------------------------------------------------------------------------- #
# Endpoints                                                                   #
# --------------------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "endpoints": ["/health", "/ready", "/predict", "/predict/batch", "/metrics", "/docs"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not MODEL_PATH.exists():
        return JSONResponse(
            {"status": "not-ready", "reason": "model file not found"},
            status_code=503,
        )
    return {"status": "ready", "model_path": str(MODEL_PATH)}


def _to_dataframe(payload: HeartDiseaseInput) -> pd.DataFrame:
    record = {feature: getattr(payload, feature) for feature in INPUT_FEATURES}
    return pd.DataFrame([record])[INPUT_FEATURES]


def _predict_one(payload: HeartDiseaseInput, request_id: str) -> PredictionOutput:
    df = _to_dataframe(payload)
    with PREDICTION_LATENCY.time():
        proba = float(model_store.model.predict_proba(df)[0, 1])
        prediction = int(proba >= 0.5)

    confidence = proba if prediction == 1 else 1.0 - proba
    label = "heart_disease" if prediction == 1 else "no_heart_disease"

    PREDICTION_COUNTER.labels(prediction=label).inc()
    return PredictionOutput(
        prediction=prediction,
        label=label,
        confidence=round(confidence, 6),
        probability_of_disease=round(proba, 6),
        model_version=SERVICE_VERSION,
        request_id=request_id,
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(payload: HeartDiseaseInput, request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        result = _predict_one(payload, request_id)
        logger.info(
            "prediction",
            extra={
                "request_id": request_id,
                "prediction": result.prediction,
                "label": result.label,
                "confidence": result.confidence,
            },
        )
        return result
    except FileNotFoundError as exc:
        PREDICTION_ERRORS.labels(error_type="model_missing").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        PREDICTION_ERRORS.labels(error_type="inference_error").inc()
        logger.exception("inference_error", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="Internal inference error") from exc


@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(payload: BatchInput, request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    if not payload.instances:
        raise HTTPException(status_code=400, detail="`instances` must be a non-empty list.")
    try:
        results = [_predict_one(item, request_id) for item in payload.instances]
        logger.info(
            "batch_prediction",
            extra={
                "request_id": request_id,
                "batch_size": len(results),
            },
        )
        return BatchOutput(predictions=results)
    except FileNotFoundError as exc:
        PREDICTION_ERRORS.labels(error_type="model_missing").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc


__all__ = ["app"]
