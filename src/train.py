"""Train heart-disease classifiers and log everything to MLflow.

Trains:
    - Logistic Regression (with grid search over `C`, `penalty`)
    - Random Forest        (with grid search over `n_estimators`, `max_depth`)

Each candidate is evaluated with stratified 5-fold cross-validation on the
training set; the best estimator is then refit on the full training set and
final test metrics + plots are logged.

The best of the two models (by ROC-AUC) is also serialised as `models/best_model.pkl`
together with the fitted preprocessor for use by the FastAPI service.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # no display required (CI/headless safe)
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data_processing import RANDOM_SEED, build_preprocessor, load_data, train_test_split_data  # noqa: E402
from src.download_data import download  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")

EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Author identity recorded in MLflow runs. Overridable via env so the local OS
# username (which on some machines equals an employer-issued ID) is never
# exposed in screenshots, run tags, or the public report.
MLFLOW_AUTHOR = os.getenv("MLFLOW_USER_NAME", "P Sai Madhav")

MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")


def _evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def _save_plots(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, prefix: str) -> Tuple[Path, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = ARTIFACTS_DIR / f"{prefix}_confusion_matrix.png"
    roc_path = ARTIFACTS_DIR / f"{prefix}_roc_curve.png"

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax_cm, cmap="Blues")
    ax_cm.set_title(f"{prefix} - Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
    ax_roc.set_title(f"{prefix} - ROC Curve")
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=150)
    plt.close(fig_roc)

    return cm_path, roc_path


def _log_run(
    model_name: str,
    grid: GridSearchCV,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int,
) -> Dict[str, float]:
    """Run grid search, evaluate, log to MLflow, return final metrics."""
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("dataset", "heart-disease-uci")
        # Override the auto-detected mlflow.user / mlflow.source.name so the
        # local OS username and absolute home-directory path are not leaked
        # in the MLflow UI or exported run data.
        mlflow.set_tag("mlflow.user", MLFLOW_AUTHOR)
        mlflow.set_tag("mlflow.source.name", "src/train.py")
        mlflow.log_param("cv_splits", n_splits)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))

        logger.info("[%s] Running grid search...", model_name)
        grid.fit(X_train, y_train)

        best_estimator: Pipeline = grid.best_estimator_

        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("cv_best_roc_auc", float(grid.best_score_))

        # Cross-validation breakdown for the BEST hyperparameter set, all metrics
        cv_metrics = {}
        for scoring in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            scores = cross_val_score(
                best_estimator,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED),
                scoring=scoring,
                n_jobs=-1,
            )
            cv_metrics[f"cv_{scoring}_mean"] = float(scores.mean())
            cv_metrics[f"cv_{scoring}_std"] = float(scores.std())
        for k, v in cv_metrics.items():
            mlflow.log_metric(k, v)

        # Final test-set metrics
        test_metrics = _evaluate(best_estimator, X_test, y_test)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
            logger.info("[%s] test_%s = %.4f", model_name, k, v)

        cm_path, roc_path = _save_plots(best_estimator, X_test, y_test, prefix=model_name)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # MLflow model logging (full pipeline, preprocessor + estimator)
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="model",
            registered_model_name=None,
            input_example=X_test.head(2),
        )

        return {
            "model_name": model_name,
            "estimator": best_estimator,
            "test_metrics": test_metrics,
            "cv_metrics": cv_metrics,
            "best_params": grid.best_params_,
        }


def _build_logistic_search(preprocessor) -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l1", "l2"],
    }
    return GridSearchCV(
        pipe,
        param_grid=grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        refit=True,
    )


def _build_random_forest_search(preprocessor) -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    grid = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    }
    return GridSearchCV(
        pipe,
        param_grid=grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        refit=True,
    )


def main(args: argparse.Namespace) -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = download(Path(args.data_dir))
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=args.test_size)
    logger.info("Train shape=%s | Test shape=%s", X_train.shape, X_test.shape)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)

    preprocessor = build_preprocessor()

    runs = []
    runs.append(
        _log_run(
            "logistic_regression",
            _build_logistic_search(preprocessor),
            X_train,
            y_train,
            X_test,
            y_test,
            n_splits=5,
        )
    )
    runs.append(
        _log_run(
            "random_forest",
            _build_random_forest_search(preprocessor),
            X_train,
            y_train,
            X_test,
            y_test,
            n_splits=5,
        )
    )

    best = max(runs, key=lambda r: r["test_metrics"]["roc_auc"])
    logger.info(
        "Best model: %s (test_roc_auc=%.4f)",
        best["model_name"],
        best["test_metrics"]["roc_auc"],
    )

    best_model_path = MODELS_DIR / "best_model.pkl"
    metadata_path = MODELS_DIR / "best_model_metadata.json"
    joblib.dump(best["estimator"], best_model_path)

    metadata = {
        "model_name": best["model_name"],
        "best_params": {
            k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in best["best_params"].items()
        },
        "test_metrics": best["test_metrics"],
        "cv_metrics": best["cv_metrics"],
        "feature_order_input": [
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
        ],
        "random_seed": RANDOM_SEED,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved best model -> %s", best_model_path)
    logger.info("Saved metadata   -> %s", metadata_path)

    summary_md = ARTIFACTS_DIR / "training_summary.md"
    with summary_md.open("w") as f:
        f.write("# Training Summary\n\n")
        f.write(f"**Best model:** `{best['model_name']}`\n\n")
        f.write("## Test metrics (held-out set)\n\n")
        for run in runs:
            f.write(f"### {run['model_name']}\n\n")
            f.write("| Metric | Value |\n|---|---|\n")
            for k, v in run["test_metrics"].items():
                f.write(f"| {k} | {v:.4f} |\n")
            f.write("\n")
    logger.info("Saved markdown summary -> %s", summary_md)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Heart-Disease classifiers and log to MLflow.")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
