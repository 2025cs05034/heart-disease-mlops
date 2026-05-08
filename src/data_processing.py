"""Data processing utilities for the Heart Disease dataset.

Exposes:
    - `FEATURE_NAMES`         : ordered list of model input columns
    - `NUMERIC_FEATURES`      : continuous features
    - `CATEGORICAL_FEATURES`  : ordinal/categorical features
    - `load_data()`           : read CSV, basic cleaning, return X, y
    - `build_preprocessor()`  : sklearn `ColumnTransformer` for full reproducibility
    - `train_test_split_data()` : stratified split with deterministic seed
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"

RANDOM_SEED = 42


def load_data(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and return (X, y).

    Drops fully duplicated rows; coerces all columns to numeric.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run `python -m src.download_data` first.")
    df = pd.read_csv(csv_path)
    logger.info("Loaded dataframe with shape %s", df.shape)

    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing from dataset.")
    df[TARGET] = (df[TARGET] > 0).astype(int)

    missing_features = set(FEATURE_NAMES) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    X = df[FEATURE_NAMES].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build a deterministic preprocessing pipeline.

    Numeric features  -> median imputation + StandardScaler
    Categorical feats -> most-frequent imputation + OneHotEncoder
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
):
    """Stratified train/test split with a fixed random seed."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def basic_eda_summary(X: pd.DataFrame, y: pd.Series) -> dict:
    """Return a small summary used by training/EDA scripts."""
    return {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "missing_per_column": {c: int(X[c].isna().sum()) for c in X.columns},
        "target_balance": {int(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "numeric_describe": X[NUMERIC_FEATURES].describe().to_dict(),
    }


__all__ = [
    "FEATURE_NAMES",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET",
    "RANDOM_SEED",
    "load_data",
    "build_preprocessor",
    "train_test_split_data",
    "basic_eda_summary",
]
