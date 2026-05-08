"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_dataset() -> pd.DataFrame:
    """Tiny synthetic dataset that mimics the real schema for unit tests."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "age": rng.integers(29, 78, size=n),
            "sex": rng.integers(0, 2, size=n),
            "cp": rng.integers(0, 4, size=n),
            "trestbps": rng.integers(94, 200, size=n),
            "chol": rng.integers(126, 564, size=n),
            "fbs": rng.integers(0, 2, size=n),
            "restecg": rng.integers(0, 3, size=n),
            "thalach": rng.integers(71, 202, size=n),
            "exang": rng.integers(0, 2, size=n),
            "oldpeak": rng.uniform(0, 6, size=n).round(1),
            "slope": rng.integers(0, 3, size=n),
            "ca": rng.integers(0, 5, size=n),
            "thal": rng.integers(0, 4, size=n),
        }
    )
    # Make target depend on a few features so a model can actually learn.
    score = 0.05 * df["age"] + 0.4 * df["cp"] + 0.5 * df["oldpeak"] - 0.04 * df["thalach"]
    df["target"] = (score > score.median()).astype(int)
    return df


@pytest.fixture()
def synthetic_csv(synthetic_dataset, tmp_path):
    csv = tmp_path / "heart_disease.csv"
    synthetic_dataset.to_csv(csv, index=False)
    return csv
