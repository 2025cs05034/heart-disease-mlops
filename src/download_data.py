"""Download Heart Disease UCI dataset.

Source: UCI Machine Learning Repository, Dataset ID 45
https://archive.ics.uci.edu/dataset/45/heart+disease

Two strategies are tried in order:
1. Primary  : UCI direct CSV URL (processed.cleveland.data) + manual schema
2. Fallback : `ucimlrepo` Python package (which already returns a DataFrame)

The script is idempotent - if the file already exists locally it is reused.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("download_data")

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" "heart-disease/processed.cleveland.data"

COLUMNS = [
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
    "target",
]


def _download_via_uci(out_path: Path) -> pd.DataFrame:
    logger.info("Downloading from UCI: %s", UCI_URL)
    response = requests.get(UCI_URL, timeout=30)
    response.raise_for_status()
    out_path.write_text(response.text)
    df = pd.read_csv(out_path, header=None, names=COLUMNS, na_values="?")
    return df


def _download_via_ucimlrepo() -> pd.DataFrame:
    logger.info("Falling back to ucimlrepo package (dataset id=45)")
    from ucimlrepo import fetch_ucirepo  # type: ignore

    bundle = fetch_ucirepo(id=45)
    X = bundle.data.features
    y = bundle.data.targets
    df = pd.concat([X, y], axis=1)
    df = df.rename(columns={"num": "target"})
    return df


def download(output_dir: Path) -> Path:
    """Download (or reuse) the cleaned Heart Disease dataset.

    Returns the path of the saved CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "processed.cleveland.data"
    csv_path = output_dir / "heart_disease.csv"

    if csv_path.exists():
        logger.info("Dataset already exists at %s, reusing.", csv_path)
        return csv_path

    df: pd.DataFrame
    try:
        df = _download_via_uci(raw_path)
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning("UCI direct download failed: %s", exc)
        df = _download_via_ucimlrepo()

    # The original target ranges 0-4 (severity); collapse to binary 0/1
    # 0 = no disease, 1-4 = disease present
    df["target"] = (df["target"].astype(float) > 0).astype(int)

    # Coerce to numeric, allow NaNs to be handled in preprocessing
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(csv_path, index=False)
    logger.info("Saved dataset (%s rows, %s columns) -> %s", len(df), df.shape[1], csv_path)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UCI Heart Disease dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        path = download(args.output_dir)
        print(f"Dataset ready: {path}")
        return 0
    except Exception as exc:  # pragma: no cover
        logger.exception("Download failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
