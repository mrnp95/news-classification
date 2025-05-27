from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def vectorize_and_split(
    clean_path: Path,
    output_dir: Path,
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Vectorise clean text using TF-IDF and split into train/test sets.

    Saves:
        • vectorizer.joblib – fitted TfidfVectorizer
        • X_train.npz / X_test.npz – sparse TF-IDF matrices
        • y_train.npy / y_test.npy – label arrays
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cleaned dataset from %s", clean_path)
    df = pd.read_parquet(clean_path)

    X = df["clean_text"].fillna("").values
    y = df["label"].values

    logger.info("Splitting dataset (test_size=%.2f, stratify=y)...", test_size)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Fitting TfidfVectorizer (max_features=%d, ngram_range=%s)...", max_features, ngram_range)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",  # extra safety – already removed but avoids new words
        dtype=np.float32,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Save artifacts
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")
    sparse.save_npz(output_dir / "X_train.npz", X_train)
    sparse.save_npz(output_dir / "X_test.npz", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)

    logger.info("Saved vectorizer and datasets to %s", output_dir)
    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TF-IDF vectorisation and train-test split.")
    parser.add_argument(
        "--clean_path",
        type=Path,
        default=Path("workdir/processed_data_clean.parquet"),
        help="Path to cleaned parquet produced in preprocessing step.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("workdir/features"), help="Artifact output directory")
    parser.add_argument("--max_features", type=int, default=50000, help="Maximum TF-IDF vocabulary size")
    parser.add_argument("--ngram_min", type=int, default=1, help="Minimum n-gram size")
    parser.add_argument("--ngram_max", type=int, default=2, help="Maximum n-gram size")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for split")

    args = parser.parse_args()
    vectorize_and_split(
        args.clean_path,
        args.output_dir,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        test_size=args.test_size,
        random_state=args.random_state,
    ) 