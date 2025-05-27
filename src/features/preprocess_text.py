from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure required NLTK resources are downloaded (only the first time).
_RESOURCES = [
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
]
for pkg, path in _RESOURCES:
    try:
        nltk.data.find(path)
    except LookupError:
        logger.info("Downloading NLTK resource '%s'...", pkg)
        nltk.download(pkg)

STOP_WORDS: set[str] = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Pre-compiled regex patterns for speed
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """Clean, tokenize, remove stopwords and lemmatize a single document.

    Steps:
        1. Lower-case.
        2. Remove punctuation, digits & special characters.
        3. Collapse multiple whitespaces.
        4. Remove stop-words.
        5. Lemmatize each remaining token.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        A space-separated string of processed tokens.
    """
    if not isinstance(text, str):
        text = str(text)

    # Lowercase and strip
    text = text.lower().strip()

    # Remove non-alphabetic characters
    text = NON_ALPHA_PATTERN.sub(" ", text)
    # Collapse multi-whitespace
    text = MULTI_SPACE_PATTERN.sub(" ", text)

    tokens = [t for t in text.split() if t not in STOP_WORDS]
    lemmas = [LEMMATIZER.lemmatize(tok) for tok in tokens]

    return " ".join(lemmas)


def preprocess_dataset(df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
    """Apply `preprocess_text` to a DataFrame column.

    Adds a new column `clean_text`.
    """
    logger.info("Preprocessing %d rows (column '%s')...", len(df), text_col)
    df = df.copy()
    df["clean_text"] = df[text_col].astype(str).apply(preprocess_text)
    return df


def run_cli(processed_path: Path, output_path: Path | None = None) -> None:
    """Command-line entry: read parquet, preprocess, write parquet."""
    if output_path is None:
        output_path = processed_path.with_stem(processed_path.stem + "_clean")

    logger.info("Loading processed dataset from %s", processed_path)
    df = pd.read_parquet(processed_path)

    df_clean = preprocess_dataset(df)

    logger.info("Saving cleaned dataset to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    logger.info("Saved cleaned dataset with shape %s", df_clean.shape)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
    )
    import argparse

    parser = argparse.ArgumentParser(description="Text preprocessing for fake news dataset.")
    parser.add_argument(
        "--processed_path",
        type=Path,
        default=Path("workdir/processed_data.parquet"),
        help="Path to parquet file with raw full_text column.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("workdir/processed_data_clean.parquet"),
        help="Destination parquet for cleaned data.",
    )

    args = parser.parse_args()
    run_cli(args.processed_path, args.output_path) 