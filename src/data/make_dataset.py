from pathlib import Path
import logging
from typing import Tuple

import pandas as pd

# Set up basic logging configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_raw_data(true_path: Path, fake_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw True and Fake news datasets.

    Parameters
    ----------
    true_path : Path
        Path to True.csv file.
    fake_path : Path
        Path to Fake.csv file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames for authentic (True) and fake news articles, respectively.
    """
    logger.info("Loading raw datasets...")
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    logger.info("Loaded %d real news rows and %d fake news rows", len(true_df), len(fake_df))
    return true_df, fake_df


def add_label_column(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Add a binary label column to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    label : int
        Label to assign to all rows (0 for real, 1 for fake).

    Returns
    -------
    pd.DataFrame
        DataFrame with an added `label` column.
    """
    df = df.copy()
    df["label"] = label
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning steps on the concatenated DataFrame.

    Steps:
        1. Drop rows with any null values in critical columns (`title`, `text`).
        2. Combine `title` and `text` into a new column `full_text`.

    Parameters
    ----------
    df : pd.DataFrame
        Concatenated DataFrame containing both real and fake news.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a new `full_text` column.
    """
    logger.info("Cleaning dataset...")
    # Drop rows with nulls in title or text
    before_rows = len(df)
    df = df.dropna(subset=["title", "text"]).reset_index(drop=True)
    logger.info("Dropped %d rows containing missing title/text", before_rows - len(df))

    # Combine title and text into single column
    df["full_text"] = df["title"].astype(str).str.strip() + " " + df["text"].astype(str).str.strip()

    return df


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save the processed DataFrame to disk as a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame.
    output_path : Path
        Destination path for the output file. The parent directory will be created if it doesn't exist.
    """
    logger.info("Saving processed data to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved processed data with %d rows.", len(df))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and preprocess fake news dataset.")
    parser.add_argument("--true_path", type=Path, default=Path("data/True.csv"), help="Path to True.csv file")
    parser.add_argument("--fake_path", type=Path, default=Path("data/Fake.csv"), help="Path to Fake.csv file")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("workdir/processed_data.parquet"),
        help="Destination for processed data parquet file",
    )

    args = parser.parse_args()

    true_df, fake_df = load_raw_data(args.true_path, args.fake_path)

    true_df = add_label_column(true_df, 0)
    fake_df = add_label_column(fake_df, 1)

    combined_df = pd.concat([true_df, fake_df], ignore_index=True)

    cleaned_df = basic_cleaning(combined_df)

    save_processed_data(cleaned_df, args.output_path) 