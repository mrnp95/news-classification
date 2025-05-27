from pathlib import Path
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load the parquet file produced by the data processing step."""
    logger.info("Loading processed data from %s", path)
    return pd.read_parquet(path)


def plot_class_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot the distribution of labels (real vs fake)."""
    counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=counts.index.map({0: "Real", 1: "Fake"}), y=counts.values, palette="viridis")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.01 * max(counts.values), f"{v:,}", ha="center")
    plt.tight_layout()
    file_path = output_dir / "class_distribution.png"
    plt.savefig(file_path)
    plt.close()
    logger.info("Saved class distribution plot to %s", file_path)


def plot_text_length_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot histogram of full_text lengths in tokens for each class."""
    df = df.copy()
    df["token_len"] = df["full_text"].str.split().str.len()
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="token_len", hue="label", bins=50, kde=True, palette="viridis", log_scale=(False, True))
    plt.title("Token Length Distribution by Class")
    plt.xlabel("Number of Tokens in full_text")
    plt.ylabel("Frequency (log scale)")
    plt.tight_layout()
    file_path = output_dir / "text_length_distribution.png"
    plt.savefig(file_path)
    plt.close()
    logger.info("Saved text length distribution plot to %s", file_path)


def plot_subject_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot subject/topic counts per class if 'subject' column exists."""
    if "subject" not in df.columns:
        logger.warning("No 'subject' column found, skipping subject distribution plot.")
        return
    plt.figure(figsize=(10, 6))
    # Prepare counts: aggregate by subject & label, then sort by total occurrences
    subject_counts = df.groupby(["subject", "label"]).size().unstack(fill_value=0)
    subject_counts["total"] = subject_counts.sum(axis=1)
    subject_counts = subject_counts.sort_values("total", ascending=False).drop(columns="total")
    subject_counts.rename(columns={0: "Real", 1: "Fake"}, inplace=True)
    subject_counts.plot(kind="bar", stacked=True, colormap="viridis")
    plt.title("Subject Distribution by Class")
    plt.xlabel("Subject")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    file_path = output_dir / "subject_distribution.png"
    plt.savefig(file_path)
    plt.close()
    logger.info("Saved subject distribution plot to %s", file_path)


def generate_eda_report(processed_path: Path, output_dir: Path):
    """Run all EDA analyses and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_processed_data(processed_path)

    # Basic dataset summary saved as CSV
    summary_path = output_dir / "dataset_info.txt"
    with summary_path.open("w") as f:
        f.write("Data shape: " + str(df.shape) + "\n")
        f.write("\nLabel counts:\n")
        f.write(df["label"].value_counts().to_string() + "\n")
        if "subject" in df.columns:
            f.write("\nSubject counts (top 20):\n")
            f.write(df["subject"].value_counts().head(20).to_string() + "\n")
    logger.info("Saved dataset summary to %s", summary_path)

    # Visualizations
    plot_class_distribution(df, output_dir)
    plot_text_length_distribution(df, output_dir)
    plot_subject_distribution(df, output_dir)

    logger.info("EDA completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate EDA for fake news dataset.")
    parser.add_argument(
        "--processed_path",
        type=Path,
        default=Path("workdir/processed_data.parquet"),
        help="Path to processed parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("workdir/eda"),
        help="Directory to save EDA reports/plots",
    )

    args = parser.parse_args()

    generate_eda_report(args.processed_path, args.output_dir) 