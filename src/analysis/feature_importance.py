from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_vectorizer(vectorizer_path: Path):
    logger.info("Loading vectorizer from %s", vectorizer_path)
    return joblib.load(vectorizer_path)


def load_model(model_path: Path):
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Return a DataFrame with feature and weight, handling various model types."""
    if hasattr(model, "coef_"):
        # Linear models (LogisticRegression, LinearSVC)
        coef = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        # MultinomialNB: difference of log probabilities between classes
        log_prob = model.feature_log_prob_
        coef = log_prob[1] - log_prob[0]
    elif hasattr(model, "feature_importances_"):
        # Tree-based / XGBoost
        coef = model.feature_importances_
    else:
        raise ValueError("Model type not supported for feature importance extraction.")

    df_imp = pd.DataFrame({"feature": feature_names, "importance": coef})
    return df_imp


def plot_top_features(df_imp: pd.DataFrame, model_name: str, output_dir: Path, top_n: int = 20):
    """Plot top positive and negative features for linear models; just top for others."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # For linear models with signed coefficients, show both directions
    if df_imp["importance"].min() < 0:
        df_sorted_pos = df_imp.nlargest(top_n, "importance")
        df_sorted_neg = df_imp.nsmallest(top_n, "importance")

        plt.figure(figsize=(8, 6))
        sns.barplot(x="importance", y="feature", data=pd.concat([df_sorted_neg, df_sorted_pos]),
                    palette=["#d62728" if v < 0 else "#1f77b4" for v in pd.concat([df_sorted_neg, df_sorted_pos])["importance"]])
        plt.title(f"Top ±{top_n} Features – {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_top_features.png")
        plt.close()
    else:
        # Non-signed importances (NB ratio, XGB importance)
        df_sorted = df_imp.nlargest(top_n, "importance")
        plt.figure(figsize=(8, 6))
        sns.barplot(x="importance", y="feature", data=df_sorted, palette="viridis")
        plt.title(f"Top {top_n} Features – {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_top_features.png")
        plt.close()

    # Save raw importance values
    df_imp.sort_values("importance", ascending=False).to_csv(output_dir / f"{model_name}_feature_importance.csv", index=False)
    logger.info("Saved feature importance for %s", model_name)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def run_feature_importance(vectorizer_path: Path, model_dir: Path, output_dir: Path, models: List[str] | None = None, top_n: int = 20):
    vectorizer = load_vectorizer(vectorizer_path)
    feature_names: List[str] = vectorizer.get_feature_names_out()

    if models is None:
        # autodetect .joblib files in model_dir
        models = [p.stem for p in model_dir.glob("*.joblib")]

    for model_name in models:
        model_path = model_dir / f"{model_name}.joblib"
        if not model_path.exists():
            logger.warning("Model file %s not found, skipping.", model_path)
            continue

        model = load_model(model_path)

        try:
            df_imp = get_feature_importance(model, feature_names)
        except ValueError as e:
            logger.warning("%s – %s", model_name, e)
            continue

        plot_top_features(df_imp, model_name, output_dir, top_n=top_n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate feature importance visualizations.")
    parser.add_argument(
        "--vectorizer_path",
        type=Path,
        default=Path("workdir/features/vectorizer.joblib"),
        help="Path to fitted TfidfVectorizer joblib file.",
    )
    parser.add_argument(
        "--model_dir", type=Path, default=Path("workdir/models"), help="Directory containing model joblib files."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("workdir/reports/importance"), help="Directory to save feature importance plots."
    )
    parser.add_argument("--top_n", type=int, default=20, help="Number of top features per direction to visualize.")
    parser.add_argument(
        "--models", nargs="*", default=None, help="Specific model names (stems) to analyze. If omitted, analyze all."
    )

    args = parser.parse_args()

    run_feature_importance(args.vectorizer_path, args.model_dir, args.output_dir, args.models, args.top_n) 