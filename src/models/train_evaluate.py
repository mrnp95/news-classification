from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.base import clone

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception as _xgb_exc:  # pragma: no cover
    # Catch ImportError or library loading errors (e.g., missing libomp on macOS).
    import warnings

    warnings.warn(f"xgboost is not available and will be skipped: {_xgb_exc}")
    XGBClassifier = None  # type: ignore

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

ModelArtifact = Tuple[str, object]  # (model_name, fitted_model)

def load_features(feature_dir: Path) -> Tuple[sparse.spmatrix, sparse.spmatrix, np.ndarray, np.ndarray]:
    """Load train/test sparse matrices and labels."""
    X_train = sparse.load_npz(feature_dir / "X_train.npz")
    X_test = sparse.load_npz(feature_dir / "X_test.npz")
    y_train = np.load(feature_dir / "y_train.npy")
    y_test = np.load(feature_dir / "y_test.npy")
    return X_train, X_test, y_train, y_test


def fit_models(X_train: sparse.spmatrix, y_train: np.ndarray, X_test: sparse.spmatrix | None=None, y_test: np.ndarray | None=None) -> Dict[str, object]:
    """Fit baseline and enhanced models."""
    models: Dict[str, object] = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", C=1.0),
        "LinearSVC": LinearSVC(C=1.0),
    }
    if XGBClassifier is not None:
        models["XGBClassifier"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            verbosity=0,
        )
    else:
        logger.warning("xgboost is not installed; skipping XGBClassifier.")

    fitted: Dict[str, object] = {}
    for name, model in models.items():
        logger.info("Training %s...", name)
        if name == "XGBClassifier" and X_test is not None and y_test is not None:
            model.set_params(eval_metric="logloss")
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )
            model._learning_curve = model.evals_result_  # attach for later plotting
        else:
            model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def evaluate_model(model_name: str, model, X_test: sparse.spmatrix, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a model and return metrics."""
    logger.info("Evaluating %s...", model_name)
    y_pred = model.predict(X_test)

    # Probabilities / scores for ROC AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None  # type: ignore

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_score) if y_score is not None else float("nan")

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }
    logger.info("%s metrics: %s", model_name, metrics)
    return metrics


def plot_confusion(model_name: str, y_test: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    file_path = output_dir / f"{model_name}_confusion.png"
    plt.savefig(file_path)
    plt.close()


def plot_roc(model_name: str, y_test: np.ndarray, y_score: np.ndarray, output_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    file_path = output_dir / f"{model_name}_roc.png"
    plt.savefig(file_path)
    plt.close()


def save_metrics(all_metrics: Dict[str, Dict[str, float]], output_path: Path):
    with output_path.open("w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved metrics to %s", output_path)


def plot_learning_curve_generic(model_name: str, estimator, X, y, output_dir: Path):
    """Plot learning curve (train vs validation F1) for non-boosting models."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=clone(estimator),
        X=X,
        y=y,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="f1",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, label="Train", marker="o")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, label="Validation", marker="o")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel("Training samples")
    plt.ylabel("F1-score")
    plt.title(f"Learning Curve – {model_name}")
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{model_name}_learning_curve.png")
    plt.close()


def train_and_evaluate(feature_dir: Path, model_dir: Path, report_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_features(feature_dir)

    fitted_models = fit_models(X_train, y_train, X_test, y_test)

    agg_metrics: Dict[str, Dict[str, float]] = {}

    for name, model in fitted_models.items():
        # Save model
        joblib.dump(model, model_dir / f"{name}.joblib")

        # Metrics
        metrics = evaluate_model(name, model, X_test, y_test)
        agg_metrics[name] = metrics

        # Confusion matrix
        y_pred = model.predict(X_test)
        plot_confusion(name, y_test, y_pred, report_dir)

        # Learning curve for XGB
        if name == "XGBClassifier" and hasattr(model, "_learning_curve"):
            lc = model._learning_curve["validation_0"]["logloss"]
            lc_val = model._learning_curve["validation_1"]["logloss"]
            epochs = range(1, len(lc) + 1)
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, lc, label="Train")
            plt.plot(epochs, lc_val, label="Validation")
            plt.xlabel("Boosting round")
            plt.ylabel("Logloss")
            plt.title("XGBoost Learning Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(report_dir / "XGB_learning_curve.png")
            plt.close()

        # ROC curve (only if we have scores)
        if metrics["roc_auc"] == metrics["roc_auc"]:  # not nan
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = None
            if y_score is not None:
                plot_roc(name, y_test, y_score, report_dir)

        # After evaluation, plot learning curve for classical models
        if name != "XGBClassifier":
            plot_learning_curve_generic(name, model, X_train, y_train, report_dir)

    # Bar plot of F1 scores
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(agg_metrics.keys()), y=[m["f1"] for m in agg_metrics.values()], palette="viridis")
    plt.ylabel("F1 Score")
    plt.title("Model F1 Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(report_dir / "model_f1_comparison.png")
    plt.close()

    save_metrics(agg_metrics, report_dir / "metrics.json")
    logger.info("Training & evaluation completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate models for fake news detection.")
    parser.add_argument(
        "--feature_dir",
        type=Path,
        default=Path("workdir/features"),
        help="Directory containing vectorizer and train/test files.",
    )
    parser.add_argument(
        "--model_dir", type=Path, default=Path("workdir/models"), help="Directory to save fitted models"
    )
    parser.add_argument(
        "--report_dir", type=Path, default=Path("workdir/reports"), help="Directory to save evaluation plots & metrics"
    )

    args = parser.parse_args()

    train_and_evaluate(args.feature_dir, args.model_dir, args.report_dir) 