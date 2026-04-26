"""Evaluation utilities for baseline and transformer models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.utils import save_json


def compute_classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]
) -> dict[str, float]:
    """Compute core classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def plot_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: list[str], out_path: Path
) -> None:
    """Save a confusion matrix figure."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_metrics_json(metrics: dict[str, float], out_path: Path) -> None:
    """Save metrics dictionary."""
    save_json(metrics, out_path)


def compare_models_table(rows: list[dict[str, float | str]], out_path: Path) -> pd.DataFrame:
    """Save model comparison table to CSV."""
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
