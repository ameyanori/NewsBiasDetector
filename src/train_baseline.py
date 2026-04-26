"""Train and tune TF-IDF + Logistic Regression baseline."""

# AI acknowledgement: portions of this training module were AI-assisted,
# then reviewed and adapted by the project author.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline  # kept for compatibility in type hints

from src.config import BaselineTuningConfig, CLASS_NAMES
from src.evaluate import compute_classification_metrics
from src.features import build_tfidf_vectorizer, extract_stance_features_from_texts


@dataclass
class HybridBaselineModel:
    """Hybrid baseline using TF-IDF text + handcrafted stance features."""

    tfidf_vectorizer: Any
    clf: LogisticRegression
    use_stance_features: bool = True
    stance_feature_names: list[str] | None = None
    feature_names_: list[str] | None = None
    classes_: np.ndarray | None = None

    def _stance_matrix(self, texts: pd.Series) -> sp.csr_matrix:
        stance_df = extract_stance_features_from_texts(texts)
        if self.stance_feature_names is None:
            self.stance_feature_names = stance_df.columns.tolist()
        stance_df = stance_df.reindex(columns=self.stance_feature_names, fill_value=0.0)
        return sp.csr_matrix(stance_df.values.astype(float))

    def fit(self, texts: pd.Series, labels: pd.Series) -> "HybridBaselineModel":
        x_text = self.tfidf_vectorizer.fit_transform(texts)
        if self.use_stance_features:
            x_stance = self._stance_matrix(texts)
            x = sp.hstack([x_text, x_stance], format="csr")
            self.feature_names_ = list(self.tfidf_vectorizer.get_feature_names_out()) + [
                f"STANCE:{n}" for n in (self.stance_feature_names or [])
            ]
        else:
            x = x_text
            self.feature_names_ = list(self.tfidf_vectorizer.get_feature_names_out())
        self.clf.fit(x, labels)
        self.classes_ = self.clf.classes_
        return self

    def transform_texts(self, texts: pd.Series | list[str]) -> sp.csr_matrix:
        if not isinstance(texts, pd.Series):
            texts = pd.Series(texts)
        x_text = self.tfidf_vectorizer.transform(texts)
        if self.use_stance_features:
            x_stance = self._stance_matrix(texts)
            return sp.hstack([x_text, x_stance], format="csr")
        return x_text

    def predict(self, texts: pd.Series | list[str]) -> np.ndarray:
        return self.clf.predict(self.transform_texts(texts))

    def predict_proba(self, texts: pd.Series | list[str]) -> np.ndarray:
        return self.clf.predict_proba(self.transform_texts(texts))


def train_logistic_regression(
    train_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    max_features: int = 40_000,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
    c_value: float = 1.0,
) -> HybridBaselineModel:
    # AI acknowledgement: baseline training setup was AI-assisted and then
    # manually tuned/verified against validation and test metrics.
    """Train baseline pipeline."""
    vectorizer = build_tfidf_vectorizer(max_features=max_features, min_df=min_df, ngram_range=ngram_range)
    clf = LogisticRegression(C=c_value, max_iter=2_000, class_weight="balanced", random_state=42)
    model = HybridBaselineModel(tfidf_vectorizer=vectorizer, clf=clf, use_stance_features=True)
    model.fit(train_df[text_col], train_df[label_col])
    return model


def tune_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    tuning: BaselineTuningConfig,
) -> tuple[HybridBaselineModel, dict[str, Any], dict[str, float]]:
    """Simple grid search on validation macro F1."""
    best_pipe: HybridBaselineModel | None = None
    best_cfg: dict[str, Any] = {}
    best_metrics = {"macro_f1": -1.0}

    for ngram in tuning.ngram_ranges:
        for max_features in tuning.max_features:
            for min_df in tuning.min_dfs:
                for c_value in tuning.c_values:
                    pipe = train_logistic_regression(
                        train_df, text_col, label_col, max_features, min_df, ngram, c_value
                    )
                    preds = pipe.predict(val_df[text_col])
                    metrics = compute_classification_metrics(val_df[label_col], preds, CLASS_NAMES)
                    if metrics["macro_f1"] > best_metrics["macro_f1"]:
                        best_pipe = pipe
                        best_metrics = metrics
                        best_cfg = {
                            "ngram_range": ngram,
                            "max_features": max_features,
                            "min_df": min_df,
                            "C": c_value,
                            "use_stance_features": True,
                        }
    if best_pipe is None:
        raise RuntimeError("Baseline tuning failed to produce a model.")
    return best_pipe, best_cfg, best_metrics


def save_model(model: HybridBaselineModel, output_path: Path) -> None:
    """Serialize trained baseline pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
