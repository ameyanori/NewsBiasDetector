"""Explainability helpers for baseline and optional LIME analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
import re

from src.config import CLASS_NAMES


def _extract_components(model: Any) -> tuple[Any, Any, list[str]]:
    """Support sklearn Pipeline and custom hybrid baseline model."""
    if hasattr(model, "named_steps"):
        vectorizer = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        feature_names = vectorizer.get_feature_names_out().tolist()
        return vectorizer, clf, feature_names
    vectorizer = getattr(model, "tfidf_vectorizer")
    clf = getattr(model, "clf")
    feature_names = getattr(model, "feature_names_", vectorizer.get_feature_names_out().tolist())
    return vectorizer, clf, feature_names


def top_weighted_terms_by_class(model: Pipeline, top_k: int = 20) -> pd.DataFrame:
    """Extract top TF-IDF coefficient terms for each class."""
    _vectorizer, clf, feature_names = _extract_components(model)
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(clf.classes_):
        coef = clf.coef_[class_index]
        top_idx = coef.argsort()[-top_k:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "term": feature_names[idx],
                    "weight": float(coef[idx]),
                }
            )
    return pd.DataFrame(rows)


def explain_single_baseline_prediction(model: Pipeline, text: str, top_k: int = 10) -> dict[str, Any]:
    """Return class prediction and top contributing sentences."""
    _vectorizer, clf, feature_names = _extract_components(model)
    if hasattr(model, "transform_texts"):
        x = model.transform_texts([text])
    else:
        x = _vectorizer.transform([text])
    probs = model.predict_proba([text])[0]
    pred_idx = probs.argmax()
    pred_label = model.classes_[pred_idx]

    contributions = (x.toarray()[0] * clf.coef_[pred_idx]).tolist()
    top_idx = sorted(range(len(contributions)), key=lambda i: contributions[i], reverse=True)[:top_k]
    top_terms = [
        {"term": feature_names[i], "contribution": float(contributions[i])}
        for i in top_idx
        if contributions[i] > 0
    ]

    # Sentence-level explanation: score each sentence with the predicted-class linear margin.
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sentence_rows: list[dict[str, Any]] = []
    for sentence in raw_sentences:
        if hasattr(model, "transform_texts"):
            x_sent = model.transform_texts([sentence])
        else:
            x_sent = _vectorizer.transform([sentence])
        sentence_score = float(x_sent.toarray()[0] @ clf.coef_[pred_idx])
        if sentence_score > 0:
            sentence_rows.append({"sentence": sentence, "contribution": sentence_score})
    sentence_rows = sorted(sentence_rows, key=lambda r: r["contribution"], reverse=True)[:top_k]

    return {
        "prediction": pred_label,
        "probabilities": {cls: float(p) for cls, p in zip(model.classes_, probs)},
        "top_sentences": sentence_rows,
        "top_terms": top_terms,
    }


def run_lime_explanation(model: Pipeline, text: str, top_k: int = 10) -> dict[str, Any]:
    """Run optional LIME explanation; fallback to coefficient explanation."""
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception:
        return explain_single_baseline_prediction(model, text, top_k=top_k)

    explainer = LimeTextExplainer(class_names=list(CLASS_NAMES))
    exp = explainer.explain_instance(text, model.predict_proba, num_features=top_k)
    return {"lime_explanation": exp.as_list()}


def save_explanation_examples(explanations: list[dict[str, Any]], out_path: Path) -> None:
    """Save explanation dictionaries as JSON lines."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(explanations).to_json(out_path, orient="records", lines=True)
