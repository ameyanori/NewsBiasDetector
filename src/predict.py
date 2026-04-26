"""Inference utilities for saved baseline and transformer models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CLASS_NAMES
from src.preprocess import mask_outlet_names_in_text


def load_saved_model(model_path: Path, model_type: str = "baseline") -> Any:
    """Load serialized model based on type."""
    if model_type == "baseline":
        return joblib.load(model_path)
    if model_type == "transformer":
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        return {"tokenizer": tokenizer, "model": model}
    raise ValueError(f"Unsupported model_type: {model_type}")


def predict_text(model: Any, text: str, model_type: str = "baseline") -> str:
    """Predict class label for input text."""
    text = mask_outlet_names_in_text(text)
    if model_type == "baseline":
        return str(model.predict([text])[0])
    probs = predict_proba_text(model, text, model_type=model_type)
    return max(probs, key=probs.get)


def predict_proba_text(model: Any, text: str, model_type: str = "baseline") -> dict[str, float]:
    """Return per-class probabilities."""
    text = mask_outlet_names_in_text(text)
    if model_type == "baseline":
        probs = model.predict_proba([text])[0]
        return {cls: float(p) for cls, p in zip(model.classes_, probs)}

    tokenizer = model["tokenizer"]
    transformer_model = model["model"]
    encoded = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    logits = transformer_model(**encoded).logits.detach().numpy()[0]
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    return {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
