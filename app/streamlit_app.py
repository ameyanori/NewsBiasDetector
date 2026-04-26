"""Streamlit app for ideological lean prediction."""

# AI acknowledgement: app scaffolding and iteration were AI-assisted,
# with human review of UX text and prediction behavior.

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.explain import explain_single_baseline_prediction
from src.predict import load_saved_model
from src.preprocess import mask_outlet_names_in_text


st.set_page_config(page_title="News Bias Detector", page_icon="📰", layout="centered")
st.title("Detecting Ideological Lean in News Text")
st.caption(
    "Predictions reflect outlet-rated ideological lean patterns in language. "
    "They do not determine objective truth or absolute bias."
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline" / "baseline.joblib"

if not MODEL_PATH.exists():
    st.warning("Baseline model not found. Train first with `python main.py --skip-transformer`.")
else:
    model = load_saved_model(MODEL_PATH, model_type="baseline")
    user_text = st.text_area("Paste article text", height=220)
    if st.button("Predict") and user_text.strip():
        masked_text = mask_outlet_names_in_text(user_text)
        explanation = explain_single_baseline_prediction(model, masked_text, top_k=10)
        st.subheader(f"Predicted lean: {explanation['prediction']}")
        st.write("Confidence scores")
        st.json(explanation["probabilities"])
        st.write("Top contributing words")
        st.dataframe(explanation["top_terms"], use_container_width=True)
