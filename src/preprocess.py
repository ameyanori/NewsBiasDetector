"""Text preprocessing, label normalization, leakage masking, and dataset splitting."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_NAMES,
    LABEL_NORMALIZATION_MAP,
    OUTLET_NAME_REGEXES,
    OUTLET_RESIDUAL_PATTERN,
    RANDOM_SEED,
)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip noisy URL artifacts."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Map raw labels to left/center/right and drop unknown labels."""
    out = df.copy()
    out[label_col] = out[label_col].astype(str).str.lower().str.strip()
    out[label_col] = out[label_col].map(LABEL_NORMALIZATION_MAP)
    out = out[out[label_col].isin(CLASS_NAMES)].copy()
    return out


def drop_missing_and_duplicates(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Remove rows with missing fields and duplicate text."""
    out = df.dropna(subset=[text_col, label_col]).copy()
    out[text_col] = out[text_col].astype(str).map(clean_text)
    out = out[out[text_col].str.len() > 0].copy()
    out = out.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    return out


def mask_outlet_names_in_text(text: str) -> str:
    """Replace common U.S. news outlet names with a neutral placeholder."""
    s = str(text)
    for pattern in OUTLET_NAME_REGEXES:
        s = re.sub(pattern, " [OUTLET] ", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:\[OUTLET\]\s*){2,}", "[OUTLET] ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def mask_outlet_names(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Mask outlet names in a text column."""
    out = df.copy()
    out[text_col] = out[text_col].astype(str).map(mask_outlet_names_in_text)
    return out


def mask_source_names_if_needed(
    df: pd.DataFrame, text_col: str, source_col: Optional[str] = None
) -> pd.DataFrame:
    """Mask source names in text to reduce direct leakage from outlet names."""
    out = df.copy()
    if not source_col or source_col not in out.columns:
        return out

    source_names = (
        out[source_col]
        .dropna()
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.split()
        .explode()
        .value_counts()
    )
    common_source_tokens = set(source_names[source_names > 30].index.tolist())
    if not common_source_tokens:
        return out

    pattern = r"\b(" + "|".join(re.escape(token) for token in sorted(common_source_tokens)) + r")\b"
    out[text_col] = out[text_col].str.replace(pattern, " [SOURCE] ", regex=True, case=False)
    out[text_col] = out[text_col].str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def split_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test split."""
    if abs((val_size + test_size) - 0.30) > 0.20:
        # Guardrail for odd split values; function still works for any valid fractions.
        pass

    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), stratify=df[label_col], random_state=RANDOM_SEED
    )
    rel_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test_size, stratify=temp_df[label_col], random_state=RANDOM_SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def leakage_report(
    df: pd.DataFrame, text_col: str, label_col: str, source_col: Optional[str] = None
) -> pd.DataFrame:
    """Simple leakage checks for label tokens and source mentions in text."""
    rows: list[dict[str, object]] = []
    lower_text = df[text_col].astype(str).str.lower()
    for label in ["left", "center", "right"]:
        rows.append(
            {
                "check": f"label_token_{label}",
                "matches": int(lower_text.str.contains(fr"\b{label}\b", regex=True).sum()),
                "fraction": float(lower_text.str.contains(fr"\b{label}\b", regex=True).mean()),
            }
        )
    if source_col and source_col in df.columns:
        source_strings = df[source_col].dropna().astype(str).str.lower().unique().tolist()[:50]
        source_hits = 0
        for src in source_strings:
            src = src.strip()
            if len(src) < 3:
                continue
            source_hits += int(lower_text.str.contains(re.escape(src), regex=True).sum())
        rows.append(
            {
                "check": "source_name_mentions_top50",
                "matches": int(source_hits),
                "fraction": float(source_hits / max(len(df), 1)),
            }
        )
    residual = lower_text.str.contains(OUTLET_RESIDUAL_PATTERN, regex=True)
    rows.append(
        {
            "check": "outlet_residual_tokens_after_pipeline",
            "matches": int(residual.sum()),
            "fraction": float(residual.mean()),
        }
    )
    return pd.DataFrame(rows)
