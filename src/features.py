"""Feature engineering and EDA helpers."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

STANCE_ENTITIES = {
    "trump": ("trump", "donald trump", "president trump"),
    "biden": ("biden", "joe biden", "president biden"),
    "democrat": ("democrat", "democrats", "democratic"),
    "republican": ("republican", "republicans", "gop"),
    "iran": ("iran", "iranian"),
}

POSITIVE_WORDS = {
    "strong",
    "success",
    "successful",
    "improve",
    "improved",
    "growth",
    "benefit",
    "safe",
    "peace",
    "win",
    "wins",
    "winning",
}

NEGATIVE_WORDS = {
    "weak",
    "fail",
    "failed",
    "failing",
    "chaos",
    "crisis",
    "corrupt",
    "bad",
    "worse",
    "worst",
    "terror",
    "terrorism",
}


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_stance_features_for_text(text: str, window_size: int = 8) -> dict[str, float]:
    """Extract entity-centric sentiment and framing features from one text."""
    tokens = _tokenize_words(text)
    token_count = max(len(tokens), 1)
    features: dict[str, float] = {
        "text_len_tokens": float(token_count),
        "sent_pos_count": float(sum(1 for t in tokens if t in POSITIVE_WORDS)),
        "sent_neg_count": float(sum(1 for t in tokens if t in NEGATIVE_WORDS)),
        "sent_pos_minus_neg": 0.0,
    }
    features["sent_pos_minus_neg"] = features["sent_pos_count"] - features["sent_neg_count"]

    joined = " ".join(tokens)
    for entity, aliases in STANCE_ENTITIES.items():
        mentions = 0
        local_pos = 0
        local_neg = 0
        for alias in aliases:
            alias_tokens = alias.split()
            if not alias_tokens:
                continue
            if len(alias_tokens) == 1:
                idxs = [i for i, tok in enumerate(tokens) if tok == alias_tokens[0]]
            else:
                idxs = [
                    i
                    for i in range(len(tokens) - len(alias_tokens) + 1)
                    if tokens[i : i + len(alias_tokens)] == alias_tokens
                ]
            for idx in idxs:
                mentions += 1
                left = max(0, idx - window_size)
                right = min(len(tokens), idx + len(alias_tokens) + window_size)
                window = tokens[left:right]
                local_pos += sum(1 for w in window if w in POSITIVE_WORDS)
                local_neg += sum(1 for w in window if w in NEGATIVE_WORDS)
        features[f"mention_{entity}"] = float(mentions)
        features[f"local_pos_{entity}"] = float(local_pos)
        features[f"local_neg_{entity}"] = float(local_neg)
        features[f"local_pos_minus_neg_{entity}"] = float(local_pos - local_neg)

    # Coarse sentence-level polarity share.
    sentence_splits = re.split(r"[.!?]+", text)
    sentence_scores = []
    for sentence in sentence_splits:
        stoks = _tokenize_words(sentence)
        if not stoks:
            continue
        spos = sum(1 for t in stoks if t in POSITIVE_WORDS)
        sneg = sum(1 for t in stoks if t in NEGATIVE_WORDS)
        sentence_scores.append(spos - sneg)
    if sentence_scores:
        features["sentence_polarity_mean"] = float(np.mean(sentence_scores))
        features["sentence_polarity_std"] = float(np.std(sentence_scores))
    else:
        features["sentence_polarity_mean"] = 0.0
        features["sentence_polarity_std"] = 0.0

    # Presence of explicit ideological cue terms.
    lower = joined
    features["cue_conservative_count"] = float(
        sum(lower.count(w) for w in ["america first", "deep state", "patriot", "border crisis"])
    )
    features["cue_progressive_count"] = float(
        sum(lower.count(w) for w in ["systemic racism", "climate justice", "equity", "inclusion"])
    )
    return features


def extract_stance_features_from_texts(texts: pd.Series) -> pd.DataFrame:
    """Extract stance feature dataframe from a text series."""
    rows = [extract_stance_features_for_text(str(text)) for text in texts.tolist()]
    df = pd.DataFrame(rows).fillna(0.0)
    return df


def build_tfidf_vectorizer(
    max_features: int = 40_000, min_df: int = 2, ngram_range: tuple[int, int] = (1, 2)
) -> TfidfVectorizer:
    """Construct a TF-IDF vectorizer with common defaults."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
    )


def extract_article_length(df: pd.DataFrame, text_col: str) -> pd.Series:
    """Compute article lengths by token count."""
    return df[text_col].astype(str).str.split().str.len()


def get_text_stats(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """Return basic text length statistics grouped by class."""
    stats_df = df.copy()
    stats_df["article_length"] = extract_article_length(stats_df, text_col)
    return stats_df.groupby(label_col)["article_length"].describe()


def plot_class_distribution(df: pd.DataFrame, label_col: str, out_path: Path) -> None:
    """Save class distribution bar chart."""
    counts = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_article_length_distribution(
    df: pd.DataFrame, text_col: str, label_col: str, out_path: Path
) -> None:
    """Save article length distribution per class."""
    tmp = df.copy()
    tmp["article_length"] = extract_article_length(tmp, text_col)
    plt.figure(figsize=(10, 5))
    for label in sorted(tmp[label_col].unique()):
        tmp.loc[tmp[label_col] == label, "article_length"].plot(kind="kde", label=label)
    plt.title("Article Length Distribution by Class")
    plt.xlabel("Token Count")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
