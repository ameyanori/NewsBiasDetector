"""Project configuration constants and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RANDOM_SEED = 42
CLASS_NAMES = ["left", "center", "right"]

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

DEFAULT_TEXT_COLUMN_CANDIDATES = ["text", "article", "content", "body", "article_text"]
DEFAULT_LABEL_COLUMN_CANDIDATES = [
    "label",
    "bias",
    "lean",
    "ideology",
    "bias_rating",
    "allsides_bias",
    "political_bias",
]
DEFAULT_SOURCE_COLUMN_CANDIDATES = ["source", "outlet", "publisher"]
DEFAULT_TITLE_COLUMN_CANDIDATES = ["title", "headline"]

TRANSFORMER_CHECKPOINT = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EPOCHS = 2
DEFAULT_LEARNING_RATE = 2e-5

# Multi-word outlets first. Replaced with [OUTLET] to reduce outlet-name shortcut learning.
# Omit very short tokens (e.g. AP, Vox) to limit false positives in general English.
OUTLET_NAME_REGEXES: Tuple[str, ...] = (
    r"\bOne America News Network\b",
    r"\bOne America News\b",
    r"\bSan Francisco Chronicle\b",
    r"\bAssociated Press\b",
    r"\bLos Angeles Times\b",
    r"\bHuffington Post\b",
    r"\bWall Street Journal\b",
    r"\bThe New York Times\b",
    r"\bNew York Times\b",
    r"\bThe Washington Post\b",
    r"\bWashington Post\b",
    r"\bFox News\b",
    r"\bFox Business\b",
    r"\bNBC News\b",
    r"\bABC News\b",
    r"\bCBS News\b",
    r"\bUSA Today\b",
    r"\bMother Jones\b",
    r"\bThe Daily Caller\b",
    r"\bDaily Caller\b",
    r"\bThe Federalist\b",
    r"\bBreitbart News\b",
    r"\bThe Guardian\b",
    r"\bChicago Tribune\b",
    r"\bPolitico\b",
    r"\bBloomberg News\b",
    r"\bMSNBC\b",
    r"\bReuters\b",
    r"\bBreitbart\b",
    r"\bNewsmax\b",
    r"\bOANN\b",
    r"\bCNN\b",
    r"\bCNBC\b",
    r"\bNPR\b",
    r"\bBBC News\b",
    r"\bBBC\b",
    r"\bAxios\b",
    r"\bNYTimes\b",
    r"\bNYT\b",
    r"\bWSJ\b",
    r"\bWaPo\b",
    r"\bThe Hill\b",
    r"\bThe Atlantic\b",
    r"\bNational Review\b",
    r"\bThe Epoch Times\b",
    r"\bEpoch Times\b",
    r"\bRealClearPolitics\b",
    r"\bFiveThirtyEight\b",
    r"\bThe Intercept\b",
    r"\bJacobin\b",
)

# Post-mask residual check (substring hits should be near zero if masking worked).
OUTLET_RESIDUAL_PATTERN: str = r"\b(cnn|msnbc|fox news|reuters|nyt|wsj|npr|bbc)\b"

LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    "left": "left",
    "lean left": "left",
    "center": "center",
    "centre": "center",
    "lean right": "right",
    "right": "right",
}


@dataclass(frozen=True)
class BaselineTuningConfig:
    """Hyperparameter grid for TF-IDF + Logistic Regression tuning."""

    ngram_ranges: List[tuple] = ((1, 1), (1, 2))
    max_features: List[int] = (20_000, 40_000)
    min_dfs: List[int] = (2, 5)
    c_values: List[float] = (0.5, 1.0, 2.0)


def ensure_directories() -> None:
    """Create all required project directories if missing."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        BASELINE_MODEL_DIR,
        TRANSFORMER_MODEL_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
