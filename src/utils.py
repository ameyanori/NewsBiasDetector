"""General utility helpers for IO and reproducibility."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


# PyTorch MPS allocator defaults (unified memory); see torch docs "MPS Environment Variables".
_DEFAULT_MPS_HIGH_WATERMARK = 1.7
_DEFAULT_MPS_LOW_WATERMARK = 1.4


def prepare_mps_watermark_env() -> None:
    """Validate MPS watermark env vars and fix the common HIGH/LOW mismatch before training.

    If only ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` is lowered (e.g. to 0.7), PyTorch still uses its
    default *low* watermark (1.4). That makes low > high and triggers
    ``RuntimeError: invalid low watermark ratio 1.4``. When HIGH is set below that default and LOW
    is unset, we set LOW to match HIGH so low <= high.
    """
    high_raw = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    low_raw = os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")

    high: float | None = None
    low: float | None = None
    if high_raw not in (None, ""):
        try:
            high = float(high_raw)
        except ValueError as exc:
            raise ValueError(
                f"PYTORCH_MPS_HIGH_WATERMARK_RATIO must be a number; got {high_raw!r}"
            ) from exc
        if high < 0:
            raise ValueError(
                f"PYTORCH_MPS_HIGH_WATERMARK_RATIO must be >= 0 (got {high}). "
                "See PyTorch docs: 0.0 disables the high watermark limit."
            )

    if low_raw not in (None, ""):
        try:
            low = float(low_raw)
        except ValueError as exc:
            raise ValueError(
                f"PYTORCH_MPS_LOW_WATERMARK_RATIO must be a number; got {low_raw!r}"
            ) from exc
        if low < 0:
            raise ValueError(f"PYTORCH_MPS_LOW_WATERMARK_RATIO must be >= 0 (got {low}).")

    if high is not None and low is not None and low > high:
        raise ValueError(
            f"PYTORCH_MPS_LOW_WATERMARK_RATIO ({low}) must be <= "
            f"PYTORCH_MPS_HIGH_WATERMARK_RATIO ({high}). "
            f"PyTorch defaults are high={_DEFAULT_MPS_HIGH_WATERMARK}, low={_DEFAULT_MPS_LOW_WATERMARK}; "
            "if you only lower HIGH, also set LOW to at most HIGH (the training code can set this for you)."
        )

    if high is None and low is not None and low > _DEFAULT_MPS_HIGH_WATERMARK:
        raise ValueError(
            f"PYTORCH_MPS_LOW_WATERMARK_RATIO ({low}) is above PyTorch's default high watermark "
            f"({_DEFAULT_MPS_HIGH_WATERMARK}). Set PYTORCH_MPS_HIGH_WATERMARK_RATIO >= LOW, or lower LOW."
        )

    if high is not None and low is None and high < _DEFAULT_MPS_LOW_WATERMARK:
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(high)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
