"""Dataset loading functions for local files and Hugging Face datasets."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download

from src.config import (
    DEFAULT_LABEL_COLUMN_CANDIDATES,
    DEFAULT_SOURCE_COLUMN_CANDIDATES,
    DEFAULT_TEXT_COLUMN_CANDIDATES,
    DEFAULT_TITLE_COLUMN_CANDIDATES,
)


def _pick_first_existing(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    colset = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in colset:
            return colset[candidate]
    return None


def validate_columns(df: pd.DataFrame) -> tuple[str, str, Optional[str], Optional[str]]:
    """Find text/label columns and optional source/title columns."""
    text_col = _pick_first_existing(df.columns, DEFAULT_TEXT_COLUMN_CANDIDATES)
    label_col = _pick_first_existing(df.columns, DEFAULT_LABEL_COLUMN_CANDIDATES)
    source_col = _pick_first_existing(df.columns, DEFAULT_SOURCE_COLUMN_CANDIDATES)
    title_col = _pick_first_existing(df.columns, DEFAULT_TITLE_COLUMN_CANDIDATES)

    if text_col is None or label_col is None:
        raise ValueError(
            "Dataset must contain text and label columns. "
            f"Found columns: {list(df.columns)}"
        )
    return text_col, label_col, source_col, title_col


def load_csv_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset."""
    return pd.read_csv(path)


def load_json_dataset(path: Path) -> pd.DataFrame:
    """Load a JSON dataset."""
    return pd.read_json(path)


def load_parquet_dataset(path: Path) -> pd.DataFrame:
    """Load a parquet dataset."""
    return pd.read_parquet(path)


def load_huggingface_dataset(dataset_name: str) -> pd.DataFrame:
    """Load Hugging Face dataset and merge splits."""
    ds = load_dataset(dataset_name)
    if isinstance(ds, DatasetDict):
        parts = [split.to_pandas() for split in ds.values()]
        return pd.concat(parts, ignore_index=True)
    return ds.to_pandas()


def _load_allsides_zip_from_hub() -> pd.DataFrame:
    """Load AllSides.zip and infer labels from folder names."""
    archive_path = hf_hub_download(
        repo_id="valurank/PoliticalBias_AllSides_Txt",
        repo_type="dataset",
        filename="AllSides.zip",
    )

    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.lower().endswith(".txt"):
                continue
            lowered = name.lower()
            if "/left/" in lowered or "/left data/" in lowered:
                label = "left"
            elif (
                "/center/" in lowered
                or "/centre/" in lowered
                or "/center data/" in lowered
                or "/centre data/" in lowered
            ):
                label = "center"
            elif "/right/" in lowered or "/right data/" in lowered:
                label = "right"
            else:
                continue
            with zf.open(name) as fp:
                raw = fp.read()
            # Handle mixed encodings found in raw text dumps.
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                rows.append({"text": text, "label": label})

    if not rows:
        raise ValueError("Could not derive labeled rows from AllSides.zip.")
    return pd.DataFrame(rows)


def load_huggingface_dataset_with_fallback(dataset_name: str) -> pd.DataFrame:
    """Load HF dataset and auto-fallback for known text-only AllSides variant."""
    df = load_huggingface_dataset(dataset_name)
    has_text = _pick_first_existing(df.columns, DEFAULT_TEXT_COLUMN_CANDIDATES) is not None
    has_label = _pick_first_existing(df.columns, DEFAULT_LABEL_COLUMN_CANDIDATES) is not None
    if has_text and has_label:
        return df

    if dataset_name == "valurank/PoliticalBias_AllSides_Txt" and has_text and not has_label:
        return _load_allsides_zip_from_hub()

    return df


def load_dataset_auto(path_or_name: str) -> pd.DataFrame:
    """Load dataset from local path or Hugging Face dataset name."""
    path = Path(path_or_name)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return load_csv_dataset(path)
        if suffix == ".json":
            return load_json_dataset(path)
        if suffix in {".parquet", ".pq"}:
            return load_parquet_dataset(path)
        raise ValueError(f"Unsupported file type: {suffix}")
    return load_huggingface_dataset_with_fallback(path_or_name)
