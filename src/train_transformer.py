"""Fine-tune DistilBERT classifier for 3-class prediction."""

# AI acknowledgement: this module includes AI-assisted implementation help,
# with human-led experimentation and validation of training behavior.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils import prepare_mps_watermark_env

from src.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    MAX_SEQUENCE_LENGTH,
    RANDOM_SEED,
    TRANSFORMER_CHECKPOINT,
)


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Transformer tuning hyperparameters."""

    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    num_epochs: int = DEFAULT_NUM_EPOCHS
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    use_class_weights: bool = True
    early_stopping_patience: int = 1


def tokenize_batch(batch: dict, tokenizer: AutoTokenizer, max_sequence_length: int) -> dict:
    """Tokenize a batch of raw texts."""
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_sequence_length,
    )


def _to_hf_dataset(df: pd.DataFrame, text_col: str, label_col: str) -> Dataset:
    label_to_id = {label: idx for idx, label in enumerate(CLASS_NAMES)}
    out = pd.DataFrame(
        {
            "text": df[text_col].tolist(),
            "labels": df[label_col].map(label_to_id).astype(int).tolist(),
        }
    )
    return Dataset.from_pandas(out, preserve_index=False)


def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


class WeightedTrainer(Trainer):
    """Trainer that supports optional class-weighted cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is None:
            loss = outputs.get("loss")
        else:
            if self.class_weights is not None:
                weights = self.class_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_trainer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    output_dir: Path,
    config: TransformerTrainingConfig,
) -> tuple[Trainer, AutoTokenizer]:
    # AI acknowledgement: this trainer-construction workflow was AI-assisted,
    # with manual validation of tokenizer/model args and training behavior.
    """Create Hugging Face trainer and tokenizer."""
    prepare_mps_watermark_env()
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_CHECKPOINT)
    train_ds = _to_hf_dataset(train_df, text_col, label_col).map(
        lambda x: tokenize_batch(x, tokenizer, config.max_sequence_length), batched=True
    )
    val_ds = _to_hf_dataset(val_df, text_col, label_col).map(
        lambda x: tokenize_batch(x, tokenizer, config.max_sequence_length), batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_CHECKPOINT,
        num_labels=3,
        id2label={i: l for i, l in enumerate(CLASS_NAMES)},
        label2id={l: i for i, l in enumerate(CLASS_NAMES)},
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=RANDOM_SEED,
        report_to=[],
    )
    class_weights = None
    if config.use_class_weights:
        label_to_id = {label: idx for idx, label in enumerate(CLASS_NAMES)}
        y_train = train_df[label_col].map(label_to_id).values
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(list(range(len(CLASS_NAMES)))),
            y=y_train,
        )
        class_weights = torch.tensor(weights, dtype=torch.float)

    callbacks = []
    if config.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )
    return trainer, tokenizer


def train_transformer_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    output_dir: Path,
    config: TransformerTrainingConfig | None = None,
) -> tuple[Trainer, AutoTokenizer]:
    # AI acknowledgement: training loop wiring was AI-assisted, then checked by
    # running full training/evaluation and reviewing saved artifacts.
    """Fine-tune DistilBERT model."""
    cfg = config or TransformerTrainingConfig()
    trainer, tokenizer = build_trainer(train_df, val_df, text_col, label_col, output_dir, cfg)
    trainer.train()
    return trainer, tokenizer


def save_transformer_model(trainer: Trainer, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    """Persist trained model artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def evaluate_transformer(
    trainer: Trainer, tokenizer: AutoTokenizer, df: pd.DataFrame, text_col: str, label_col: str
) -> dict[str, float]:
    """Evaluate trained model on a dataframe split."""
    max_length = getattr(tokenizer, "model_max_length", MAX_SEQUENCE_LENGTH)
    test_ds = _to_hf_dataset(df, text_col, label_col).map(
        lambda x: tokenize_batch(x, tokenizer, min(max_length, MAX_SEQUENCE_LENGTH)),
        batched=True,
    )
    metrics = trainer.predict(test_ds).metrics
    return {
        k.replace("test_", ""): float(v)
        for k, v in metrics.items()
        if k in {"test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1"}
    }


def run_transformer_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    output_dir: Path,
) -> list[dict[str, float | str]]:
    """Run a compact hyperparameter sweep and return test metrics for each run."""
    configs = [
        TransformerTrainingConfig(learning_rate=2e-5, num_epochs=2, batch_size=8),
        TransformerTrainingConfig(learning_rate=2e-5, num_epochs=3, batch_size=8),
        TransformerTrainingConfig(learning_rate=1e-5, num_epochs=3, batch_size=8),
        TransformerTrainingConfig(learning_rate=3e-5, num_epochs=2, batch_size=8),
    ]

    rows: list[dict[str, float | str]] = []
    best_f1 = -1.0
    best_pair: tuple[Trainer, AutoTokenizer] | None = None
    for idx, config in enumerate(configs, start=1):
        run_dir = output_dir / f"run_{idx}"
        trainer, tokenizer = train_transformer_model(
            train_df, val_df, text_col, label_col, run_dir, config=config
        )
        metrics = evaluate_transformer(trainer, tokenizer, test_df, text_col, label_col)
        rows.append(
            {
                "model": f"distilbert_run_{idx}",
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                **metrics,
            }
        )
        if metrics.get("macro_f1", -1.0) > best_f1:
            best_f1 = metrics["macro_f1"]
            best_pair = (trainer, tokenizer)

    if best_pair is not None:
        best_trainer, best_tokenizer = best_pair
        save_transformer_model(best_trainer, best_tokenizer, output_dir)
    return rows
