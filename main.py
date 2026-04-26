"""Main training pipeline for news bias detection."""

# AI acknowledgement: this file was developed with AI-assisted drafting/debugging,
# with human review and validation of design decisions and outputs.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    BASELINE_MODEL_DIR,
    CLASS_NAMES,
    FIGURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TRANSFORMER_MODEL_DIR,
    BaselineTuningConfig,
    ensure_directories,
)
from src.data_loader import load_dataset_auto, validate_columns
from src.evaluate import (
    compare_models_table,
    compute_classification_metrics,
    plot_confusion_matrix,
    save_metrics_json,
)
from src.explain import save_explanation_examples, top_weighted_terms_by_class
from src.features import (
    get_text_stats,
    plot_article_length_distribution,
    plot_class_distribution,
)
from src.preprocess import (
    drop_missing_and_duplicates,
    leakage_report,
    mask_outlet_names,
    mask_source_names_if_needed,
    normalize_labels,
    split_dataset,
)
from src.train_baseline import save_model, tune_baseline
from src.train_transformer import (
    TransformerTrainingConfig,
    evaluate_transformer,
    run_transformer_experiments,
    save_transformer_model,
    train_transformer_model,
)
from src.utils import save_json, set_seed


def _save_error_analysis(test_df: pd.DataFrame, pred_col: str, out_path: Path) -> None:
    wrong = test_df[test_df["label"] != test_df[pred_col]].copy()
    wrong.to_csv(out_path, index=False)


def _try_load_saved_metrics(path: Path) -> Optional[dict[str, float]]:
    if not path.exists():
        return None
    try:
        import json

        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return {k: float(v) for k, v in payload.items()}
    except Exception:
        return None


def _phase1_baseline_train_eval(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[Pipeline, dict[str, float]]:
    # AI acknowledgement: function scaffold was AI-assisted, then validated by
    # running end-to-end baseline training and checking saved artifacts.
    """Phase 1: baseline train/eval with tuned configuration."""
    best_baseline, best_cfg, val_metrics = tune_baseline(
        train_df, val_df, "text", "label", BaselineTuningConfig()
    )
    baseline_test_preds = best_baseline.predict(test_df["text"])
    baseline_test_metrics = compute_classification_metrics(test_df["label"], baseline_test_preds, CLASS_NAMES)

    save_model(best_baseline, BASELINE_MODEL_DIR / "baseline.joblib")
    save_metrics_json(val_metrics, METRICS_DIR / "baseline_val_metrics.json")
    save_metrics_json(baseline_test_metrics, METRICS_DIR / "baseline_test_metrics.json")
    save_json(best_cfg, METRICS_DIR / "baseline_best_params.json")
    plot_confusion_matrix(
        test_df["label"], baseline_test_preds, CLASS_NAMES, FIGURES_DIR / "baseline_confusion_matrix.png"
    )
    test_with_pred = test_df.copy()
    test_with_pred["baseline_pred"] = baseline_test_preds
    test_with_pred.to_csv(PREDICTIONS_DIR / "baseline_test_predictions.csv", index=False)
    return best_baseline, baseline_test_metrics


def _phase2_eda_and_leakage(df: pd.DataFrame, source_col: Optional[str]) -> None:
    """Phase 2: EDA and leakage checks."""
    plot_class_distribution(df, "label", FIGURES_DIR / "class_distribution.png")
    plot_article_length_distribution(df, "text", "label", FIGURES_DIR / "article_length_by_class.png")
    get_text_stats(df, "text", "label").to_csv(REPORTS_DIR / "text_stats_by_class.csv")
    leak = leakage_report(df, "text", "label", source_col)
    leak.to_csv(REPORTS_DIR / "leakage_report.csv", index=False)


def _phase3_baseline_analysis(best_baseline: Pipeline, test_df: pd.DataFrame) -> None:
    """Phase 3: tuned baseline feature analysis and report artifacts."""
    top_features = top_weighted_terms_by_class(best_baseline, top_k=20)
    top_features.to_csv(REPORTS_DIR / "baseline_top_features.csv", index=False)
    with (REPORTS_DIR / "baseline_top_features.md").open("w", encoding="utf-8") as fp:
        fp.write("# Top Weighted Terms by Class\n\n")
        for cls in CLASS_NAMES:
            fp.write(f"## {cls}\n\n")
            for _, row in top_features[top_features["class"] == cls].iterrows():
                fp.write(f"- {row['term']}: {row['weight']:.4f}\n")
            fp.write("\n")

    explanation_examples = []
    for _, row in test_df.head(5).iterrows():
        explanation_examples.append(
            {
                "text": row["text"][:500],
                "true_label": row["label"],
                "predicted_label": best_baseline.predict([row["text"]])[0],
            }
        )
    save_explanation_examples(explanation_examples, REPORTS_DIR / "baseline_example_explanations.jsonl")


def _phase4_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TransformerTrainingConfig,
    run_sweep: bool,
) -> dict[str, float]:
    # AI acknowledgement: transformer orchestration was drafted with AI help
    # and refined via manual hyperparameter experiments and metric checks.
    """Phase 4: DistilBERT fine-tuning and test evaluation."""
    if run_sweep:
        sweep_rows = run_transformer_experiments(
            train_df, val_df, test_df, "text", "label", TRANSFORMER_MODEL_DIR
        )
        compare_models_table(sweep_rows, METRICS_DIR / "transformer_sweep_results.csv")
        best = max(sweep_rows, key=lambda row: float(row.get("macro_f1", -1.0)))
        best_metrics = {
            "accuracy": float(best["accuracy"]),
            "macro_precision": float(best["macro_precision"]),
            "macro_recall": float(best["macro_recall"]),
            "macro_f1": float(best["macro_f1"]),
        }
        save_metrics_json(best_metrics, METRICS_DIR / "transformer_test_metrics.json")
        return best_metrics

    trainer, tokenizer = train_transformer_model(
        train_df, val_df, "text", "label", TRANSFORMER_MODEL_DIR, config=config
    )
    save_transformer_model(trainer, tokenizer, TRANSFORMER_MODEL_DIR)
    transformer_metrics = evaluate_transformer(trainer, tokenizer, test_df, "text", "label")
    save_metrics_json(transformer_metrics, METRICS_DIR / "transformer_test_metrics.json")
    return transformer_metrics


def _phase5_error_analysis(test_df: pd.DataFrame) -> None:
    """Phase 5: save common baseline misclassifications."""
    pred_path = PREDICTIONS_DIR / "baseline_test_predictions.csv"
    if not pred_path.exists():
        return
    pred_df = pd.read_csv(pred_path)
    if "baseline_pred" in pred_df.columns:
        _save_error_analysis(
            pred_df.rename(columns={"baseline_pred": "pred"}),
            "pred",
            REPORTS_DIR / "baseline_errors.csv",
        )


def run_pipeline(
    dataset_path_or_name: str,
    skip_transformer: bool = False,
    run_phase: str = "all",
    transformer_config: TransformerTrainingConfig | None = None,
    run_transformer_sweep: bool = False,
    mask_outlets: bool = True,
) -> None:
    ensure_directories()
    set_seed(42)

    raw_df = load_dataset_auto(dataset_path_or_name)
    text_col, label_col, source_col, _ = validate_columns(raw_df)

    df = raw_df.rename(columns={text_col: "text", label_col: "label"}).copy()
    if source_col:
        df = df.rename(columns={source_col: "source"})
        source_col = "source"

    df = normalize_labels(df, "label")
    df = drop_missing_and_duplicates(df, "text", "label")
    df = mask_source_names_if_needed(df, "text", source_col)
    if mask_outlets:
        df = mask_outlet_names(df, "text")

    df.to_csv(PROCESSED_DATA_DIR / "cleaned_dataset.csv", index=False)

    train_df, val_df, test_df = split_dataset(df, "text", "label")
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    best_baseline: Optional[Pipeline] = None
    baseline_test_metrics: Optional[dict[str, float]] = None
    transformer_metrics: Optional[dict[str, float]] = None

    if run_phase in {"all", "1"}:
        best_baseline, baseline_test_metrics = _phase1_baseline_train_eval(train_df, val_df, test_df)
    else:
        baseline_path = BASELINE_MODEL_DIR / "baseline.joblib"
        if baseline_path.exists():
            import joblib

            best_baseline = joblib.load(baseline_path)

    if run_phase in {"all", "2"}:
        _phase2_eda_and_leakage(df, source_col)

    if run_phase in {"all", "3"}:
        if best_baseline is None:
            raise FileNotFoundError("Baseline model not found. Run phase 1 first.")
        _phase3_baseline_analysis(best_baseline, test_df)

    if run_phase in {"all", "4"} and not skip_transformer:
        transformer_metrics = _phase4_transformer(
            train_df,
            val_df,
            test_df,
            config=transformer_config or TransformerTrainingConfig(),
            run_sweep=run_transformer_sweep,
        )

    if run_phase in {"all", "5"}:
        _phase5_error_analysis(test_df)

    rows: list[dict[str, float | str]] = []
    if baseline_test_metrics is None:
        baseline_test_metrics = _try_load_saved_metrics(METRICS_DIR / "baseline_test_metrics.json")
    if baseline_test_metrics:
        rows.append({"model": "tfidf_logreg", **baseline_test_metrics})
    if transformer_metrics:
        rows.append({"model": "distilbert", **transformer_metrics})
    if rows:
        compare_models_table(rows, METRICS_DIR / "model_comparison.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train news bias models.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="valurank/PoliticalBias_AllSides_Txt",
        help="Dataset path or Hugging Face dataset name.",
    )
    parser.add_argument("--skip-transformer", action="store_true", help="Skip transformer training.")
    parser.add_argument("--transformer-lr", type=float, default=2e-5, help="Transformer learning rate.")
    parser.add_argument("--transformer-epochs", type=int, default=2, help="Transformer training epochs.")
    parser.add_argument("--transformer-batch-size", type=int, default=8, help="Transformer batch size.")
    parser.add_argument(
        "--transformer-warmup-ratio", type=float, default=0.1, help="Transformer warmup ratio."
    )
    parser.add_argument(
        "--transformer-weight-decay", type=float, default=0.01, help="Transformer weight decay."
    )
    parser.add_argument(
        "--transformer-max-length", type=int, default=512, help="Transformer max sequence length."
    )
    parser.add_argument(
        "--transformer-early-stopping-patience",
        type=int,
        default=1,
        help="Early stopping patience in eval epochs.",
    )
    parser.add_argument(
        "--transformer-no-class-weights",
        action="store_true",
        help="Disable class-weighted loss for transformer.",
    )
    parser.add_argument(
        "--run-transformer-sweep",
        action="store_true",
        help="Run a compact transformer hyperparameter sweep (phase 4).",
    )
    parser.add_argument(
        "--no-mask-outlets",
        action="store_true",
        help="Disable masking of common news outlet names in text (ablation only).",
    )
    parser.add_argument(
        "--run-phase",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "5"],
        help="Run a specific phase only: 1-baseline, 2-EDA, 3-baseline analysis, 4-transformer, 5-error analysis.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TransformerTrainingConfig(
        learning_rate=args.transformer_lr,
        batch_size=args.transformer_batch_size,
        num_epochs=args.transformer_epochs,
        warmup_ratio=args.transformer_warmup_ratio,
        weight_decay=args.transformer_weight_decay,
        max_sequence_length=args.transformer_max_length,
        use_class_weights=not args.transformer_no_class_weights,
        early_stopping_patience=args.transformer_early_stopping_patience,
    )
    run_pipeline(
        args.dataset,
        skip_transformer=args.skip_transformer,
        run_phase=args.run_phase,
        transformer_config=cfg,
        run_transformer_sweep=args.run_transformer_sweep,
        mask_outlets=not args.no_mask_outlets,
    )
