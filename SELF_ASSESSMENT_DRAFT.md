# Self-Assessment Draft (CS372 Final Project)

This draft is a starting point for the Gradescope self-assessment. Update any item wording or evidence paths as needed before submission.

## Category 1: Machine Learning (strict best 15 selections)

Submit exactly these 15 claims:

1. **Completed project individually without a partner (10 pts)**
   - Evidence: solo submission claim in self-assessment form; no partner listed in project docs/submission metadata.

2. **Modified/Adapted a transformer model by fine-tuning for task (7 pts)**
   - Evidence: `src/train_transformer.py`, `models/transformer/`, `outputs/metrics/transformer_test_metrics.json`.

3. **Compared multiple approaches quantitatively (7 pts)**
   - Evidence: baseline metrics (`outputs/metrics/baseline_test_metrics.json`) vs transformer metrics (`outputs/metrics/transformer_test_metrics.json`).

4. **Performed error analysis with failure cases (7 pts)**
   - Evidence: `outputs/reports/baseline_errors.csv`, `outputs/predictions/baseline_test_predictions.csv`.

5. **Interpretable model design / explainability analysis (7 pts)**
   - Evidence: `src/explain.py`, `outputs/reports/baseline_top_features.csv`, `outputs/reports/baseline_example_explanations.jsonl`.

6. **Applied regularization techniques (5 pts)**
   - Evidence: `src/train_baseline.py` (`class_weight="balanced"`); `src/train_transformer.py` (`weight_decay`, early stopping callback).

7. **Conducted systematic hyperparameter tuning (5 pts)**
   - Evidence: `src/config.py` (`BaselineTuningConfig`), `src/train_baseline.py` (`tune_baseline`), `outputs/metrics/baseline_best_params.json`.

8. **Applied feature engineering (5 pts)**
   - Evidence: `src/features.py` (handcrafted stance/sentiment features), `outputs/reports/baseline_top_features.csv` (`STANCE:*` terms).

9. **Documented iterative model improvement decisions (5 pts)**
   - Evidence: `README.md` tuning notes + saved best config in `outputs/metrics/baseline_best_params.json`; transformer metrics in `outputs/metrics/transformer_test_metrics.json`.

10. **Modular code design with reusable functions/classes (3 pts)**
    - Evidence: `src/data_loader.py`, `src/preprocess.py`, `src/train_baseline.py`, `src/train_transformer.py`, `src/evaluate.py`, `main.py`.

11. **Proper train/validation/test split (3 pts)**
    - Evidence: `src/preprocess.py` (`split_dataset`), `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`.

12. **Created baseline model for comparison (3 pts)**
    - Evidence: `src/train_baseline.py`, `outputs/metrics/baseline_test_metrics.json`.

13. **Applied basic preprocessing for text modality (3 pts)**
    - Evidence: `src/preprocess.py` (label normalization, dedup/missing handling, outlet masking).

14. **Trained using GPU/accelerator hardware (3 pts)**
    - Evidence: transformer MPS training setup in `src/train_transformer.py` + run logs.

15. **Used at least three appropriate evaluation metrics (3 pts)**
    - Evidence: `src/evaluate.py`; metrics JSON files include accuracy, macro precision, macro recall, macro F1.

---

## Category 2: Following Directions (check all that apply)

- `SETUP.md` exists with install/run instructions.
- `ATTRIBUTION.md` exists with dataset/library attributions.
- `requirements.txt` is included.
- `README.md` includes What It Does + Quick Start + Evaluation + Video Links section (fill links before submit).
- Self-assessment submitted with evidence pointers (this document can be adapted for that).

---

## Category 3: Cohesion and Motivation (check all that apply)

- Unified project goal in `README.md`: predict outlet-rated ideological lean from article text.
- Real-world motivation included in `README.md` ("Why It Matters").
- Clear pipeline progression in `main.py`: preprocess -> train/eval -> analysis -> reporting.
- Evaluation metrics directly match classification objective.

---

## Final pre-submit checklist

- Add real demo + walkthrough URLs in `README.md` Video Links.
- Ensure self-assessment item count is <= 15 in Category 1.
- For any claim requiring "impact", include metric deltas or a brief comparison note.
