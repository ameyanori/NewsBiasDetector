# Detecting Ideological Bias in U.S. News Articles Using NLP
This project predicts outlet-rated ideological lean (`left`, `center`, `right`) from article text using a TF-IDF baseline and a fine-tuned DistilBERT classifier, then exports evaluation and explainability artifacts for analysis.

## What It Does
The pipeline loads political news text, normalizes labels, applies data cleaning and optional outlet-name masking, and trains two classifiers: a TF-IDF + Logistic Regression baseline and a DistilBERT transformer model. It evaluates both with accuracy and macro-averaged metrics, exports comparison/error-analysis reports, and provides a Streamlit app for interactive baseline inference with contribution-based explanations. The model predicts outlet-rated lean patterns in language and is not an objective truth detector.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# full pipeline (phases 1-5)
python main.py --dataset valurank/PoliticalBias_AllSides_Txt

# run the app
PYTHONPATH=. python -m streamlit run app/streamlit_app.py
```

Phased execution (`--run-phase`) supports:
- `1` baseline train/eval
- `2` EDA + leakage checks
- `3` baseline explainability exports
- `4` transformer train/eval (or sweep)
- `5` baseline error analysis

## Video Links
- Demo video (3-5 min): 
- Technical walkthrough (5-10 min): 

## Evaluation
Current saved test metrics:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.6467 | 0.6362 | 0.6359 | 0.6344 |
| DistilBERT (best saved run) | 0.8531 | 0.8550 | 0.8454 | 0.8498 |

Primary metric artifacts:
- `outputs/metrics/baseline_test_metrics.json`
- `outputs/metrics/transformer_test_metrics.json`
- `outputs/metrics/model_comparison.csv`

Additional analysis artifacts:
- `outputs/reports/leakage_report.csv`
- `outputs/reports/baseline_top_features.csv`
- `outputs/reports/baseline_errors.csv`

## Why It Matters
Media framing can shape political perception. A language-based classifier helps analyze patterns associated with source-rated ideological lean while explicitly avoiding claims about objective truth.

## Dataset
- Default dataset: [valurank/PoliticalBias_AllSides_Txt](https://huggingface.co/datasets/valurank/PoliticalBias_AllSides_Txt)
- Labels are normalized to `left`, `center`, `right`

## Project Structure
- `src/` core source code
- `data/` processed dataset artifacts
- `models/` saved baseline and transformer artifacts
- `notebooks/` exploratory and analysis notebooks
- `videos/` demo and technical walkthrough videos
- `docs/` additional documentation
- `SETUP.md` installation and run details
- `ATTRIBUTION.md` dataset/library/AI attributions
# Detecting Ideological Bias in U.S. News Articles Using NLP

## What It Does
This project classifies article text into `left`, `center`, or `right` outlet-rated ideological lean. It compares a TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT model, then provides interpretable signals behind predictions.

## Why It Matters
Media framing can shape political perception. A language-based classifier can help analyze patterns associated with source-rated ideological lean while explicitly avoiding claims about objective truth.

## Dataset
- Default dataset: [valurank/PoliticalBias_AllSides_Txt](https://huggingface.co/datasets/valurank/PoliticalBias_AllSides_Txt)
- Labels are derived from source-level AllSides ratings.
- This project normalizes labels to `left`, `center`, `right`.

## Method
- Text cleaning, duplicate/missing removal, label normalization
- Optional masking of common U.S. news outlet names in article text (default **on**) to reduce shortcut learning from strings like “CNN” or “Fox News”
- Leakage checks (label-token and source-name mentions)
- Train/validation/test split
- Baseline: TF-IDF + Logistic Regression with tuning
- Transformer: DistilBERT fine-tuning for 3-class classification
- Metrics: accuracy, macro precision, macro recall, macro F1, confusion matrix

## Results
Training writes artifacts to:
- `outputs/metrics/` for JSON metrics + model comparison table
- `outputs/figures/` for class distribution, length distribution, confusion matrix
- `outputs/reports/` for top terms, leakage report, error examples

## Error Analysis
The pipeline exports misclassified examples and summary reports to support confusion-pair and failure-mode analysis.

## Explainability
- Baseline: top weighted terms per class + per-example feature contributions
- Optional: LIME explanation fallback

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --dataset valurank/PoliticalBias_AllSides_Txt
PYTHONPATH=. python -m streamlit run app/streamlit_app.py
```

### Transformer Tuning
Run a single tuned configuration:
```bash
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 4 \
  --transformer-epochs 3 --transformer-lr 2e-5 --transformer-batch-size 8 \
  --transformer-warmup-ratio 0.1 --transformer-weight-decay 0.01
```

Run a compact sweep and save comparison table:
```bash
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 4 --run-transformer-sweep
```

### Outlet masking + phased retrain (after changing masking defaults)
Phases are intentionally scoped to baseline + transformer:

- `1`: baseline tune/train/eval
- `2`: EDA + leakage checks
- `3`: baseline analysis/explainability exports
- `4`: transformer train/eval (or sweep)
- `5`: baseline error analysis export

Typical rerun sequence:
```bash
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 1
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 2
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 3
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 4 \
  --transformer-epochs 3 --transformer-lr 2e-5 --transformer-batch-size 8
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 5
```
Ablation without outlet masking: add `--no-mask-outlets`.

## Evaluation
Current saved test metrics:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.6467 | 0.6362 | 0.6359 | 0.6344 |
| DistilBERT (best saved run) | 0.8531 | 0.8550 | 0.8454 | 0.8498 |

## Project Structure
See `SETUP.md` for full file tree and execution details.

## Limitations
This model predicts outlet-rated ideological lean patterns from language. It does **not** determine objective truth or absolute bias.

## Future Work
- Same-event cross-outlet comparisons
- Framing-focused discourse features
- Multilingual extension
- Improved transformer explanation methods

## Video Links
- Demo: TBD
- Walkthrough: TBD
