# Setup Guide

## Python Version
- Recommended: Python 3.11+

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
Default data source is Hugging Face:
```python
from datasets import load_dataset
ds = load_dataset("valurank/PoliticalBias_AllSides_Txt")
```

You can also pass local CSV/JSON/parquet paths with `--dataset`.

## Outlet name masking
By default, `main.py` replaces common U.S. news outlet strings in article text with `[OUTLET]` before train/val/test splits (see `src/config.py` → `OUTLET_NAME_REGEXES`). This reduces models learning shortcuts like “CNN” → left.

Disable for ablation studies:
```bash
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --no-mask-outlets --run-phase 1
```

## Phased retrain (outlet-masked models)
After masking is on, retrain in order:
```bash
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 1
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 2
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 3
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 4 \
  --transformer-epochs 3 --transformer-lr 2e-5 --transformer-batch-size 8
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --run-phase 5
```
`--run-phase` supports `1` through `5` (plus `all`).

## Train Baseline + Transformer
```bash
python main.py --dataset valurank/PoliticalBias_AllSides_Txt
```

## Train Baseline Only
```bash
python main.py --dataset valurank/PoliticalBias_AllSides_Txt --skip-transformer
```

## Launch App
From the project root so `src` imports resolve:
```bash
cd /path/to/NewsBiasDetector
PYTHONPATH=. python -m streamlit run app/streamlit_app.py
```
The app applies the same outlet masking as training before baseline prediction.

## Key Outputs
- `outputs/metrics/`: metrics JSON + comparison CSV
- `outputs/figures/`: EDA and confusion matrix figures
- `outputs/reports/`: top features, leakage report, error analysis
- `models/`: saved baseline and transformer models
