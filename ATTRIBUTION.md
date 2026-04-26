# Attribution

## Dataset
- Hugging Face dataset: [valurank/PoliticalBias_AllSides_Txt](https://huggingface.co/datasets/valurank/PoliticalBias_AllSides_Txt)
- Loaded via:
```python
from datasets import load_dataset
ds = load_dataset("valurank/PoliticalBias_AllSides_Txt")
```

## Libraries
- `scikit-learn` for baseline modeling and metrics
- `transformers` + `torch` for DistilBERT fine-tuning
- `matplotlib` for figures
- `streamlit` for the demo app
- `lime` / `shap` for optional explainability extensions

## AI Assistance Acknowledgement
- AI coding assistants (Cursor + LLM support) were used to accelerate implementation, refactoring, bug fixing, and documentation drafting.
- Human-directed work included project design choices, experiment planning, metric interpretation, and verification of generated code outputs.
- AI-generated suggestions were reviewed, edited, and adapted before being kept in the final codebase.
