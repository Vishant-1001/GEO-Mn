# Legacy prototype artifacts (not loaded at runtime)

Retained for historical reference and reproducibility of the audit only.

| File | Was | Status |
|---|---|---|
| `manganese_model.pkl`, `feature_columns.pkl`, `X_train.pkl` | earlier exploration RandomForest (features without labels/coordinates) | read only by `ml/train_exploration.py` → `legacy_check` (offline); never served |
| `production_model.pkl`, `prod_feature_columns.pkl` | earlier efficiency-ratio production model | superseded; unused |

No API endpoint loads these files. The current models are in `models/` and described in `models/model_manifest.json`.
