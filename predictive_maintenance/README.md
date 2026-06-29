# Predictive Maintenance System

Production-grade RUL prediction for industrial equipment using XGBoost and Streamlit.

## Project structure

```
predictive_maintenance/
├── config.py               # Centralised configuration dataclasses
├── data_generator.py       # Synthetic industrial sensor data
├── data_preprocessing.py   # Cleaning, validation, group-safe splitting
├── feature_engineering.py  # Rolling, lag, cumulative, interaction features + RUL
├── train.py                # Scaler + XGBoost training with SHAP explanations
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI entry point (generate | train | all)
├── requirements.txt
├── README.md
├── data/
├── models/
└── logs/
```

## Quick start

```bash
pip install -r requirements.txt
python main.py --mode all        # generate data + train model
streamlit run app.py             # launch dashboard
```

## Key fixes over the original code

| File | Bug | Fix |
|------|-----|-----|
| `data_preprocessing.py` | `fillna(method="ffill")` deprecated (pandas 2.x) | `.ffill().bfill()` |
| `feature_engineering.py` | `groupby()[col].fillna(method="bfill")` deprecated | `.transform(lambda s: s.bfill())` |
| `feature_engineering.py` | `data.fillna(method="bfill")` at end | `.bfill()` |
| `train.py` | `eval_set` built before scaler was fitted (wrong values) | Fit scaler first, then build `eval_set` |
| `train.py` | `early_stopping_rounds` patched via `set_params()` post-construction | Moved to `ModelConfig.xgboost_params` |
| `train.py` | Pipeline hides scaler so `eval_set` can't be scaled correctly | Replaced Pipeline with explicit scaler + model |
| `app.py` | `self.model.feature_names_in_` on pipeline (XGBoost strips names after scaling) | Feature names saved explicitly in artefact dict |
| `app.py` | `argparse` in `main()` crashes when launched via `streamlit run` | `argparse` removed; paths use module-level constants |
