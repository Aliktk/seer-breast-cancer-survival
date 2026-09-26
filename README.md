# SEER Breast Cancer Survival

Survival analysis and 5-year mortality prediction on the SEER breast cancer cohort (4,024 women, diagnosed 2006–2010).

Two questions:

1. What is a patient's survival curve, given what is known at diagnosis?
2. What is her risk of death (any cause) within 5 years?

A short overview is in [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md).

## Results

Held-out test set (20 %), evaluated once after model selection.

| Task | Model | Discrimination | Calibration |
|---|---|---|---|
| 5-year mortality | L1 logistic regression | ROC-AUC 0.734 (95 % CI 0.673–0.791) | slope 0.99 |
| Survival curve | Elastic-net Cox | C-index 0.711 (95 % CI 0.663–0.759) | good at 60 months |

- Tuned CatBoost, LightGBM, XGBoost, random survival forests and a stacked ensemble did not beat the linear models. The simplest model within one standard error of the best was kept.
- `Survival Months` is part of the outcome and is never used as an input. Adding it raises AUC from 0.745 to 0.864, which is leakage.
- 754 "Alive" patients were followed for less than 5 years. The survival model uses them as censored cases. The 5-year classifier leaves them out, since their label is unknown.
- Main risk factors: number of positive nodes (and node ratio), ER/PR-negative status, grade and age.
- A published leakage-free result on this cohort reports ROC-AUC 0.75–0.78 (Cruz-Fernandez et al., 2026).

On the request for >95 % precision, recall, F1 and accuracy, see [`docs/sota_and_95_percent_target.md`](docs/sota_and_95_percent_target.md).

## Quick start

Python 3.10–3.13 (tested on 3.11).

```bash
git clone https://github.com/Aliktk/seer-breast-cancer-survival.git
cd seer-breast-cancer-survival
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt

streamlit run app/streamlit_app.py    # app (models are already in models/)
python scripts/train.py               # retrain, ~2 min
pytest                                # tests
python scripts/run_notebooks.py       # re-run all notebooks, ~14 min
```

## Notebooks

| # | Notebook | Content |
|---|---|---|
| 01 | Raw data audit | file structure, header issues, duplicates, outcome columns |
| 02 | Cleaning and validation | cleaning, 17 checks, clinical consistency flags |
| 03 | EDA | distributions, group death rates, tests with effect sizes, correlation |
| 04 | Survival EDA | Kaplan–Meier, log-rank, Cox model, proportional-hazards check |
| 05 | Feature engineering | engineered features and ablation tests |
| 06 | 5-year classification | label definition, leakage test, model comparison, test evaluation |
| 07 | Survival models | Cox, Coxnet, RSF, gradient-boosted survival; C-index, IBS, calibration |
| 08 | Explainability | odds ratios, hazard ratios, permutation importance, SHAP |
| 09 | Inference and audit | predictor class, input checks, subgroup performance |
| 10 | SOTA benchmark | tuned boosting models, ensemble, threshold sweep, learning curve |

01, 02, 04, 06 and 07 cover the main analysis. The others add supporting evidence.

## App

Multi-page Streamlit app built on the same package as the notebooks (`src/seer_survival`).

- **Data lab**: raw file → cleaning → validation, run live
- **EDA**: filterable plots, tests and Kaplan–Meier curves
- **Feature pipeline**: which features the models use; one patient traced to the final risk
- **Verify results**: recomputes test metrics and compares them with the stored values; retrain button; threshold explorer
- **Leakage lab**: shows how `Survival Months` and SMOTE-before-split inflate scores
- **Patient / Batch**: predictions with survival curves

Deployment steps: [`docs/deployment.md`](docs/deployment.md).

## Layout

```
app/            Streamlit app (entry point: app/streamlit_app.py)
data/           raw/SEER.csv (interim/processed files are created by notebooks 02 and 05)
docs/           summary, method notes, data dictionary, references
models/         trained pipelines (.joblib), metadata.json, model card
notebooks/      01–10, executed
reports/        result tables (figures are created when the notebooks run)
scripts/        train.py, run_notebooks.py
src/            seer_survival package: data, features, targets, models, training, evaluate, inference
tests/          pytest
```

## Limitations

- Outcome is all-cause death. Cause of death is not in the data.
- No treatment or HER2 variables.
- Node-positive women aged 30–69 only. Other inputs are extrapolations, and the app warns about them.
- Internal validation only.
- Possible under-prediction for Black patients (small subgroup, CI includes zero).

## Data

Teng, J. (2019). *SEER Breast Cancer Data*. IEEE DataPort. https://doi.org/10.21227/a9qy-ph35

References: [`docs/references.md`](docs/references.md). Code licence: MIT.

Author: Ali Nawaz
