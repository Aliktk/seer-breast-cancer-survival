# Model Card

## Overview

| | 5-year mortality classifier | Survival model |
|---|---|---|
| File | `mortality_5y_classifier.joblib` | `survival_model.joblib` |
| Algorithm | L1-penalised logistic regression (C = 0.5, liblinear) | Elastic-net Cox (Coxnet, l1_ratio = 0.5, α = 0.01) |
| Output | P(death from any cause within 60 months) | survival function S(t), t ≤ 107 months |
| Libraries | scikit-learn 1.5.2 | scikit-survival 0.23.1 |
| Inputs | 12 clinical variables at diagnosis (see `docs/data_dictionary.md`) | same |
| Preprocessing | `FeatureEngineer` + `ColumnTransformer`, inside the pickled pipeline | same |

Exact selection details, the decision threshold, risk-group cut-offs, training ranges and all metrics are in `metadata.json`.

## Intended use

- Teaching, research and methodological demonstration on the public SEER breast cancer cohort.
- **Not** intended for clinical decision-making. It is not a medical device and has not been externally validated.

## Training data

- SEER Breast Cancer Data (Teng, IEEE DataPort 2019): women aged 30–69, infiltrating duct and lobular carcinoma, node-positive, diagnosed 2006–2010, US.
- Training split: 3,219 patients (493 deaths). The classifier uses the 2,559 with a known 5-year status (366 deaths within 60 months).

## Evaluation (held-out test split, used once)

| Classifier (657 labelled, 92 events) | Value |
|---|---|
| ROC-AUC | 0.734 (95 % CI 0.673–0.791) |
| PR-AUC | 0.394 |
| Brier score | 0.104 |
| Calibration slope / intercept | 0.99 / −0.02 |
| Sensitivity / specificity at threshold 0.147 | 0.62 / 0.74 |

| Survival model (805 patients, 123 events) | Value |
|---|---|
| Harrell's C | 0.711 (95 % CI 0.663–0.759) |
| Uno's C (τ = 84 months) | 0.693 |
| Time-dependent AUC 36 / 60 / 84 months | 0.760 / 0.729 / 0.695 |
| Integrated Brier score (36–84 months) | 0.092 (Kaplan–Meier reference 0.103) |

## Known limitations and risks

- Outcome is all-cause mortality. Older patients' deaths from other causes reduce discrimination (C 0.64 in the 60–69 age group).
- No treatment, HER2, or comorbidity information.
- Possible risk under-estimation for Black patients: observed 27.8 % vs predicted 20.5 % 5-year mortality in 54 labelled test patients. The 95 % CI includes no difference.
- Ages outside 30–69 and other out-of-range inputs are extrapolations; the inference code warns about them.
- ER and PR violate the proportional-hazards assumption (notebook 04). Their hazard ratios are time-averaged.

## Retraining

`python scripts/train.py` reproduces both artefacts deterministically (seed 42).
