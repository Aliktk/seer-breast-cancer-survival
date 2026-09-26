# Methods and Decisions

Numbers in brackets refer to [`references.md`](references.md).

## Outcome

`Status` and `Survival Months` together form a right-censored survival outcome. Because of that:

- `Survival Months` cannot be an input [26].
- Alive patients have different follow-up times, so a 5-year label is only defined for patients who died within 60 months or were followed for more than 60 months.
- Survival models (Kaplan–Meier [4], Cox [5]) can use every patient, including censored ones.

This leads to two tasks. The main task is survival analysis. The second is 5-year mortality classification. The two final models rank test patients almost identically (Spearman 0.94).

## Protocol

1. Stratified 80/20 split, seed 42, made before any fitting.
2. All preprocessing sits inside an sklearn `Pipeline`, so it is refitted in every CV fold.
3. 5-fold CV on the training part. The simplest model within one standard error of the best is kept [24].
4. The decision threshold and risk-group cut-offs come from out-of-fold training predictions.
5. The test set is used once. The selection was not changed after seeing test results.

Metrics:
- ROC-AUC, PR-AUC, Brier score and calibration [16] for the 5-year risk model.
- Harrell's C [13], Uno's C [14], time-dependent AUC and integrated Brier score [15] for the survival model.
- Decision-curve analysis [17].

## Models compared

| Family | Models | Kept? |
|---|---|---|
| Linear | L1/L2 logistic regression, Cox, Coxnet [8] | yes (L1 logistic, Coxnet) |
| Tree ensembles | random forest, XGBoost [19], LightGBM [20], CatBoost, RSF [9], gradient-boosted Cox | no gain |
| Other | SVM, MLP, stacked ensemble | no gain |
| Not run | DeepSurv [11], DeepHit [12], TabPFN [22] | see note |

Tree models, which can capture non-linear effects and interactions, did not beat the linear models. With about 490 training events, deep survival models were not expected to help either. That is a judgement call and was not tested. TabPFN could not be downloaded in the build environment.

## Decisions

| Decision | Reason |
|---|---|
| Treat `Dead` as all-cause death | the data source does not give the cause of death |
| Keep the one duplicate row | there is no patient ID, and identical coarse records are plausible |
| Flag, do not edit, 107 clinically inconsistent rows | dropping them changes test AUC by −0.005 |
| Keep extreme tumour sizes and node counts | they are real and informative; `log1p` handles the skew |
| Exclude `6th Stage` | it is fully determined by T and N (VIF 13.2) |
| Add node ratio, log transforms, ER/PR combination | node ratio is prognostic [33]; small gain, no harm |
| No age² term | +0.002 C-index, better in only 2 of 5 folds |
| Drop censored patients from the 5-year label | counting them as survivors is biased; the survival model covers everyone |
| No SMOTE or class weights | they distort calibration [27] |
| Keep L1 logistic and Coxnet | same accuracy as complex models, better calibrated, easy to read |
| Risk groups = training tertiles | there are no validated clinical cut-offs for this model |

## Hypotheses tested

| Hypothesis | Result |
|---|---|
| `Survival Months` inflates performance | yes: +0.12 to +0.15 AUC |
| `6th Stage` = f(T, N) | yes: 100 % agreement |
| Node ratio adds information beyond node counts | no (ΔC +0.0006) |
| Non-linear age helps a linear model | no |
| Non-PH / non-linear models beat Cox | no: within 0.003 C-index |
| Flagged rows change conclusions | no |

## Ceiling

With 12 variables known at diagnosis, AUC reaches about 0.73–0.75 and C-index about 0.71–0.74 across all model families. A published leakage-free study on the same cohort reports AUC 0.75–0.78 [3]. The main missing information is treatment, HER2 status, comorbidity and cause of death.
