# State-of-the-Art Models and the 95 % Target

Request: apply recent models and reach more than 95 % precision, recall, F1 and accuracy.

Code and outputs: `notebooks/10_sota_benchmark_and_95_percent_target.ipynb`.

## Leakage-free results

Test set: 657 patients, 92 deaths within 5 years. The threshold for each model was chosen on training data.

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| L1 logistic regression | 0.734 | 0.769 | 0.317 | 0.565 | 0.406 |
| CatBoost (default) | 0.727 | 0.779 | 0.320 | 0.511 | 0.393 |
| CatBoost (Optuna, 40 trials) | 0.734 | 0.791 | 0.345 | 0.543 | 0.422 |
| LightGBM (Optuna) | 0.719 | 0.813 | 0.370 | 0.478 | 0.417 |
| XGBoost (Optuna) | 0.715 | 0.801 | 0.344 | 0.467 | 0.396 |
| Stacked ensemble | 0.729 | 0.802 | 0.352 | 0.489 | 0.409 |

Precision, recall and F1 are for the Dead class.

- **Threshold sweep:** across every threshold on the test set, the best F1 is 0.42. No threshold gets all four metrics to 95 %.
- **Majority baseline:** predicting "survives" for everyone already gives 0.86 accuracy.
- **Learning curve:** AUC stays flat at about 0.73 from roughly 1,300 training patients onwards.
- **TabPFN:** not run, because its weights could not be downloaded in the build environment. The notebook runs it automatically where the package is installed.

## Where 95 % comes from

The table uses the same CatBoost model throughout, with raw `Status` as the target and a threshold of 0.5.

| Pipeline | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Correct | 0.841 | 0.439 | 0.146 | 0.220 |
| + `Survival Months` as input | 0.911 | 0.800 | 0.553 | 0.654 |
| SMOTE before the split | 0.902 | 0.955 | 0.843 | 0.896 |
| Both | 0.947 | 0.978 | 0.915 | 0.945 |

- **`Survival Months`:** it is the time to death or last follow-up, so it is not known at diagnosis.
- **SMOTE before the split:** it puts synthetic copies of training patients into the test set and balances the test set to 50/50.

## Conclusion

With these 12 variables, the achievable AUC is about 0.73–0.75. A published leakage-free result on the same cohort reports 0.75–0.78 (Cruz-Fernandez et al., *Technologies* 2026). Higher performance needs more information, such as treatment, HER2 status and cause of death.
