# Project Summary

**Data.** SEER breast cancer cohort: 4,024 women, 616 deaths (15.3 %), diagnosed 2006–2010 (Teng, IEEE DataPort 2019).

**Aim.** Predict the survival curve and the 5-year risk of death from 12 variables known at diagnosis.

## Steps

1. **Audit and clean.** Fixed headers and a typo, dropped an empty column, ran 17 checks. No imputation was needed.
2. **Outcome.** `Status` and `Survival Months` together form a censored time-to-event outcome. 754 patients who are alive were followed for less than 5 years.
3. **No leakage.** `Survival Months` is never an input, and the train/test split is made before any fitting.
4. **Features.** Node ratio, log-transformed size and node count, and ER/PR combination. All of them sit inside the model pipeline.
5. **Model choice.** 5-fold CV. The simplest model within one standard error of the best was kept.
6. **Test.** One evaluation on a 20 % held-out set.

## Results

| Model | Test result |
|---|---|
| L1 logistic regression (5-year risk) | ROC-AUC 0.734 (0.673–0.791), calibration slope 0.99 |
| Elastic-net Cox (survival) | C-index 0.711 (0.663–0.759) |

- Tuned boosting models and a stacked ensemble reach the same AUC (0.715–0.734).
- The learning curve is flat, so more data of the same kind would not help.
- A published leakage-free result on this cohort is ROC-AUC 0.75–0.78.

## The 95 % target

- No valid setup reaches 95 %. The best F1 for death, at any threshold, is 0.42.
- About 95 % appears only when `Survival Months` is used as an input and SMOTE is applied before the split. The app's Leakage lab reproduces this.
- Better results would need more information, such as treatment, HER2 status and cause of death.

## Limitations

- All-cause mortality only.
- No treatment or HER2 data.
- Internal validation only.
- The Black patient subgroup is small, and the model may under-predict for it.
