import streamlit as st

from common import clean_data, meta

m = meta()
cm, sm = m["test_metrics_classifier"], m["test_metrics_survival"]
df = clean_data()

st.title("🎗️ SEER Breast Cancer Survival")
st.caption("Survival curve and 5-year risk of death from 12 variables known at diagnosis.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{len(df):,}")
c2.metric("Deaths", f"{df.event.sum()} ({df.event.mean():.1%})")
c3.metric("5-year risk · test AUC", f"{cm['roc_auc']:.3f}",
          help=f"95% CI {cm['roc_auc_ci95'][0]:.3f}–{cm['roc_auc_ci95'][1]:.3f}")
c4.metric("Survival · test C-index", f"{sm['c_harrell']:.3f}")

st.markdown(
    """
| Step | What was done | Page |
|---|---|---|
| Clean | fix headers and typo, drop the empty column, 17 checks | Data lab |
| Explore | death rates, tests, Kaplan–Meier curves | EDA |
| Avoid leakage | `Survival Months` never an input; split before fitting | Leakage lab |
| Features | node ratio, log transforms, ER/PR combination, inside the pipeline | Feature pipeline |
| Model | 5-fold CV; simplest model within 1 SE of the best | Verify results |
| Test | held-out 20 %, evaluated once | Verify results |
| Predict | single patient or CSV | Patient / Batch |
"""
)

st.markdown(
    f"""
**Findings**
- Models: `{m['classifier_name']}` (5-year risk) and `{m['survival_model_name']}` (survival curve).
- Tuned boosting models and ensembles reach the same AUC as logistic regression (about 0.73).
- Scores near 95 % appear only with leakage and SMOTE before the split.
- Main risk factors: nodal burden, ER/PR-negative status, grade, age.
"""
)
st.caption("The Verify results page recomputes these numbers from the raw CSV and the saved models.")
