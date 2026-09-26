import pandas as pd
import streamlit as st

st.title("🧭 Method check")

st.subheader("Standard checks")
st.dataframe(pd.DataFrame([
    ("Train/test split before any fitting", "✅", "Verify results"),
    ("Outcome columns never used as inputs", "✅", "Feature pipeline"),
    ("Censoring handled", "✅", "Data lab, step 5"),
    ("Preprocessing fitted inside CV folds", "✅", "Feature pipeline"),
    ("Model chosen by CV; test set used once", "✅", "Verify results"),
    ("Calibration and confidence intervals reported", "✅", "Verify results; notebooks 06–07"),
    ("No resampling that distorts probabilities", "✅", "Leakage lab"),
    ("Reproducible (retrain gives identical scores)", "✅", "Verify results"),
    ("Subgroup check", "✅", "notebook 09"),
    ("External validation", "❌", "no second cohort available"),
    ("Treatment, HER2, cause of death", "❌", "not in the dataset"),
], columns=["check", "status", "where"]), hide_index=True, use_container_width=True)

st.subheader("Scope")
st.markdown("""
- The final models are the simplest candidates: L1 logistic regression and elastic-net Cox.
- Complex models were run only for comparison. None of them did better.
- Extra ideas (age², tuning, SMOTE) were tested and dropped when they did not help.
- Main analysis: notebooks 01, 02, 04, 06, 07. Notebooks 03, 05, 08, 09 and 10 are supporting evidence.
""")

st.subheader("Likely questions")
st.markdown("""
| Question | Answer |
|---|---|
| Why is accuracy only 0.73–0.80? | Predicting "survives" for everyone gives 0.86. With 14 % deaths, AUC and calibration are the useful measures. |
| Why not a newer model? | Tuned CatBoost, LightGBM, XGBoost and an ensemble give the same AUC, and the learning curve is flat. |
| Does dropping censored patients bias the 5-year label? | Censoring looks administrative, and the survival model, which uses all patients, ranks patients the same way (Spearman 0.94). |
| Only one test split? | Model choice used 5-fold CV, and the test CI is reported (AUC 0.673–0.791). |
| Any subgroup concerns? | The model may under-predict for Black patients (27.8 % observed vs 20.5 % predicted, n = 54; CI includes zero). |
""")

st.info("With the 12 variables available at diagnosis, AUC is about 0.73–0.75 for every model tried. "
        "Scores near 0.95 on this dataset require leakage or test contamination (see Leakage lab).")
