import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import BLUE, GREY, ORANGE, STATUS_COLORS, raw_data
from seer_survival import config as C
from seer_survival.data import clean, consistency_flags, validate

st.title("🧪 Data lab: raw → clean, step by step")
st.caption("Each step runs live with the same functions used in notebooks 01–02 and in training.")

src = st.radio("Data source", ["Bundled SEER.csv", "Upload a CSV with the same columns"], horizontal=True)
raw = raw_data()
if src.startswith("Upload"):
    up = st.file_uploader("CSV in the original SEER format", type="csv")
    if up is None:
        st.stop()
    raw = pd.read_csv(up)

# ---------------------------------------------------------------- Step 1
st.header("Step 1 · Raw file audit")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{raw.shape[0]:,}")
c2.metric("Columns", raw.shape[1])
c3.metric("Exact duplicate rows", int(raw.duplicated().sum()))
c4.metric("Fully empty columns", int(raw.isna().all().sum()))

issues = []
for c in raw.columns:
    if c != c.strip():
        issues.append({"column": repr(c), "issue": "leading/trailing space in header", "action": "strip"})
    if raw[c].isna().all():
        issues.append({"column": repr(c), "issue": "100 % empty", "action": "drop"})
    if "Reginol" in c:
        issues.append({"column": repr(c), "issue": "typo 'Reginol'", "action": "rename → nodes_positive"})
st.markdown("**Problems detected automatically**")
st.dataframe(pd.DataFrame(issues) if issues else pd.DataFrame([{"issue": "none"}]), hide_index=True,
             use_container_width=True)
with st.expander("Raw data (first 50 rows, exactly as in the file)"):
    st.dataframe(raw.head(50), use_container_width=True)

# ---------------------------------------------------------------- Step 2
st.header("Step 2 · Cleaning and type conversion")
try:
    df = clean(raw)
except Exception as e:
    st.error(f"Cleaning failed: {e}")
    st.stop()
st.markdown(
    "- headers stripped and renamed to snake_case · `Reginol` typo fixed · empty column dropped\n"
    "- `Grade` text → ordinal 1–4 · long race / marital labels shortened\n"
    "- new `event` column (1 = Dead) for survival models · **nothing imputed** (no real missing values)"
)
c1, c2 = st.columns(2)
c1.markdown("**Grade mapping applied**")
c1.dataframe(pd.crosstab(raw[[c for c in raw.columns if c.strip() == "Grade"][0]], df["grade"]),
             use_container_width=True)
c2.markdown("**Cleaned data (first 10 rows)**")
c2.dataframe(df.head(10), use_container_width=True, height=250)

# ---------------------------------------------------------------- Step 3
st.header("Step 3 · Validation checks")
rep = validate(df).to_frame()
n_hard = rep.name.str.startswith("[hard]").sum()
n_hard_ok = (rep.name.str.startswith("[hard]") & rep.passed).sum()
c1, c2 = st.columns(2)
c1.metric("Hard checks passed", f"{n_hard_ok} / {n_hard}")
c2.metric("Soft (clinical) checks flagged", int((~rep.passed & rep.name.str.startswith("[soft]")).sum()))
st.dataframe(rep.style.apply(lambda r: ["background-color:#fde0d9" if not r.passed else "" for _ in r], axis=1),
             hide_index=True, use_container_width=True)
st.caption("[hard] = data would be unusable if it failed. [soft] = clinically unexpected but possible, "
           "so it is flagged and not edited (AJCC 6th-edition cut-offs).")

# ---------------------------------------------------------------- Step 4
st.header("Step 4 · Clinical consistency flags")
flags = consistency_flags(df)
st.write(f"**{int(flags.any(axis=1).sum())} rows** carry at least one flag. They are **kept**; removing them changes "
         "test AUC by only −0.005 (notebook 09).")
c1, c2 = st.columns(2)
d = df.assign(flag=np.where(flags.flag_n_count, "flagged", "ok"))
c1.plotly_chart(px.strip(d, x="n_stage", y="nodes_positive", color="flag", category_orders={"n_stage": ["N1", "N2", "N3"]},
                         color_discrete_map={"ok": BLUE, "flagged": ORANGE}, title="Positive nodes vs N stage"),
                use_container_width=True)
d = df.assign(flag=np.where(flags.flag_t3_size, "flagged", "ok"))
c2.plotly_chart(px.strip(d, x="t_stage", y="tumor_size", color="flag", category_orders={"t_stage": ["T1", "T2", "T3", "T4"]},
                         color_discrete_map={"ok": BLUE, "flagged": ORANGE}, title="Tumour size vs T stage"),
                use_container_width=True)

# ---------------------------------------------------------------- Step 5
st.header("Step 5 · The outcome: why this is a survival problem")
alive = df.status == "Alive"
short = int((alive & (df.survival_months < 60)).sum())
c1, c2, c3 = st.columns(3)
c1.metric("Mean follow-up · Alive", f"{df.loc[alive, 'survival_months'].mean():.1f} mo")
c2.metric("Mean follow-up · Dead", f"{df.loc[~alive, 'survival_months'].mean():.1f} mo")
c3.metric("Alive but followed < 60 months", short, help="Their 5-year outcome is unknown (censored).")
st.plotly_chart(px.histogram(df, x="survival_months", color="status", nbins=40, barmode="overlay", opacity=0.6,
                             color_discrete_map=STATUS_COLORS,
                             title="Follow-up time by status: survival months depends on the outcome"),
                use_container_width=True)
st.warning("Because `Survival Months` is shorter for patients who died, using it as an input would leak the answer. "
           "It is used only to **define** the outcome.", icon="⚠️")
