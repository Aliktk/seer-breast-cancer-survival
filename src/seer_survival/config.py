"""Paths, column names and constants shared by notebooks, scripts and the app."""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "SEER.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DATA_PATH = INTERIM_DIR / "seer_clean.csv"
FEATURES_PATH = PROCESSED_DIR / "seer_features.csv"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5

# --------------------------------------------------------------------------
# Column names after cleaning (snake_case, typos fixed)
# --------------------------------------------------------------------------
RAW_TO_CLEAN = {
    "Age": "age",
    "Race": "race",
    "Marital Status": "marital_status",
    "T Stage": "t_stage",
    "N Stage": "n_stage",
    "6th Stage": "stage_6th",
    "Grade": "grade",
    "A Stage": "a_stage",
    "Tumor Size": "tumor_size",
    "Estrogen Status": "estrogen_status",
    "Progesterone Status": "progesterone_status",
    "Regional Node Examined": "nodes_examined",
    "Reginol Node Positive": "nodes_positive",  # typo in the source file
    "Survival Months": "survival_months",
    "Status": "status",
}

TIME_COL = "survival_months"
EVENT_COL = "event"          # 1 = Dead, 0 = Alive (censored)
STATUS_COL = "status"

NUMERIC_COLS = ["age", "tumor_size", "nodes_examined", "nodes_positive"]
NOMINAL_COLS = ["race", "marital_status"]
BINARY_COLS = ["estrogen_status", "progesterone_status", "a_stage"]
ORDINAL_COLS = ["t_stage", "n_stage", "stage_6th", "grade"]

# Clinical ordering of the ordinal variables (AJCC 6th edition / SEER grade).
ORDINAL_LEVELS = {
    "t_stage": ["T1", "T2", "T3", "T4"],
    "n_stage": ["N1", "N2", "N3"],
    "stage_6th": ["IIA", "IIB", "IIIA", "IIIB", "IIIC"],
    "grade": [1, 2, 3, 4],
}

GRADE_MAP = {
    "Well differentiated; Grade I": 1,
    "Moderately differentiated; Grade II": 2,
    "Poorly differentiated; Grade III": 3,
    "Undifferentiated; anaplastic; Grade IV": 4,
}

RACE_SHORT = {
    "White": "White",
    "Black": "Black",
    "Other (American Indian/AK Native, Asian/Pacific Islander)": "Other",
}

MARITAL_SHORT = {
    "Married (including common law)": "Married",
    "Single (never married)": "Single",
    "Divorced": "Divorced",
    "Widowed": "Widowed",
    "Separated": "Separated",
}

# Horizons (months) used for fixed-time evaluation.
HORIZON_MONTHS = 60
EVAL_TIMES = [36, 60, 84]

# Features that a clinician can know at diagnosis. survival_months and status
# are OUTCOMES and must never be used as inputs.
LEAKAGE_COLS = ["survival_months", "status", "event"]
