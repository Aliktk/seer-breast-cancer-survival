"""Feature engineering and preprocessing as sklearn transformers (same code for CV, training and the app)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C

# Raw clinical inputs a user provides (what the app form asks for).
INPUT_COLS = [
    "age", "race", "marital_status", "t_stage", "n_stage", "grade", "a_stage",
    "tumor_size", "estrogen_status", "progesterone_status",
    "nodes_examined", "nodes_positive",
]

# Columns produced by FeatureEngineer and consumed by the ColumnTransformer.
NUM_FEATURES = [
    "age", "log_tumor_size", "nodes_examined", "log_nodes_positive", "node_ratio",
]
ORD_FEATURES = ["t_stage", "n_stage", "grade"]
BIN_FEATURES = ["er_positive", "pr_positive", "distant_a_stage"]
NOM_FEATURES = ["race", "marital_status", "hr_status"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-wise features: node ratio, log size and node count, ER/PR combination, 0/1 recodes.

    `stage_6th` is not used because it is derived from T and N.
    """

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(INPUT_COLS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["node_ratio"] = X["nodes_positive"] / X["nodes_examined"].clip(lower=1)
        X["log_tumor_size"] = np.log1p(X["tumor_size"])
        X["log_nodes_positive"] = np.log1p(X["nodes_positive"])
        X["er_positive"] = (X["estrogen_status"] == "Positive").astype(int)
        X["pr_positive"] = (X["progesterone_status"] == "Positive").astype(int)
        X["distant_a_stage"] = (X["a_stage"] == "Distant").astype(int)
        er = np.where(X["er_positive"] == 1, "ER+", "ER-")
        pr = np.where(X["pr_positive"] == 1, "PR+", "PR-")
        X["hr_status"] = pd.Series(er, index=X.index) + "/" + pd.Series(pr, index=X.index)
        return X[NUM_FEATURES + ORD_FEATURES + BIN_FEATURES + NOM_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(NUM_FEATURES + ORD_FEATURES + BIN_FEATURES + NOM_FEATURES)


def make_column_transformer(scale: bool = True) -> ColumnTransformer:
    """Scale numeric/ordinal columns and one-hot encode nominal ones."""
    num_steps = [("scale", StandardScaler())] if scale else [("identity", "passthrough")]
    ord_enc = OrdinalEncoder(
        categories=[C.ORDINAL_LEVELS["t_stage"], C.ORDINAL_LEVELS["n_stage"], C.ORDINAL_LEVELS["grade"]],
        handle_unknown="use_encoded_value", unknown_value=np.nan,
    )
    ord_steps = [("ordinal", ord_enc)] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUM_FEATURES),
            ("ord", Pipeline(ord_steps), ORD_FEATURES),
            ("bin", "passthrough", BIN_FEATURES),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None), NOM_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_preprocessor(scale: bool = True) -> Pipeline:
    """FeatureEngineer -> ColumnTransformer, ready to prepend to any model."""
    return Pipeline([("engineer", FeatureEngineer()), ("encode", make_column_transformer(scale))])


def split_xy(df: pd.DataFrame):
    """Inputs, survival time and event indicator from a cleaned frame."""
    return df[INPUT_COLS].copy(), df[C.TIME_COL].to_numpy(), df[C.EVENT_COL].to_numpy()
