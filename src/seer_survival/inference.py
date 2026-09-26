"""Inference wrapper used by notebook 09 and the Streamlit app.

    >>> from seer_survival.inference import SeerPredictor
    >>> pred = SeerPredictor.load()
    >>> pred.predict(pd.DataFrame([patient_dict]))
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import INPUT_COLS

ALLOWED = {
    "race": ["White", "Black", "Other"],
    "marital_status": ["Married", "Single", "Divorced", "Widowed", "Separated"],
    "t_stage": C.ORDINAL_LEVELS["t_stage"],
    "n_stage": C.ORDINAL_LEVELS["n_stage"],
    "grade": C.ORDINAL_LEVELS["grade"],
    "a_stage": ["Regional", "Distant"],
    "estrogen_status": ["Positive", "Negative"],
    "progesterone_status": ["Positive", "Negative"],
}
NUMERIC = ["age", "tumor_size", "nodes_examined", "nodes_positive"]


def validate_input(df: pd.DataFrame, ranges: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Check columns, types and categories. Raise on errors; warn when outside the training range."""
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[INPUT_COLS].copy()
    errors, warns = [], []
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            errors.append(f"'{c}' must be numeric (rows {df.index[df[c].isna()].tolist()})")
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    for c, levels in ALLOWED.items():
        bad = ~df[c].isin(levels)
        if bad.any():
            errors.append(f"'{c}' has invalid values {sorted(df.loc[bad, c].astype(str).unique())}; "
                          f"allowed: {levels}")
    if not errors:
        if (df["nodes_positive"] > df["nodes_examined"]).any():
            errors.append("nodes_positive cannot exceed nodes_examined")
        if (df[NUMERIC] <= 0).any().any():
            errors.append("age, tumor_size, nodes_examined and nodes_positive must be > 0")
    if errors:
        raise ValueError("; ".join(errors))
    if ranges:
        for c, (lo, hi) in ranges.get("numeric", {}).items():
            if c in df and ((df[c] < lo) | (df[c] > hi)).any():
                warns.append(f"{c} outside training range [{lo:g}, {hi:g}] - prediction is an extrapolation")
    return df, warns


def from_raw_format(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either the original SEER.csv headers or the cleaned snake_case names.

    Extra columns (e.g. Status, Survival Months) are ignored for prediction.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in C.RAW_TO_CLEAN.items() if k in df.columns})
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()
    if "grade" in df and df["grade"].dtype == object:
        mapped = df["grade"].map(C.GRADE_MAP)
        df["grade"] = mapped.where(mapped.notna(), df["grade"])
    if "race" in df:
        df["race"] = df["race"].replace(C.RACE_SHORT)
    if "marital_status" in df:
        df["marital_status"] = df["marital_status"].replace(C.MARITAL_SHORT)
    return df


@dataclass
class SeerPredictor:
    classifier: object
    survival: object
    metadata: dict

    @classmethod
    def load(cls, models_dir: str | Path = C.MODELS_DIR) -> "SeerPredictor":
        models_dir = Path(models_dir)
        with open(models_dir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            classifier=joblib.load(models_dir / "mortality_5y_classifier.joblib"),
            survival=joblib.load(models_dir / "survival_model.joblib"),
            metadata=meta,
        )

    # ------------------------------------------------------------------
    def risk_group(self, risk: np.ndarray) -> np.ndarray:
        lo, hi = self.metadata["risk_group_cutoffs"]
        return np.select([risk < lo, risk < hi], ["Low", "Intermediate"], default="High")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X, warns = validate_input(df, self.metadata.get("training_ranges"))
        risk = self.classifier.predict_proba(X)[:, 1]
        times = np.asarray(C.EVAL_TIMES, dtype=float)
        surv = np.vstack([fn(times) for fn in self.survival.predict_survival_function(X)])
        out = pd.DataFrame({
            "risk_5y": risk,
            "risk_group": self.risk_group(risk),
            "flag_high_risk": (risk >= self.metadata["threshold"]).astype(int),
        }, index=df.index)
        for i, t in enumerate(C.EVAL_TIMES):
            out[f"survival_{t}m"] = surv[:, i]
        out.attrs["warnings"] = warns
        return out

    def survival_curve(self, row: pd.DataFrame, grid=None) -> pd.DataFrame:
        X, _ = validate_input(row)
        fn = self.survival.predict_survival_function(X.iloc[[0]])[0]
        grid = np.arange(1, int(fn.x.max()) + 1) if grid is None else np.asarray(grid)
        grid = grid[(grid >= fn.x.min()) & (grid <= fn.x.max())]
        return pd.DataFrame({"month": grid, "survival": fn(grid)})

    def explain(self, row: pd.DataFrame) -> pd.DataFrame:
        """Change in 5-year risk when each input is set to a typical value.

        Positive means the patient's value raises her risk. Interactions are ignored.
        """
        X, _ = validate_input(row)
        X = X.iloc[[0]]
        ref = self.metadata["reference_patient"]
        base = float(self.classifier.predict_proba(X)[:, 1][0])
        rows = []
        for c in INPUT_COLS:
            Xr = X.copy()
            Xr[c] = ref[c]
            if c == "nodes_positive" and Xr["nodes_positive"].iloc[0] > Xr["nodes_examined"].iloc[0]:
                Xr["nodes_positive"] = Xr["nodes_examined"]
            r = float(self.classifier.predict_proba(Xr)[:, 1][0])
            rows.append({"feature": c, "value": X[c].iloc[0], "reference": ref[c], "contribution": base - r})
        return pd.DataFrame(rows).sort_values("contribution", key=np.abs, ascending=False)
