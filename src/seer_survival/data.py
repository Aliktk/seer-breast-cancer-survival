"""Loading, cleaning and validating the SEER breast cancer extract."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(path: str | Path = C.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as shipped (no cleaning)."""
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# AJCC 6th edition stage grouping (only the groups present in this cohort)
# --------------------------------------------------------------------------
def derive_stage_6th(t_stage: str, n_stage: str) -> str:
    """AJCC 6th-edition stage group from T and N (M0, N1+ as in this cohort)."""
    if n_stage == "N3":
        return "IIIC"
    if t_stage == "T4":
        return "IIIB"
    if n_stage == "N2":
        return "IIIA"
    # remaining: N1 with T1-T3
    return {"T1": "IIA", "T2": "IIB", "T3": "IIIA"}[t_stage]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean(df_raw: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
    """Strip headers, drop the empty column, rename, map grade/labels and add `event`.

    Nothing is imputed (no values are missing). Clinically odd values are flagged
    by `consistency_flags`, not edited.
    """
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols)

    unknown = set(df.columns) - set(C.RAW_TO_CLEAN)
    if unknown:
        raise ValueError(f"Unexpected columns in raw file: {sorted(unknown)}")
    df = df.rename(columns=C.RAW_TO_CLEAN)

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()

    df["grade"] = df["grade"].map(C.GRADE_MAP).astype("int64")
    df["race"] = df["race"].map(C.RACE_SHORT)
    df["marital_status"] = df["marital_status"].map(C.MARITAL_SHORT)
    df[C.EVENT_COL] = (df[C.STATUS_COL] == "Dead").astype("int64")

    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Unmapped category values produced NaN in: {bad}")

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    n_violations: int
    detail: str = ""


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, mask_or_bool, detail: str = "") -> None:
        if isinstance(mask_or_bool, (pd.Series, np.ndarray)):
            n = int(np.asarray(mask_or_bool).sum())
            self.checks.append(CheckResult(name, n == 0, n, detail))
        else:
            self.checks.append(CheckResult(name, bool(mask_or_bool), 0 if mask_or_bool else 1, detail))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks])

    @property
    def hard_failures(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed and c.name.startswith("[hard]")]


def validate(df: pd.DataFrame) -> ValidationReport:
    """Schema, range and consistency checks on a cleaned frame.

    `[hard]` failures make the data unusable; `[soft]` ones are reported only.
    """
    r = ValidationReport()
    expected = set(C.RAW_TO_CLEAN.values()) | {C.EVENT_COL}
    r.add("[hard] schema matches", set(df.columns) == expected,
          f"missing={expected - set(df.columns)}, extra={set(df.columns) - expected}")
    r.add("[hard] no missing values", df.isna().any(axis=1))
    r.add("[hard] age within 18-100", ~df["age"].between(18, 100))
    r.add("[hard] tumor_size > 0", df["tumor_size"] <= 0)
    r.add("[hard] nodes_examined >= 1", df["nodes_examined"] < 1)
    r.add("[hard] nodes_positive <= nodes_examined", df["nodes_positive"] > df["nodes_examined"])
    r.add("[hard] survival_months >= 1", df["survival_months"] < 1)
    for col, levels in C.ORDINAL_LEVELS.items():
        r.add(f"[hard] {col} in allowed levels", ~df[col].isin(levels))

    derived = [derive_stage_6th(t, n) for t, n in zip(df["t_stage"], df["n_stage"])]
    r.add("[soft] stage_6th equals AJCC(T, N)", df["stage_6th"].to_numpy() != np.array(derived),
          "stage_6th is fully determined by T and N -> redundant feature")

    flags = consistency_flags(df)
    r.add("[soft] T1 implies size <= 20 mm", flags["flag_t1_size"])
    r.add("[soft] T2 implies 20 < size <= 50 mm", flags["flag_t2_size"])
    r.add("[soft] T3 implies size > 50 mm", flags["flag_t3_size"])
    r.add("[soft] N stage agrees with positive-node count", flags["flag_n_count"],
          "pN1: 1-3, pN2: 4-9, pN3: >=10 axillary nodes (AJCC 6th)")
    r.add("[soft] no exact duplicate rows", df.duplicated())
    return r


def consistency_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where T does not match tumour size or N does not match the node count.

    Cut-offs follow AJCC 6th ed. SEER staging can use information not in these
    columns, so a flag is not proof of an error.
    """
    size, t = df["tumor_size"], df["t_stage"]
    pos, n = df["nodes_positive"], df["n_stage"]
    out = pd.DataFrame(index=df.index)
    out["flag_t1_size"] = (t == "T1") & (size > 20)
    out["flag_t2_size"] = (t == "T2") & ~size.between(21, 50)
    out["flag_t3_size"] = (t == "T3") & (size <= 50)
    out["flag_n_count"] = (
        ((n == "N1") & (pos > 3))
        | ((n == "N2") & ~pos.between(4, 9))
        | ((n == "N3") & (pos < 10))
    )
    return out


def load_clean(path: str | Path = C.RAW_DATA_PATH) -> pd.DataFrame:
    """Shortcut used by the later notebooks and the training script."""
    return clean(load_raw(path))
