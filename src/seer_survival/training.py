"""Cross-validation, model selection and final fitting (used by notebooks and scripts/train.py)."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from . import config as C
from .features import INPUT_COLS, split_xy
from .models import (classifier_zoo, make_classifier_pipeline, make_survival_pipeline,
                     survival_zoo)
from .targets import horizon_label, make_surv

# Lower number = simpler / more interpretable. Used for the 1-SE rule.
COMPLEXITY = {
    "LogReg (L1)": 1, "LogReg (L2)": 1, "Cox PH (ridge)": 1, "Coxnet (elastic net)": 1,
    "SVM (RBF)": 3, "MLP": 3, "Random Forest": 2, "Random Survival Forest": 2,
    "HistGradBoost": 2, "XGBoost": 2, "LightGBM": 2, "Gradient Boosted Survival": 2,
}


# --------------------------------------------------------------------------
# Data split (shared by both tasks so the test patients are identical)
# --------------------------------------------------------------------------
def make_split(df: pd.DataFrame, test_size: float = C.TEST_SIZE, seed: int = C.RANDOM_STATE):
    X, t, e = split_xy(df)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size,
                                      stratify=e, random_state=seed)
    return {
        "X_train": X.iloc[idx_tr].reset_index(drop=True), "X_test": X.iloc[idx_te].reset_index(drop=True),
        "t_train": t[idx_tr], "t_test": t[idx_te], "e_train": e[idx_tr], "e_test": e[idx_te],
        "idx_train": idx_tr, "idx_test": idx_te,
    }


# --------------------------------------------------------------------------
# Task B: classification
# --------------------------------------------------------------------------
def cv_classifiers(X, y, zoo: dict | None = None, n_splits: int = C.N_SPLITS_CV,
                   seed: int = C.RANDOM_STATE):
    """Stratified K-fold CV. Returns (per-fold scores, out-of-fold probabilities)."""
    zoo = zoo or classifier_zoo()
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    rows, oof = [], {}
    for name, (est, scale) in zoo.items():
        p_oof = np.zeros(len(y))
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            pipe = make_classifier_pipeline(clone(est), scale)
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            p_oof[va] = p
            rows.append({"model": name, "fold": fold, "roc_auc": roc_auc_score(y[va], p),
                         "brier": brier_score_loss(y[va], p)})
        oof[name] = p_oof
    return pd.DataFrame(rows), pd.DataFrame(oof)


def cv_survival(X, time, event, zoo: dict | None = None, times=C.EVAL_TIMES,
                n_splits: int = C.N_SPLITS_CV, seed: int = C.RANDOM_STATE):
    zoo = zoo or survival_zoo()
    times = np.asarray(times, dtype=float)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, (est, scale) in zoo.items():
        for fold, (tr, va) in enumerate(cv.split(X, event)):
            y_tr, y_va = make_surv(time[tr], event[tr]), make_surv(time[va], event[va])
            pipe = make_survival_pipeline(clone(est), scale).fit(X.iloc[tr], y_tr)
            risk = pipe.predict(X.iloc[va])
            c = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]
            surv = np.vstack([fn(times) for fn in pipe.predict_survival_function(X.iloc[va])])
            ibs = integrated_brier_score(y_tr, y_va, surv, times)
            rows.append({"model": name, "fold": fold, "c_index": c, "ibs": ibs})
    return pd.DataFrame(rows)


def summarise_cv(scores: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
    g = scores.groupby("model")[metric]
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n_folds": g.size()})
    out["se"] = out["sd"] / np.sqrt(out["n_folds"])
    out["complexity"] = out.index.map(COMPLEXITY).fillna(9).astype(int)
    return out.sort_values("mean", ascending=not higher_is_better)


def select_one_se(summary: pd.DataFrame, higher_is_better: bool = True) -> str:
    """Simplest model whose mean score is within one SE of the best (ESL, section 7.10)."""
    best = summary["mean"].idxmax() if higher_is_better else summary["mean"].idxmin()
    bound = summary.loc[best, "mean"] - summary.loc[best, "se"] if higher_is_better \
        else summary.loc[best, "mean"] + summary.loc[best, "se"]
    ok = summary[summary["mean"] >= bound] if higher_is_better else summary[summary["mean"] <= bound]
    ok = ok.sort_values(["complexity", "mean"], ascending=[True, not higher_is_better])
    return ok.index[0]


# --------------------------------------------------------------------------
# Final fit + artefacts
# --------------------------------------------------------------------------
def fit_final_classifier(name: str, X, y):
    est, scale = classifier_zoo()[name]
    return make_classifier_pipeline(clone(est), scale).fit(X, y)


def fit_final_survival(name: str, X, time, event):
    est, scale = survival_zoo()[name]
    return make_survival_pipeline(clone(est), scale).fit(X, make_surv(time, event))


def training_ranges(X: pd.DataFrame) -> dict:
    num = {c: [float(X[c].min()), float(X[c].max())] for c in X.select_dtypes("number")}
    cat = {c: sorted(map(str, X[c].unique())) for c in X.columns if c not in num}
    return {"numeric": num, "categorical": cat}


def save_artifacts(clf, surv, metadata: dict, models_dir=C.MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, models_dir / "mortality_5y_classifier.joblib", compress=3)
    joblib.dump(surv, models_dir / "survival_model.joblib", compress=3)
    metadata = {
        **metadata,
        "input_columns": INPUT_COLS,
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def horizon_subset(X, time, event, horizon: int = C.HORIZON_MONTHS):
    y, known = horizon_label(time, event, horizon)
    return X[known].reset_index(drop=True), y[known], known
