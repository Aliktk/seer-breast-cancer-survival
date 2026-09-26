"""Metrics for the 5-year classifier and the survival models."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, roc_auc_score, roc_curve,
)
from sksurv.metrics import (
    concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
    integrated_brier_score,
)

from .config import RANDOM_STATE


# ==========================================================================
# Classification
# ==========================================================================
def youden_threshold(y_true, p) -> float:
    """Threshold maximising sensitivity + specificity - 1 (fit on training OOF)."""
    fpr, tpr, thr = roc_curve(y_true, p)
    return float(thr[np.argmax(tpr - fpr)])


def calibration_intercept_slope(y_true, p, eps: float = 1e-6) -> tuple[float, float]:
    """Calibration intercept and slope from logit(P(y=1)) = a + b * logit(p).

    Ideal values: a = 0, b = 1.
    """
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p))
    slope = LogisticRegression(C=1e6, max_iter=10000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp)))
        grad = np.sum(y - mu)
        hess = -np.sum(mu * (1 - mu))
        step = grad / hess
        a -= step
        if abs(step) < 1e-10:
            break
    return float(a), float(slope)


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    """Quantile-binned ECE (weighted mean |observed - predicted|)."""
    y = np.asarray(y_true)
    p = np.asarray(p)
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    idx = np.digitize(p, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(y_true, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(p)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    a, b = calibration_intercept_slope(y, p)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "ece": expected_calibration_error(y, p),
        "cal_intercept": a,
        "cal_slope": b,
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "f1": f1_score(y, yhat, zero_division=0),
        "balanced_acc": np.nanmean([sens, spec]),
        "accuracy": (tp + tn) / len(y),
    }


def bootstrap_ci(metric: Callable, y_true, p, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = RANDOM_STATE) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI for ``metric(y, p)``."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(p)
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap_diff(metric: Callable, y_true, p_a, p_b, n_boot: int = 1000,
                          seed: int = RANDOM_STATE) -> dict:
    """Bootstrap distribution of metric(A) - metric(B) on the same patients."""
    rng = np.random.default_rng(seed)
    y, pa, pb = map(np.asarray, (y_true, p_a, p_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], pa[idx]) - metric(y[idx], pb[idx]))
    diffs = np.array(diffs)
    return {
        "diff": float(metric(y, pa) - metric(y, pb)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }


def net_benefit(y_true, p, thresholds) -> pd.DataFrame:
    """Decision-curve analysis (Vickers & Elkin, Med Decis Making 2006)."""
    y = np.asarray(y_true)
    p = np.asarray(p)
    n = len(y)
    prev = y.mean()
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        w = t / (1 - t)
        rows.append({
            "threshold": t,
            "model": tp / n - fp / n * w,
            "treat_all": prev - (1 - prev) * w,
            "treat_none": 0.0,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# Survival
# ==========================================================================
def survival_metrics(model, X_test, y_train_surv, y_test_surv, times) -> dict:
    """Harrell's and Uno's C-index, time-dependent AUC and integrated Brier score.

    `times` must lie inside the test follow-up range.
    """
    risk = model.predict(X_test)
    ev, tm = y_test_surv["event"], y_test_surv["time"]
    harrell = concordance_index_censored(ev, tm, risk)[0]
    tau = float(np.max(times))
    uno = concordance_index_ipcw(y_train_surv, y_test_surv, risk, tau=tau)[0]
    auc_t, mean_auc = cumulative_dynamic_auc(y_train_surv, y_test_surv, risk, times)

    surv_fns = model.predict_survival_function(X_test)
    surv_mat = np.vstack([fn(times) for fn in surv_fns])
    ibs = integrated_brier_score(y_train_surv, y_test_surv, surv_mat, times)
    out = {"c_harrell": harrell, "c_uno": uno, "mean_td_auc": mean_auc, "ibs": ibs}
    for t, a in zip(times, auc_t):
        out[f"td_auc_{int(t)}m"] = a
    return out


def km_baseline_ibs(y_train_surv, y_test_surv, times) -> float:
    """IBS of the no-covariate Kaplan-Meier model (reference for 'skill')."""
    from sksurv.nonparametric import kaplan_meier_estimator

    t, s = kaplan_meier_estimator(y_train_surv["event"], y_train_surv["time"])
    idx = np.searchsorted(t, times, side="right") - 1   # right-continuous step function
    s_at = np.where(idx >= 0, s[np.clip(idx, 0, None)], 1.0)
    surv_mat = np.tile(s_at, (len(y_test_surv), 1))
    return float(integrated_brier_score(y_train_surv, y_test_surv, surv_mat, times))
