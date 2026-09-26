"""Train and save both models.

    python scripts/train.py
    python scripts/train.py --fast   # fewer models (overwrites models/ and reports/tables/)

Split -> 5-fold CV -> one-SE selection -> refit on train -> single test evaluation -> save.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seer_survival import config as C  # noqa: E402
from seer_survival.data import load_clean  # noqa: E402
from seer_survival.evaluate import (bootstrap_ci, classification_metrics, km_baseline_ibs,  # noqa: E402
                                    survival_metrics, youden_threshold)
from seer_survival.features import INPUT_COLS  # noqa: E402
from seer_survival.models import classifier_zoo, survival_zoo  # noqa: E402
from seer_survival.targets import make_surv  # noqa: E402
from seer_survival.training import (cv_classifiers, cv_survival, fit_final_classifier,  # noqa: E402
                                    fit_final_survival, horizon_subset, make_split,
                                    save_artifacts, select_one_se, summarise_cv, training_ranges)
from sklearn.metrics import roc_auc_score  # noqa: E402

warnings.filterwarnings("ignore")


def main(fast: bool = False) -> dict:
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    s = make_split(df)
    print(f"train={len(s['X_train'])}  test={len(s['X_test'])}  "
          f"events train/test={s['e_train'].sum()}/{s['e_test'].sum()}")

    # ---------------- Task B: 5-year mortality classifier -----------------
    Xtr_h, ytr_h, _ = horizon_subset(s["X_train"], s["t_train"], s["e_train"])
    Xte_h, yte_h, _ = horizon_subset(s["X_test"], s["t_test"], s["e_test"])
    zoo = classifier_zoo()
    if fast:
        zoo = {k: v for k, v in zoo.items() if k in ("LogReg (L2)", "LogReg (L1)", "XGBoost")}
    cls_scores, oof = cv_classifiers(Xtr_h, ytr_h, zoo)
    cls_summary = summarise_cv(cls_scores, "roc_auc")
    cls_summary.to_csv(C.TABLES_DIR / "cv_classification_summary.csv")
    best_cls = select_one_se(cls_summary)
    print("\nCV ROC-AUC (5-year mortality)\n", cls_summary.round(4))
    print("selected:", best_cls)

    thr = youden_threshold(ytr_h, oof[best_cls])
    cutoffs = np.quantile(oof[best_cls], [1 / 3, 2 / 3]).tolist()
    clf = fit_final_classifier(best_cls, Xtr_h, ytr_h)
    p_test = clf.predict_proba(Xte_h)[:, 1]
    cls_test = classification_metrics(yte_h, p_test, threshold=thr)
    auc, lo, hi = bootstrap_ci(roc_auc_score, yte_h, p_test)
    cls_test.update({"roc_auc_ci95": [lo, hi], "n_test": int(len(yte_h)), "events_test": int(yte_h.sum())})

    # ---------------- Task A: survival model --------------------------------
    szoo = survival_zoo()
    if fast:
        szoo = {k: v for k, v in szoo.items() if k in ("Cox PH (ridge)", "Random Survival Forest")}
    surv_scores = cv_survival(s["X_train"], s["t_train"], s["e_train"], szoo)
    surv_summary = summarise_cv(surv_scores, "c_index")
    surv_summary.to_csv(C.TABLES_DIR / "cv_survival_summary.csv")
    best_surv = select_one_se(surv_summary)
    print("\nCV C-index (survival)\n", surv_summary.round(4))
    print("selected:", best_surv)

    surv = fit_final_survival(best_surv, s["X_train"], s["t_train"], s["e_train"])
    y_tr, y_te = make_surv(s["t_train"], s["e_train"]), make_surv(s["t_test"], s["e_test"])
    times = np.asarray(C.EVAL_TIMES, dtype=float)
    surv_test = survival_metrics(surv, s["X_test"], y_tr, y_te, times)
    surv_test["ibs_km_reference"] = km_baseline_ibs(y_tr, y_te, times)

    # ---------------- Save --------------------------------------------------
    Xtr = s["X_train"]
    reference = {c: (float(Xtr[c].median()) if Xtr[c].dtype.kind in "if" else Xtr[c].mode()[0])
                 for c in INPUT_COLS}
    reference["grade"] = int(Xtr["grade"].mode()[0])
    meta = {
        "classifier_name": best_cls,
        "survival_model_name": best_surv,
        "horizon_months": C.HORIZON_MONTHS,
        "threshold": thr,
        "threshold_rule": "Youden J on 5-fold out-of-fold training predictions",
        "risk_group_cutoffs": cutoffs,
        "risk_group_rule": "tertiles of out-of-fold training risk",
        "training_ranges": training_ranges(Xtr),
        "reference_patient": reference,
        "test_metrics_classifier": cls_test,
        "test_metrics_survival": surv_test,
        "cv_classification": cls_summary.reset_index().to_dict(orient="records"),
        "cv_survival": surv_summary.reset_index().to_dict(orient="records"),
        "n_train": int(len(Xtr)), "n_test": int(len(s["X_test"])),
        "random_state": C.RANDOM_STATE,
    }
    save_artifacts(clf, surv, meta)
    with open(C.TABLES_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"classifier": cls_test, "survival": surv_test}, f, indent=2, default=float)
    print("\nTest (held-out) classifier:", {k: (round(float(v), 4) if isinstance(v, float) else v)
                                            for k, v in cls_test.items()})
    print("Test (held-out) survival:  ", {k: round(float(v), 4) for k, v in surv_test.items()})
    print(f"\nSaved models to {C.MODELS_DIR}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="only a few quick models")
    main(**vars(ap.parse_args()))
