"""Feature matrix, hyperparameter search, gradient-boosting models and
probability recalibration."""
from __future__ import annotations

import gc
import json

import joblib
import numpy as np
import pandas as pd

from utils import (FULL_FEATURES, GLOBAL_SEED, MODELS, REF_LABEL, Paths,
                   chrono_split, ece, load_features, nll)

N_TRIALS = 40


def present_features(df) -> list[str]:
    return [f for f in FULL_FEATURES if f in df.columns]


def make_model(kind, params, gpu=False, seed=GLOBAL_SEED, train_dir=None):
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, random_seed=seed, verbose=0,
                                  task_type="GPU" if gpu else "CPU",
                                  devices="0" if gpu else None,
                                  train_dir=str(train_dir) if train_dir else None,
                                  allow_writing_files=False)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, random_state=seed, n_jobs=-1,
                             tree_method="hist",
                             device="cuda" if gpu else "cpu",
                             eval_metric="logloss")
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, random_state=seed, n_jobs=-1,
                              deterministic=True, force_row_wise=True,
                              verbosity=-1)
    raise ValueError(f"unknown model: {kind}")


def make_xgb(params, gpu=False, seed=GLOBAL_SEED):
    """The workhorse model, used by most of the secondary analyses."""
    return make_model("xgboost", params, gpu=gpu, seed=seed)


def search_space(kind, trial):
    p = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3,
                                              log=True),
         "n_estimators": trial.suggest_int("n_estimators", 200, 800)}
    if kind == "catboost":
        p["depth"] = trial.suggest_int("depth", 4, 10)
        p["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
        p["iterations"] = p.pop("n_estimators")
    else:
        p["max_depth"] = trial.suggest_int("max_depth", 4, 10)
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree",
                                                    0.6, 1.0)
        p["reg_lambda"] = trial.suggest_float("reg_lambda", 1.0, 10.0)
        if kind == "lightgbm":
            p["num_leaves"] = trial.suggest_int("num_leaves", 31, 255)
    return p


def tune(P: Paths, cat, kind, Xtr, ytr, Xva, yva, gpu=False,
         n_trials=N_TRIALS, seed=GLOBAL_SEED):
    """Select hyperparameters on the validation block and cache them."""
    import optuna
    from sklearn.metrics import f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    path = P.models / f"{cat}__optuna_best.json"
    best = json.loads(path.read_text()) if path.exists() else {}
    if kind in best:
        return best[kind]

    def objective(trial):
        m = make_model(kind, search_space(kind, trial), gpu=gpu, seed=seed,
                       train_dir=P.ckpt / "catboost_info")
        m.fit(Xtr, ytr)
        return f1_score(yva, m.predict(Xva), zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best[kind] = study.best_params
    path.write_text(json.dumps(best, indent=2))
    return study.best_params


def tune_xgb(P: Paths, tag, Xtr, ytr, Xva, yva, gpu=False, n_trials=25,
             seed=GLOBAL_SEED):
    """A smaller search, used when a secondary analysis re-tunes."""
    import optuna
    from sklearn.metrics import f1_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ck = P.ckpt / f"{tag}__retune.json"
    if ck.exists():
        return json.loads(ck.read_text())

    def objective(trial):
        p = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3,
                                                 log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                    0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        }
        m = make_xgb(p, gpu=gpu, seed=seed)
        m.fit(Xtr, ytr)
        return f1_score(yva, m.predict(Xva), zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    ck.write_text(json.dumps(study.best_params, indent=2))
    return study.best_params


def best_params(P: Paths, cat, kind="xgboost"):
    return json.loads((P.models / f"{cat}__optuna_best.json").read_text())[kind]


def fit_imputer(X_train, seed=GLOBAL_SEED):
    """Iterative imputation, fitted on the training block only.

    keep_empty_features retains a column that is missing throughout the
    training block (it is imputed as a constant) so the transformed matrix
    always has the same columns as the input. Without it a catalog that lacks
    one field entirely comes back one column short.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    imp = IterativeImputer(max_iter=10, random_state=seed,
                           sample_posterior=False, keep_empty_features=True)
    imp.fit(X_train)
    return imp


def imputed_matrix(P: Paths, cat, df=None, feats=None):
    """Feature matrix for a catalog, using the imputer fitted during
    training."""
    if df is None:
        df = load_features(P, cat)
    if feats is None:
        feats = present_features(df)
    imp = joblib.load(P.models / f"{cat}__imputer.joblib")
    X = df[feats].astype(np.float32)
    return pd.DataFrame(imp.transform(X), columns=feats).astype(np.float32)


def fit_recalibrators(p_val, y_val):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize_scalar
    out = {}
    z = np.log(np.clip(p_val, 1e-7, 1 - 1e-7)
               / np.clip(1 - p_val, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    out["platt"] = ("platt", LogisticRegression(C=1e6).fit(z, y_val))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    out["isotonic"] = ("isotonic", iso)
    zv = z.ravel()
    res = minimize_scalar(
        lambda T: nll(y_val, 1.0 / (1.0 + np.exp(-zv / max(T, 1e-3)))),
        bounds=(0.05, 20.0), method="bounded")
    out["temperature"] = ("temperature", float(res.x))
    return out


def apply_recal(recalibrator, p):
    kind, obj = recalibrator
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1 - 1e-7))
    if kind == "platt":
        return obj.predict_proba(z.reshape(-1, 1))[:, 1]
    if kind == "isotonic":
        return obj.predict(p)
    return 1.0 / (1.0 + np.exp(-z / obj))


def build_models(P: Paths, cat, gpu=False, models=MODELS, seed=GLOBAL_SEED):
    """Train every model on one catalog and store raw and recalibrated
    probabilities for the whole series."""
    from sklearn.metrics import f1_score, roc_auc_score

    df = load_features(P, cat)
    feats = present_features(df)
    X = df[feats].astype(np.float32)
    y = df[REF_LABEL].to_numpy(np.int8)
    n = len(df)
    b1, b2, b3 = chrono_split(n)
    np.savez(P.models / f"{cat}__splits.npz", b1=b1, b2=b2, b3=b3, n=n)
    print(f"=== {cat}: n={n}, train {b1} / val {b2 - b1} / calib {b3 - b2} / "
          f"test {n - b3}, {len(feats)} features ===")

    imputer_path = P.models / f"{cat}__imputer.joblib"
    if imputer_path.exists():
        imp = joblib.load(imputer_path)
    else:
        imp = fit_imputer(X.iloc[:b1], seed=seed)
        joblib.dump(imp, imputer_path)
    Xi = pd.DataFrame(imp.transform(X), columns=feats).astype(np.float32)

    for kind in models:
        probs_path = P.models / f"{cat}__{kind}__probs.npz"
        if probs_path.exists():
            prob = np.load(probs_path)["prob"].astype(np.float64)
        else:
            print(f"  training {kind} ...")
            params = tune(P, cat, kind, Xi.iloc[:b1], y[:b1],
                          Xi.iloc[b1:b2], y[b1:b2], gpu=gpu, seed=seed)
            m = make_model(kind, params, gpu=gpu, seed=seed,
                           train_dir=P.ckpt / "catboost_info")
            m.fit(Xi.iloc[:b1], y[:b1])
            prob = m.predict_proba(Xi)[:, 1]
            np.savez(probs_path, prob=prob.astype(np.float32))
            del m
            gc.collect()

        recal_path = P.models / f"{cat}__{kind}__recal_probs.npz"
        if recal_path.exists():
            continue
        # The recalibrator is chosen on the calibration block, never on test.
        recals = fit_recalibrators(prob[b1:b2], y[b1:b2])
        best_name, best_e = None, np.inf
        for name, r in recals.items():
            e = ece(y[b2:b3], apply_recal(r, prob[b2:b3]))
            if e < best_e:
                best_name, best_e = name, e
        np.savez(recal_path,
                 prob=apply_recal(recals[best_name], prob).astype(np.float32),
                 method=np.array(best_name))
        print(f"  {kind}: test AUC {roc_auc_score(y[b3:], prob[b3:]):.4f} "
              f"F1 {f1_score(y[b3:], (prob[b3:] >= .5).astype(int)):.4f} | "
              f"recalibrator {best_name} (ECE {best_e:.4f})")

    del df, X, Xi
    gc.collect()


def train_all(P: Paths, gpu=False, models=MODELS, seed=GLOBAL_SEED):
    from utils import CATS
    for cat in CATS:
        build_models(P, cat, gpu=gpu, models=models, seed=seed)

    missing = []
    for cat in CATS:
        ok = all((P.models / f"{cat}__{m}__recal_probs.npz").exists()
                 for m in models)
        if not (ok and (P.models / f"{cat}__splits.npz").exists()):
            missing.append(cat)
    if missing:
        raise RuntimeError(
            "training did not complete for: " + ", ".join(missing)
            + "\nIf a boosting library failed to import, install "
              "requirements.txt and run again.")
