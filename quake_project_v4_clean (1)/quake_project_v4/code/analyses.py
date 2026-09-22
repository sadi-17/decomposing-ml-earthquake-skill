"""The scientific analyses. Each function returns one or more tables; the
caller decides where they are written."""
from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

import etas
from models import (best_params, fit_imputer, imputed_matrix, make_xgb,
                    present_features, tune_xgb, fit_recalibrators, apply_recal)
from utils import (ALPHA, ALPHAS, BATCH, CATS, FULL_FEATURES, GLOBAL_SEED,
                   MODELS, N_BINS, N_BOOT, RADII_KM, REF_LABEL, REF_R, REF_T,
                   T_MAX_DAYS, WINDOWS_DAYS, Paths, aci_trace, aki_b,
                   bh_adjust, chord_from_km, decide, decide_argmax,
                   delong_test, ece, load_features, load_probs,
                   load_splits, mc_maxc, mondrian_thresholds,
                   prediction_sets, set_metrics, split_threshold, to_xyz)

SEED = GLOBAL_SEED


# Shared geometry: which events have a larger neighbour before and after
def directional_flags(times_s, mw, tree, xyz, R_km, T_days):
    """Separately: is there a larger event within R km in the T days before,
    and in the T days after? The windowed label is the OR of the two."""
    n = len(mw)
    before = np.zeros(n, dtype=bool)
    after = np.zeros(n, dtype=bool)
    T_s = T_days * 86400.0
    chord = chord_from_km(R_km)
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        neigh = tree.query_ball_point(xyz[lo:hi], r=chord)
        for k in range(hi - lo):
            i = lo + k
            nb = np.asarray(neigh[k])
            neigh[k] = None
            nb = nb[nb != i]
            if nb.size == 0:
                continue
            dt = times_s[nb] - times_s[i]
            within = np.abs(dt) <= T_s
            nb, dt = nb[within], dt[within]
            if nb.size == 0:
                continue
            bigger = mw[nb] > mw[i]
            if bigger.any():
                before[i] = bool((bigger & (dt < 0)).any())
                after[i] = bool((bigger & (dt > 0)).any())
        del neigh
        if (lo // BATCH) % 200 == 0:
            gc.collect()
    return before, after


def get_directional(P: Paths, cat, R_km=REF_R, T_days=REF_T):
    ck = P.ckpt / f"{cat}__directional_R{int(R_km)}_T{int(T_days)}.npz"
    if ck.exists():
        d = np.load(ck)
        return d["before"], d["after"]
    df = load_features(P, cat, ["time", "mw", "latitude", "longitude"])
    times_s = ((df["time"] - pd.Timestamp(0, tz="UTC"))
               .dt.total_seconds().to_numpy())
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    before, after = directional_flags(times_s, df["mw"].to_numpy(float),
                                      cKDTree(xyz), xyz, R_km, T_days)
    np.savez(ck, before=before, after=after)
    del df, xyz
    gc.collect()
    return before, after


# Calibration
def reliability_points(y, p, n_bins=N_BINS):
    order = np.argsort(p)
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    rows = []
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b):
            rows.append({"mean_pred": float(p[b].mean()),
                         "frac_pos": float(y[b].mean()), "count": int(len(b))})
    return rows


def calibration_audit(P: Paths):
    """Discrimination and calibration for every catalog, model and
    recalibrator, plus reliability points and the ACI coverage trace."""
    metric_rows, rel_rows, aci_rows = [], [], []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        y = load_features(P, cat, [REF_LABEL])[REF_LABEL].to_numpy(np.int8)
        for model in MODELS:
            raw_path = P.models / f"{cat}__{model}__probs.npz"
            if not raw_path.exists():
                continue
            p = np.load(raw_path)["prob"].astype(np.float64)
            y_te = y[b3:]

            recals = fit_recalibrators(p[b1:b2], y[b1:b2])
            variants = {"raw": p[b3:]}
            for name, r in recals.items():
                variants[name] = apply_recal(r, p[b3:])

            for name, pv in variants.items():
                pred = (pv >= 0.5).astype(np.int8)
                metric_rows.append({
                    "catalog": cat, "model": model, "calibrator": name,
                    "n_test": int(len(y_te)),
                    "auc": float(roc_auc_score(y_te, pv)),
                    "f1": float(f1_score(y_te, pred, zero_division=0)),
                    "precision": float(precision_score(y_te, pred,
                                                       zero_division=0)),
                    "recall": float(recall_score(y_te, pred, zero_division=0)),
                    "ece": ece(y_te, pv),
                    "brier": float(np.mean((pv - y_te) ** 2)),
                    "nll": float(-np.mean(
                        y_te * np.log(np.clip(pv, 1e-7, 1 - 1e-7))
                        + (1 - y_te) * np.log(np.clip(1 - pv, 1e-7, 1 - 1e-7)))),
                })
                for pt in reliability_points(y_te, pv):
                    rel_rows.append({"catalog": cat, "model": model,
                                     "calibrator": name, **pt})

            # Adaptive conformal inference against a fixed split threshold.
            p_recal = load_probs(P, cat, model)
            cov_aci, _ = aci_trace(p_recal[b3:], y_te, p_recal[b2:b3],
                                   y[b2:b3])
            qfix = split_threshold(
                np.where(y[b2:b3] == 1, 1 - p_recal[b2:b3], p_recal[b2:b3]),
                ALPHA)
            in0, in1 = prediction_sets(p_recal[b3:], qfix)
            cov_fix = np.where(y_te == 1, in1, in0).astype(float)
            roll = lambda x: pd.Series(x).rolling(  # noqa: E731
                2000, min_periods=200).mean().to_numpy()
            r_aci, r_fix = roll(cov_aci), roll(cov_fix)
            step = max(1, len(y_te) // 400)
            for t in range(0, len(y_te), step):
                aci_rows.append({"catalog": cat, "model": model, "t": t,
                                 "rolling_cov_split": r_fix[t],
                                 "rolling_cov_aci": r_aci[t]})
            print(f"  {cat}/{model}: ACI mean coverage {cov_aci.mean():.3f} "
                  f"vs fixed {cov_fix.mean():.3f}")
    return (pd.DataFrame(metric_rows).round(5),
            pd.DataFrame(rel_rows).round(5),
            pd.DataFrame(aci_rows).round(4))


# Label-definition sensitivity sweep
def kendall_tau(rank_a, rank_b):
    from scipy.stats import kendalltau
    common = [f for f in rank_a if f in rank_b]
    if len(common) < 3:
        return np.nan
    ra = [rank_a.index(f) for f in common]
    rb = [rank_b.index(f) for f in common]
    return float(kendalltau(ra, rb).statistic)


def all_label_cols(df):
    cols = [f"label_R{R}_T{T}" for R in RADII_KM for T in WINDOWS_DAYS]
    cols.append("label_adaptive")
    return [c for c in cols if c in df.columns]


def _sweep_one(P, cat, col, Xi, labels, feats, splits, params, gpu):
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    ck = P.ckpt / f"sweep__{cat}__{col}.json"
    if ck.exists():
        return json.loads(ck.read_text())

    b1, b2, b3 = splits
    y = labels[col].to_numpy(np.int8)
    if len(np.unique(y[:b1])) < 2 or len(np.unique(y[b3:])) < 2:
        return None

    m = make_xgb(params, gpu=gpu)
    m.fit(Xi.iloc[:b1], y[:b1])
    prob = m.predict_proba(Xi)[:, 1].astype(np.float64)
    p_te, yte = prob[b3:], y[b3:]
    pred = (p_te >= 0.5).astype(int)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(prob[b1:b2], y[b1:b2])
    p_cal_r, p_te_r = iso.predict(prob[b2:b3]), iso.predict(p_te)
    q0, q1 = mondrian_thresholds(p_cal_r, y[b2:b3], ALPHA)
    conf = set_metrics(yte, p_te_r <= q0, (1 - p_te_r) <= q1)

    sub = Xi.iloc[b3:].iloc[:20000]
    contribs = m.get_booster().predict(xgb.DMatrix(sub), pred_contribs=True)
    ranking = [feats[i] for i in np.argsort(-np.abs(contribs[:, :-1]).mean(0))]

    row = {"catalog": cat, "config": col,
           "mainshock_rate": float(y.mean()),
           "auc": float(roc_auc_score(yte, p_te)),
           "f1": float(f1_score(yte, pred, zero_division=0)),
           "precision": float(precision_score(yte, pred, zero_division=0)),
           "recall": float(recall_score(yte, pred, zero_division=0)),
           "accuracy": float(accuracy_score(yte, pred)),
           "brier": float(np.mean((p_te - yte) ** 2)),
           "ece_raw": ece(yte, p_te), "ece_recal": ece(yte, p_te_r),
           "coverage": conf["coverage"], "avg_set_size": conf["avg_set_size"],
           "rate_empty": conf["rate_empty"],
           "rate_singleton": conf["rate_singleton"],
           "rate_both": conf["rate_both"],
           "shap_ranking": ranking, "shap_top1": ranking[0]}
    ck.write_text(json.dumps(row))
    del m, prob
    gc.collect()
    return row


def sensitivity_sweep(P: Paths, gpu=False):
    """Every label definition in the grid, one model per definition."""
    rows, shap_rows = [], []
    for cat in CATS:
        splits = load_splits(P, cat)
        df = load_features(P, cat)
        feats = present_features(df)
        Xi = imputed_matrix(P, cat, df, feats)
        cols = all_label_cols(df)
        labels = df[cols]
        ref_y = labels[REF_LABEL].to_numpy(np.int8)
        params = best_params(P, cat)

        cfg_rows = []
        for col in cols:
            r = _sweep_one(P, cat, col, Xi, labels, feats, splits, params, gpu)
            if r is None:
                print(f"  [skip] {cat}/{col}: labels are degenerate in a block")
                continue
            r["flip_rate_vs_ref"] = float(
                (labels[col].to_numpy(np.int8) != ref_y).mean())
            cfg_rows.append(r)

        ref = next((r for r in cfg_rows if r["config"] == REF_LABEL), None)
        ref_rank = ref["shap_ranking"] if ref else []
        for r in cfg_rows:
            r["shap_kendall_tau_vs_ref"] = kendall_tau(r["shap_ranking"],
                                                       ref_rank)
            r["shap_top5_overlap"] = (
                len(set(r["shap_ranking"][:5]) & set(ref_rank[:5])) / 5.0
                if ref_rank else np.nan)
            shap_rows.append({"catalog": cat, "config": r["config"],
                              "ranking": " > ".join(r["shap_ranking"][:6]),
                              "kendall_tau_vs_ref": r["shap_kendall_tau_vs_ref"],
                              "top5_overlap": r["shap_top5_overlap"]})
            rows.append({k: v for k, v in r.items() if k != "shap_ranking"})
        s = pd.DataFrame(cfg_rows)
        print(f"  {cat}: AUC {s.auc.min():.3f}-{s.auc.max():.3f}, "
              f"F1 {s.f1.min():.3f}-{s.f1.max():.3f}, "
              f"labels flipped up to {s.flip_rate_vs_ref.max():.1%}")
        del df, Xi, labels
        gc.collect()
    return pd.DataFrame(rows).round(4), pd.DataFrame(shap_rows).round(4)


# Conformal prediction sets
def conformal_audit(P: Paths):
    rows = []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        y = load_features(P, cat, [REF_LABEL])[REF_LABEL].to_numpy(np.int8)
        for model in MODELS:
            if not (P.models / f"{cat}__{model}__recal_probs.npz").exists():
                continue
            p = load_probs(P, cat, model)
            p_cal, y_cal, p_te, y_te = p[b2:b3], y[b2:b3], p[b3:], y[b3:]

            for alpha in ALPHAS:
                qhat = split_threshold(
                    np.where(y_cal == 1, 1 - p_cal, p_cal), alpha)
                in0, in1 = prediction_sets(p_te, qhat)
                rows.append({"catalog": cat, "model": model, "method": "split",
                             "alpha": alpha, **set_metrics(y_te, in0, in1)})

                q0, q1 = mondrian_thresholds(p_cal, y_cal, alpha)
                rows.append({"catalog": cat, "model": model,
                             "method": "mondrian", "alpha": alpha,
                             **set_metrics(y_te, p_te <= q0,
                                           (1 - p_te) <= q1)})

            cov, size = aci_trace(p_te, y_te, p_cal, y_cal)
            rows.append({"catalog": cat, "model": model, "method": "aci",
                         "alpha": ALPHA,
                         "coverage": float(cov.mean()),
                         "cov_class1": float(cov[y_te == 1].mean()),
                         "cov_class0": float(cov[y_te == 0].mean()),
                         "avg_set_size": float(size.mean()),
                         "rate_empty": float((size == 0).mean()),
                         "rate_singleton": float((size == 1).mean()),
                         "rate_both": float((size == 2).mean()),
                         "identity_gap": 0.0})
            np.savez(P.ckpt / f"{cat}__{model}__aci_trace.npz", cov=cov,
                     size=size)
    out = pd.DataFrame(rows).round(4)
    head = out[(out.method == "mondrian") & (out.alpha == ALPHA)
               & (out.model == "xgboost")]
    for _, r in head.iterrows():
        print(f"  {r.catalog:<22} empty {r.rate_empty:.1%} | singleton "
              f"{r.rate_singleton:.1%} | both {r.rate_both:.1%} | coverage "
              f"{r.coverage:.3f}")
    return out


# Decision rule
STATE_TABLE = pd.DataFrame([
    {"prediction_set": "{1}", "in_0": False, "in_1": True,
     "state": "RESPONSE", "reading": "mainshock; begin search and rescue"},
    {"prediction_set": "{0}", "in_0": True, "in_1": False,
     "state": "ALERT",
     "reading": "not the mainshock; a larger event is possible"},
    {"prediction_set": "{0,1}", "in_0": True, "in_1": True,
     "state": "ESCALATE",
     "reading": "role undetermined; prepare for a larger event"},
    {"prediction_set": "{}", "in_0": False, "in_1": False,
     "state": "ESCALATE", "reading": "coverage failure; treat as undetermined"},
])


def decision_rule_comparison(P: Paths, models=("xgboost",), alphas=None):
    """Routing empty prediction sets to ESCALATE against resolving them by
    argmax. The two rules coincide wherever no empty sets occur."""
    alphas = alphas or sorted({0.02, ALPHA})
    rows = []
    for cat in CATS:
        for model in models:
            if not (P.models / f"{cat}__{model}__recal_probs.npz").exists():
                continue
            b1, b2, b3 = load_splits(P, cat)
            y = load_features(P, cat, [REF_LABEL])[REF_LABEL].to_numpy(np.int8)
            p = load_probs(P, cat, model)
            p_cal, y_cal, p_te, y_te = p[b2:b3], y[b2:b3], p[b3:], y[b3:]

            for alpha in alphas:
                q0, q1 = mondrian_thresholds(p_cal, y_cal, alpha)
                in0, in1 = p_te <= q0, (1 - p_te) <= q1
                size = in0.astype(int) + in1.astype(int)
                escalate_rule = decide(in0, in1)
                argmax_rule, empty = decide_argmax(p_te, in0, in1)

                for name, act in [("empty -> ESCALATE", escalate_rule),
                                  ("empty -> argmax", argmax_rule)]:
                    resp, alert = act == "RESPONSE", act == "ALERT"
                    rows.append({
                        "catalog": cat, "model": model, "alpha": alpha,
                        "n_test": int(len(y_te)),
                        "rate_empty": float((size == 0).mean()),
                        "rate_singleton": float((size == 1).mean()),
                        "rate_both": float((size == 2).mean()),
                        "n_empty": int(empty.sum()), "rule": name,
                        "P_RESPONSE": float(resp.mean()),
                        "P_ALERT": float(alert.mean()),
                        "P_ESCALATE": float((act == "ESCALATE").mean()),
                        # a false RESPONSE declares the sequence over when a
                        # larger event followed inside the labelling window
                        "P_false_RESPONSE": float((resp & (y_te == 0)).mean()),
                        "false_RESPONSE_given_RESPONSE":
                            float((y_te[resp] == 0).mean()) if resp.any()
                            else np.nan,
                        "P_false_ALERT": float((alert & (y_te == 1)).mean()),
                    })
    out = pd.DataFrame(rows).round(4)
    for (cat, alpha), g in out.groupby(["catalog", "alpha"]):
        esc = g[g.rule.str.endswith("ESCALATE")].iloc[0]
        arg = g[g.rule.str.endswith("argmax")].iloc[0]
        print(f"  {cat:<22} alpha {alpha:.2f}: empty {esc.rate_empty:.1%}, "
              f"ESCALATE {arg.P_ESCALATE:.1%} -> {esc.P_ESCALATE:.1%}")
    return out


ALPHA_GRID = np.round(np.arange(0.005, 0.31, 0.005), 3)
COST_FN = [5, 10, 50]


def decision_costs(P: Paths, cat="comcat_global_m4", model="xgboost"):
    """Expected cost against alpha when a false RESPONSE is priced at c_FN
    times a false ALERT, under both empty-set routings."""
    b1, b2, b3 = load_splits(P, cat)
    y = load_features(P, cat, [REF_LABEL])[REF_LABEL].to_numpy(np.int8)
    p = load_probs(P, cat, model)
    p_cal, y_cal, p_te, y_te = p[b2:b3], y[b2:b3], p[b3:], y[b3:]

    rows = []
    for alpha in ALPHA_GRID:
        q0, q1 = mondrian_thresholds(p_cal, y_cal, alpha)
        in0, in1 = p_te <= q0, (1 - p_te) <= q1
        for rule, act in [("escalate", decide(in0, in1)),
                          ("argmax", decide_argmax(p_te, in0, in1)[0])]:
            resp, alert = act == "RESPONSE", act == "ALERT"
            esc = act == "ESCALATE"
            fn = float((resp & (y_te == 0)).mean())
            fp = float((alert & (y_te == 1)).mean())
            ab = float(esc.mean())
            for c in COST_FN:
                rows.append({"catalog": cat, "rule": rule, "alpha": alpha,
                             "c_FN": c,
                             "P_false_RESPONSE": round(fn, 4),
                             "P_false_ALERT": round(fp, 4),
                             "P_ESCALATE": round(ab, 4),
                             "P_empty": round(float((~(in0 | in1)).mean()), 4),
                             "expected_cost": round(c * fn + fp + ab, 4)})
    costs = pd.DataFrame(rows)
    for rule in ("escalate", "argmax"):
        for c in COST_FN:
            s = costs[(costs.c_FN == c) & (costs.rule == rule)]
            b = s.loc[s.expected_cost.idxmin()]
            print(f"  {rule:<9} c_FN={c:>2}: alpha* {b.alpha:.3f}, cost "
                  f"{b.expected_cost:.4f}, escalate {b.P_ESCALATE:.1%}")
    return costs


# Baselines
def magnitude_baselines(P: Paths):
    """The magnitude-difference baselines the model has to beat.

    For a deterministic two-point score AUC equals (sensitivity +
    specificity)/2 exactly, so both the score and that identity are reported.
    """
    rows = []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat, [REF_LABEL, "mag_diff_30d"])
        y = df[REF_LABEL].to_numpy(np.int8)
        yte = y[b3:]
        md = df["mag_diff_30d"].to_numpy(float)[b3:]
        p_model = load_probs(P, cat, "xgboost")[b3:]

        hard = (md >= 0).astype(float)
        sens = float(hard[yte == 1].mean())
        spec = float(1.0 - hard[yte == 0].mean())

        for name, score, pred in [
                ("heuristic (hard verdict)", hard, hard.astype(int)),
                ("magdiff (continuous)", md, (md >= 0).astype(int)),
                ("xgboost", p_model, (p_model >= 0.5).astype(int))]:
            row = {"catalog": cat, "score": name,
                   "auc": roc_auc_score(yte, score),
                   "accuracy": accuracy_score(yte, pred),
                   "precision": precision_score(yte, pred, zero_division=0),
                   "recall": recall_score(yte, pred, zero_division=0),
                   "f1": f1_score(yte, pred, zero_division=0)}
            if name.startswith("heuristic"):
                row["auc_identity_sens_spec"] = (sens + spec) / 2.0
            rows.append(row)

        a1, a2, pv = delong_test(yte, p_model, md)
        rows.append({"catalog": cat, "score": "DeLong: xgboost vs magdiff",
                     "auc": a1, "auc_baseline": a2, "delong_p": pv})
        a1, a2, pv = delong_test(yte, p_model, hard)
        rows.append({"catalog": cat, "score": "DeLong: xgboost vs heuristic",
                     "auc": a1, "auc_baseline": a2, "delong_p": pv})
        print(f"  {cat:<22} model {roc_auc_score(yte, p_model):.4f} vs "
              f"magdiff {roc_auc_score(yte, md):.4f}")
    return pd.DataFrame(rows).round(5)


N_LOCAL = 50
MIN_EVENTS_B = 15
BACKGROUND_N = 500
FEAT_R_KM = 50.0


def local_bvalues(df):
    """Local b-value from the most recent neighbours, and a longer-window
    background b-value for the drop that motivates traffic-light schemes."""
    n = len(df)
    times = ((df["time"] - pd.Timestamp(0, tz="UTC"))
             .dt.total_seconds().to_numpy())
    mw = df["mw"].to_numpy(float)
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    tree = cKDTree(xyz)
    chord = chord_from_km(FEAT_R_KM)
    local_b = np.full(n, np.nan)
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        neigh = tree.query_ball_point(xyz[lo:hi], r=chord)
        for k in range(hi - lo):
            i = lo + k
            nb = np.asarray(neigh[k])
            neigh[k] = None
            nb = nb[nb < i]
            if len(nb) < MIN_EVENTS_B:
                continue
            nb = nb[np.argsort(times[nb])[-N_LOCAL:]]
            mags = mw[nb]
            local_b[i] = aki_b(mags, mags.min(), min_n=MIN_EVENTS_B)
        del neigh
    bg_b = np.full(n, np.nan)
    for i in range(n):
        mags = mw[max(0, i - BACKGROUND_N):i]
        if len(mags) >= MIN_EVENTS_B:
            bg_b[i] = aki_b(mags, mags.min(), min_n=MIN_EVENTS_B)
    return local_b, bg_b


def bvalue_baselines(P: Paths):
    from sklearn.linear_model import LogisticRegression
    rows = []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat, ["time", "mw", "latitude", "longitude",
                                    REF_LABEL])
        y = df[REF_LABEL].to_numpy(np.int8)
        ck = P.ckpt / f"{cat}__bvalues.npz"
        if ck.exists():
            d = np.load(ck)
            local_b, bg_b = d["local_b"], d["bg_b"]
        else:
            print(f"  computing local b-values for {cat} ...")
            local_b, bg_b = local_bvalues(df)
            np.savez(ck, local_b=local_b, bg_b=bg_b)
        drop = bg_b - local_b
        model_p = load_probs(P, cat, "xgboost")

        te = np.arange(b3, len(df))
        valid = te[np.isfinite(drop[te]) & np.isfinite(local_b[te])]
        if len(valid) < 100 or len(np.unique(y[valid])) < 2:
            print(f"  [skip] {cat}: b-value defined for too few test events")
            continue
        coverage = len(valid) / len(te)
        yv = y[valid]

        tr = np.arange(0, b1)
        tr = tr[np.isfinite(local_b[tr])]
        p_lr = np.full(len(valid), np.nan)
        if len(tr) > 200 and len(np.unique(y[tr])) == 2:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(local_b[tr].reshape(-1, 1), y[tr])
            p_lr = lr.predict_proba(local_b[valid].reshape(-1, 1))[:, 1]

        for name, score in [("b-value drop", drop[valid]),
                            ("b-value logistic", p_lr)]:
            if not np.isfinite(score).all():
                continue
            a_m, a_b, pv = delong_test(yv, model_p[valid], score)
            rows.append({"catalog": cat, "baseline": name,
                         "test_coverage": round(coverage, 3),
                         "auc_baseline": round(a_b, 4),
                         "auc_model_same_subset": round(a_m, 4),
                         "delong_p": pv})
            print(f"  {cat} | {name}: AUC {a_b:.3f} (model {a_m:.3f} on the "
                  f"same subset, p={pv:.2e})")
        del df
        gc.collect()
    out = pd.DataFrame(rows)
    if len(out):
        out["delong_p_bh"] = bh_adjust(out["delong_p"].to_numpy())
    return out


# Negative-class decomposition and the operational subset
def subset_metrics(P: Paths):
    """Pooled metrics, the backward-local-maxima subset where the operational
    question is actually posed, and foreshock recall."""
    rows, comp = [], []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat, [REF_LABEL, "mag_diff_30d"])
        y = df[REF_LABEL].to_numpy(np.int8)
        before, after = get_directional(P, cat)

        neg = y == 0
        comp.append({
            "catalog": cat,
            "label_reconstruction_agreement":
                round(float(((~(before | after)).astype(np.int8) == y).mean()),
                      4),
            "n_events": len(y),
            "mainshock_rate": round(float(y.mean()), 4),
            "frac_neg_aftershock_only":
                round(float((before & ~after)[neg].mean()), 4),
            "frac_neg_foreshock_only":
                round(float((after & ~before)[neg].mean()), 4),
            "frac_neg_both": round(float((before & after)[neg].mean()), 4),
        })

        te = slice(b3, len(y))
        yte = y[te]
        p = load_probs(P, cat, "xgboost")[te]
        md = df["mag_diff_30d"].to_numpy(float)[te]
        bef, aft = before[te], after[te]

        pred = (p >= 0.5).astype(int)
        rows.append({"catalog": cat, "subset": "all test events",
                     "n": len(yte), "positive_rate": float(yte.mean()),
                     "auc": roc_auc_score(yte, p),
                     "f1": f1_score(yte, pred, zero_division=0),
                     "precision": precision_score(yte, pred, zero_division=0),
                     "recall": recall_score(yte, pred, zero_division=0),
                     "accuracy": accuracy_score(yte, pred),
                     "auc_magdiff": roc_auc_score(yte, md)})

        # Events with no larger predecessor: here class 0 means a foreshock
        # and nothing else.
        m = ~bef
        if m.sum() > 50 and len(np.unique(yte[m])) == 2:
            pm, ym = p[m], yte[m]
            predm = (pm >= 0.5).astype(int)
            rows.append({"catalog": cat,
                         "subset": "backward local maxima (operational)",
                         "n": int(m.sum()), "positive_rate": float(ym.mean()),
                         "auc": roc_auc_score(ym, pm),
                         "f1": f1_score(ym, predm, zero_division=0),
                         "precision": precision_score(ym, predm,
                                                      zero_division=0),
                         "recall": recall_score(ym, predm, zero_division=0),
                         "accuracy": accuracy_score(ym, predm),
                         "auc_magdiff": roc_auc_score(ym, md[m])})

        fs = aft & ~bef
        if fs.any():
            rows.append({"catalog": cat, "subset": "foreshocks only",
                         "n": int(fs.sum()),
                         "positive_rate": float(yte[fs].mean()),
                         "recall": float((p[fs] < 0.5).mean()),
                         "auc": np.nan, "f1": np.nan, "precision": np.nan,
                         "accuracy": np.nan, "auc_magdiff": np.nan})
            print(f"  {cat:<22} foreshocks in test block {int(fs.sum()):>5}, "
                  f"flagged as non-mainshock {float((p[fs] < 0.5).mean()):.1%}")
    return pd.DataFrame(rows).round(4), pd.DataFrame(comp)


def label_feature_coupling(P: Paths):
    """How much of the label the magnitude difference and the backward half of
    the labelling rule reproduce on their own."""
    from sklearn.feature_selection import mutual_info_classif
    rows = []
    for cat in CATS:
        df = load_features(P, cat, [REF_LABEL, "mag_diff_30d"])
        y = df[REF_LABEL].to_numpy(np.int8)
        md = df["mag_diff_30d"].to_numpy(float)
        before, _ = get_directional(P, cat)

        mi = float(mutual_info_classif(md.reshape(-1, 1), y,
                                       random_state=SEED)[0])
        pr = float(y.mean())
        h = -(pr * np.log(pr) + (1 - pr) * np.log(1 - pr))
        backward_rule = (~before).astype(np.int8)
        rows.append({
            "catalog": cat,
            "mutual_info_nats": round(mi, 4),
            "label_entropy_nats": round(h, 4),
            "mi_over_entropy": round(mi / h, 4),
            "auc_magdiff_alone": round(roc_auc_score(y, md), 4),
            "backward_rule_accuracy":
                round(float((backward_rule == y).mean()), 4),
            "backward_rule_f1": round(float(f1_score(y, backward_rule)), 4),
        })
        print(f"  {cat:<22} MI/H {rows[-1]['mi_over_entropy']:.3f}, backward "
              f"rule alone reproduces "
              f"{rows[-1]['backward_rule_accuracy']:.1%} of labels")
    return pd.DataFrame(rows)


# ETAS null
N_SYNTH = 5
# Every geometry- and magnitude-derived quantity a simulated catalog supports.
# It is a superset of the real model's features that survive simulation, so
# the floor it produces is the more conservative of the two.
SYNTH_FEATS = ["mw", "local_count_7d", "log_energy_30d", "mag_diff_30d",
               "time_gap_local", "seismicity_z", "latitude", "longitude"]


def _synthetic_directional(P, df, tag):
    ck = P.ckpt / f"{tag}__directional.npz"
    if ck.exists():
        d = np.load(ck)
        return d["before"], d["after"]
    times_s = ((df["time"] - pd.Timestamp(0, tz="UTC"))
               .dt.total_seconds().to_numpy())
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    before, after = directional_flags(times_s, df["mw"].to_numpy(float),
                                      cKDTree(xyz), xyz, REF_R, REF_T)
    np.savez(ck, before=before, after=after)
    return before, after


def _null_realisations(P, cat, cfg, gpu, n_runs=N_SYNTH):
    params = best_params(P, cat)
    sim_cfg = etas.simulation_config(cfg)
    rows = []
    for r in range(n_runs):
        tag = f"synth__{cat}__{r}"
        ck = P.ckpt / f"{tag}.parquet"
        if ck.exists():
            df = pd.read_parquet(ck)
        else:
            try:
                df = etas.simulate_etas(sim_cfg, seed=SEED + r)
            except etas.EtasError as e:
                print(f"    [discarded] realisation {r}: {e}")
                continue
            if len(df) < 2000:
                print(f"    [discarded] realisation {r}: only {len(df)} "
                      f"events, too few to evaluate")
                continue
            df = etas.label_and_featurize(df, REF_R, REF_T)
            df.to_parquet(ck, index=False)

        y = df["label"].to_numpy(np.int8)
        n = len(df)
        b1, b3 = int(n * 0.60), int(n * 0.85)
        if len(np.unique(y[:b1])) < 2 or len(np.unique(y[b3:])) < 2:
            print(f"    [discarded] realisation {r}: degenerate labels")
            continue

        m = make_xgb(params, gpu=gpu)
        m.fit(df[SYNTH_FEATS].iloc[:b1], y[:b1])
        p = m.predict_proba(df[SYNTH_FEATS])[:, 1]
        yte, pte = y[b3:], p[b3:]
        md = df["mag_diff_30d"].to_numpy(float)[b3:]

        before, after = _synthetic_directional(P, df, tag)
        bef, aft = before[b3:], after[b3:]
        op, fs = ~bef, aft & ~bef

        row = {"catalog": cat, "run": r, "n_events": n,
               "mainshock_rate": float(y.mean()),
               "observed_mainshock_rate":
                   cfg["observed_mainshock_rate_above_m0"],
               "calibration_status": cfg.get("calibration_status", "unknown"),
               "branching_ratio": cfg.get("branching_ratio", np.nan),
               "label_reconstruction_agreement":
                   float(((~(before | after)).astype(np.int8) == y).mean()),
               "auc_pooled": roc_auc_score(yte, pte),
               "f1_pooled": f1_score(yte, (pte >= 0.5).astype(int),
                                     zero_division=0),
               "auc_magdiff_pooled": roc_auc_score(yte, md),
               "n_operational": int(op.sum()), "n_foreshocks": int(fs.sum())}
        if op.sum() > 50 and len(np.unique(yte[op])) == 2:
            row["auc_operational"] = roc_auc_score(yte[op], pte[op])
            row["f1_operational"] = f1_score(yte[op],
                                             (pte[op] >= 0.5).astype(int),
                                             zero_division=0)
            row["auc_magdiff_operational"] = roc_auc_score(yte[op], md[op])
            row["positive_rate_operational"] = float(yte[op].mean())
        else:
            for k in ("auc_operational", "f1_operational",
                      "auc_magdiff_operational", "positive_rate_operational"):
                row[k] = np.nan
        row["foreshock_recall"] = (float((pte[fs] < 0.5).mean())
                                   if fs.any() else np.nan)
        rows.append(row)
        print(f"    realisation {r}: {n} events, pooled AUC "
              f"{row['auc_pooled']:.4f}, operational AUC "
              f"{row['auc_operational']:.4f}, mainshock rate "
              f"{row['mainshock_rate']:.3f}")
        del m
        gc.collect()
    return rows


def etas_null(P: Paths, gpu=False):
    """Fit one ETAS model per catalog and evaluate the same pipeline on
    simulated catalogs, where offspring magnitudes are independent of their
    parents and no event-role physics exists."""
    params_rows, null_rows = [], []
    for cat in CATS:
        print(f"  fitting ETAS to {cat} ...")
        cfg = etas.fit_etas(P, cat)
        params_rows.append({k: (json.dumps(v) if isinstance(v, (list, tuple))
                                else v) for k, v in cfg.items()})
        if not etas.is_usable(cfg):
            print(f"  no null for {cat}: {cfg['status']} — "
                  f"{cfg.get('status_detail', '')}")
            continue
        null_rows += _null_realisations(P, cat, cfg, gpu)
    return pd.DataFrame(params_rows), pd.DataFrame(null_rows).round(4)


# Completeness
MC_WINDOW = 500
MC_STEP = 250
STRICT_THRESHOLDS = {
    "comcat_global_m4": [4.5, 5.0],
    "scedc_socal_m25": [3.0, 3.5],
    "comcat_himalaya_m4": [4.5],
}


def mc_over_time(df):
    mw = df["mw"].to_numpy(float)
    t = df["time"].to_numpy()
    rows = []
    for lo in range(0, len(mw) - MC_WINDOW, MC_STEP):
        hi = lo + MC_WINDOW
        rows.append({"time": t[(lo + hi) // 2], "mc": mc_maxc(mw[lo:hi]),
                     "n": MC_WINDOW})
    return pd.DataFrame(rows)


def completeness_analysis(P: Paths, gpu=False):
    """Mc through time, and whether skill without magnitude features survives
    above the detection threshold.

    MAXC returns the modal bin, which for a hard-truncated catalog is close to
    the catalog floor, so fixed thresholds well above it are evaluated too.
    """
    mc_rows, ab_rows = [], []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat)
        mc_t = mc_over_time(df)
        mc_t["catalog"] = cat
        mc_rows.append(mc_t)

        mc_modern = mc_maxc(df["mw"].to_numpy(float)[b2:])
        print(f"  {cat}: modern Mc {mc_modern:.2f} "
              f"(catalog minimum {df['mw'].min():.2f})")

        feats_all = present_features(df)
        Xi = imputed_matrix(P, cat, df, feats_all)
        y = df[REF_LABEL].to_numpy(np.int8)
        params = best_params(P, cat)
        no_mag = [f for f in feats_all
                  if f not in ("mw", "mag_diff_30d", "log_energy_30d")]

        mw_arr = df["mw"].to_numpy(float)
        samples = [("all events", np.ones(len(df), bool)),
                   (f"mw >= modern Mc ({mc_modern:.2f})", mw_arr >= mc_modern)]
        for thr in STRICT_THRESHOLDS.get(cat, []):
            samples.append((f"mw >= {thr:.1f} (fixed)", mw_arr >= thr))

        for tag, mask in samples:
            idx = np.where(mask)[0]
            tr, te = idx[idx < b1], idx[idx >= b3]
            if len(tr) < 500 or len(te) < 200 or len(np.unique(y[te])) < 2:
                print(f"  [skip] {cat} / {tag}: too few events")
                continue
            for ablation, fs in [("full", feats_all),
                                 ("no_magnitude", no_mag)]:
                m = make_xgb(params, gpu=gpu)
                m.fit(Xi[fs].iloc[tr], y[tr])
                p = m.predict_proba(Xi[fs].iloc[te])[:, 1]
                ab_rows.append({
                    "catalog": cat, "sample": tag, "ablation": ablation,
                    "n_train": len(tr), "n_test": len(te),
                    "mainshock_rate_test": float(y[te].mean()),
                    "auc": roc_auc_score(y[te], p),
                    "f1": f1_score(y[te], (p >= 0.5).astype(int),
                                   zero_division=0)})
                del m
                gc.collect()
        del df, Xi
        gc.collect()
    return (pd.concat(mc_rows, ignore_index=True),
            pd.DataFrame(ab_rows).round(4))


# Sequence-level uncertainty
SEQ_R_KM = 100.0
SEQ_T_DAYS = 30.0


def sequence_ids(df, R_km=SEQ_R_KM, T_days=SEQ_T_DAYS):
    """Single-link chaining: an event joins the most recent cluster whose last
    member is within R km and T days. Crude, but it errs toward larger and so
    more conservative resampling units."""
    times_s = ((df["time"] - pd.Timestamp(0, tz="UTC"))
               .dt.total_seconds().to_numpy())
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    tree = cKDTree(xyz)
    n = len(df)
    sid = np.full(n, -1, dtype=np.int64)
    nxt = 0
    T_s = T_days * 86400.0
    chord = chord_from_km(R_km)
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        neigh = tree.query_ball_point(xyz[lo:hi], r=chord)
        for k in range(hi - lo):
            i = lo + k
            nb = np.asarray(neigh[k])
            neigh[k] = None
            nb = nb[nb < i]
            if nb.size:
                nb = nb[(times_s[i] - times_s[nb]) <= T_s]
            if nb.size:
                sid[i] = sid[nb[int(np.argmax(times_s[nb]))]]
            else:
                sid[i] = nxt
                nxt += 1
        del neigh
    return sid


def get_sequence_ids(P: Paths, cat):
    ck = P.ckpt / f"{cat}__sequence_ids.npy"
    if ck.exists():
        return np.load(ck)
    df = load_features(P, cat, ["time", "latitude", "longitude"])
    sid = sequence_ids(df)
    np.save(ck, sid)
    print(f"  {cat}: {len(np.unique(sid))} sequences from {len(sid)} events")
    return sid


def cluster_boot_ci(stat_fn, y, groups, *arrays, n_boot=N_BOOT, seed=SEED):
    """Resample whole sequences with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, len(uniq), replace=True)
        s = np.concatenate([idx_by_group[g] for g in gs])
        if len(np.unique(y[s])) < 2:
            continue
        vals.append(stat_fn(y[s], *[a[s] for a in arrays]))
    vals = np.asarray(vals)
    return (float(stat_fn(y, *arrays)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)), len(uniq))


def sequence_intervals(P: Paths):
    rows = []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat, [REF_LABEL, "mag_diff_30d"])
        y = df[REF_LABEL].to_numpy(np.int8)
        sid = get_sequence_ids(P, cat)
        p = load_probs(P, cat, "xgboost")
        yte, pte, gte = y[b3:], p[b3:], sid[b3:]
        md = df["mag_diff_30d"].to_numpy(float)[b3:]

        for name, fn, arr in [
                ("auc", lambda yy, pp: roc_auc_score(yy, pp), pte),
                ("f1", lambda yy, pp: f1_score(yy, (pp >= .5).astype(int),
                                               zero_division=0), pte),
                ("ece", ece, pte),
                ("auc_magdiff", lambda yy, pp: roc_auc_score(yy, pp), md)]:
            v, lo, hi, ng = cluster_boot_ci(fn, yte, gte, arr)
            rows.append({"catalog": cat, "statistic": name, "value": v,
                         "ci_lo": lo, "ci_hi": hi, "n_events": len(yte),
                         "n_sequences": ng, "resampling_unit": "sequence"})
            print(f"  {cat} | {name}: {v:.4f} [{lo:.4f}, {hi:.4f}] "
                  f"over {ng} sequences")
    return pd.DataFrame(rows).round(5)


# Hyperparameter robustness
CORNERS = ["label_R25_T15", "label_R25_T90", "label_R100_T15",
           "label_R100_T90", REF_LABEL]


def corner_retune(P: Paths, gpu=False):
    """Re-tune at the corners of the labelling grid and compare against the
    hyperparameters selected on the reference window."""
    rows = []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat)
        Xi = imputed_matrix(P, cat, df)
        frozen = best_params(P, cat)
        for col in [c for c in CORNERS if c in df.columns]:
            y = df[col].to_numpy(np.int8)
            tuned = tune_xgb(P, f"{cat}__{col}", Xi.iloc[:b1], y[:b1],
                             Xi.iloc[b1:b2], y[b1:b2], gpu=gpu)
            for tag, params in [("frozen (reference window)", frozen),
                                ("re-tuned for this window", tuned)]:
                m = make_xgb(params, gpu=gpu)
                m.fit(Xi.iloc[:b1], y[:b1])
                p = m.predict_proba(Xi.iloc[b3:])[:, 1]
                rows.append({"catalog": cat, "config": col,
                             "hyperparameters": tag,
                             "auc": roc_auc_score(y[b3:], p),
                             "f1": f1_score(y[b3:], (p >= 0.5).astype(int),
                                            zero_division=0)})
                del m
                gc.collect()
        del df, Xi
        gc.collect()
    out = pd.DataFrame(rows).round(4)
    for cat in CATS:
        s = out[out.catalog == cat]
        for tag in s.hyperparameters.unique():
            t = s[s.hyperparameters == tag]
            print(f"  {cat:<22} {tag:<26} F1 {t.f1.min():.3f}-{t.f1.max():.3f}")
    return out


def right_censoring(P: Paths):
    """Labels near the end of a catalog cannot see their full forward window;
    metrics are reported with and without that tail."""
    rows = []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat, ["time", REF_LABEL])
        y = df[REF_LABEL].to_numpy(np.int8)
        p = load_probs(P, cat, "xgboost")
        t = df["time"]
        cutoff = t.iloc[-1] - pd.Timedelta(days=T_MAX_DAYS)

        te = np.arange(b3, len(y))
        trimmed = te[(t.iloc[te] <= cutoff).to_numpy()]
        n_lost = len(te) - len(trimmed)
        for tag, idx in [("untrimmed", te),
                         (f"trimmed last {int(T_MAX_DAYS)} d", trimmed)]:
            if len(idx) < 100 or len(np.unique(y[idx])) < 2:
                continue
            yy, pp = y[idx], p[idx]
            rows.append({"catalog": cat, "test_block": tag, "n": len(idx),
                         "n_censored_removed": n_lost if "trim" in tag else 0,
                         "mainshock_rate": float(yy.mean()),
                         "auc": roc_auc_score(yy, pp),
                         "f1": f1_score(yy, (pp >= 0.5).astype(int),
                                        zero_division=0),
                         "ece": ece(yy, pp)})
        print(f"  {cat}: {n_lost} of {len(te)} test events fall in the "
              f"censored tail ({n_lost / len(te):.1%})")
    return pd.DataFrame(rows).round(4)


# Cross-region transfer
SOCAL_BOX = dict(min_lat=32.0, max_lat=37.0, min_lon=-122.0, max_lon=-114.0)
HIMALAYA_BOX = dict(min_lat=20.0, max_lat=32.0, min_lon=85.0, max_lon=98.0)


def in_box(df, box):
    return (df["latitude"].between(box["min_lat"], box["max_lat"])
            & df["longitude"].between(box["min_lon"], box["max_lon"]))


def _isotonic(p, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p, y)
    return iso


def _mondrian_metrics(p_cal, y_cal, p_te, y_te, alpha=ALPHA):
    q0, q1 = mondrian_thresholds(p_cal, y_cal, alpha)
    d = set_metrics(y_te, p_te <= q0, (1 - p_te) <= q1)
    d["cov_gap"] = abs(d["cov_class1"] - d["cov_class0"])
    return d


def transfer_pair(P: Paths, target_cat, box, gpu=False, label=REF_LABEL):
    """Train on the global catalog with the target region removed, then apply
    it to the target, with and without recalibration on target data."""
    g = load_features(P, "comcat_global_m4")
    b1, b2, b3 = load_splits(P, "comcat_global_m4")
    t_train_end, t_val_end = g["time"].iloc[b1], g["time"].iloc[b2]
    t_test_start = g["time"].iloc[b3]

    h = load_features(P, target_cat)
    feats = [f for f in FULL_FEATURES if f in g.columns and f in h.columns]

    donor = g[~in_box(g, box).to_numpy()].reset_index(drop=True)
    td = donor["time"]
    d1, d2, d3 = (int((td < t_train_end).sum()), int((td < t_val_end).sum()),
                  int((td < t_test_start).sum()))
    yD = donor[label].to_numpy(np.int8)
    if d1 < 100:
        raise ValueError(f"donor training block has {d1} events; the donor "
                         f"does not overlap the global training period")

    imp = fit_imputer(donor[feats].iloc[:d1])
    XD = pd.DataFrame(imp.transform(donor[feats]),
                      columns=feats).astype(np.float32)

    dm = make_xgb(best_params(P, "comcat_global_m4"), gpu=gpu)
    ck = P.ckpt / f"donor_minus_{target_cat}__xgboost.json"
    if ck.exists():
        dm.load_model(ck)
    else:
        dm.fit(XD.iloc[:d1], yD[:d1])
        dm.save_model(ck)

    pD = dm.predict_proba(XD)[:, 1].astype(float)
    isoD = _isotonic(pD[d1:d2], yD[d1:d2])
    pD_cal, yD_cal = isoD.predict(pD[d2:d3]), yD[d2:d3]

    yH = h[label].to_numpy(np.int8)
    _, h2, h3 = load_splits(P, target_cat)
    tgt = np.where((h["time"] >= t_test_start).to_numpy())[0]
    if len(tgt) < 50 or len(np.unique(yH[tgt])) < 2:
        raise ValueError(f"target has {len(tgt)} events in the donor test "
                         f"period; too few to evaluate")
    yT = yH[tgt]
    XT = pd.DataFrame(imp.transform(h[feats].iloc[tgt]),
                      columns=feats).astype(np.float32)
    pT_raw = dm.predict_proba(XT)[:, 1].astype(float)

    def disc(y, p):
        return {"auc": roc_auc_score(y, p),
                "f1": f1_score(y, (p >= 0.5).astype(int), zero_division=0),
                "ece": ece(y, p)}

    rows = []
    pN = load_probs(P, target_cat, "xgboost")
    cal_idx = np.setdiff1d(np.arange(h2, h3), tgt)
    rows.append({"pair": f"global -> {target_cat}",
                 "system": "native (target-trained)", "n_target": len(tgt),
                 **disc(yT, pN[tgt]),
                 **_mondrian_metrics(pN[cal_idx], yH[cal_idx], pN[tgt], yT)})

    pT_d = isoD.predict(pT_raw)
    rows.append({"pair": f"global -> {target_cat}",
                 "system": "transfer (donor calibration only)",
                 "n_target": len(tgt), **disc(yT, pT_d),
                 **_mondrian_metrics(pD_cal, yD_cal, pT_d, yT)})

    ta = np.where(((h["time"] < t_test_start).to_numpy())
                  & (np.arange(len(h)) >= h2))[0]
    if len(ta) >= 50:
        XC = pd.DataFrame(imp.transform(h[feats].iloc[ta]),
                          columns=feats).astype(np.float32)
        pC = dm.predict_proba(XC)[:, 1].astype(float)
        isoT = _isotonic(pC, yH[ta])
        rows.append({"pair": f"global -> {target_cat}",
                     "system": "transfer + target adaptation",
                     "n_cal_target": len(ta), "n_target": len(tgt),
                     **disc(yT, isoT.predict(pT_raw)),
                     **_mondrian_metrics(isoT.predict(pC), yH[ta],
                                         isoT.predict(pT_raw), yT)})
    else:
        print(f"  [skip] target adaptation: only {len(ta)} calibration events")
    del dm, XD, g, h
    gc.collect()
    return rows


def transfer_pairs(P: Paths, gpu=False):
    rows = []
    for target, box in [("comcat_himalaya_m4", HIMALAYA_BOX),
                        ("scedc_socal_m25", SOCAL_BOX)]:
        print(f"  global minus box -> {target}")
        try:
            rows += transfer_pair(P, target, box, gpu=gpu)
        except Exception as e:
            print(f"  [failed] {target}: {type(e).__name__}: {e}")
    out = pd.DataFrame(rows).round(4)
    for pair in out.pair.unique():
        s = out[out.pair == pair]
        nat = s[s.system.str.startswith("native")]
        zer = s[s.system.str.startswith("transfer (")]
        if len(nat) and len(zer):
            n, z = nat.iloc[0], zer.iloc[0]
            print(f"  {pair}: AUC {n.auc:.4f} -> {z.auc:.4f}, coverage "
                  f"{n.coverage:.4f} -> {z.coverage:.4f}, class gap "
                  f"{n.cov_gap:.4f} -> {z.cov_gap:.4f}")
    return out


# Case studies
CASES = {
    "Kahramanmaras_2023": dict(t0="2023-01-30", t1="2023-03-15",
                               lat=(36.0, 39.0), lon=(35.5, 39.0)),
    "Ridgecrest_2019": dict(t0="2019-06-28", t1="2019-08-10",
                            lat=(35.0, 36.5), lon=(-118.2, -117.0)),
    "Kumamoto_2016": dict(t0="2016-04-10", t1="2016-05-15",
                          lat=(32.0, 33.6), lon=(130.0, 131.5)),
}
CASE_ALPHA = 0.02
CASE_MW_MIN = 6.5


def case_replay(P: Paths, cat="comcat_global_m4", model="xgboost"):
    """Replay each case study at every label definition in the grid and record
    the operational state assigned to each event."""
    b1, b2, b3 = load_splits(P, cat)
    df = load_features(P, cat)
    label_cols = [c for c in (f"label_R{R}_T{T}" for R in RADII_KM
                              for T in WINDOWS_DAYS) if c in df.columns]
    p = load_probs(P, cat, model)

    rows = []
    for name, c in CASES.items():
        mask = (df["time"].between(c["t0"], c["t1"])
                & df["latitude"].between(*c["lat"])
                & df["longitude"].between(*c["lon"])
                & (df["mw"] >= 4.5))
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            print(f"  {name}: no events in the box")
            continue
        for col in label_cols:
            y = df[col].to_numpy(np.int8)
            cal = np.arange(b2, b3)
            cal = cal[~np.isin(cal, idx)]       # never calibrate on the case
            q0, q1 = mondrian_thresholds(p[cal], y[cal], CASE_ALPHA)
            in0, in1 = p[idx] <= q0, (1 - p[idx]) <= q1
            act = decide(in0, in1)
            for k, i in enumerate(idx):
                rows.append({
                    "sequence": name, "config": col,
                    "time": df["time"].iloc[i], "mw": df["mw"].iloc[i],
                    "place": df["place"].iloc[i] if "place" in df else "",
                    "prob": round(float(p[i]), 4),
                    "set": ("{0,1}" if in0[k] and in1[k] else
                            "{1}" if in1[k] else "{0}" if in0[k] else "{}"),
                    "state": act[k], "true_label": int(y[i]),
                    "false_RESPONSE": bool(act[k] == "RESPONSE" and y[i] == 0),
                })
    replay = pd.DataFrame(rows)
    if replay.empty:
        print("  no case-study events matched the boxes")
        return replay, pd.DataFrame()

    big = replay[replay["mw"] >= CASE_MW_MIN]
    if big.empty:
        return replay, pd.DataFrame()
    flips = (big.groupby(["sequence", "time", "mw"])["state"]
             .nunique().reset_index(name="n_distinct_states"))
    labels = (big.groupby(["sequence", "time", "mw"])["true_label"]
              .nunique().reset_index(name="n_distinct_labels"))
    instability = flips.merge(labels, on=["sequence", "time", "mw"])
    n = int((instability.n_distinct_states > 1).sum())
    print(f"  {n} of {len(instability)} events with mw >= {CASE_MW_MIN} "
          f"receive more than one state across the grid")
    del df
    gc.collect()
    return replay, instability


# Independent declustering paradigm
def zbz_labels(P: Paths, cat):
    ck = P.ckpt / f"{cat}__zbz.npz"
    if ck.exists():
        d = np.load(ck)
        return d["label"], float(d["eta0"])
    df = load_features(P, cat, ["time", "mw", "latitude", "longitude"])
    print(f"  nearest-neighbour distances for {cat} ({len(df)} events) ...")
    eta, parent = etas.nn_distances(df)
    log_eta = np.log10(np.where(np.isfinite(eta) & (eta > 0), eta, np.nan))
    eta0, sep, modes = etas.pick_eta0(log_eta)
    # Under this paradigm the weakly linked (background) events play the role
    # the windowed rule gives to mainshocks.
    label = ((log_eta >= eta0) | ~np.isfinite(log_eta)).astype(np.int8)
    np.savez(ck, label=label, eta=eta, eta0=np.array(eta0),
             separation=np.array(sep), parent=parent)
    print(f"    eta0 = 10^{eta0:.2f}, modes 10^{modes[0]:.1f} and "
          f"10^{modes[1]:.1f} ({sep:.1f} sd), background fraction "
          f"{label.mean():.3f}")
    del df
    gc.collect()
    return label, eta0


def zbz_relabelling(P: Paths, gpu=False):
    rows = []
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat)
        y_win = df[REF_LABEL].to_numpy(np.int8)
        y_zbz, _ = zbz_labels(P, cat)
        Xi = imputed_matrix(P, cat, df)
        params = best_params(P, cat)

        for tag, y in [("windowed (R=50, T=30)", y_win),
                       ("nearest-neighbour (ZBZ)", y_zbz)]:
            m = make_xgb(params, gpu=gpu)
            m.fit(Xi.iloc[:b1], y[:b1])
            p = m.predict_proba(Xi)[:, 1]
            yte, pte = y[b3:], p[b3:]
            rows.append({"catalog": cat, "labelling": tag,
                         "background_rate": float(y.mean()),
                         "flip_vs_windowed": float((y != y_win).mean()),
                         "auc": roc_auc_score(yte, pte),
                         "f1": f1_score(yte, (pte >= 0.5).astype(int),
                                        zero_division=0),
                         "ece": ece(yte, pte)})
            del m
            gc.collect()
        w, z = rows[-2], rows[-1]
        print(f"  {cat:<22} AUC {w['auc']:.3f} -> {z['auc']:.3f}, "
              f"{z['flip_vs_windowed']:.1%} of labels moved")
        del df, Xi
        gc.collect()
    return pd.DataFrame(rows).round(4)


def feature_ablation(P: Paths, gpu=False):
    """What the magnitude family contributes on its own."""
    ablations = {
        "full": FULL_FEATURES,
        "no_magdiff": [f for f in FULL_FEATURES if f != "mag_diff_30d"],
        "magdiff_only": ["mag_diff_30d"],
        "no_mw_family": [f for f in FULL_FEATURES
                         if f not in ("mag_diff_30d", "mw", "log_energy_30d")],
    }
    rows, store = [], {}
    for cat in CATS:
        b1, b2, b3 = load_splits(P, cat)
        df = load_features(P, cat)
        y = df[REF_LABEL].to_numpy(np.int8)
        yte = y[b3:]
        Xi = imputed_matrix(P, cat, df)
        params = best_params(P, cat)

        for ablation, fs in ablations.items():
            fs = [f for f in fs if f in Xi.columns]
            if not fs:
                print(f"  [skip] {cat}/{ablation}: no features present")
                continue
            ck = P.ckpt / f"{cat}__ablation_{ablation}.npz"
            if ck.exists():
                p = np.load(ck)["prob"].astype(float)
            else:
                m = make_xgb(params, gpu=gpu)
                m.fit(Xi[fs].iloc[:b1], y[:b1])
                p = m.predict_proba(Xi[fs])[:, 1].astype(float)
                np.savez(ck, prob=p.astype(np.float32))
                del m
                gc.collect()
            store[(cat, ablation)] = p[b3:]
            pred = (p[b3:] >= 0.5).astype(int)
            rows.append({"catalog": cat, "ablation": ablation,
                         "n_features": len(fs),
                         "auc": roc_auc_score(yte, p[b3:]),
                         "f1": f1_score(yte, pred, zero_division=0),
                         "recall": recall_score(yte, pred, zero_division=0)})
            print(f"  {cat} / {ablation} ({len(fs)} features): AUC "
                  f"{rows[-1]['auc']:.4f} F1 {rows[-1]['f1']:.4f}")

        for ablation in ablations:
            if ablation == "full" or (cat, ablation) not in store:
                continue
            _, _, pv = delong_test(yte, store[(cat, "full")],
                                   store[(cat, ablation)])
            for r in rows:
                if r["catalog"] == cat and r["ablation"] == ablation:
                    r["delong_p_vs_full"] = pv
        del df, Xi
        gc.collect()

    out = pd.DataFrame(rows)
    if "delong_p_vs_full" in out.columns:
        out["delong_p_bh"] = bh_adjust(out["delong_p_vs_full"].to_numpy())
    return out.round(5)


def feature_availability(P: Paths):
    """Which model features each catalog provides, for the results folder."""
    rows = []
    for cat in CATS:
        cols = set(load_features(P, cat).columns)
        rows.append({"catalog": cat,
                     "n_model_features": sum(f in cols for f in FULL_FEATURES),
                     "present": ", ".join(f for f in FULL_FEATURES
                                          if f in cols),
                     "absent": ", ".join(f for f in FULL_FEATURES
                                         if f not in cols) or "(none)"})
    return pd.DataFrame(rows)
