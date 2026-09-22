#!/usr/bin/env python3
"""Check that a finished run is complete and internally consistent.

    python code/audit_outputs.py --root /content/drive/MyDrive/quake_project_v4 \
        --run-name v4_fresh

Exits non-zero if an expected output is missing, a table is empty, the
conformal set-size decomposition is inconsistent, an ETAS null was produced
from parameters that are not subcritical, or the run does not match the
dataset it claims to come from.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CATS, Paths, sha256  # noqa: E402

REQUIRED_TABLES = [
    "calibration_metrics.csv",
    "sensitivity_grid.csv",
    "conformal_sets.csv",
    "decision_states.csv",
    "decision_rule_comparison.csv",
    "decision_costs.csv",
    "baselines.csv",
    "subset_metrics.csv",
    "negative_class_composition.csv",
    "label_feature_coupling.csv",
    "etas_parameters.csv",
    "mc_over_time.csv",
    "completeness_ablation.csv",
    "sequence_cis.csv",
    "corner_retune.csv",
    "right_censoring.csv",
    "transfer_pairs.csv",
    "zbz_relabelling.csv",
    "feature_ablation.csv",
    "feature_availability.csv",
]

REQUIRED_FIGURES = [
    "reliability.png", "aci_coverage.png", "set_composition.png",
    "counts_and_mc.png", "cost_model.png", "transfer.png",
    "shap_stability.png", "auc_vs_f1_divergence.png",
    "decision_rule_empty_set.png",
]

# Columns where a missing value is part of the result rather than a fault.
EXPECTED_NAN = {
    "subset_metrics.csv": {"auc", "f1", "precision", "accuracy",
                           "auc_magdiff"},
    "feature_ablation.csv": {"delong_p_vs_full", "delong_p_bh"},
    "transfer_pairs.csv": {"n_cal_target"},
    "decision_rule_comparison.csv": {"false_RESPONSE_given_RESPONSE"},
    # the rolling window has no value until it is full
    "aci_rolling.csv": {"rolling_cov_split", "rolling_cov_aci"},
    # score rows and DeLong rows share one table and carry different columns
    "baselines.csv": {"accuracy", "precision", "recall", "f1",
                      "auc_identity_sens_spec", "auc_baseline", "delong_p"},
    "etas_null.csv": {"auc_operational", "f1_operational",
                      "auc_magdiff_operational", "positive_rate_operational",
                      "foreshock_recall"},
    "etas_parameters.csv": None,   # failed fits legitimately leave blanks
}

ERRORS: list[str] = []
WARNINGS: list[str] = []


def err(m):
    ERRORS.append(m)
    print(f"  [error] {m}")


def warn(m):
    WARNINGS.append(m)
    print(f"  [warn ] {m}")


def ok(m):
    print(f"  [ ok  ] {m}")


def section(title):
    print(f"\n{title}")


def audit(P: Paths) -> int:
    section("dataset")
    if not P.provenance.exists():
        err("dataset/data_provenance.json is missing")
    else:
        prov = json.loads(P.provenance.read_text())
        if prov.get("catalog_end_pinned") in (None, "unpinned"):
            err("the catalog end date is not pinned")
        for cat in CATS:
            entry = prov.get("derived", {}).get(f"{cat}_features")
            if entry and P.features(cat).exists():
                if sha256(P.features(cat)) != entry["sha256"]:
                    err(f"{cat}: feature file differs from the provenance "
                        f"record")
                else:
                    ok(f"{cat}: feature file matches provenance")

    section("tables")
    present = {p.name for p in P.tables.glob("*.csv")}
    for t in REQUIRED_TABLES:
        if t not in present:
            err(f"missing table: {t}")
    ok(f"{len(present & set(REQUIRED_TABLES))} of {len(REQUIRED_TABLES)} "
       f"expected tables present")

    for csv in sorted(P.tables.glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            err(f"{csv.name}: unreadable ({e})")
            continue
        if df.empty:
            err(f"{csv.name}: empty")
            continue
        allowed = EXPECTED_NAN.get(csv.name, set())
        if allowed is None:
            continue
        bad = {c: int(df[c].isna().sum()) for c in df.columns
               if c not in allowed and df[c].isna().any()}
        if bad:
            warn(f"{csv.name}: unexpected missing values in {bad}")

    section("figures")
    figs = {p.name for p in P.figures.glob("*.png")}
    for f in REQUIRED_FIGURES:
        if f not in figs:
            err(f"missing figure: {f}")
    ok(f"{len(figs)} figures present")

    section("models")
    for cat in CATS:
        splits = (P.models / f"{cat}__splits.npz").exists()
        probs = any((P.models / f"{cat}__{m}__recal_probs.npz").exists()
                    for m in ("xgboost", "catboost", "lightgbm"))
        if splits and probs:
            ok(f"{cat}: trained and recalibrated")
        else:
            err(f"{cat}: training output incomplete")

    section("conformal sets")
    p = P.tables / "conformal_sets.csv"
    if p.exists():
        a = pd.read_csv(p)
        need = {"rate_empty", "rate_singleton", "rate_both", "coverage",
                "avg_set_size"}
        if not need.issubset(a.columns):
            err(f"conformal table missing columns: {need - set(a.columns)}")
        else:
            gap = float((a.rate_empty + a.rate_singleton
                         + a.rate_both - 1).abs().max())
            if gap > 2e-3:
                err(f"set-size rates do not sum to one (max {gap:.2e})")
            else:
                ok("set-size rates sum to one")
            implied = 1 - a.rate_empty + a.rate_both
            bad = int((implied - a.avg_set_size).abs().gt(2e-3).sum())
            if bad:
                err(f"{bad} rows where mean set size != 1 - empty + both")
            else:
                ok("mean set size matches the identity")

    section("ETAS")
    par_path = P.tables / "etas_parameters.csv"
    null_path = P.tables / "etas_null.csv"
    if not par_path.exists():
        err("etas_parameters.csv is missing")
    else:
        par = pd.read_csv(par_path)
        for _, r in par.iterrows():
            status = r.get("status", "?")
            if status != "ok":
                warn(f"{r['catalog']}: no ETAS null ({status})")
                continue
            n = r.get("branching_ratio", float("nan"))
            if not (n == n) or n >= 1.0:
                err(f"{r['catalog']}: branching ratio {n} is not subcritical "
                    f"although a null was produced")
            else:
                ok(f"{r['catalog']}: branching ratio {n:.3f}, calibration "
                   f"{r.get('calibration_status', '?')}")
        usable = [r["catalog"] for _, r in par.iterrows()
                  if r.get("status") == "ok"]
        if usable and not null_path.exists():
            err("ETAS parameters were fitted but etas_null.csv is missing")
        elif null_path.exists():
            null = pd.read_csv(null_path)
            missing = sorted(set(usable) - set(null.catalog.unique()))
            if missing:
                err(f"no null realisations for: {missing}")
            else:
                ok(f"{len(null)} realisations over "
                   f"{null.catalog.nunique()} catalogs")

    section("manifest")
    if not P.manifest.exists():
        err("run_manifest.json is missing")
    else:
        mf = json.loads(P.manifest.read_text())
        for k in ("global_seed", "environment", "code_version",
                  "dataset_provenance", "parameters"):
            if k not in mf:
                err(f"manifest does not record {k}")
        if mf.get("dataset_provenance") == "MISSING":
            err("manifest records the dataset provenance as MISSING")
        if not ERRORS:
            ok("manifest records seed, environment, code hashes and dataset "
               "provenance")

    section("run consistency")
    stamps = [q.stat().st_mtime for q in P.run.rglob("*") if q.is_file()]
    if not stamps:
        err("the run folder is empty")
    else:
        span_h = (max(stamps) - min(stamps)) / 3600.0
        print(f"  artefacts span {span_h:.1f} h")
        if span_h > 72:
            err(f"artefacts span {span_h / 24:.1f} days; they did not come "
                f"from one execution")
        else:
            ok("artefacts come from one execution window")

    print(f"\nerrors {len(ERRORS)} | warnings {len(WARNINGS)}")
    if ERRORS:
        for e in ERRORS:
            print(f"  - {e}")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default="/content/drive/MyDrive/quake_project_v4")
    ap.add_argument("--run-name", default="v4_fresh")
    a = ap.parse_args()
    P = Paths(a.root, run_name=a.run_name)
    if not P.run.is_dir():
        sys.exit(f"no such run: {P.run}")
    print(f"auditing {P.run}")
    sys.exit(audit(P))
