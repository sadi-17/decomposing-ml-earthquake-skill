#!/usr/bin/env python3
"""Run the full analysis on a dataset built by make_dataset.py.

    python code/run_analysis.py --root /content/drive/MyDrive/quake_project_v4 \
        --run-name v4_fresh --gpu

Reads dataset/ only and writes results/<run-name>/ only. A new run name starts
from an empty run folder: no model, checkpoint or table is inherited from an
earlier run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyses  # noqa: E402
import models  # noqa: E402
import report  # noqa: E402
from utils import (CATS, GLOBAL_SEED, Paths, elapsed_hours,  # noqa: E402
                   reset_clock, resolve_gpu, save_table, seed_everything,
                   sha256, stage, start_log, write_manifest)

warnings.filterwarnings("ignore")


def check_dataset(P: Paths):
    """The analysis must run on the dataset whose provenance was recorded."""
    if not P.provenance.exists():
        sys.exit(f"{P.provenance} is missing; rebuild the dataset before "
                 f"running the analysis")
    prov = json.loads(P.provenance.read_text())
    derived = prov.get("derived", {})
    for cat in CATS:
        entry = derived.get(f"{cat}_features")
        if not entry:
            continue
        if sha256(P.features(cat)) != entry["sha256"]:
            sys.exit(f"{P.features(cat).name} does not match the hash in "
                     f"data_provenance.json. Re-run make_dataset.py so the "
                     f"analysis and the provenance record agree.")
    print(f"dataset    : built {prov.get('generated_utc', '?')[:19]}, "
          f"catalog end {prov.get('catalog_end_pinned', '?')}, hashes match")
    for cat, e in prov.get("inputs", {}).items():
        print(f"             {cat:<22} {e.get('n_events', '?'):>8} events  "
              f"sha {str(e.get('sha256', ''))[:12]}")


def wipe_run(P: Paths):
    for p in (P.models, P.tables, P.figures, P.ckpt):
        if p.is_dir():
            shutil.rmtree(p)
    if P.manifest.exists():
        P.manifest.unlink()
    P.mkdirs(analysis=True)
    print("run folder cleared")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default="/content/drive/MyDrive/quake_project_v4")
    ap.add_argument("--run-name", default="v4_fresh")
    ap.add_argument("--gpu", action="store_true",
                    help="train the boosting models on the GPU where the "
                         "library supports it")
    ap.add_argument("--wipe", action="store_true",
                    help="clear this run folder before starting")
    args = ap.parse_args()

    seed_everything(GLOBAL_SEED)
    gpu = resolve_gpu(args.gpu)

    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        pass

    P = Paths(args.root, run_name=args.run_name)
    P.require_dataset()
    P.mkdirs(analysis=True)
    if args.wipe:
        wipe_run(P)

    log = start_log(P.logs, "analysis")
    reset_clock()
    print(f"log        -> {log}")
    print(f"root       : {P.root}")
    print(f"run name   : {P.run_name}")
    print(f"seed       : {GLOBAL_SEED}")
    print(f"GPU        : {gpu}")
    check_dataset(P)

    stage(1, "train models")
    models.train_all(P, gpu=gpu)

    stage(2, "calibration audit")
    calib, reliability, aci = analyses.calibration_audit(P)
    save_table(P, calib, "calibration_metrics.csv")
    save_table(P, reliability, "reliability_points.csv")
    save_table(P, aci, "aci_rolling.csv")

    stage(3, "label-definition sensitivity sweep")
    grid, shap_stability = analyses.sensitivity_sweep(P, gpu=gpu)
    save_table(P, grid, "sensitivity_grid.csv")
    save_table(P, shap_stability, "shap_stability.csv")

    stage(4, "conformal prediction sets")
    save_table(P, analyses.conformal_audit(P), "conformal_sets.csv")

    stage(5, "decision rule and cost model")
    save_table(P, analyses.STATE_TABLE, "decision_states.csv")
    save_table(P, analyses.decision_rule_comparison(P),
               "decision_rule_comparison.csv")
    save_table(P, analyses.decision_costs(P), "decision_costs.csv")

    stage(6, "baselines")
    save_table(P, analyses.magnitude_baselines(P), "baselines.csv")
    bvals = analyses.bvalue_baselines(P)
    if len(bvals):
        save_table(P, bvals, "bvalue_baselines.csv")

    stage(7, "negative-class decomposition and the operational subset")
    subset, composition = analyses.subset_metrics(P)
    save_table(P, subset, "subset_metrics.csv")
    save_table(P, composition, "negative_class_composition.csv")
    save_table(P, analyses.label_feature_coupling(P),
               "label_feature_coupling.csv")

    stage(8, "ETAS null")
    etas_params, etas_rows = analyses.etas_null(P, gpu=gpu)
    save_table(P, etas_params, "etas_parameters.csv")
    if len(etas_rows):
        save_table(P, etas_rows, "etas_null.csv")
    else:
        print("  no ETAS null was produced; see etas_parameters.csv for why")

    stage(9, "completeness")
    mc_table, mc_ablation = analyses.completeness_analysis(P, gpu=gpu)
    save_table(P, mc_table, "mc_over_time.csv")
    save_table(P, mc_ablation, "completeness_ablation.csv")

    stage(10, "sequence-level intervals")
    save_table(P, analyses.sequence_intervals(P), "sequence_cis.csv")

    stage(11, "hyperparameters at the grid corners")
    save_table(P, analyses.corner_retune(P, gpu=gpu), "corner_retune.csv")

    stage(12, "right-censoring")
    save_table(P, analyses.right_censoring(P), "right_censoring.csv")

    stage(13, "cross-region transfer")
    save_table(P, analyses.transfer_pairs(P, gpu=gpu), "transfer_pairs.csv")

    stage(14, "case studies across the grid")
    replay, instability = analyses.case_replay(P)
    if len(replay):
        save_table(P, replay, "case_replay_grid.csv")
    if len(instability):
        save_table(P, instability, "case_state_instability.csv")

    stage(15, "nearest-neighbour relabelling")
    save_table(P, analyses.zbz_relabelling(P, gpu=gpu), "zbz_relabelling.csv")

    stage(16, "feature ablation")
    save_table(P, analyses.feature_ablation(P, gpu=gpu), "feature_ablation.csv")
    save_table(P, analyses.feature_availability(P), "feature_availability.csv")

    stage(17, "figures and report")
    figs = report.build_figures(P)

    stage(18, "run manifest")
    etas_summary = {}
    if len(etas_params):
        for _, r in etas_params.iterrows():
            etas_summary[r["catalog"]] = {
                k: r[k] for k in ("status", "m0", "b", "alpha", "K",
                                  "branching_ratio", "calibration_status")
                if k in etas_params.columns}
    manifest = write_manifest(P, extra={
        "gpu_used": bool(gpu),
        "etas": etas_summary,
        "figures_written": sorted(p.name for p in P.figures.glob("*.png")),
    })
    report.write_report(P, figs)

    print(f"\n  manifest -> {manifest}")
    print(f"  runtime  : {elapsed_hours():.2f} h")
    print(f"  tables   : {len(list(P.tables.glob('*.csv')))}")
    print(f"  figures  : {len(list(P.figures.glob('*.png')))}")
    print(f"\nnext:  python code/audit_outputs.py --root {P.root} "
          f"--run-name {P.run_name}")


if __name__ == "__main__":
    main()
