"""Figures and the self-contained HTML results file. Everything here is built
from the tables written by the analysis, so figures and tables cannot drift
apart."""
from __future__ import annotations

import base64
import html as _html
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import (ALPHA, CATS, GLOBAL_SEED, REF_R, REF_T, Paths,  # noqa: E402
                   load_features)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})

COST_FN = [5, 10, 50]

TABLE_ORDER = [
    ("Primary performance and calibration", "calibration_metrics.csv",
     "AUC, F1, precision, recall, ECE, Brier and NLL for every catalog, model "
     "and recalibrator, from the same test-set probabilities."),
    ("Sensitivity to the label definition", "sensitivity_grid.csv",
     "Every labelling window in the grid, with its conformal set-size "
     "breakdown and the fraction of labels that change class."),
    ("ETAS null", "etas_null.csv",
     "The same pipeline applied to simulated catalogs in which offspring "
     "magnitudes are independent of their parents."),
    ("Fitted ETAS parameters", "etas_parameters.csv",
     "Per catalog: completeness magnitude, b, kernel parameters, implied "
     "branching ratio and the calibration status of the productivity."),
    ("Magnitude baselines", "baselines.csv",
     "The hard-verdict and continuous magnitude-difference scores against the "
     "model, with DeLong tests."),
    ("Classical b-value baselines", "bvalue_baselines.csv",
     "Aki (1965) local b-value and the b-value drop."),
    ("Conformal prediction sets", "conformal_sets.csv",
     "Coverage and the rates of empty, singleton and two-class sets."),
    ("Prediction set to operational state", "decision_states.csv",
     "The decision rule as a total function on prediction sets."),
    ("Decision rule: empty-set routing", "decision_rule_comparison.csv",
     "Escalating empty sets against resolving them by argmax, per catalog and "
     "risk level."),
    ("Operational cost model", "decision_costs.csv",
     "Expected cost against alpha under both routings."),
    ("Operational subset and foreshock recall", "subset_metrics.csv",
     "Pooled metrics, metrics on events with no larger predecessor, and the "
     "share of foreshocks declined as mainshocks."),
    ("Negative-class composition", "negative_class_composition.csv",
     "Aftershocks against foreshocks within class 0."),
    ("Label and feature coupling", "label_feature_coupling.csv",
     "How much of the label the magnitude difference and the backward half of "
     "the rule reproduce alone."),
    ("Feature ablation", "feature_ablation.csv",
     "Full, no-magdiff, magdiff-only and no-magnitude-family models with "
     "BH-adjusted DeLong p-values."),
    ("Completeness-controlled ablation", "completeness_ablation.csv",
     "Whether skill without magnitude features survives above the detection "
     "threshold."),
    ("Magnitude of completeness over time", "mc_over_time.csv",
     "MAXC estimate in sliding windows of events."),
    ("Sequence-level intervals", "sequence_cis.csv",
     "Bootstrap intervals resampled over sequences rather than events."),
    ("Hyperparameters at the grid corners", "corner_retune.csv",
     "Frozen against per-window hyperparameter selection."),
    ("Right-censoring", "right_censoring.csv",
     "Metrics with and without the final T_max days of the test block."),
    ("Cross-region transfer", "transfer_pairs.csv",
     "Two donor-target pairs, with marginal and class-conditional coverage "
     "reported separately."),
    ("Nearest-neighbour relabelling", "zbz_relabelling.csv",
     "The windowed labels against Zaliapin & Ben-Zion declustering."),
    ("Case replay across the grid", "case_replay_grid.csv",
     "Kahramanmaras, Ridgecrest and Kumamoto at every label definition."),
    ("Case-study state instability", "case_state_instability.csv",
     "Large events whose assigned state depends on the labelling window."),
    ("Feature availability", "feature_availability.csv",
     "Which model features each source catalog provides."),
    ("SHAP stability", "shap_stability.csv",
     "How far the feature ranking moves when only the label moves."),
]

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
max-width:1180px;margin:0 auto;padding:28px 22px;color:#1a1a1a;line-height:1.5}
h1{border-bottom:3px solid #1a1a1a;padding-bottom:8px;margin-bottom:4px}
h2{margin-top:38px;border-bottom:1px solid #ccc;padding-bottom:5px}
h3{margin-top:26px;color:#333}
.sub{color:#666;font-size:.9em;margin-bottom:24px}
.note{background:#f6f7f9;border-left:4px solid #4C72B0;padding:10px 14px;
margin:12px 0;font-size:.92em}
table{border-collapse:collapse;width:100%;font-size:.82em;margin:12px 0;
overflow-x:auto;display:block}
th,td{border:1px solid #ddd;padding:4px 7px;text-align:right;white-space:nowrap}
th{background:#f0f1f3;text-align:left;position:sticky;top:0}
td:first-child,th:first-child{text-align:left}
tr:nth-child(even){background:#fafafa}
img{max-width:100%;border:1px solid #e0e0e0;margin:10px 0}
.cap{font-size:.85em;color:#555;font-style:italic;margin-bottom:18px}
.toc{background:#f6f7f9;padding:14px 20px;border-radius:5px;font-size:.9em}
.toc a{color:#2a5db0;text-decoration:none}
code{background:#f0f1f3;padding:1px 5px;border-radius:3px;font-size:.9em}
.missing{color:#999;font-style:italic}
"""


def _read(P, name):
    p = P.tables / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _parse_time(col):
    # Timestamps survive a CSV round-trip with and without fractional seconds,
    # so the format is given explicitly rather than inferred from row one.
    return pd.to_datetime(col, format="ISO8601", utc=True)


def _short(name):
    return (name.replace("comcat_", "").replace("_m4", "").replace("_m25", ""))


def build_figures(P: Paths):
    figs = []

    def save(fig, name, caption):
        path = P.figures / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figs.append({"name": name, "path": str(path), "caption": caption})
        print(f"  {name}.png")

    grid = _read(P, "sensitivity_grid.csv")

    def heatmaps(value, title, fmt="{:.3f}", cmap="viridis", shared=False):
        if grid is None or value not in grid.columns:
            return
        vmin = vmax = None
        if shared:
            vmin, vmax = float(grid[value].min()), float(grid[value].max())
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-9
        fig, axes = plt.subplots(1, len(CATS), figsize=(4.2 * len(CATS), 3.4))
        axes = np.atleast_1d(axes)
        for ax, cat in zip(axes, CATS):
            s = grid[(grid.catalog == cat)
                     & (grid.config.str.startswith("label_R"))].copy()
            if s.empty:
                ax.axis("off")
                continue
            s["R"] = s.config.str.extract(r"R(\d+)").astype(int)
            s["T"] = s.config.str.extract(r"T(\d+)").astype(int)
            m = s.pivot_table(index="R", columns="T", values=value)
            im = ax.imshow(m.values, cmap=cmap, aspect="auto", vmin=vmin,
                           vmax=vmax)
            ax.set_xticks(range(len(m.columns)), m.columns)
            ax.set_yticks(range(len(m.index)), m.index)
            ax.set_xlabel("half-window T (days)")
            ax.set_ylabel("radius R (km)")
            ax.set_title(_short(cat), fontsize=8)
            ax.grid(False)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    v = m.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, fmt.format(v), ha="center", va="center",
                                fontsize=7, color="w")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(title, fontsize=10)
        save(fig, f"grid_{value}", title)

    heatmaps("auc", "Test AUC across the labelling grid")
    heatmaps("f1", "Test F1 across the labelling grid", cmap="magma")
    heatmaps("flip_rate_vs_ref",
             "Fraction of labels that change class against the reference "
             "window", cmap="inferno")
    heatmaps("rate_empty", f"Empty prediction sets at alpha = {ALPHA:.2f}",
             cmap="cividis", shared=True)

    if grid is not None:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for cat in CATS:
            s = grid[grid.catalog == cat]
            if s.empty:
                continue
            ax.scatter(s.flip_rate_vs_ref, s.auc, marker="o",
                       label=f"{_short(cat)} AUC")
            ax.scatter(s.flip_rate_vs_ref, s.f1, marker="x",
                       label=f"{_short(cat)} F1")
        ax.set_xlabel("fraction of labels flipped against the reference")
        ax.set_ylabel("test metric")
        ax.set_title("AUC and F1 against label movement")
        ax.legend(fontsize=6, ncol=2)
        save(fig, "auc_vs_f1_divergence",
             "Ranking skill and thresholded utility respond differently to a "
             "change in the labelling window.")

    rel = _read(P, "reliability_points.csv")
    if rel is not None and len(rel):
        fig, axes = plt.subplots(1, len(CATS), figsize=(4.0 * len(CATS), 3.4))
        axes = np.atleast_1d(axes)
        for ax, cat in zip(axes, CATS):
            s = rel[(rel.catalog == cat) & (rel.model == "xgboost")]
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
            for v in s.calibrator.unique():
                t = s[s.calibrator == v]
                ax.plot(t.mean_pred, t.frac_pos, marker="o", ms=3, label=v)
            ax.set_xlabel("mean predicted probability")
            ax.set_ylabel("observed frequency")
            ax.set_title(_short(cat), fontsize=8)
            ax.legend(fontsize=6)
        fig.suptitle("Reliability diagrams, raw and recalibrated", fontsize=10)
        save(fig, "reliability", "Calibration before and after recalibration.")

    aci = _read(P, "aci_rolling.csv")
    if aci is not None and len(aci):
        fig, axes = plt.subplots(1, len(CATS), figsize=(4.0 * len(CATS), 3.0))
        axes = np.atleast_1d(axes)
        for ax, cat in zip(axes, CATS):
            s = aci[(aci.catalog == cat) & (aci.model == "xgboost")]
            ax.plot(s.t, s.rolling_cov_split, label="fixed threshold")
            ax.plot(s.t, s.rolling_cov_aci, label="adaptive (ACI)")
            ax.axhline(1 - ALPHA, ls="--", c="k", lw=1,
                       label=f"nominal {1 - ALPHA:.2f}")
            ax.set_xlabel("test event index (chronological)")
            ax.set_ylabel("rolling coverage")
            ax.set_title(_short(cat), fontsize=8)
            ax.legend(fontsize=6)
        fig.suptitle("Rolling empirical coverage over the test block",
                     fontsize=10)
        save(fig, "aci_coverage",
             "Fixed-threshold and adaptive conformal coverage through time.")

    conf = _read(P, "conformal_sets.csv")
    if conf is not None:
        s = conf[(conf.method == "mondrian") & (conf.alpha == ALPHA)
                 & (conf.model == "xgboost")]
        if len(s):
            fig, ax = plt.subplots(figsize=(5.4, 3.2))
            x = np.arange(len(s))
            ax.bar(x, s.rate_empty, label="empty", color="#C44E52")
            ax.bar(x, s.rate_singleton, bottom=s.rate_empty,
                   label="singleton", color="#4C72B0")
            ax.bar(x, s.rate_both, bottom=s.rate_empty + s.rate_singleton,
                   label="both classes", color="#DD8452")
            ax.set_xticks(x, [_short(t) for t in s.catalog], rotation=20)
            ax.set_ylabel("fraction of test events")
            ax.set_title("Prediction-set composition by set size")
            ax.legend(fontsize=7)
            save(fig, "set_composition",
                 "An empty set excludes both classes and has coverage zero by "
                 "construction; a two-class set is an abstention.")

    null = _read(P, "etas_null.csv")
    base = _read(P, "baselines.csv")
    if null is not None and len(null) and base is not None:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        real = base[base.score == "xgboost"]
        md = base[base.score == "magdiff (continuous)"]
        x = np.arange(len(real))
        ax.bar(x - 0.2, real.auc, 0.35, label="model (real catalog)")
        ax.bar(x + 0.2, md.auc, 0.35, label="magdiff baseline (real)")
        ax.axhspan(null.auc_pooled.min(), null.auc_pooled.max(), color="grey",
                   alpha=0.35, label="ETAS null, pooled")
        ax.axhline(null.auc_pooled.mean(), color="k", ls="--", lw=1)
        ax.set_xticks(x, [_short(t) for t in real.catalog], rotation=20)
        ax.set_ylabel("test AUC")
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Discrimination against the ETAS null")
        ax.legend(fontsize=7)
        save(fig, "etas_null_floor",
             "Pooled AUC on real catalogs and on simulated catalogs where no "
             "event-role physics exists by construction.")

    shap = _read(P, "shap_stability.csv")
    if shap is not None and len(shap):
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        for cat in CATS:
            s = shap[shap.catalog == cat]
            if s.empty:
                continue
            ax.plot(range(len(s)), s.kendall_tau_vs_ref, marker="o", ms=3,
                    label=_short(cat))
        ax.set_xlabel("label configuration index")
        ax.set_ylabel("Kendall tau against the reference ranking")
        ax.set_title("Feature-attribution stability across label definitions")
        ax.legend(fontsize=7)
        save(fig, "shap_stability",
             "How far the explanation moves when only the label moves.")

    costs = _read(P, "decision_costs.csv")
    if costs is not None and len(costs):
        fig, axes = plt.subplots(1, len(COST_FN),
                                 figsize=(3.6 * len(COST_FN), 3.0))
        axes = np.atleast_1d(axes)
        for ax, cfn in zip(axes, COST_FN):
            for rule in ("escalate", "argmax"):
                s = costs[(costs.c_FN == cfn) & (costs.rule == rule)]
                ax.plot(s.alpha, s.expected_cost, label=rule)
            ax.set_xlabel("alpha")
            ax.set_ylabel("expected cost")
            ax.set_title(f"c_FN = {cfn}", fontsize=8)
            ax.legend(fontsize=6)
        fig.suptitle("Operating point under each empty-set routing",
                     fontsize=10)
        save(fig, "cost_model",
             "Expected cost against the conformal risk level.")

    mc = _read(P, "mc_over_time.csv")
    if mc is not None and len(mc):
        fig, axes = plt.subplots(2, len(CATS), figsize=(4.4 * len(CATS), 6),
                                 sharex="col")
        axes = np.atleast_2d(axes)
        for j, cat in enumerate(CATS):
            years = load_features(P, cat, ["time"])["time"].dt.year
            counts = years.value_counts().sort_index()
            axes[0, j].bar(counts.index, counts.values, width=1.0,
                           color="#4C72B0")
            axes[0, j].set_yscale("log")
            axes[0, j].set_title(f"{_short(cat)}\nevents per year", fontsize=9)
            s = mc[mc.catalog == cat]
            axes[1, j].plot(_parse_time(s.time).dt.year, s.mc, color="#C44E52")
            axes[1, j].set_title("magnitude of completeness (MAXC + 0.2)",
                                 fontsize=9)
            axes[1, j].set_xlabel("year")
        save(fig, "counts_and_mc",
             "Event counts and the magnitude of completeness are distinct "
             "quantities and are plotted separately.")

    tf = _read(P, "transfer_pairs.csv")
    if tf is not None and len(tf):
        panels = [(m, t) for m, t in [("auc", "discrimination (AUC)"),
                                      ("coverage", "marginal coverage"),
                                      ("cov_gap", "class-conditional gap")]
                  if m in tf.columns]
        fig, axes = plt.subplots(1, len(panels),
                                 figsize=(4.0 * len(panels), 3.3))
        axes = np.atleast_1d(axes)
        pairs = list(tf.pair.unique())
        systems = list(tf.system.unique())
        width = 0.8 / max(len(systems), 1)
        for ax, (metric, title) in zip(axes, panels):
            for k, system in enumerate(systems):
                sv = tf[tf.system == system]
                vals = [sv[sv.pair == p][metric].mean() for p in pairs]
                off = (k - (len(systems) - 1) / 2.0) * width
                ax.bar(np.arange(len(pairs)) + off, vals, width, label=system)
            ax.set_xticks(range(len(pairs)),
                          [_short(p.replace("global -> ", ""))
                           for p in pairs], rotation=12)
            ax.set_ylabel(metric)
            ax.set_title(title, fontsize=9)
            if metric == "coverage":
                ax.axhline(1 - ALPHA, ls="--", c="k", lw=1,
                           label=f"nominal {1 - ALPHA:.2f}")
            ax.legend(fontsize=6)
        fig.suptitle("Cross-region transfer", fontsize=10)
        save(fig, "transfer",
             "Discrimination, marginal coverage and the class-conditional "
             "coverage gap for each donor-target pair.")

    dec = _read(P, "decision_rule_comparison.csv")
    if dec is not None and len(dec):
        d = dec[np.isclose(dec.alpha, ALPHA)]
        cats = list(d.catalog.unique())
        if cats:
            x = np.arange(len(cats))
            w = 0.35
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
            for ax, (col, title) in zip(axes, [
                    ("P_ESCALATE", "Escalation rate"),
                    ("P_false_RESPONSE", "False RESPONSE rate"),
                    (None, "Prediction-set composition")]):
                if col is None:
                    emp = [d[d.catalog == c].rate_empty.iloc[0] for c in cats]
                    sing = [d[d.catalog == c].rate_singleton.iloc[0]
                            for c in cats]
                    both = [d[d.catalog == c].rate_both.iloc[0] for c in cats]
                    ax.bar(x, emp, label="empty")
                    ax.bar(x, sing, bottom=emp, label="singleton")
                    ax.bar(x, both, bottom=np.array(emp) + np.array(sing),
                           label="both classes")
                else:
                    esc = [d[(d.catalog == c)
                             & d.rule.str.endswith("ESCALATE")][col].iloc[0]
                           for c in cats]
                    arg = [d[(d.catalog == c)
                             & d.rule.str.endswith("argmax")][col].iloc[0]
                           for c in cats]
                    ax.bar(x - w / 2, arg, w, label="empty -> argmax")
                    ax.bar(x + w / 2, esc, w, label="empty -> ESCALATE")
                ax.set_xticks(x)
                ax.set_xticklabels([_short(c) for c in cats], rotation=15,
                                   ha="right", fontsize=8)
                ax.set_title(title, fontsize=10)
                ax.legend(fontsize=7)
            save(fig, "decision_rule_empty_set",
                 f"Effect of routing empty prediction sets to ESCALATE at "
                 f"alpha = {ALPHA}. The rules coincide wherever the empty-set "
                 f"rate is zero.")

    print(f"  {len(figs)} figures written to {P.figures}")
    return figs


def _table_html(df, max_rows=200):
    if df is None or df.empty:
        return '<p class="missing">not produced in this run</p>'
    note = ""
    if len(df) > max_rows:
        note = (f'<p class="cap">first {max_rows} of {len(df)} rows; the CSV '
                f'holds the rest</p>')
        df = df.head(max_rows)
    return df.to_html(index=False, float_format=lambda x: f"{x:.4g}",
                      na_rep="-", border=0) + note


def _img_b64(path):
    try:
        return base64.b64encode(Path(path).read_bytes()).decode()
    except Exception:
        return None


def build_report(P: Paths, figs):
    """One HTML file with every table and figure embedded."""
    parts = [f"<style>{CSS}</style>",
             "<h1>Mainshock discrimination — results</h1>",
             f'<p class="sub">Run {_html.escape(P.run_name)}, generated '
             f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Seed '
             f'{GLOBAL_SEED}. Reference window R = {REF_R:.0f} km, '
             f'T = {REF_T:.0f} d.</p>']

    parts.append("<h2>Contents</h2><div class='toc'><ol>")
    for i, (title, _, _) in enumerate(TABLE_ORDER):
        parts.append(f"<li><a href='#t{i}'>{_html.escape(title)}</a></li>")
    parts.append("</ol><ol start='100' style='list-style:none;padding:0'>")
    for f in figs:
        parts.append(f"<li><a href='#f{f['name']}'>Figure: "
                     f"{_html.escape(f['name'])}</a></li>")
    parts.append("</ol></div>")

    parts.append("<h2>Definitions</h2>")
    parts.append(
        '<div class="note"><b>Prediction sets.</b> A conformal set can be '
        '<code>{0}</code>, <code>{1}</code>, <code>{0,1}</code> or empty. '
        'Since <code>E[|C|] = 1 - P(empty) + P(both)</code> the mean size does '
        'not identify the composition, so the three rates are reported '
        'separately.</div>')
    parts.append(
        '<div class="note"><b>Operational subset.</b> Backward local maxima '
        'are events with no larger predecessor inside the labelling window. '
        'Within that subset a negative label means the event was followed by '
        'something larger, which is the operational question.</div>')
    parts.append(
        '<div class="note"><b>ETAS null.</b> Offspring magnitudes are drawn '
        'independently of their parents, so no event-role information exists '
        'in a simulated catalog. The fitted parameters, the implied branching '
        'ratio and the calibration status of each null are in the ETAS '
        'parameter table.</div>')

    parts.append("<h2>Tables</h2>")
    for i, (title, fname, blurb) in enumerate(TABLE_ORDER):
        df = _read(P, fname)
        parts.append(f"<h3 id='t{i}'>{i + 1}. {_html.escape(title)}</h3>")
        parts.append(f"<p class='cap'>{_html.escape(blurb)} "
                     f"<code>{_html.escape(fname)}</code></p>")
        parts.append(_table_html(df))

    parts.append("<h2>Figures</h2>")
    for f in figs:
        b64 = _img_b64(f["path"])
        parts.append(f"<h3 id='f{f['name']}'>{_html.escape(f['name'])}</h3>")
        parts.append(f'<img src="data:image/png;base64,{b64}">' if b64
                     else '<p class="missing">figure not available</p>')
        parts.append(f"<p class='cap'>{_html.escape(f['caption'])}</p>")

    parts.append("<h2>Run record</h2>")
    if P.manifest.exists():
        parts.append("<pre>" + _html.escape(P.manifest.read_text()[:8000])
                     + "</pre>")
    return "\n".join(parts)


def write_report(P: Paths, figs):
    out = P.run / "ALL_RESULTS.html"
    out.write_text(build_report(P, figs), encoding="utf-8")
    print(f"  report -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
