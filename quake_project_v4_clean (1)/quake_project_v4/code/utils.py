"""Configuration, paths, provenance and the numerical helpers shared by the
dataset builder and the analysis."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GLOBAL_SEED = 42

CATS = ["comcat_global_m4", "scedc_socal_m25", "comcat_himalaya_m4"]
MODELS = ["catboost", "xgboost", "lightgbm"]

# Reference labelling window. An event is a mainshock unless a larger event
# occurs within REF_R km and REF_T days either side of it.
REF_LABEL = "label_R50_T30"
REF_R, REF_T = 50.0, 30.0

RADII_KM = [25, 50, 75, 100]
WINDOWS_DAYS = [15, 30, 60, 90]
T_MAX_DAYS = max(WINDOWS_DAYS)

# Candidate model features. Each catalog uses the subset its source provides;
# the per-catalog difference is recorded in feature_availability.csv.
FULL_FEATURES = ["mw", "depth", "gap", "rms", "dmin", "magError", "depthError",
                 "horizontalError", "local_count_7d", "log_energy_30d",
                 "mag_diff_30d"]

ALPHAS = [0.05, 0.10, 0.20]
ALPHA = 0.10
ACI_GAMMA = 0.005
N_BINS = 15
N_BOOT = 2000

EARTH_R = 6371.0
BATCH = 200

# Chronological train / validation / calibration / test.
SPLITS = (0.60, 0.10, 0.15, 0.15)

# Pinned so the FDSN query is identical whichever day it runs, and so year
# chunks downloaded in different sessions agree with one another.
CATALOG_END = "2026-09-01"

LGBM_DETERMINISM = {"deterministic": True, "force_row_wise": True,
                    "verbosity": -1}


class Paths:
    """Every path in the project derives from one project root."""

    def __init__(self, root, run_name: str = "v4_fresh"):
        self.root = Path(root).expanduser().resolve()
        self.run_name = run_name

        self.code = self.root / "code"
        self.raw = self.root / "raw_data"
        self.chunks = self.raw / "chunks"
        self.dataset = self.root / "dataset"
        # The dataset build owns these, so they sit outside results/ and a
        # re-run of the analysis cannot invalidate them.
        self.dataset_ckpt = self.dataset / "ckpt"
        self.dataset_logs = self.dataset / "logs"

        self.run = self.root / "results" / run_name
        self.models = self.run / "models"
        self.tables = self.run / "tables"
        self.figures = self.run / "figures"
        self.logs = self.run / "logs"
        self.ckpt = self.run / "ckpt"
        self.manifest = self.run / "run_manifest.json"

        self.provenance = self.dataset / "data_provenance.json"
        self.summary = self.dataset / "dataset_summary.csv"

    def mkdirs(self, analysis: bool = True):
        for p in (self.raw, self.chunks, self.dataset, self.dataset_ckpt,
                  self.dataset_logs):
            p.mkdir(parents=True, exist_ok=True)
        if analysis:
            for p in (self.run, self.models, self.tables, self.figures,
                      self.logs, self.ckpt):
                p.mkdir(parents=True, exist_ok=True)
        return self

    def raw_catalog(self, cat):
        return self.raw / f"{cat}.parquet"

    def labeled(self, cat):
        return self.dataset / f"{cat}_labeled.parquet"

    def features(self, cat):
        return self.dataset / f"{cat}_features.parquet"

    def require_dataset(self):
        missing = [c for c in CATS if not self.features(c).exists()]
        if missing:
            sys.exit("dataset not found for: " + ", ".join(missing)
                     + f"\nBuild it first:  python code/make_dataset.py "
                       f"--root {self.root}")

    def __repr__(self):
        return f"<Paths root={self.root} run={self.run_name}>"


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def resolve_gpu(requested: bool) -> bool:
    """GPU is used only when asked for and actually present."""
    if not requested:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return False
    try:
        found = subprocess.run(["nvidia-smi"],
                               capture_output=True).returncode == 0
    except Exception:
        found = False
    if not found:
        print("--gpu given but no GPU was found; continuing on CPU")
    return found


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1, encoding="utf-8")
        self.out = sys.__stdout__

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def start_log(log_dir: Path, tag: str) -> Path:
    p = log_dir / f"{tag}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.txt"
    sys.stdout = sys.stderr = Tee(p)
    return p


_T0 = [time.time()]
STAGES: list = []


def reset_clock():
    _T0[0] = time.time()
    STAGES.clear()


def stage(n, label):
    minutes = (time.time() - _T0[0]) / 60
    print(f"\n[stage {n}] {label}   (+{minutes:.1f} min)")
    STAGES.append((n, label, minutes))


def elapsed_hours() -> float:
    return (time.time() - _T0[0]) / 3600.0


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def environment() -> dict:
    env = {"python": sys.version.split()[0],
           "platform": sys.platform,
           "utc": datetime.now(timezone.utc).isoformat()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
                "catboost", "optuna", "matplotlib", "pyarrow"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "absent"
    return env


def code_version(code_dir: Path) -> dict:
    return {p.name: sha256(p)[:16] for p in sorted(code_dir.glob("*.py"))}


TABLES: list = []


def save_table(P: Paths, df: pd.DataFrame, name: str, note: str = ""):
    if not name.endswith(".csv"):
        name += ".csv"
    p = P.tables / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    TABLES.append({"name": name, "rows": int(len(df)), "note": note})
    print(f"  saved {name} ({len(df)} rows)")
    return p


def load_table(P: Paths, name: str):
    p = P.tables / (name if name.endswith(".csv") else name + ".csv")
    return pd.read_csv(p) if p.exists() else None


def load_features(P: Paths, cat: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(P.features(cat), columns=columns)


def load_splits(P: Paths, cat: str):
    s = np.load(P.models / f"{cat}__splits.npz")
    return int(s["b1"]), int(s["b2"]), int(s["b3"])


def load_probs(P: Paths, cat: str, model: str, recal: bool = True):
    tag = "recal_probs" if recal else "probs"
    return np.load(P.models / f"{cat}__{model}__{tag}.npz")["prob"] \
        .astype(np.float64)


def chrono_split(n: int, fracs=SPLITS):
    """Chronological split; returns the three cut indices."""
    b1 = int(n * fracs[0])
    b2 = int(n * (fracs[0] + fracs[1]))
    b3 = int(n * (fracs[0] + fracs[1] + fracs[2]))
    return b1, b2, b3


def write_manifest(P: Paths, extra: dict | None = None) -> Path:
    prov = (json.loads(P.provenance.read_text())
            if P.provenance.exists() else "MISSING")
    manifest = {
        "run_name": P.run_name,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_hours": round(elapsed_hours(), 3),
        "global_seed": GLOBAL_SEED,
        "project_root": str(P.root),
        "code_version": code_version(P.code),
        "environment": environment(),
        "dataset_provenance": prov,
        "parameters": {
            "reference_label": REF_LABEL,
            "radii_km": RADII_KM,
            "windows_days": WINDOWS_DAYS,
            "splits": SPLITS,
            "alpha": ALPHA,
            "alphas": ALPHAS,
            "n_boot": N_BOOT,
            "catalog_end": CATALOG_END,
            "model_features": FULL_FEATURES,
        },
        "tables_written": TABLES,
        "stages": [{"n": n, "label": lab, "elapsed_min": round(e, 2)}
                   for n, lab, e in STAGES],
    }
    if extra:
        manifest.update(extra)
    P.manifest.parent.mkdir(parents=True, exist_ok=True)
    P.manifest.write_text(json.dumps(manifest, indent=2, default=str))
    return P.manifest


# Geometry. Distances are computed as chord lengths on the unit sphere, which
# keeps the neighbour search in a KD-tree.
def to_xyz(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)])


def chord_from_km(d_km):
    return 2.0 * np.sin(d_km / (2.0 * EARTH_R))


def km_from_chord(chord):
    return 2.0 * EARTH_R * np.arcsin(np.clip(np.asarray(chord) / 2.0, 0, 1))


# Seismological helpers
def aki_b(mags, mc, min_n=50):
    """Aki (1965) maximum-likelihood b-value for events at or above mc."""
    m = np.asarray(mags, dtype=float)
    m = m[m >= mc - 1e-9]
    if len(m) < min_n or (m.mean() - mc) <= 1e-6:
        return np.nan
    return float(1.0 / (np.log(10.0) * (m.mean() - mc)))


def mc_maxc(mags, bin_w=0.1, correction=0.2, min_n=50):
    """Magnitude of completeness: modal bin of the frequency-magnitude
    distribution plus the usual MAXC offset."""
    m = np.asarray(mags, dtype=float)
    m = m[np.isfinite(m)]
    if len(m) < min_n:
        return np.nan
    edges = np.arange(np.floor(m.min() * 10) / 10,
                      np.ceil(m.max() * 10) / 10 + bin_w, bin_w)
    h, e = np.histogram(m, bins=edges)
    if h.sum() == 0:
        return np.nan
    return float(e[int(np.argmax(h))] + correction)


# Calibration and conformal prediction
def ece(y, p, n_bins=N_BINS):
    order = np.argsort(p)
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    e = 0.0
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b):
            e += (len(b) / len(p)) * abs(y[b].mean() - p[b].mean())
    return float(e)


def nll(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def split_threshold(scores, alpha):
    n = len(scores)
    if n == 0:
        return 1.0
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, min(q, 1.0), method="higher"))


def prediction_sets(p, qhat):
    return p <= qhat, (1.0 - p) <= qhat


def mondrian_thresholds(p_cal, y_cal, alpha):
    q1 = split_threshold(1.0 - p_cal[y_cal == 1], alpha)
    q0 = split_threshold(p_cal[y_cal == 0], alpha)
    return q0, q1


def set_metrics(y, in0, in1):
    """Coverage and the three set sizes. Size 0, 1 and 2 are operationally
    distinct and the mean size alone does not distinguish them."""
    size = in0.astype(int) + in1.astype(int)
    covered = np.where(y == 1, in1, in0)
    empty = float((size == 0).mean())
    both = float((size == 2).mean())
    mean_size = float(size.mean())
    return {
        "coverage": float(covered.mean()),
        "cov_class1": float(covered[y == 1].mean()) if (y == 1).any() else np.nan,
        "cov_class0": float(covered[y == 0].mean()) if (y == 0).any() else np.nan,
        "avg_set_size": mean_size,
        "rate_empty": empty,
        "rate_singleton": float((size == 1).mean()),
        "rate_both": both,
        "identity_gap": abs(mean_size - (1.0 - empty + both)),
    }


def aci_trace(p_te, y_te, p_cal, y_cal, alpha=ALPHA, gamma=ACI_GAMMA):
    """Adaptive conformal inference: the miscoverage level walks in response
    to realised coverage over the test block."""
    cal_scores = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    a_t = alpha
    cov = np.zeros(len(y_te))
    size = np.zeros(len(y_te), dtype=int)
    for t in range(len(y_te)):
        qhat = split_threshold(cal_scores, float(np.clip(a_t, 1e-4, 0.999)))
        in0, in1 = p_te[t] <= qhat, (1.0 - p_te[t]) <= qhat
        covered = in1 if y_te[t] == 1 else in0
        cov[t] = covered
        size[t] = int(in0) + int(in1)
        a_t = a_t + gamma * (alpha - (0.0 if covered else 1.0))
    return cov, size


# Prediction set -> operational state. An empty set carries no information,
# so it escalates rather than resolving to the more likely class.
DECISION_MAP = {
    (False, True): "RESPONSE",
    (True, False): "ALERT",
    (True, True): "ESCALATE",
    (False, False): "ESCALATE",
}


def decide(in0, in1):
    a = np.empty(len(in0), dtype=object)
    for k, v in DECISION_MAP.items():
        a[(in0 == k[0]) & (in1 == k[1])] = v
    return a


def decide_argmax(p, in0, in1):
    """Alternative rule: empty sets resolve to the more likely class."""
    empty = ~(in0 | in1)
    i1 = in1 | (empty & (p >= 0.5))
    i0 = in0 | (empty & (p < 0.5))
    return decide(i0, i1), empty


def bh_adjust(p):
    """Benjamini-Hochberg step-up adjustment."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    q = p[ok]
    n = len(q)
    if n == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, p1, p2):
    """Two-sided p-value for AUC(p1) == AUC(p2) on the same sample.
    DeLong, DeLong & Clarke-Pearson (1988)."""
    from scipy.stats import norm
    y = np.asarray(y)
    m, n = int((y == 1).sum()), int((y == 0).sum())
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan
    preds = np.vstack([p1, p2])
    tx, ty = np.empty((2, m)), np.empty((2, n))
    tz = np.empty((2, m + n))
    for r in range(2):
        px, py = preds[r][y == 1], preds[r][y == 0]
        tz[r] = _midrank(np.concatenate([px, py]))
        tx[r] = _midrank(px)
        ty[r] = _midrank(py)
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    d = np.array([1.0, -1.0])
    var = float(d @ S @ d)
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z)))
