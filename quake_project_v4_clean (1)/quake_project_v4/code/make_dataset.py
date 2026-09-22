#!/usr/bin/env python3
"""Build the earthquake dataset from FDSN catalogs.

    python code/make_dataset.py --root /content/drive/MyDrive/quake_project_v4

Stages: download and clean the catalogs, apply the windowed labelling rules,
compute the event features, then record provenance and a summary.

Writes raw_data/<catalog>.parquet, dataset/<catalog>_labeled.parquet,
dataset/<catalog>_features.parquet, dataset/data_provenance.json and
dataset/dataset_summary.csv. It never writes into results/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    BATCH, CATALOG_END, CATS, EARTH_R, FULL_FEATURES, GLOBAL_SEED, RADII_KM,
    REF_LABEL, T_MAX_DAYS, WINDOWS_DAYS, Paths, chord_from_km, chrono_split,
    elapsed_hours, environment, reset_clock, seed_everything, sha256, stage,
    start_log, to_xyz,
)

COMCAT_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
COMCAT_COUNT = "https://earthquake.usgs.gov/fdsnws/event/1/count"
SCEDC_QUERY = "https://service.scedc.caltech.edu/fdsnws/event/1/query"
HIMALAYA_BOX = dict(min_lat=20.0, max_lat=32.0, min_lon=85.0, max_lon=98.0)
# The SCEDC event service returns its whole event table, which includes
# regional and teleseismic entries, so the download is constrained to the
# Southern California study region.
SOCAL_BOX = dict(min_lat=32.0, max_lat=37.0, min_lon=-122.0, max_lon=-114.0)

ROW_CAP = 19000
POLITE_SLEEP = 0.5
MAX_RETRIES = 4
MAX_DEPTH = 14

MW_FAMILY = {"mw", "mww", "mwc", "mwb", "mwr", "mi", "mwp"}
ML_FAMILY = {"ml", "mlr", "mlg", "ml(texnet)", "mb_lg", "mblg", "md", "mh",
             "mc", "me"}
MB_FAMILY = {"mb", "mb1", "mbmle"}
MS_FAMILY = {"ms", "ms_20", "msz"}

# SCEDC single-letter magnitude codes. Coda (c) and duration (d) magnitudes
# are calibrated against local magnitude, so they join the ML family.
SCEDC_MAGTYPE = {"w": "mw", "l": "ml", "b": "mb", "s": "ms", "c": "mc",
                 "d": "md", "h": "mh", "e": "me", "n": "ml"}

ADAPT_R = lambda m: np.minimum(10 ** (0.1238 * m + 0.983), 100.0)  # noqa: E731
ADAPT_T = lambda m: np.minimum(10 ** (0.5 * m - 0.547), 365.0)     # noqa: E731

FEAT_R_KM = 50.0
GAP_CAP_DAYS = 3650.0
EPS = 1e-9
CARRY = ["depth", "gap", "rms", "dmin", "horizontalError", "depthError",
         "magError", "magNst", "latitude", "longitude"]


# Magnitude homogenisation
def normalise_magtype(t):
    s = str(t).strip().lower()
    if not s or s in ("nan", "none"):
        return "unknown"
    if len(s) <= 2 and s not in (MB_FAMILY | MS_FAMILY):
        return SCEDC_MAGTYPE.get(s, s)
    return s


def to_mw(mag, mtype):
    t = normalise_magtype(mtype)
    if t in MW_FAMILY:
        return mag, "direct"
    if t in MB_FAMILY:
        return 0.85 * mag + 0.33, "mb"
    if t in ML_FAMILY:
        return 0.85 * mag + 0.15, "ml"
    if t in MS_FAMILY:
        return 0.67 * mag + 2.07, "ms"
    return mag, "unconverted"


# FDSN download
def bbox_params(box):
    """FDSN spatial constraint for a bounding box."""
    return {"minlatitude": box["min_lat"], "maxlatitude": box["max_lat"],
            "minlongitude": box["min_lon"], "maxlongitude": box["max_lon"]}


def _request(url, params):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=180)
            if r.status_code in (200, 204, 400, 413):
                return r
            print(f"    HTTP {r.status_code}, retry {attempt + 1}")
        except requests.RequestException as e:
            print(f"    {type(e).__name__}, retry {attempt + 1}")
        time.sleep(2 ** attempt * 2)
    return None


def parse_fdsn_text(txt):
    # Pipe-delimited FDSN text. SCEDC misspells "Longtitude" and reports the
    # event type in an ET column, so columns are mapped rather than assumed.
    lines = [line for line in txt.splitlines() if line.strip()]
    hdr, rows = None, []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if hdr is None:
            hdr = [p.lstrip("#").strip() for p in parts]
            continue
        if line.lstrip().startswith("#"):
            continue
        if len(parts) == len(hdr):
            rows.append(parts)
    if hdr is None or not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=hdr)
    ren = {"EventID": "id", "Time": "time", "Latitude": "latitude",
           "Longtitude": "longitude", "Longitude": "longitude",
           "Depth/km": "depth", "Depth/Km": "depth", "MagType": "magType",
           "Magnitude": "mag", "EventLocationName": "place", "ET": "type"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for c in ["latitude", "longitude", "depth", "mag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"].astype(str).str.replace("/", "-", regex=False),
            format="mixed", utc=True, errors="coerce")
    if "type" not in df.columns:
        df["type"] = "earthquake"
    return df


def parse_response(r, fmt):
    if fmt == "csv":
        return pd.read_csv(io.StringIO(r.text))
    return parse_fdsn_text(r.text)


def detect_format(query_url, minmag):
    probe = {"starttime": "2019-07-05T00:00:00",
             "endtime": "2019-07-06T00:00:00", "minmagnitude": minmag}
    for fmt in ("csv", "text"):
        r = _request(query_url, {**probe, "format": fmt})
        if r is not None and r.status_code in (200, 204):
            print(f"  endpoint format: {fmt}")
            return fmt
    raise RuntimeError(f"{query_url} accepted neither csv nor text")


def fetch_window(query_url, minmag, start, end, count_url=None, depth=0,
                 fmt="csv", extra=None):
    if depth > MAX_DEPTH or (end - start) < timedelta(hours=6):
        raise RuntimeError(
            f"cannot retrieve {start} to {end} by splitting further. This is "
            f"almost always a rejected request rather than an oversized one — "
            f"check the endpoint and format, not the window size.")

    def split():
        mid = start + (end - start) / 2
        return (fetch_window(query_url, minmag, start, mid, count_url,
                             depth + 1, fmt, extra)
                + fetch_window(query_url, minmag, mid, end, count_url,
                               depth + 1, fmt, extra))

    base = {"starttime": start.isoformat(), "endtime": end.isoformat(),
            "minmagnitude": minmag, **(extra or {})}
    if count_url:
        r = _request(count_url, base)
        if r is None:
            raise RuntimeError(f"count failed {start}-{end}")
        n = int(r.text.strip()) if r.text.strip() else 0
        if n == 0:
            return []
        if n > ROW_CAP:
            return split()

    r = _request(query_url, {**base, "format": fmt, "orderby": "time-asc"})
    if r is None:
        raise RuntimeError(f"query failed {start}-{end}")
    if r.status_code == 413:
        return split()
    if r.status_code == 400:
        # A 400 is either "too many rows" or "bad request"; only the first is
        # fixable by splitting, so inspect the body before recursing.
        body = r.text[:300].lower()
        if any(k in body for k in ("too many", "exceed", "limit", "413")):
            return split()
        raise RuntimeError(f"endpoint rejected the request ({start}-{end}): "
                           f"{r.text[:200].strip()}")
    if r.status_code == 204 or not r.text.strip():
        return []
    df = parse_response(r, fmt)
    if len(df) >= ROW_CAP:
        return split()
    print(f"  {'  ' * depth}{start.date()} -> {end.date()}: {len(df):>6}")
    time.sleep(POLITE_SLEEP)
    return [df]


def chunk_key(name, minmag, extra):
    """Cache key for the per-year downloads. It includes the query, so a
    changed request is never served from chunks fetched under the old one."""
    q = json.dumps({"minmagnitude": minmag, **(extra or {})}, sort_keys=True)
    return f"{name}_{hashlib.sha256(q.encode()).hexdigest()[:8]}"


def fetch_catalog(P, name, query_url, minmag, start_year, end,
                  count_url=None, extra=None):
    """Year by year, cached in raw_data/chunks so a dropped session resumes."""
    P.chunks.mkdir(parents=True, exist_ok=True)
    fmt = detect_format(query_url, minmag)
    key = chunk_key(name, minmag, extra)
    frames = []
    for year in range(start_year, end.year + 1):
        ck = P.chunks / f"{key}_{year}.parquet"
        if ck.exists():
            frames.append(pd.read_parquet(ck))
            continue
        start = datetime(year, 1, 1)
        stop = min(datetime(year + 1, 1, 1), end)
        if stop <= start:
            continue
        year_frames = fetch_window(query_url, minmag, start, stop, count_url,
                                   0, fmt, extra)
        df = (pd.concat(year_frames, ignore_index=True) if year_frames
              else pd.DataFrame())
        df.to_parquet(ck, index=False)
        frames.append(df)
    frames = [f for f in frames if len(f)]
    if not frames:
        raise RuntimeError(f"{name}: the endpoint returned no events at all")
    return pd.concat(frames, ignore_index=True)


def clean(df):
    n0 = len(df)
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower().isin(["earthquake", "eq"])]
    df = df.dropna(subset=["time", "latitude", "longitude", "mag"])
    # nst is present in ComCat but not SCEDC; dropping it keeps the feature
    # set comparable across catalogs.
    df = df.drop(columns=["nst"], errors="ignore")
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id", keep="first")
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    df = df.sort_values("time").reset_index(drop=True)
    conv = [to_mw(m, t) for m, t in zip(df["mag"].to_numpy(),
                                        df.get("magType", pd.Series(
                                            ["unknown"] * len(df))).to_numpy())]
    df["mw"] = [c[0] for c in conv]
    df["mw_source"] = [c[1] for c in conv]
    unconv = int((df["mw_source"] == "unconverted").sum())
    print(f"  cleaned {n0} -> {len(df)} ({unconv} unconverted magnitude types)")
    if unconv > 0.5 * len(df):
        print(f"  [warn] over half of magnitudes were left unconverted; check "
              f"magType values: {sorted(set(df['magType'].astype(str)))[:12]}")
    return df


def save_catalog(df, path: Path, meta: dict):
    df.to_parquet(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved -> {path}")


def needs_download(path: Path, query: dict, rebuild: bool) -> bool:
    """A stored catalog is reused only when it was built by the query that
    is configured now; otherwise it is downloaded again."""
    if rebuild or not path.exists():
        return True
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return True
    meta = json.loads(meta_path.read_text())
    return any(meta.get(k) != v for k, v in query.items())


def build_catalogs(P, end, rebuild):
    """Download and clean the three catalogs into raw_data/."""
    global_path = P.raw_catalog("comcat_global_m4")
    global_query = {"minmagnitude": 4.0, "start_year": 1930}
    if not needs_download(global_path, global_query, rebuild):
        print("[1/3] comcat_global_m4 present")
        comcat = pd.read_parquet(global_path)
    else:
        print("[1/3] ComCat global, 1930 onwards, M>=4.0")
        comcat = clean(fetch_catalog(P, "comcat", COMCAT_QUERY, 4.0, 1930, end,
                                     count_url=COMCAT_COUNT))
        save_catalog(comcat, global_path, {
            "source": "comcat", **global_query,
            "n_events": int(len(comcat)),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "credit": "U.S. Geological Survey ANSS ComCat"})

    socal_path = P.raw_catalog("scedc_socal_m25")
    socal_query = {"minmagnitude": 2.5, "start_year": 1981,
                   "bbox": SOCAL_BOX}
    if not needs_download(socal_path, socal_query, rebuild):
        print("[2/3] scedc_socal_m25 present")
    else:
        b = SOCAL_BOX
        print(f"[2/3] SCEDC, 1981 onwards, M>=2.5, {b['min_lat']}-"
              f"{b['max_lat']}N {b['min_lon']}-{b['max_lon']}E "
              f"(no /count endpoint, so windows split reactively)")
        scedc = clean(fetch_catalog(P, "scedc", SCEDC_QUERY, 2.5, 1981, end,
                                    extra=bbox_params(SOCAL_BOX)))
        save_catalog(scedc, socal_path, {
            "source": "scedc", **socal_query,
            "n_events": int(len(scedc)),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "credit": "SCEDC / Caltech-USGS SCSN, doi:10.7909/C3WD3xH1"})

    him_path = P.raw_catalog("comcat_himalaya_m4")
    if not needs_download(him_path, {"bbox": HIMALAYA_BOX}, rebuild):
        print("[3/3] comcat_himalaya_m4 present")
    else:
        print("[3/3] Himalaya subset of ComCat (spatial filter, no download)")
        b = HIMALAYA_BOX
        him = comcat[comcat["latitude"].between(b["min_lat"], b["max_lat"])
                     & comcat["longitude"].between(b["min_lon"],
                                                   b["max_lon"])] \
            .reset_index(drop=True)
        save_catalog(him, him_path, {
            "source": "comcat_subset_himalaya", "bbox": b,
            "n_events": int(len(him)),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "note": "derived product; credit USGS ANSS ComCat"})


# Labels
def label_fixed(times_s, mw, tree, xyz, R_km, T_days):
    """Label 0 when a larger event occurs within R km and T days either side."""
    n = len(mw)
    lab = np.ones(n, dtype=np.int8)
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
            nb = nb[np.abs(times_s[nb] - times_s[i]) <= T_s]
            if nb.size and (mw[nb] > mw[i]).any():
                lab[i] = 0
        del neigh
        if (lo // BATCH) % 200 == 0:
            gc.collect()
    return lab


def label_adaptive(times_s, mw, tree, xyz):
    """Magnitude-dependent window instead of a fixed one."""
    n = len(mw)
    lab = np.ones(n, dtype=np.int8)
    max_chord = chord_from_km(100.0)
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        neigh = tree.query_ball_point(xyz[lo:hi], r=max_chord)
        for k in range(hi - lo):
            i = lo + k
            nb = np.asarray(neigh[k])
            neigh[k] = None
            nb = nb[nb != i]
            if nb.size == 0:
                continue
            bigger = nb[mw[nb] > mw[i]]
            if bigger.size == 0:
                continue
            dch = np.linalg.norm(xyz[bigger] - xyz[i], axis=1)
            d_km = 2.0 * EARTH_R * np.arcsin(np.clip(dch / 2.0, 0, 1))
            dt_d = np.abs(times_s[bigger] - times_s[i]) / 86400.0
            if ((d_km <= ADAPT_R(mw[bigger]))
                    & (dt_d <= ADAPT_T(mw[bigger]))).any():
                lab[i] = 0
        del neigh
        if (lo // BATCH) % 200 == 0:
            gc.collect()
    return lab


def label_configs():
    cfg = [(f"label_R{R}_T{T}", ("fixed", R, T))
           for R in RADII_KM for T in WINDOWS_DAYS]
    cfg.append(("label_adaptive", ("adaptive", None, None)))
    return cfg


def build_labels(P, cat, source_key):
    out = P.labeled(cat)
    raw = P.raw_catalog(cat)
    if out.exists() and out.stat().st_mtime >= raw.stat().st_mtime:
        print(f"[skip] {out.name}")
        return
    print(f"=== labelling {cat} ===")
    df = pd.read_parquet(raw).sort_values("time").reset_index(drop=True)
    times_s = ((df["time"] - pd.Timestamp(0, tz="UTC"))
               .dt.total_seconds().to_numpy())
    mw = df["mw"].to_numpy(float)
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    tree = cKDTree(xyz)
    for col, (kind, R, T) in label_configs():
        # Checkpoints carry the source hash, so a changed catalog cannot be
        # labelled from a stale array.
        ck = P.dataset_ckpt / f"{cat}_{source_key}__{col}.npy"
        if ck.exists():
            lab = np.load(ck)
        else:
            lab = (label_fixed(times_s, mw, tree, xyz, R, T) if kind == "fixed"
                   else label_adaptive(times_s, mw, tree, xyz))
            np.save(ck, lab)
        df[col] = lab
        print(f"  {col}: mainshock rate {lab.mean():.3f}")
        gc.collect()
    df.to_parquet(out, index=False)
    print(f"  saved -> {out}")
    del df, xyz, tree
    gc.collect()


# Features
def compute_features(times_s, mw, tree, xyz):
    """Local rate, released energy, magnitude deficit and quiescence, all from
    past events within FEAT_R_KM."""
    n = len(mw)
    f = {k: np.zeros(n, dtype=np.float32) for k in
         ["local_count_7d", "log_energy_30d", "mag_diff_30d", "seismicity_z"]}
    f["time_gap_local"] = np.full(n, GAP_CAP_DAYS, dtype=np.float32)
    chord = chord_from_km(FEAT_R_KM)
    D7, D30, D365 = 7 * 86400.0, 30 * 86400.0, 365 * 86400.0
    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        neigh = tree.query_ball_point(xyz[lo:hi], r=chord)
        for k in range(hi - lo):
            i = lo + k
            nb = np.asarray(neigh[k])
            neigh[k] = None
            nb = nb[nb != i]
            if nb.size == 0:
                f["mag_diff_30d"][i] = mw[i]
                continue
            dt = times_s[i] - times_s[nb]
            past = nb[dt > 0]
            dt = dt[dt > 0]
            if past.size == 0:
                f["mag_diff_30d"][i] = mw[i]
                continue
            f["time_gap_local"][i] = min(dt.min() / 86400.0, GAP_CAP_DAYS)
            m7, m30, m365 = dt <= D7, dt <= D30, dt <= D365
            x7 = float(m7.sum())
            f["local_count_7d"][i] = x7
            if m30.any():
                mw30 = mw[past[m30]]
                f["log_energy_30d"][i] = np.log10(
                    np.sum(10.0 ** (1.5 * mw30)) + EPS)
                f["mag_diff_30d"][i] = mw[i] - mw30.max()
            else:
                f["mag_diff_30d"][i] = mw[i]
            mu = (float(m365.sum()) / 365.0) * 7.0
            f["seismicity_z"][i] = (x7 - mu) / np.sqrt(mu + EPS)
        del neigh
        if (lo // BATCH) % 200 == 0:
            gc.collect()
    return f


def build_features(P, cat, source_key):
    out = P.features(cat)
    labeled = P.labeled(cat)
    if out.exists() and out.stat().st_mtime >= labeled.stat().st_mtime:
        print(f"[skip] {out.name}")
        return
    print(f"=== features {cat} ===")
    df = pd.read_parquet(labeled).sort_values("time").reset_index(drop=True)
    times_s = ((df["time"] - pd.Timestamp(0, tz="UTC"))
               .dt.total_seconds().to_numpy())
    mw = df["mw"].to_numpy(float)
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    ck = P.dataset_ckpt / f"{cat}_{source_key}__features.npz"
    if ck.exists():
        feats = dict(np.load(ck))
    else:
        feats = compute_features(times_s, mw, cKDTree(xyz), xyz)
        np.savez(ck, **feats)
    for c, a in feats.items():
        df[c] = a
    keep = (["id", "time", "mw", "mw_source", "place"]
            + [c for c in CARRY if c in df.columns]
            + [c for c in df.columns if c.startswith("label")]
            + list(feats.keys()))
    keep = list(dict.fromkeys(c for c in keep if c in df.columns))
    df[keep].to_parquet(out, index=False)
    print(f"  saved -> {out} ({len(keep)} columns)")
    del df, xyz
    gc.collect()


def feature_availability(P):
    """Which model features each catalog's source actually provides."""
    rows = []
    for cat in CATS:
        cols = set(pd.read_parquet(P.features(cat)).columns)
        rows.append({
            "catalog": cat,
            "n_model_features": sum(f in cols for f in FULL_FEATURES),
            "present": ", ".join(f for f in FULL_FEATURES if f in cols),
            "absent": ", ".join(f for f in FULL_FEATURES
                                if f not in cols) or "(none)"})
    out = pd.DataFrame(rows)
    out.to_csv(P.dataset / "feature_availability.csv", index=False)
    return out


def write_provenance(P, catalog_end, seed):
    prov = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_end_pinned": catalog_end,
        "global_seed": seed,
        "environment": environment(),
        "parameters": {
            "radii_km": RADII_KM,
            "windows_days": WINDOWS_DAYS,
            "t_max_days": T_MAX_DAYS,
            "reference_label": REF_LABEL,
            "feature_radius_km": FEAT_R_KM,
            "adaptive_R": "min(10**(0.1238*m + 0.983), 100) km",
            "adaptive_T": "min(10**(0.5*m - 0.547), 365) d",
            "mw_conversions": {"mb": "0.85*m + 0.33", "ml": "0.85*m + 0.15",
                               "ms": "0.67*m + 2.07", "mw_family": "direct"},
        },
        "endpoints": {"comcat": COMCAT_QUERY, "scedc": SCEDC_QUERY},
        "spatial_bounds": {"comcat_global_m4": "unrestricted",
                           "scedc_socal_m25": SOCAL_BOX,
                           "comcat_himalaya_m4": HIMALAYA_BOX},
        "inputs": {},
        "derived": {},
    }
    for cat in CATS:
        raw = P.raw_catalog(cat)
        entry = {"file": raw.name, "status": "ABSENT"}
        if raw.exists():
            df = pd.read_parquet(raw)
            t = pd.to_datetime(df["time"], utc=True, errors="coerce")
            entry = {
                "file": raw.name,
                "sha256": sha256(raw),
                "bytes": int(raw.stat().st_size),
                "n_events": int(len(df)),
                "t_start": str(t.min()),
                "t_end": str(t.max()),
                "mw_min": round(float(df["mw"].min()), 3),
                "mw_max": round(float(df["mw"].max()), 3),
            }
            for col in ("magType", "mw_source"):
                if col in df.columns:
                    entry[f"{col}_counts"] = {
                        str(k): int(v)
                        for k, v in df[col].value_counts().head(12).items()}
        prov["inputs"][cat] = entry
        print(f"  {cat:<22} {entry.get('n_events', 0):>8} events  "
              f"sha {entry.get('sha256', '')[:12]}")

    for cat in CATS:
        for kind, path in (("labeled", P.labeled(cat)),
                           ("features", P.features(cat))):
            if path.exists():
                prov["derived"][f"{cat}_{kind}"] = {
                    "file": path.name, "sha256": sha256(path),
                    "bytes": int(path.stat().st_size)}

    P.provenance.write_text(json.dumps(prov, indent=2))
    print(f"  provenance -> {P.provenance}")


def write_summary(P):
    rows, problems = [], []
    for cat in CATS:
        fea = P.features(cat)
        if not fea.exists():
            problems.append(f"{cat}: {fea.name} was not produced")
            continue
        df = pd.read_parquet(fea)
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        n = len(df)
        b1, b2, b3 = chrono_split(n)
        nonmono = int((t.diff().dt.total_seconds() < 0).sum())
        if nonmono:
            problems.append(f"{cat}: {nonmono} non-monotonic timestamps; the "
                            f"chronological split assumes sorted time")
        label_cols = [c for c in df.columns if c.startswith("label_")]
        feature_cols = [c for c in df.columns
                        if c not in label_cols and c != "time"]
        rows.append({
            "catalog": cat,
            "n_events": n,
            "n_columns": len(feature_cols),
            "n_label_defs": len(label_cols),
            "t_start": str(t.min().date()),
            "t_end": str(t.max().date()),
            "mw_min": round(float(df.mw.min()), 2),
            "mw_max": round(float(df.mw.max()), 2),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_events": int(df.duplicated(
                subset=["time", "latitude", "longitude", "mw"]).sum()),
            "non_monotonic_t": nonmono,
            "n_mainshock_ref": int(df[REF_LABEL].sum()),
            "mainshock_rate_ref": round(float(df[REF_LABEL].mean()), 4),
            "train": b1, "validation": b2 - b1,
            "calibration": b3 - b2, "test": n - b3,
            "features_sha16": sha256(fea)[:16],
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(P.summary, index=False)

    bal = []
    for cat in CATS:
        if not P.features(cat).exists():
            continue
        df = pd.read_parquet(P.features(cat))
        for c in sorted(x for x in df.columns if x.startswith("label_")):
            bal.append({"catalog": cat, "label": c,
                        "n_positive": int(df[c].sum()),
                        "n_negative": int((1 - df[c]).sum()),
                        "positive_rate": round(float(df[c].mean()), 4)})
    if bal:
        pd.DataFrame(bal).to_csv(P.dataset / "label_counts.csv", index=False)
    return problems


def clear_dataset(P):
    targets = (sorted(P.raw.glob("*.parquet")) + sorted(P.raw.glob("*.json"))
               + sorted(P.dataset.glob("*.parquet"))
               + sorted(P.dataset.glob("*.csv"))
               + [P.provenance, P.chunks, P.dataset_ckpt])
    targets = [p for p in targets if p.exists()]
    print(f"removing {len(targets)} dataset entries (results/ is not touched)")
    for p in targets:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    P.mkdirs(analysis=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default="/content/drive/MyDrive/quake_project_v4")
    ap.add_argument("--catalog-end", default=CATALOG_END,
                    help="pinned catalog end date, YYYY-MM-DD")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete raw_data/ and dataset/ and start again")
    args = ap.parse_args()

    seed_everything(GLOBAL_SEED)
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        pass

    P = Paths(args.root).mkdirs(analysis=False)
    log = start_log(P.dataset_logs, "dataset")
    end = pd.Timestamp(args.catalog_end).to_pydatetime()

    print(f"log          -> {log}")
    print(f"project root : {P.root}")
    print(f"catalog end  : {args.catalog_end} (pinned)")
    print(f"seed         : {GLOBAL_SEED}")

    if args.rebuild:
        clear_dataset(P)
    reset_clock()

    stage(1, "download and clean catalogs")
    build_catalogs(P, end, args.rebuild)
    for cat in CATS:
        d = pd.read_parquet(P.raw_catalog(cat),
                            columns=["time", "mw", "mw_source"])
        print(f"  {cat}: {len(d)} events, {d['time'].min().date()} -> "
              f"{d['time'].max().date()}, mw {d['mw'].min():.2f}-"
              f"{d['mw'].max():.2f}")

    stage(2, "windowed labels")
    keys = {cat: sha256(P.raw_catalog(cat))[:16] for cat in CATS}
    for cat in CATS:
        build_labels(P, cat, keys[cat])

    stage(3, "event features")
    for cat in CATS:
        build_features(P, cat, keys[cat])
    avail = feature_availability(P)
    for _, r in avail.iterrows():
        print(f"  {r.catalog}: {r.n_model_features} model features | "
              f"absent: {r.absent}")

    stage(4, "provenance")
    write_provenance(P, args.catalog_end, GLOBAL_SEED)

    stage(5, "dataset summary")
    problems = write_summary(P)
    if problems:
        print("\nproblems:")
        for p in problems:
            print("  -", p)
        sys.exit("dataset build incomplete — do not proceed to the analysis")

    print(f"\nbuilt in {elapsed_hours():.2f} h")
    print(f"next:  python code/run_analysis.py --root {P.root} "
          f"--run-name v4_fresh --gpu")


if __name__ == "__main__":
    main()
