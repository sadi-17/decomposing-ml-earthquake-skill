"""ETAS null model: nearest-neighbour linkage, parameter fitting under the
stability condition of the simulated magnitude distribution, productivity
calibration, and the branching simulator.

Magnitudes are drawn from an unbounded Gutenberg-Richter law above m0 and the
productivity of an event of magnitude m is K * 10^(alpha * (m - m0)). The
expected number of direct offspring of a randomly chosen event is then

    n = K * b / (b - alpha)        for alpha < b,

and the branching process is subcritical only when n < 1. Both conditions are
enforced while fitting and calibrating, not merely checked at simulation time:
a parameter set that violates either is reported as a failure instead of being
simulated and truncated.
"""
from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from make_dataset import compute_features, label_fixed
from utils import (BATCH, GLOBAL_SEED, REF_LABEL, REF_R, REF_T, Paths,
                   aki_b, chord_from_km, km_from_chord, load_features,
                   mc_maxc, to_xyz)

# Nearest-neighbour linkage (Zaliapin & Ben-Zion)
ZBZ_D = 1.6            # fractal dimension of epicentres
ZBZ_B = 1.0            # Gutenberg-Richter b used in the distance definition
ZBZ_MAXR_KM = 1000.0
ZBZ_MAXT_YR = 20.0

# Fitting and calibration
FIT_MAX_EVENTS = 60000
MIN_LINKED_PAIRS = 200
ALPHA_FIT_RANGE = (0.2, 1.6)
BRANCHING_MAX = 0.95   # numerical margin below the critical value n = 1
CAL_ITERS = 8
CAL_TOL = 0.02
K_MIN = 1e-5
CAL_TARGET_EVENTS = 3000     # background events wanted in a calibration run
CAL_MIN_DAYS = 400.0

# Memory guard only. A realisation that trips it is discarded, never used.
MAX_SIM_EVENTS = 3_000_000


class EtasError(RuntimeError):
    """Raised when a parameter set or a realisation is not usable."""


def branching_ratio(K, alpha, b):
    """Expected direct offspring per event under unbounded GR magnitudes."""
    if not np.isfinite(K) or not np.isfinite(alpha) or not np.isfinite(b):
        return np.inf
    if alpha >= b:
        return np.inf
    return float(K * b / (b - alpha))


def max_stable_K(alpha, b, n_max=BRANCHING_MAX):
    """Largest productivity constant with branching ratio at most n_max."""
    if alpha >= b:
        return 0.0
    return float(n_max * (b - alpha) / b)


def nn_distances(df, d=ZBZ_D, b=ZBZ_B):
    """For each event, the smallest nearest-neighbour distance eta over all
    preceding events, and the index of the event that attains it. The search
    neighbourhood only keeps the problem finite; it is far wider than any
    plausible eta0."""
    times_yr = ((df["time"] - pd.Timestamp(0, tz="UTC"))
                .dt.total_seconds().to_numpy()) / (365.25 * 86400.0)
    mw = df["mw"].to_numpy(float)
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    tree = cKDTree(xyz)
    n = len(df)
    eta = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    chord = chord_from_km(ZBZ_MAXR_KM)

    for lo in range(0, n, BATCH):
        hi = min(lo + BATCH, n)
        neigh = tree.query_ball_point(xyz[lo:hi], r=chord)
        for k in range(hi - lo):
            j = lo + k
            nb = np.asarray(neigh[k])
            neigh[k] = None
            nb = nb[nb < j]
            if nb.size == 0:
                continue
            t_ij = times_yr[j] - times_yr[nb]
            keep = (t_ij > 0) & (t_ij <= ZBZ_MAXT_YR)
            nb, t_ij = nb[keep], t_ij[keep]
            if nb.size == 0:
                continue
            r_ij = np.maximum(km_from_chord(
                np.linalg.norm(xyz[nb] - xyz[j], axis=1)), 0.1)
            e = t_ij * (r_ij ** d) * 10.0 ** (-b * mw[nb])
            a = int(np.argmin(e))
            eta[j] = e[a]
            parent[j] = nb[a]
        del neigh
        if (lo // BATCH) % 200 == 0:
            gc.collect()
    return eta, parent


def pick_eta0(log_eta, seed=GLOBAL_SEED):
    """Split clustered from background events at the crossing point of a
    two-component Gaussian mixture on log10(eta). The mode separation is
    returned so a catalog that is not clearly bimodal is visible as such."""
    from sklearn.mixture import GaussianMixture
    v = log_eta[np.isfinite(log_eta)].reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=seed, n_init=5).fit(v)
    mu = gm.means_.ravel()
    sd = np.sqrt(gm.covariances_.ravel())
    lo_i, hi_i = int(np.argmin(mu)), int(np.argmax(mu))
    sep = float((mu[hi_i] - mu[lo_i])
                / np.sqrt(0.5 * (sd[lo_i] ** 2 + sd[hi_i] ** 2)))

    grid = np.linspace(mu[lo_i], mu[hi_i], 2000).reshape(-1, 1)
    post = gm.predict_proba(grid)[:, hi_i]
    eta0 = float(grid[int(np.argmin(np.abs(post - 0.5))), 0])

    if sep < 1.0:
        h, edges = np.histogram(v.ravel(), bins=120)
        centres = 0.5 * (edges[1:] + edges[:-1])
        smooth = np.convolve(h.astype(float), np.ones(5) / 5.0, mode="same")
        band = (centres > mu[lo_i]) & (centres < mu[hi_i])
        if band.any():
            eta0 = float(centres[band][int(np.argmin(smooth[band]))])
        print(f"    [warn] weak bimodality ({sep:.2f} sd): the clustered / "
              f"background split is poorly resolved for this catalog")
    return eta0, sep, (float(mu[lo_i]), float(mu[hi_i]))


def gr_magnitudes(n, m0, b, rng):
    return m0 + rng.exponential(1.0 / (b * np.log(10.0)), size=n)


def simulate_etas(cfg, seed=0, max_events=MAX_SIM_EVENTS):
    """Ogata ETAS by branching: Poisson background, Omori-Utsu waiting times,
    isotropic power-law spatial kernel, Gutenberg-Richter magnitudes.

    Offspring magnitudes are drawn independently of the parent, so a triggered
    event may exceed its trigger and nothing in the catalog anticipates an
    event's future role. That independence is what makes the catalog a null
    for role prediction.
    """
    m0, b, alpha, K = cfg["m0"], cfg["b"], cfg["alpha"], cfg["K"]
    n_branch = branching_ratio(K, alpha, b)
    if not np.isfinite(n_branch) or n_branch >= 1.0:
        raise EtasError(
            f"branching ratio {n_branch:.3f} is not subcritical "
            f"(alpha={alpha:.3f}, b={b:.3f}, K={K:.5f})")

    rng = np.random.default_rng(seed)
    lat0, lat1, lon0, lon1 = cfg["box"]

    n_bg = rng.poisson(cfg["mu_per_day"] * cfg["days"])
    t = rng.uniform(0, cfg["days"], n_bg)
    lat = rng.uniform(lat0, lat1, n_bg)
    lon = rng.uniform(lon0, lon1, n_bg)
    m = gr_magnitudes(n_bg, m0, b, rng)

    out_t, out_lat, out_lon, out_m = [t], [lat], [lon], [m]
    total = n_bg
    while len(t):
        n_off = rng.poisson(K * 10.0 ** (alpha * (m - m0)))
        has_kids = n_off > 0
        if not has_kids.any():
            break
        pt = np.repeat(t[has_kids], n_off[has_kids])
        plat = np.repeat(lat[has_kids], n_off[has_kids])
        plon = np.repeat(lon[has_kids], n_off[has_kids])
        pm = np.repeat(m[has_kids], n_off[has_kids])
        n = len(pt)

        u = rng.uniform(0, 1, n)
        dt = cfg["c"] * ((1 - u) ** (1.0 / (1.0 - cfg["p"])) - 1.0)

        scale = cfg["d_km"] * 10.0 ** (0.5 * (pm - m0))
        v = rng.uniform(0, 1, n)
        r_km = scale * ((1 - v) ** (1.0 / (1.0 - cfg["q"])) - 1.0)
        theta = rng.uniform(0, 2 * np.pi, n)
        dlat = (r_km * np.sin(theta)) / 111.2
        dlon = (r_km * np.cos(theta)) / (111.2 * np.cos(np.radians(plat))
                                         + 1e-9)

        t, lat, lon = pt + dt, plat + dlat, plon + dlon
        m = gr_magnitudes(n, m0, b, rng)

        inside = ((t < cfg["days"]) & (lat > lat0) & (lat < lat1)
                  & (lon > lon0) & (lon < lon1))
        t, lat, lon, m = t[inside], lat[inside], lon[inside], m[inside]
        if len(t) == 0:
            break
        total += len(t)
        if total > max_events:
            raise EtasError(
                f"realisation exceeded the {max_events:,}-event memory guard "
                f"at branching ratio {n_branch:.3f}; discarded")
        out_t.append(t)
        out_lat.append(lat)
        out_lon.append(lon)
        out_m.append(m)

    return pd.DataFrame({
        "time": (pd.Timestamp("1980-01-01", tz="UTC")
                 + pd.to_timedelta(np.concatenate(out_t), unit="D")),
        "latitude": np.concatenate(out_lat),
        "longitude": np.concatenate(out_lon),
        "mw": np.concatenate(out_m),
    }).sort_values("time").reset_index(drop=True)


def label_and_featurize(df, R_km=REF_R, T_days=REF_T):
    """Apply the labelling rule and the feature definitions of the real
    pipeline to a simulated catalog."""
    times_s = ((df["time"] - pd.Timestamp(0, tz="UTC"))
               .dt.total_seconds().to_numpy())
    mw = df["mw"].to_numpy(float)
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    tree = cKDTree(xyz)
    df = df.copy()
    df["label"] = label_fixed(times_s, mw, tree, xyz, R_km, T_days)
    for name, values in compute_features(times_s, mw, tree, xyz).items():
        df[name] = values
    return df


def clustered_fraction(labels):
    """Fraction of events the labelling rule demotes from mainshock."""
    return float(1.0 - np.mean(labels))


def _omori_fit(dt_days, tmax=100.0):
    dt = dt_days[(dt_days > 1e-4) & (dt_days <= tmax)]
    if len(dt) < 150:
        raise EtasError("too few linked pairs to fit the Omori-Utsu kernel")
    edges = np.logspace(-3, np.log10(tmax), 22)
    h, e = np.histogram(dt, bins=edges)
    centres = np.sqrt(e[1:] * e[:-1])
    density = h / np.diff(e)
    ok = h >= 5
    if ok.sum() < 5:
        raise EtasError("linked-pair time differences are too sparse to fit")

    def f(t, log_a, c, p):
        return log_a - p * np.log(t + c)

    popt, _ = curve_fit(f, centres[ok], np.log(density[ok]),
                        p0=[np.log(density[ok][0]), 0.01, 1.1],
                        bounds=([-30, 1e-4, 0.7], [30, 2.0, 2.2]),
                        maxfev=30000)
    return float(popt[1]), float(popt[2])


def _spatial_fit(r_km, m_parent, m0):
    """Power-law kernel whose scale grows as 10^(0.5 (m - m0))."""
    scaled = r_km / np.maximum(10.0 ** (0.5 * (m_parent - m0)), 1e-6)
    s = scaled[(scaled > 0.05) & np.isfinite(scaled)]
    if len(s) < 150:
        raise EtasError("too few linked pairs to fit the spatial kernel")
    d_km = float(np.clip(np.percentile(s, 50), 0.5, 30.0))
    tail = s[s > d_km]
    if len(tail) < 50:
        return d_km, 1.5
    q = 1.0 + 1.0 / max(float(np.mean(np.log(tail / d_km))), 1e-3)
    return d_km, float(np.clip(q, 1.05, 2.5))


def _productivity_fit(mw, parent_of, m0):
    """Regress log10(mean linked children) on magnitude to get alpha and K."""
    counts = np.bincount(parent_of[parent_of >= 0],
                         minlength=len(mw)).astype(float)
    tab = pd.DataFrame({"m": mw, "k": counts})
    tab["bin"] = (tab.m / 0.25).round() * 0.25
    grp = tab.groupby("bin")["k"].agg(["mean", "size"])
    grp = grp[(grp["size"] >= 30) & (grp["mean"] > 0)]
    if len(grp) < 3:
        raise EtasError("too few populated magnitude bins to fit productivity")
    slope, intercept = np.polyfit(grp.index.to_numpy() - m0,
                                  np.log10(grp["mean"].to_numpy()), 1)
    alpha = float(np.clip(slope, *ALPHA_FIT_RANGE))
    K = float(10.0 ** intercept)
    return alpha, K


def calibrate_productivity(cfg, target, seed=GLOBAL_SEED):
    """Scale K so the null reproduces the observed clustered fraction, without
    leaving the subcritical regime. The clustered fraction increases with K,
    so the search keeps a bracket and takes a proportional step inside it,
    falling back to the geometric midpoint when the step would leave it.
    Returns the calibrated configuration, the achieved fraction and a status."""
    k_max = max_stable_K(cfg["alpha"], cfg["b"])
    if k_max <= 0:
        return cfg, np.nan, "unstable_fit"

    # Simulate long enough for the clustered fraction to be well determined,
    # but no longer: this sets the sample size, not the parameters.
    days = float(np.clip(CAL_TARGET_EVENTS / max(cfg["mu_per_day"], 1e-6),
                         CAL_MIN_DAYS, cfg["days"]))
    trial = dict(cfg)
    trial["days"] = days

    lo, hi = K_MIN, k_max
    K = float(min(cfg["K"], k_max))
    best_K, best_got = K, np.nan
    status = "not_converged"

    for it in range(CAL_ITERS):
        trial["K"] = K
        sim = label_and_featurize(simulate_etas(trial, seed=seed + 100 + it))
        got = clustered_fraction(sim["label"].to_numpy())
        if not np.isfinite(best_got) or abs(got - target) < abs(best_got
                                                                - target):
            best_K, best_got = K, got
        print(f"    calibration {it}: clustered {got:.3f} "
              f"(target {target:.3f}) at K={K:.5f}, "
              f"n={branching_ratio(K, cfg['alpha'], cfg['b']):.3f}")
        if abs(got - target) <= CAL_TOL:
            status = "matched"
            break
        if got < target:
            if K >= k_max * (1 - 1e-9):
                status = "stability_limited"
                break
            lo = K
        else:
            hi = K
        step = K * (target + 1e-3) / (got + 1e-3)
        K = step if lo < step < hi else float(np.sqrt(lo * hi))
        K = float(np.clip(K, K_MIN, k_max))

    out = dict(cfg)
    out["K"] = best_K
    return out, best_got, status


def observed_statistics(df, m0):
    """Reference-window statistics of the real catalog on the same population
    the null simulates, namely events at or above m0.

    An event is demoted only by a *larger* neighbour, so restricting to
    mw >= m0 leaves the labels of the retained events unchanged; the subset
    statistic is exact rather than approximate.
    """
    above = df["mw"].to_numpy(float) >= m0
    labels = df[REF_LABEL].to_numpy(np.int8)
    return {
        "observed_mainshock_rate_above_m0": float(labels[above].mean()),
        "observed_clustered_fraction_above_m0": clustered_fraction(
            labels[above]),
        "observed_mainshock_rate_all": float(labels.mean()),
        "n_events_above_m0": int(above.sum()),
    }


def fit_etas(P: Paths, cat, seed=GLOBAL_SEED):
    """Fit an ETAS configuration to one catalog and calibrate its
    productivity. The returned dict always carries a status; only a
    configuration whose status is not a failure may be simulated."""
    ck = P.ckpt / f"{cat}__etas_fit.json"
    if ck.exists():
        return json.loads(ck.read_text())

    # ETAS is fitted with all the columns it needs, event time included.
    df = load_features(P, cat, ["time", "mw", "latitude", "longitude",
                                REF_LABEL])

    # The simulated catalog is complete above m0 by construction, so m0 is the
    # completeness magnitude of the real catalog and b is estimated over the
    # same range. Estimating b from the catalog floor instead biases it low
    # and can make alpha < b fail for reasons of incompleteness alone.
    m0 = float(mc_maxc(df["mw"].to_numpy(float)))
    if not np.isfinite(m0):
        cfg = {"catalog": cat, "status": "no_completeness_estimate",
               "status_detail": "Mc could not be estimated for this catalog"}
        ck.write_text(json.dumps(cfg, indent=2))
        print(f"  [fail] {cat}: Mc could not be estimated")
        return cfg
    obs = observed_statistics(df, m0)

    df = df[df["mw"] >= m0].reset_index(drop=True)
    if len(df) > FIT_MAX_EVENTS:
        df = df.iloc[-FIT_MAX_EVENTS:].reset_index(drop=True)

    mw = df["mw"].to_numpy(float)
    times_d = ((df["time"] - pd.Timestamp(0, tz="UTC"))
               .dt.total_seconds().to_numpy()) / 86400.0
    span = float(times_d.max() - times_d.min())
    b = aki_b(mw, m0)

    cfg = {"catalog": cat, "m0": round(m0, 2), "b": None, "alpha": None,
           "K": None, "c": None, "p": None, "d_km": None, "q": None,
           "mu_per_day": None, "days": round(span, 1), "box": None,
           "n_fit_events": int(len(df)), "status": "ok", **obs}

    def fail(reason, message):
        cfg["status"] = reason
        cfg["status_detail"] = message
        print(f"  [fail] {cat}: {message}")
        ck.write_text(json.dumps(cfg, indent=2))
        return cfg

    if not np.isfinite(b):
        return fail("no_b_value",
                    f"b-value is not estimable above Mc = {m0:.2f}")
    cfg["b"] = round(float(b), 4)

    eta, parent = nn_distances(df)
    log_eta = np.log10(np.where(np.isfinite(eta) & (eta > 0), eta, np.nan))
    eta0, sep, _ = pick_eta0(log_eta, seed=seed)
    clustered = np.isfinite(log_eta) & (log_eta < eta0)
    mu = float(max(int((~clustered).sum()), 1) / max(span, 1.0))
    cfg.update({"mu_per_day": round(mu, 4), "eta0_log10": round(eta0, 3),
                "linkage_separation_sd": round(float(sep), 3),
                "linked_clustered_fraction": round(float(clustered.mean()), 4)})
    print(f"  Mc = {m0:.2f}, b = {b:.3f}, linkage eta0 = 10^{eta0:.2f} "
          f"({sep:.1f} sd), {clustered.mean():.1%} clustered")

    linked = np.where(clustered & (parent >= 0))[0]
    if len(linked) < MIN_LINKED_PAIRS:
        return fail("insufficient_linkage",
                    f"only {len(linked)} linked pairs; the triggering kernels "
                    f"cannot be estimated for this catalog")

    par = parent[linked]
    xyz = to_xyz(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    try:
        c, p = _omori_fit(times_d[linked] - times_d[par])
        d_km, q = _spatial_fit(
            km_from_chord(np.linalg.norm(xyz[linked] - xyz[par], axis=1)),
            mw[par], m0)
        alpha, K = _productivity_fit(mw, parent, m0)
    except EtasError as e:
        return fail("kernel_fit_failed", str(e))

    cfg.update({"alpha": round(alpha, 4), "c": round(c, 5), "p": round(p, 4),
                "d_km": round(d_km, 3), "q": round(q, 4)})

    # Stability of the fitted magnitude model, before any simulation runs.
    if alpha >= b:
        return fail("supercritical_fit",
                    f"fitted alpha = {alpha:.3f} is not below b = {b:.3f}, so "
                    f"the expected offspring count diverges under unbounded "
                    f"Gutenberg-Richter magnitudes and no stable ETAS null "
                    f"exists for this catalog")

    k_max = max_stable_K(alpha, b)
    if K > k_max:
        print(f"  fitted K = {K:.5f} implies branching ratio "
              f"{branching_ratio(K, alpha, b):.3f}; capped at K = {k_max:.5f} "
              f"(n = {BRANCHING_MAX})")
    cfg["K_fitted"] = round(float(K), 5)
    cfg["K_max_stable"] = round(float(k_max), 5)
    cfg["K"] = float(min(K, k_max))

    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    cfg["box"] = [float(np.percentile(lat, 0.5)),
                  float(np.percentile(lat, 99.5)),
                  float(np.percentile(lon, 0.5)),
                  float(np.percentile(lon, 99.5))]

    target = obs["observed_clustered_fraction_above_m0"]
    sim_cfg = {k: cfg[k] for k in ("m0", "b", "alpha", "K", "c", "p", "d_km",
                                   "q", "mu_per_day", "days", "box")}
    try:
        calibrated, achieved, status = calibrate_productivity(
            sim_cfg, target, seed=seed)
    except EtasError as e:
        return fail("calibration_failed", str(e))

    cfg["K"] = round(float(calibrated["K"]), 5)
    cfg["branching_ratio"] = round(
        branching_ratio(cfg["K"], cfg["alpha"], cfg["b"]), 4)
    cfg["calibration_status"] = status
    cfg["calibrated_clustered_fraction"] = (round(float(achieved), 4)
                                            if np.isfinite(achieved) else None)
    cfg["clustered_fraction_target"] = round(float(target), 4)
    cfg["calibration_tolerance"] = CAL_TOL
    print(f"  K = {cfg['K']:.5f}, branching ratio {cfg['branching_ratio']:.3f}"
          f", calibration {status} (clustered "
          f"{cfg['calibrated_clustered_fraction']} vs target {target:.3f})")

    ck.write_text(json.dumps(cfg, indent=2))
    del df
    gc.collect()
    return cfg


def simulation_config(cfg):
    """The parameters the simulator needs, as a plain dict."""
    keys = ("m0", "b", "alpha", "K", "c", "p", "d_km", "q", "mu_per_day",
            "days", "box")
    out = {k: cfg[k] for k in keys}
    out["box"] = tuple(out["box"])
    return out


def is_usable(cfg) -> bool:
    """True when a stable parameter set was obtained. The calibration status
    is reported separately: an uncalibrated null is still a valid ETAS
    process, it is simply not matched to the catalog's clustered fraction."""
    return cfg.get("status") == "ok"
