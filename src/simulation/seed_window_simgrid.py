"""
Extended simulation study for the crossed seed x window uncertainty analysis
of a paired architectural ablation.

Data-generating process (matches Eq. 2 of the manuscript):

    d[s, w] = mu + a[s] + b[w] + c[s, w]

    a[s]     ~ seed effect,           Var = sig2_seed
    b[w]     ~ common window effect,  Var = sig2_win,  AR(1) with parameter phi
    c[s, w]  ~ seed x window inter.,  Var = sig2_int  (optionally heteroscedastic in w)

Estimators evaluated (exactly those used in the paper):

  * balanced crossed two-way random-effects ANOVA without replication
  * moving-block bootstrap (block length 3) of the seed-averaged window series
    for the evaluation contribution V_win
  * seed share  = (sig2_seed / S) / (sig2_seed / S + V_win)
  * F-based model interval for the seed share, from the exact conditional
    F pivot for theta = sig2_seed / sig2_int
  * full-propagation CI for the mean effect vs. the window-only CI that a
    conventional (single-training-run) analysis would report

Outputs: results_grid.csv, fig_S_precision.png, fig_coverage.png
"""

import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED = 20260813
ALPHA = 0.05
BLOCK = 3
NBOOT = 300


# ----------------------------------------------------------------------
# data generation
# ----------------------------------------------------------------------
def draw_standardised(dist, size, rng):
    """Zero-mean, unit-variance draws from the requested distribution."""
    if dist == "normal":
        return rng.standard_normal(size)
    if dist == "t3":                       # heavy-tailed, Var = 3
        return rng.standard_t(3, size=size) / np.sqrt(3.0)
    if dist == "lognormal":                # right-skewed
        s = 0.75
        x = rng.lognormal(mean=0.0, sigma=s, size=size)
        m = np.exp(s ** 2 / 2.0)
        v = (np.exp(s ** 2) - 1.0) * np.exp(s ** 2)
        return (x - m) / np.sqrt(v)
    raise ValueError(dist)


def ar1(nrep, W, phi, rng):
    """AR(1) sequence with unit marginal variance, shape (nrep, W)."""
    if phi == 0.0:
        return rng.standard_normal((nrep, W))
    e = rng.standard_normal((nrep, W)) * np.sqrt(1.0 - phi ** 2)
    out = np.empty((nrep, W))
    out[:, 0] = rng.standard_normal(nrep)
    for t in range(1, W):
        out[:, t] = phi * out[:, t - 1] + e[:, t]
    return out


def simulate(nrep, S, W, sig2_seed, sig2_win, sig2_int, phi, dist,
             hetero, rng):
    """Return d of shape (nrep, S, W) and the per-rep interaction sd profile."""
    a = draw_standardised(dist, (nrep, S, 1), rng) * np.sqrt(sig2_seed)
    b = ar1(nrep, W, phi, rng)[:, None, :] * np.sqrt(sig2_win)
    c = rng.standard_normal((nrep, S, W))
    if hetero:
        # interaction sd rises fourfold across the evaluation segment,
        # rescaled so that the average variance is still sig2_int
        prof = np.linspace(0.5, 2.0, W)
        prof = prof / np.sqrt(np.mean(prof ** 2))
        c = c * prof[None, None, :]
    c = c * np.sqrt(sig2_int)
    return a + b + c


# ----------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------
def crossed_anova(d):
    """Balanced two-way random-effects ANOVA without replication.

    d : (nrep, S, W).  Returns MS_seed, MS_int, sig2_seed_hat, sig2_int_hat.
    """
    nrep, S, W = d.shape
    grand = d.mean(axis=(1, 2), keepdims=True)
    m_s = d.mean(axis=2, keepdims=True)
    m_w = d.mean(axis=1, keepdims=True)
    ss_seed = W * ((m_s - grand) ** 2).sum(axis=(1, 2))
    ss_win = S * ((m_w - grand) ** 2).sum(axis=(1, 2))
    ss_int = ((d - m_s - m_w + grand) ** 2).sum(axis=(1, 2))
    ms_seed = ss_seed / (S - 1)
    ms_int = ss_int / ((S - 1) * (W - 1))
    sig2_int = ms_int
    sig2_seed = np.maximum(0.0, (ms_seed - ms_int) / W)
    return ms_seed, ms_int, sig2_seed, sig2_int, ss_win


def block_bootstrap_vwin(dbar, block, nboot, rng, chunk=100):
    """Moving-block bootstrap variance of the mean of the seed-averaged series.

    dbar : (nrep, W).  Blocks of window indices are applied jointly to all
    seeds upstream, which is algebraically equivalent to resampling dbar.
    """
    nrep, W = dbar.shape
    nblocks = int(np.ceil(W / block))
    offs = np.arange(block)
    out = np.empty(nrep)
    for i in range(0, nrep, chunk):
        j = min(i + chunk, nrep)
        m = j - i
        starts = rng.integers(0, W - block + 1, size=(m, nboot, nblocks))
        idx = (starts[..., None] + offs).reshape(m, nboot, -1)[:, :, :W]
        boot = np.take_along_axis(
            np.broadcast_to(dbar[i:j, None, :], (m, nboot, W)), idx, axis=2
        ).mean(axis=2)
        out[i:j] = boot.var(axis=1, ddof=1)
    return out


def f_interval_share(ms_seed, ms_int, S, W, vwin, alpha=ALPHA):
    """F-based model interval for the seed share.

    Exact conditional F pivot for theta = sig2_seed / sig2_int:
        (MS_seed / MS_int) / (1 + W theta) ~ F(S-1, (S-1)(W-1))
    The mapping to the share holds V_win at its block-bootstrap value, so the
    resulting interval is model-based rather than exact.
    """
    df1, df2 = S - 1, (S - 1) * (W - 1)
    R = ms_seed / ms_int
    fl = stats.f.ppf(1 - alpha / 2, df1, df2)
    fu = stats.f.ppf(alpha / 2, df1, df2)
    th_lo = np.maximum(0.0, (R / fl - 1.0) / W)
    th_hi = np.maximum(0.0, (R / fu - 1.0) / W)
    s_lo = (th_lo * ms_int / S) / (th_lo * ms_int / S + vwin)
    s_hi = (th_hi * ms_int / S) / (th_hi * ms_int / S + vwin)
    return s_lo, s_hi


def true_vwin(W, sig2_win, sig2_int, phi, S, hetero):
    """Exact variance of the mean of the seed-averaged window series."""
    lags = np.arange(W)
    w = (W - lags) * (phi ** lags)
    gamma = (w[0] + 2.0 * w[1:].sum()) / W ** 2 if phi > 0 else 1.0 / W
    v_b = sig2_win * gamma
    v_c = sig2_int / (S * W)
    return v_b + v_c


# ----------------------------------------------------------------------
# one grid cell
# ----------------------------------------------------------------------
def run_cell(nrep, S, W, ratio, phi, dist, hetero, rng,
             sig2_int=1.0, sig2_win=1.0, mu=0.0):
    sig2_seed = ratio * sig2_int
    d = simulate(nrep, S, W, sig2_seed, sig2_win, sig2_int, phi, dist,
                 hetero, rng)

    ms_seed, ms_int, sig2_seed_h, sig2_int_h, _ = crossed_anova(d)
    dbar = d.mean(axis=1)
    vwin = block_bootstrap_vwin(dbar, BLOCK, NBOOT, rng)
    ghat = d.mean(axis=(1, 2))

    share_hat = (sig2_seed_h / S) / (sig2_seed_h / S + vwin)
    s_lo, s_hi = f_interval_share(ms_seed, ms_int, S, W, vwin)

    vw_true = true_vwin(W, sig2_win, sig2_int, phi, S, hetero)
    share_true = (sig2_seed / S) / (sig2_seed / S + vw_true) if sig2_seed > 0 else 0.0

    se_full = np.sqrt(sig2_seed_h / S + vwin)
    se_win = np.sqrt(vwin)
    z = stats.norm.ppf(1 - ALPHA / 2)
    t_seed = stats.t.ppf(1 - ALPHA / 2, S - 1)

    cov_full_z = np.mean(np.abs(ghat - mu) <= z * se_full)
    cov_full_t = np.mean(np.abs(ghat - mu) <= t_seed * se_full)
    cov_win = np.mean(np.abs(ghat - mu) <= z * se_win)

    cov_share = np.mean((s_lo <= share_true) & (share_true <= s_hi))
    if sig2_seed == 0:
        cov_share = np.mean(s_lo <= 1e-12)

    return dict(
        S=S, W=W, ratio=ratio, phi=phi, dist=dist,
        hetero=int(hetero),
        share_true=round(100 * share_true, 1),
        share_median=round(100 * np.median(share_hat), 1),
        share_iqr_lo=round(100 * np.percentile(share_hat, 25), 1),
        share_iqr_hi=round(100 * np.percentile(share_hat, 75), 1),
        trunc_zero=round(100 * np.mean(sig2_seed_h <= 0), 1),
        ci_width_pp=round(100 * np.median(s_hi - s_lo), 1),
        cov_share=round(100 * cov_share, 1),
        cov_full_z=round(100 * cov_full_z, 1),
        cov_full_t=round(100 * cov_full_t, 1),
        cov_window_only=round(100 * cov_win, 1),
        se_ratio=round(float(np.median(se_full / se_win)), 2),
    )


# ----------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------
def main(nrep=3000):
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    # G1 -- seed count vs precision (the design question)
    for S in [3, 5, 10, 20, 50]:
        r = run_cell(nrep, S, 120, 2.0, 0.5, "normal", False, rng)
        r["grid"] = "G1_seeds"
        rows.append(r)

    # G2 -- distributional robustness and heteroscedastic interaction
    for S in [3, 5, 10, 20]:
        for dist in ["normal", "t3", "lognormal"]:
            for het in [False, True]:
                r = run_cell(nrep, S, 120, 2.0, 0.5, dist, het, rng)
                r["grid"] = "G2_distribution"
                rows.append(r)

    # G3 -- evaluation-set size and serial dependence
    for W in [50, 120, 250]:
        for phi in [0.0, 0.5, 0.8]:
            r = run_cell(nrep, 5, W, 2.0, phi, "normal", False, rng)
            r["grid"] = "G3_windows"
            rows.append(r)

    # G4 -- null and non-null seed component
    for ratio in [0.0, 0.25, 1.0, 4.0, 16.0]:
        for S in [5, 20]:
            r = run_cell(nrep, S, 120, ratio, 0.5, "normal", False, rng)
            r["grid"] = "G4_component"
            rows.append(r)

    df = pd.DataFrame(rows)
    cols = ["grid"] + [c for c in df.columns if c != "grid"]
    df = df[cols]
    df.to_csv("/home/claude/results_grid.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
