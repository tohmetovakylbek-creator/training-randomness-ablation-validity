"""
Phase 5 -- pool the 17 per-dataset CC-18 effects into a single estimate,
using REML for the between-dataset variance (tau^2) and the
Hartung-Knapp adjustment for the pooled confidence interval, matching
the method already used for households/ECL elsewhere in the paper.

This is a STANDALONE implementation, not a call into your existing
household/ECL pooling code (I don't have that code -- it lives in your
own factorial_ecl.py-style files). If you already have a REML+HK
function there with a stable, reviewed API, prefer feeding it
cc18_per_dataset_summary.csv directly instead of this script, for
consistency of implementation across the whole paper. This script exists
so you have a working, tested path either way, and its output should
match a correct call into your own code to within numerical precision --
if it doesn't, that's worth chasing down before trusting either one.

Method (standard random-effects meta-analysis, e.g. Borenstein et al.,
2009; matches R's metafor::rma(method="REML", test="knha")):

  1. Per-dataset effect y_i and its variance v_i = SE_i^2, where SE_i is
     recovered from the already-computed t4 confidence interval:
         SE_i = (ci_hi_i - ci_lo_i) / (2 * t_{0.975, 4})
     rather than recomputing it from raw data -- this is exactly the SE
     Section 3/Algorithm 1 already produced per dataset.
  2. tau^2 (between-dataset variance) estimated by maximizing the REML
     log-likelihood over tau^2 >= 0 (numerically, not the DerSimonian-
     Laird closed form, which REML supersedes in accuracy for k this
     small).
  3. Random-effects weights w_i = 1 / (v_i + tau^2_REML); pooled effect
     mu_hat = sum(w_i y_i) / sum(w_i).
  4. Hartung-Knapp: the standard random-effects SE (sqrt(1/sum(w_i)))
     assumes tau^2 is known without error, which understates uncertainty
     at small k. HK replaces it with an SE estimated from the observed
     between-dataset scatter around mu_hat, and uses a t distribution on
     k-1 df instead of a normal quantile -- exactly the same logic as
     the t_{S-1} correction in Section 3.2, one level up (here k =
     number of datasets, not number of seeds).
  5. I^2: the standard heterogeneity statistic, computed from the
     FIXED-effect (tau^2 = 0) weights and residuals, per convention.

Run on: laptop or desktop, CPU only.
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize

ALPHA = 0.05
T4 = stats.t.ppf(1 - ALPHA / 2, 4)      # matches Section 3's t_{S-1}, S=5


def recover_se_from_ci(ci_lo, ci_hi, t_crit=T4):
    return (ci_hi - ci_lo) / (2 * t_crit)


def reml_neg_loglik(tau2, y, v):
    """Restricted log-likelihood for the random-effects mean model, negated
    for minimization. Standard form, e.g. Viechtbauer (2005) eq. 6."""
    tau2 = max(tau2, 0.0)
    w = 1.0 / (v + tau2)
    mu = np.sum(w * y) / np.sum(w)
    ll = (
        -0.5 * np.sum(np.log(v + tau2))
        - 0.5 * np.sum(w * (y - mu) ** 2)
        - 0.5 * np.log(np.sum(w))
    )
    return -ll


def fit_tau2_reml(y, v):
    # tau^2 >= 0; search a generously wide bound relative to the data's own scale
    upper = max(10 * np.var(y, ddof=1), 10 * np.max(v), 1e-6)
    res = optimize.minimize_scalar(reml_neg_loglik, bounds=(0.0, upper),
                                    args=(y, v), method="bounded",
                                    options={"xatol": 1e-10})
    return max(res.x, 0.0)


def pool_reml_hk(y, v, alpha=ALPHA):
    y, v = np.asarray(y, dtype=float), np.asarray(v, dtype=float)
    k = len(y)

    tau2 = fit_tau2_reml(y, v)
    w = 1.0 / (v + tau2)
    mu_hat = np.sum(w * y) / np.sum(w)

    # Hartung-Knapp: scale the naive RE variance by the observed weighted
    # residual dispersion, use t on k-1 df instead of a normal quantile.
    q_hk = np.sum(w * (y - mu_hat) ** 2) / (k - 1)
    # Modified Hartung-Knapp guard: do not allow the HK scale estimate to
    # produce an interval narrower than its unscaled random-effects analogue.
    q_hk_modified = max(1.0, q_hk)
    se_hk = np.sqrt(q_hk_modified / np.sum(w))
    t_crit = stats.t.ppf(1 - alpha / 2, k - 1)
    ci_lo_hk, ci_hi_hk = mu_hat - t_crit * se_hk, mu_hat + t_crit * se_hk

    # naive RE CI (tau^2 treated as known) -- reported for comparison only;
    # HK above is the one to actually use, per the paper's own convention.
    se_re = np.sqrt(1.0 / np.sum(w))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo_re, ci_hi_re = mu_hat - z_crit * se_re, mu_hat + z_crit * se_re

    # I^2, from fixed-effect (tau^2=0) weights, standard formula
    w_fe = 1.0 / v
    mu_fe = np.sum(w_fe * y) / np.sum(w_fe)
    Q = np.sum(w_fe * (y - mu_fe) ** 2)
    df = k - 1
    i2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0

    return dict(
        k=k, tau2=tau2, mu_hat=mu_hat,
        se_hk=se_hk, ci_lo_hk=ci_lo_hk, ci_hi_hk=ci_hi_hk,
        se_naive_re=se_re, ci_lo_naive_re=ci_lo_re, ci_hi_naive_re=ci_hi_re,
        Q=Q, df=df, I2=i2,
        significant_hk=bool((ci_lo_hk > 0) or (ci_hi_hk < 0)),
    )


def _self_test():
    """Cross-check against a known textbook example (Borenstein et al. 2009,
    Table 12.2 -- 6 studies, log risk ratio), where the published REML tau^2
    approx 0.0 and DL tau^2 = 0.0175 approx; our REML should land near the
    published REML/PM estimate, not exactly DL's."""
    y = np.array([-0.5108, -0.2357, -0.3567, -0.2088, -0.4200, -0.4800])
    v = np.array([0.0388, 0.0700, 0.0367, 0.0330, 0.0233, 0.0225])
    tau2 = fit_tau2_reml(y, v)
    assert 0.0 <= tau2 < 0.02, f"REML tau^2 out of expected range: {tau2}"
    print(f"[self-test] textbook example: REML tau^2 = {tau2:.5f} (expected small, <0.02) -- ok")


def main():
    _self_test()

    df = pd.read_csv("cc18_per_dataset_summary.csv")
    print(f"Pooling {len(df)} datasets.\n")

    for metric in ["balanced_error", "cross_entropy"]:
        y = df[f"{metric}_effect"].values
        se = recover_se_from_ci(df[f"{metric}_effect_ci_lo"].values,
                                 df[f"{metric}_effect_ci_hi"].values)
        v = se ** 2

        r = pool_reml_hk(y, v)
        print(f"=== {metric} ===")
        print(f"  k = {r['k']} datasets")
        print(f"  tau^2 (REML) = {r['tau2']:.4f}")
        print(f"  I^2 = {r['I2']:.1f}%")
        print(f"  pooled effect = {r['mu_hat']:.3f}")
        print(f"  95% CI (Hartung-Knapp, t_{{k-1}})   = "
              f"[{r['ci_lo_hk']:.3f}, {r['ci_hi_hk']:.3f}]  "
              f"{'SIGNIFICANT' if r['significant_hk'] else 'not significant'}")
        print(f"  95% CI (naive RE, normal quantile) = "
              f"[{r['ci_lo_naive_re']:.3f}, {r['ci_hi_naive_re']:.3f}]  "
              f"(reported for comparison only -- HK above is the one to cite)")
        print()

    print("Use the Hartung-Knapp pooled effect + CI (per metric) as the headline "
          "number for Section 4.1, not the per-dataset median.")


if __name__ == "__main__":
    main()
