"""
meta_estimators.py
==================
Два замечания научрука разом:

  №2 — устойчивость пулинга. DerSimonian–Laird при 17 юнитах может быть смещён
       и даёт слишком узкие интервалы. Считаются четыре варианта:
       DL, Paule–Mandel, REML и поправка Хартунга–Кнаппа к каждому из них.

  №4 — порог практической значимости. Сейчас 2 % задано и не варьируется;
       здесь считается число домохозяйств выше порога при 1, 2 и 5 %.

Обучения не требует: эффекты и их стандартные ошибки берутся из
variance_model.decompose, масштаб домохозяйства (MAE) — из factorial_<ds>.json.

Запуск:
    python meta_estimators.py --npz results/factorial_2comp/factorial_*.npz ^
        --json results/factorial_2comp/factorial_ukdale.json ^
               results/factorial_2comp/factorial_refit.json ^
               results/factorial_2comp/factorial_sheerm.json ^
        --cell aux-per_mode_agg-convex --path y_final --exclude ukdale_house_4 ^
        --out results/meta_estimators.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import variance_model as vm

Z = 1.959963985


def tau2_dl(y, v):
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    Q = (w * (y - mu) ** 2).sum()
    df = len(y) - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    return max(0.0, (Q - df) / C), Q, df


def tau2_pm(y, v):
    """Paule–Mandel: подбор tau2 так, чтобы обобщённая Q равнялась df."""
    df = len(y) - 1
    f = lambda t: ((1 / (v + t)) * (y - ((1 / (v + t)) * y).sum() / (1 / (v + t)).sum()) ** 2).sum() - df
    lo, hi = 0.0, max(10.0, 10 * np.var(y, ddof=1) + 10 * v.max())
    if f(lo) <= 0:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tau2_reml(y, v, tol=1e-10, it=500):
    t2 = max(0.0, np.var(y, ddof=1) - v.mean())
    for _ in range(it):
        w = 1 / (v + t2)
        mu = (w * y).sum() / w.sum()
        num = (w ** 2 * ((y - mu) ** 2 - v)).sum() + (1 / w.sum())
        den = (w ** 2).sum()
        new = max(0.0, num / den)
        if abs(new - t2) < tol:
            return new
        t2 = new
    return t2


def pool(y, v, t2, hk=False):
    w = 1 / (v + t2)
    mu = (w * y).sum() / w.sum()
    se = np.sqrt(1 / w.sum())
    k = len(y)
    if hk:                                        # Hartung–Knapp–Sidik–Jonkman
        q = (w * (y - mu) ** 2).sum() / (k - 1)
        se = np.sqrt(max(q, 1.0) / w.sum())       # с защитой от занижения (Röver et al.)
        crit = stats.t.ppf(0.975, k - 1)
    else:
        crit = Z
    lo, hi = mu - crit * se, mu + crit * se
    p = 2 * (1 - (stats.t.cdf(abs(mu / se), k - 1) if hk else stats.norm.cdf(abs(mu / se))))
    pi = (mu - crit * np.sqrt(t2 + se ** 2), mu + crit * np.sqrt(t2 + se ** 2))
    return {"mu": float(mu), "se": float(se), "ci": [float(lo), float(hi)],
            "p": float(p), "tau2": float(t2), "tau": float(np.sqrt(t2)),
            "pred_int": [float(pi[0]), float(pi[1])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--cell", default="aux-per_mode_agg-convex")
    ap.add_argument("--path", default="y_final", choices=["y_final", "y_vmd"])
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--exclude", default="ukdale_house_4")
    ap.add_argument("--thresholds", default="1,2,5")
    ap.add_argument("--out", default="results/meta_estimators.json")
    args = ap.parse_args()

    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    paths = vm.resolve_npz(args.npz) if hasattr(vm, "resolve_npz") else args.npz
    eff = vm.load_effects(paths, args.cell, args.path, exclude=exclude)

    scale = {}
    for f in args.json:
        for h, rec in json.loads(Path(f).read_text(encoding="utf-8")).items():
            cell = rec.get("cells", {}).get(f"{args.cell}_emb-on")
            if cell:
                key = "y" if args.path == "y_final" else "y_vmd"
                scale[h] = cell["paths"][key]["MAE"]

    houses = sorted(h for h in eff if h in scale)
    print(f"домохозяйств: {len(houses)}")
    y, v, rel = [], [], []
    for h in houses:
        r = vm.decompose(eff[h], args.block, seed=0)
        se2 = r["var_win_of_mean"] + r["sigma_seed2"] / r["n_seeds"]
        y.append(r["effect"]); v.append(se2)
        rel.append(abs(r["effect"]) / scale[h] * 100)
    y, v, rel = np.array(y), np.array(v), np.array(rel)

    # ---------- пулинг -------------------------------------------------------
    t2_dl, Q, df = tau2_dl(y, v)
    ests = {"DerSimonian-Laird": t2_dl, "Paule-Mandel": tau2_pm(y, v), "REML": tau2_reml(y, v)}
    print(f"\nCochran Q = {Q:.1f} на {df} ст. свободы, p = {1 - stats.chi2.cdf(Q, df):.3f}")
    print(f"\n{'оценка tau^2':22s}{'tau, W':>8}{'pooled, W':>11}{'95% CI':>22}"
          f"{'p':>8}{'prediction interval':>26}")
    table = {}
    for name, t2 in ests.items():
        for hk in (False, True):
            r = pool(y, v, t2, hk=hk)
            label = name + (" + Hartung-Knapp" if hk else "")
            table[label] = r
            print(f"{label:22s}{r['tau']:8.2f}{r['mu']:11.3f}"
                  f"   [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]{r['p']:8.3f}"
                  f"      [{r['pred_int'][0]:+.2f}, {r['pred_int'][1]:+.2f}]")
    w = 1 / (v + t2_dl)
    i2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    print(f"\nI^2 (DL) = {i2:.1f} %")

    # ---------- порог --------------------------------------------------------
    print(f"\nЧувствительность к порогу практической значимости "
          f"(|эффект| в процентах от MAE домохозяйства):")
    print(f"  медиана {np.median(rel):.2f} %, диапазон {rel.min():.2f}–{rel.max():.2f} %")
    thr_out = {}
    for t in [float(x) for x in args.thresholds.split(",")]:
        n = int((rel > t).sum())
        thr_out[t] = n
        print(f"  порог {t:g} %: выше него {n} из {len(rel)} домохозяйств")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"cell": args.cell, "path": args.path, "houses": houses,
         "effects": y.tolist(), "var": v.tolist(), "rel_pct": rel.tolist(),
         "Q": float(Q), "df": int(df), "I2_DL": float(i2),
         "pooling": table, "threshold_counts": thr_out},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
