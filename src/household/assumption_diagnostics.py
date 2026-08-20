"""
assumption_diagnostics.py
=========================
Проверка предпосылок точного F-интервала (замечание научрука №2, §6.2.3).

Интервал опирается на две вещи, которые в статье названы, но не показаны:
    (1) нормальность сидовых эффектов a(h,s);
    (2) одинаковая оконная дисперсия по сидам (гомоскедастичность внутри дома).

Скрипт ничего не обучает: читает per-seed массивы через variance_model.load_effects.

Что считает:
    * по каждому домохозяйству — стандартизованные сидовые эффекты, их асимметрию
      и эксцесс, тест Шапиро–Уилка (при S = 5 он крайне маломощен, поэтому
      основной вывод делается по объединённой выборке 17 x 5 значений);
    * объединённый QQ-график стандартизованных сидовых эффектов;
    * по каждому домохозяйству — оконные дисперсии внутри каждого сида, их
      отношение max/min и тест Левена (устойчив к ненормальности);
    * долю домохозяйств, где гомоскедастичность отвергается на уровне 0.05
      и после поправки Холма на множественность.

Запуск:
    python assumption_diagnostics.py --npz results/factorial_2comp/factorial_*.npz ^
        --cell aux-per_mode_agg-convex --path y_final --exclude ukdale_house_4 ^
        --out results/diagnostics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import variance_model as vm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--cell", default="aux-per_mode_agg-convex")
    ap.add_argument("--path", default="y_final", choices=["y_final", "y_vmd"])
    ap.add_argument("--exclude", default="ukdale_house_4")
    ap.add_argument("--out", default="results/diagnostics")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    paths = vm.resolve_npz(args.npz) if hasattr(vm, "resolve_npz") else args.npz
    eff = vm.load_effects(paths, args.cell, args.path, exclude=exclude)
    houses = sorted(eff)
    if not houses:
        raise SystemExit("ничего не загружено — проверьте --npz, --cell, --path")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows, pooled_z = [], []
    print(f"\n=== {args.cell} | {args.path} | {len(houses)} домохозяйств ===\n")
    print(f"{'домохозяйство':20s}{'S':>3}{'W':>6}{'скос':>8}{'экс':>8}{'Шапиро p':>10}"
          f"{'var max/min':>13}{'Левен p':>10}")
    for h in houses:
        d = eff[h]                                   # (S, W)
        S, W = d.shape
        seed_eff = d.mean(axis=1)
        z = (seed_eff - seed_eff.mean()) / (seed_eff.std(ddof=1) + 1e-12)
        pooled_z.extend(z.tolist())

        resid = d - seed_eff[:, None]                # остатки внутри сида
        var_by_seed = resid.var(axis=1, ddof=1)
        lev = stats.levene(*[resid[s] for s in range(S)], center="median")
        sw = stats.shapiro(seed_eff) if S >= 3 else (np.nan, np.nan)

        rows.append({"house": h, "S": int(S), "W": int(W),
                     "skew": float(stats.skew(seed_eff)),
                     "kurtosis": float(stats.kurtosis(seed_eff)),
                     "shapiro_p": float(sw[1]),
                     "var_ratio": float(var_by_seed.max() / var_by_seed.min()),
                     "levene_p": float(lev.pvalue)})
        r = rows[-1]
        print(f"{h:20s}{S:3d}{W:6d}{r['skew']:8.2f}{r['kurtosis']:8.2f}"
              f"{r['shapiro_p']:10.3f}{r['var_ratio']:13.2f}{r['levene_p']:10.4f}")

    # ---- нормальность: объединённая выборка --------------------------------
    z = np.array(pooled_z)
    sw_p = stats.shapiro(z).pvalue
    ad = stats.anderson(z, dist="norm")
    print(f"\nОбъединённые стандартизованные сидовые эффекты (n = {len(z)}):")
    print(f"  асимметрия {stats.skew(z):+.3f}, эксцесс {stats.kurtosis(z):+.3f}")
    print(f"  Шапиро–Уилк p = {sw_p:.4f}")
    print(f"  Андерсон–Дарлинг A2 = {ad.statistic:.3f} "
          f"(критическое при 5%: {ad.critical_values[2]:.3f})")
    print("  Внимание: при S = 5 в отдельном доме тест почти не имеет мощности;")
    print("  содержательный вывод даёт только объединённая выборка.")

    # ---- гомоскедастичность ------------------------------------------------
    p = np.array([r["levene_p"] for r in rows])
    order = np.argsort(p)
    holm = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        holm[idx] = min(1.0, running)
    vr = np.array([r["var_ratio"] for r in rows])
    print(f"\nГомогенность оконной дисперсии по сидам:")
    print(f"  отношение max/min дисперсий: медиана {np.median(vr):.2f}, "
          f"диапазон {vr.min():.2f}–{vr.max():.2f}")
    print(f"  тест Левена отвергается при 0.05 у {(p < 0.05).sum()} из {m} домохозяйств; "
          f"после поправки Холма — у {(holm < 0.05).sum()}")
    for r, hv in zip(rows, holm):
        r["levene_p_holm"] = float(hv)

    # ---- графики -----------------------------------------------------------
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
            stats.probplot(z, dist="norm", plot=ax[0])
            ax[0].set_title("Seed effects, pooled and standardised")
            ax[0].get_lines()[0].set_markersize(3)
            ax[1].hist(vr, bins=12, color="0.35")
            ax[1].set_xlabel("max/min of within-seed window variance")
            ax[1].set_ylabel("households")
            ax[1].set_title("Homogeneity across seeds")
            fig.tight_layout()
            f = out / f"diagnostics_{args.path}.png"
            fig.savefig(f, dpi=300)
            print(f"\nграфик: {f}")
        except Exception as e:                                    # noqa: BLE001
            print(f"[warn] график не построен: {e}")

    (out / f"diagnostics_{args.path}.json").write_text(json.dumps(
        {"cell": args.cell, "path": args.path,
         "pooled": {"n": int(len(z)), "skew": float(stats.skew(z)),
                    "kurtosis": float(stats.kurtosis(z)), "shapiro_p": float(sw_p),
                    "anderson_A2": float(ad.statistic),
                    "anderson_crit_5pct": float(ad.critical_values[2])},
         "houses": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {out / f'diagnostics_{args.path}.json'}")


if __name__ == "__main__":
    main()
