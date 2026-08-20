"""
coverage_simulation.py
======================
Симуляция покрытия точного F-интервала (замечание научрука №2).

Интервал выведен для сбалансированной однофакторной модели со случайными
эффектами при нормальных сидовых эффектах и одинаковой оконной дисперсии.
В реальных данных ни то, ни другое не гарантировано. Скрипт измеряет
фактическое покрытие 95-процентного интервала для theta = sigma_seed^2 /
sigma_win^2 при отклонениях от предпосылок:

    seed_dist   : normal | t3 | lognormal (сильная асимметрия)
    window      : iid | ar1 (автокорреляция соседних окон)
    hetero      : 1.0 (гомоскедастично) | 3.0 (дисперсия по сидам различается втрое)

Границы считаются ВАШЕЙ функцией variance_model.exact_share_ci, поэтому
проверяется именно та процедура, что используется в статье.

Запуск (несколько минут):
    python coverage_simulation.py --n_rep 2000 --S 5 --W 120 --out results/coverage.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import variance_model as vm


def draw_seed_effects(dist, S, sd, rng):
    if dist == "normal":
        x = rng.normal(size=S)
    elif dist == "t3":
        x = rng.standard_t(3, size=S) / np.sqrt(3.0)          # дисперсия 1
    elif dist == "lognormal":
        x = rng.lognormal(0.0, 0.75, size=S)
        x = (x - np.exp(0.75 ** 2 / 2)) / np.sqrt(
            (np.exp(0.75 ** 2) - 1) * np.exp(0.75 ** 2))      # центр 0, дисперсия 1
    else:
        raise ValueError(dist)
    return sd * x


def draw_windows(window, S, W, sd_by_seed, rng, rho=0.35):
    e = rng.normal(size=(S, W))
    if window == "ar1":
        for w in range(1, W):
            e[:, w] = rho * e[:, w - 1] + np.sqrt(1 - rho ** 2) * e[:, w]
    return e * sd_by_seed[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rep", type=int, default=2000)
    ap.add_argument("--S", type=int, default=5)
    ap.add_argument("--W", type=int, default=120)
    ap.add_argument("--theta", type=float, default=0.03,
                    help="истинное sigma_seed^2/sigma_win^2; 0.03 близко к наблюдаемому")
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=400,
                    help="повторов блочного бутстрэпа внутри каждой реплики")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--out", default="results/coverage.json")
    args = ap.parse_args()

    S, W, theta = args.S, args.W, args.theta
    sd_win, sd_seed = 1.0, np.sqrt(theta)
    rng = np.random.default_rng(args.seed)

    scenarios = list(itertools.product(("normal", "t3", "lognormal"),
                                       ("iid", "ar1"),
                                       (1.0, 3.0)))
    print(f"S = {S}, W = {W}, истинное theta = {theta}, реплик = {args.n_rep}\n")
    print(f"{'сидовые эффекты':16s}{'окна':6s}{'гетеро':8s}{'покрытие theta':>16s}"
          f"{'ниже':>7s}{'выше':>7s}{'медиана оценки':>16s}")
    results = []
    for dist, window, hetero in scenarios:
        cover = below = above = 0
        est = []
        # дисперсии по сидам: среднее сохраняется, отношение max/min = hetero
        ratios = np.linspace(1.0, hetero, S)
        ratios = ratios / ratios.mean()
        sd_by_seed = sd_win * np.sqrt(ratios)
        for _ in range(args.n_rep):
            a = draw_seed_effects(dist, S, sd_seed, rng)
            d = a[:, None] + draw_windows(window, S, W, sd_by_seed, rng)
            vwin = vm.block_var_of_mean(d.mean(axis=0), args.block,
                                        n_boot=args.n_boot, seed=int(rng.integers(1e9)))
            ci = vm.exact_share_ci(d, args.block, vwin, n_boot=args.n_boot, seed=0)
            lo, hi = ci.get("theta_lo", np.nan), ci.get("theta_hi", np.nan)
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if lo <= theta <= hi:
                cover += 1
            elif hi < theta:
                below += 1
            else:
                above += 1
            seed_means = d.mean(axis=1)
            ms_b = W * ((seed_means - d.mean()) ** 2).sum() / (S - 1)
            ms_w = ((d - seed_means[:, None]) ** 2).sum() / (S * (W - 1))
            est.append(max(0.0, (ms_b - ms_w) / W) / ms_w if ms_w > 0 else np.nan)
        n = cover + below + above
        row = {"seed_dist": dist, "window": window, "hetero": hetero,
               "coverage": cover / n, "miss_low": below / n, "miss_high": above / n,
               "median_theta_hat": float(np.nanmedian(est)), "n": n}
        results.append(row)
        print(f"{dist:16s}{window:6s}{hetero:8.1f}{row['coverage']:16.1%}"
              f"{row['miss_low']:7.1%}{row['miss_high']:7.1%}{row['median_theta_hat']:16.4f}")

    cov = np.array([r["coverage"] for r in results])
    print(f"\nноминальный уровень 95%; фактическое покрытие от {cov.min():.1%} до {cov.max():.1%}")
    print("Интерпретация: покрытие заметно ниже 95% означает, что интервал слишком узок")
    print("при данном нарушении предпосылок; выше — что он консервативен.")
    print("'ниже'/'выше' показывают, с какой стороны интервал промахивается.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"S": S, "W": W, "theta": theta, "n_rep": args.n_rep, "block": args.block,
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
