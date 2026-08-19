"""
block_sensitivity.py
====================
Материал для §6.2.3: две вещи, которые сейчас в статье названы, но не измерены.

1. НЕСОГЛАСОВАННОСТЬ ДВУХ МОДЕЛЕЙ ОКНА. MS_within в ANOVA предполагает
   независимость тестовых окон внутри сида, а V_win оценивается блочным
   бутстрэпом именно потому, что окна зависимы. Мера расхождения — коэффициент

       g = V_win * W / MS_within

   (тот самый, что уже считается внутри variance_model.exact_share_ci).
   ВНИМАНИЕ: нейтральное значение g равно 1/S, а НЕ 1. V_win — дисперсия среднего
   ряда, уже усреднённого по S сидам, поэтому при независимых окнах она равна
   sigma_win^2/(S*W). Коэффициент инфляции относительно независимости — это g*S:
   значение 1 = модели согласны, > 1 = окна положительно зависимы (независимая
   модель занижает оконную компоненту), < 1 = наоборот. Формула
   share = theta*W/(theta*W + g*S) в exact_share_ci корректна, множитель S там уже есть.

2. ЧУВСТВИТЕЛЬНОСТЬ К ДЛИНЕ БЛОКА. В §6.2.3 сказано, что блок 3 выбран из
   содержательных соображений и не настраивался, а чувствительность не
   проверялась. Скрипт пересчитывает долю сидовой дисперсии и точный интервал
   при нескольких длинах блока, включая 1 (независимые окна).

Обучения не требует: читает те же per-seed массивы, что variance_model.py, и
использует ЕГО функции decompose и exact_share_ci, поэтому расхождение
соглашений исключено по построению.

Запуск из корня проекта:

    python block_sensitivity.py --npz results/factorial_2comp/factorial_ukdale.npz ^
        results/factorial_2comp/factorial_refit.npz ^
        results/factorial_2comp/factorial_sheerm.npz ^
        --cell aux-per_mode_agg-convex --path y_final --exclude ukdale_house_4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import variance_model as vm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--cell", default="aux-per_mode_agg-convex")
    ap.add_argument("--path", default="y_final", choices=["y_final", "y_vmd"])
    ap.add_argument("--blocks", default="1,2,3,5,7,10")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="ukdale_house_4")
    ap.add_argument("--out", default=None, help="куда сохранить JSON со всеми числами")
    args = ap.parse_args()

    blocks = [int(b) for b in args.blocks.split(",")]
    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())

    paths = vm.resolve_npz(args.npz) if hasattr(vm, "resolve_npz") else args.npz
    eff = vm.load_effects(paths, args.cell, args.path, exclude=exclude)
    if not eff:
        raise SystemExit("ничего не загружено: проверьте --npz, --cell, --path")
    houses = sorted(eff)
    print(f"\n=== {args.cell} | {args.path} | {len(houses)} домохозяйств ===\n")

    res = {h: {} for h in houses}
    for h in houses:
        d = eff[h]
        for b in blocks:
            r = vm.decompose(d, b, n_boot=args.n_boot, seed=args.seed)
            ci = vm.exact_share_ci(d, b, r["var_win_of_mean"],
                                   n_boot=args.n_boot, seed=args.seed)
            S, W = d.shape
            g = r["var_win_of_mean"] * W / r["sigma_win2"] if r["sigma_win2"] > 0 else np.nan
            res[h][b] = {"share": r["share_seed"], "g": float(g),
                         "lo": ci.get("lo", np.nan), "hi": ci.get("hi", np.nan),
                         "W": W, "S": S,
                         "sigma_seed2": r["sigma_seed2"], "sigma_win2": r["sigma_win2"],
                         "V_win": r["var_win_of_mean"]}

    # ---- 1. коэффициент инфляции при рабочей длине блока -----------------
    ref = 3 if 3 in blocks else blocks[0]
    print(f"Коэффициент инфляции оконной дисперсии g при длине блока {ref}")
    print(f"{'домохозяйство':22s}{'W':>5}{'g':>8}{'g*S':>7}{'доля сида':>11}{'точный CI':>16}")
    for h in houses:
        r = res[h][ref]
        ci = (f"[{r['lo']:.0%},{r['hi']:.0%}]" if np.isfinite(r["lo"]) else "—")
        print(f"{h:22s}{r['W']:5d}{r['g']:8.3f}{r['g']*r['S']:7.2f}{r['share']:11.1%}{ci:>16s}")
    gs = np.array([res[h][ref]["g"] for h in houses])
    S = res[houses[0]][ref]["S"]
    infl = gs * S
    print(f"\n  медиана g = {np.median(gs):.3f} (нейтральное значение 1/S = {1/S:.3f})")
    print(f"  коэффициент инфляции g*S: медиана {np.median(infl):.2f}, "
          f"диапазон {infl.min():.2f}–{infl.max():.2f}; выше 1 на "
          f"{(infl > 1).sum()} домохозяйствах из {len(infl)}")
    print("  g*S > 1 означает положительную зависимость соседних окон (независимая модель")
    print("  занижает оконную компоненту и завышает долю сида); < 1 — обратное.")

    # ---- 2. чувствительность к длине блока -------------------------------
    print(f"\nЧувствительность к длине блока (доля сидовой дисперсии, {len(houses)} домохозяйств)")
    print(f"{'блок':>6}{'медиана g':>12}{'медиана доли':>15}{'мин доли':>10}{'макс доли':>11}"
          f"{'нижн. гр. > 1/2':>17}{'обрезано в 0':>14}")
    for b in blocks:
        sh = np.array([res[h][b]["share"] for h in houses])
        lo = np.array([res[h][b]["lo"] for h in houses])
        gb = np.array([res[h][b]["g"] for h in houses])
        tr = sum(1 for h in houses if res[h][b]["sigma_seed2"] == 0.0)
        print(f"{b:6d}{np.nanmedian(gb):12.3f}{np.median(sh):15.1%}{sh.min():10.1%}"
              f"{sh.max():11.1%}{int((lo > 0.5).sum()):12d}/{len(houses):<4}{tr:14d}")

    ref_sh = np.array([res[h][ref]["share"] for h in houses])
    for b in blocks:
        if b == ref:
            continue
        sh = np.array([res[h][b]["share"] for h in houses])
        print(f"  блок {b} против {ref}: медиана |разности| "
              f"{np.median(np.abs(sh - ref_sh)):.1%}, максимум {np.max(np.abs(sh - ref_sh)):.1%}")

    print("\nЧто из этого идёт в §6.2.3:")
    print("  — медиана и диапазон g*S: количественная мера того, насколько две модели окна")
    print("    расходятся, вместо нынешней фразы «мы этого не проверяли»;")
    print("  — строка «блок 1» показывает, что было бы при независимых окнах;")
    print("  — медиана |разности| доли между длинами блока: если она мала, вывод не")
    print("    зависит от выбора блока, и это надо сказать прямо.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"cell": args.cell, "path": args.path, "blocks": blocks, "houses": res},
            ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
