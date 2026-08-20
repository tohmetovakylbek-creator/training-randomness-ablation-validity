"""
compare_gate.py
===============
Изолирующий эксперимент по гейту (замечание научрука №1).

Вопрос: экстремальное доминирование сидовой компоненты (design-free ratio 51,
доля 83 %) — свойство архитектуры вообще или именно обучаемого слияния?

В §5.2.6 меняются одновременно архитектура, аблируемый компонент и число сидов,
поэтому вклад гейта там изолировать нельзя. Здесь меняется РОВНО ОДНО: тот же
объект, тот же аблируемый компонент (identity embeddings), те же пять сидов, те
же домохозяйства, тот же протокол обучения — с гейтом и без него.

Скрипт ничего не обучает: читает per-seed массивы, которые записал
factorial_aux_agg.py, и раскладывает дисперсию функциями ВАШЕГО
variance_model.py, поэтому соглашения совпадают по построению.

Запуск из корня проекта:

    python compare_gate.py --npz results/factorial_2comp/factorial_ukdale.npz ^
        results/factorial_2comp/factorial_refit.npz ^
        results/factorial_2comp/factorial_sheerm.npz ^
        --cell-gate aux-per_mode_agg-convex ^
        --cell-nogate aux-per_mode_agg-convex_nogate ^
        --path y_final --block 3 --exclude ukdale_house_4 ^
        --out results/gate_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import variance_model as vm


def decompose_all(paths, cell, path_key, block, exclude, n_boot, seed):
    eff = vm.load_effects(paths, cell, path_key, exclude=exclude)
    out = {}
    for h, d in eff.items():
        r = vm.decompose(d, block, n_boot=n_boot, seed=seed)
        ci = vm.exact_share_ci(d, block, r["var_win_of_mean"], n_boot=n_boot, seed=seed)
        S, W = d.shape
        out[h] = {
            "share": r["share_seed"],
            "lo": ci.get("lo", np.nan), "hi": ci.get("hi", np.nan),
            "sigma_seed2": r["sigma_seed2"], "sigma_win2": r["sigma_win2"],
            "V_win": r["var_win_of_mean"],
            "design_free": (r["sigma_seed2"] / r["var_win_of_mean"]
                            if r["var_win_of_mean"] > 0 else np.nan),
            "effect": r["effect"], "S": S, "W": W,
            "truncated": bool(r["sigma_seed2"] == 0.0),
        }
    return out


def summarise(name, res):
    sh = np.array([r["share"] for r in res.values()])
    lo = np.array([r["lo"] for r in res.values()])
    hi = np.array([r["hi"] for r in res.values()])
    dfr = np.array([r["design_free"] for r in res.values()])
    tr = sum(r["truncated"] for r in res.values())
    print(f"\n{name}  (n = {len(res)})")
    print(f"  доля сидовой дисперсии: медиана {np.median(sh):.1%}, "
          f"IQR [{np.percentile(sh,25):.1%}, {np.percentile(sh,75):.1%}]")
    print(f"  design-free ratio:      медиана {np.nanmedian(dfr):.1f}, "
          f"диапазон {np.nanmin(dfr):.1f}–{np.nanmax(dfr):.1f}")
    print(f"  нижняя граница > 1/2:   {int((lo > 0.5).sum())} из {len(res)}")
    print(f"  верхняя граница < 1/2:  {int((hi < 0.5).sum())} из {len(res)}")
    print(f"  компонента обрезана в ноль: {tr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--cell-gate", default="aux-per_mode_agg-convex")
    ap.add_argument("--cell-nogate", default="aux-per_mode_agg-convex_nogate")
    ap.add_argument("--path", default="y_final", choices=["y_final", "y_vmd"])
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="ukdale_house_4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    paths = vm.resolve_npz(args.npz) if hasattr(vm, "resolve_npz") else args.npz

    gate = decompose_all(paths, args.cell_gate, args.path, args.block,
                         exclude, args.n_boot, args.seed)
    nogate = decompose_all(paths, args.cell_nogate, args.path, args.block,
                           exclude, args.n_boot, args.seed)
    if not nogate:
        raise SystemExit(f"ячейка {args.cell_nogate} не найдена — прогон без гейта не выполнен?")

    houses = sorted(set(gate) & set(nogate))
    print(f"домохозяйств с обеими конфигурациями: {len(houses)}")
    if len(houses) < 3:
        raise SystemExit("слишком мало для парных тестов")

    summarise("С ГЕЙТОМ   " + args.cell_gate, {h: gate[h] for h in houses})
    summarise("БЕЗ ГЕЙТА  " + args.cell_nogate, {h: nogate[h] for h in houses})

    print(f"\n{'домохозяйство':20s}{'доля(гейт)':>12}{'доля(без)':>11}"
          f"{'dfr(гейт)':>11}{'dfr(без)':>10}{'эффект(гейт)':>14}{'эффект(без)':>13}")
    for h in houses:
        g, n = gate[h], nogate[h]
        print(f"{h:20s}{g['share']:12.1%}{n['share']:11.1%}"
              f"{g['design_free']:11.1f}{n['design_free']:10.1f}"
              f"{g['effect']:+14.2f}{n['effect']:+13.2f}")

    print("\nпарные сравнения (Wilcoxon signed-rank):")
    for key, label in (("share", "доля сидовой дисперсии"),
                       ("design_free", "design-free ratio"),
                       ("sigma_seed2", "sigma_seed^2"),
                       ("V_win", "дисперсия среднего по окнам")):
        a = np.array([gate[h][key] for h in houses])
        b = np.array([nogate[h][key] for h in houses])
        if np.allclose(a, b):
            print(f"  {label:28s}: идентичны")
            continue
        st, p = stats.wilcoxon(a, b)
        print(f"  {label:28s}: медиана с гейтом {np.nanmedian(a):.4g}, "
              f"без гейта {np.nanmedian(b):.4g}, p = {p:.5f}, "
              f"выше без гейта у {int((b > a).sum())} из {len(houses)}")

    print("\nКак читать:")
    print("  Если доля и design-free ratio БЕЗ гейта заметно ниже, гейт действительно")
    print("  усиливает сидовую компоненту на выходе модели, и гипотеза §5.2.6 становится")
    print("  измерением. Если разницы нет, доминирование — свойство обучающего режима,")
    print("  а не слияния, и формулировку в §5.2.6 и §6.2.1 надо смягчить.")
    print("  Смотрите отдельно sigma_seed^2 и V_win: доля может измениться и потому,")
    print("  что изменился знаменатель, а не потому, что изменилась сидовая компонента.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"cell_gate": args.cell_gate, "cell_nogate": args.cell_nogate,
             "path": args.path, "block": args.block,
             "gate": gate, "nogate": nogate}, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8")
        print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
