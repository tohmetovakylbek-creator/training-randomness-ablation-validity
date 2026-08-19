"""
crosscheck_variance2.py
=======================
Прямая сверка: матрица эффектов d строится ВАШИМ variance_model.load_effects(),
затем раскладывается двумя реализациями — variance_model.decompose() и
generality_variance.variance_components() — и результаты печатаются рядом.

Кладётся в корень проекта, рядом с variance_model.py.

Шаг 1. Посмотреть, что именно ждёт load_effects (какие cell / path_key бывают):

    python crosscheck_variance2.py --show-source

Шаг 2. Сверка на одном домохозяйстве:

    python crosscheck_variance2.py --npz results_full/house_2.npz --cell <cell> --path-key <key>

Если load_effects принимает список путей — перечислите их через пробел после --npz.
Значения --cell и --path-key берутся из того, что напечатал --show-source
(это те же значения, с которыми вы запускаете variance_model.py обычно).
"""
from __future__ import annotations

import argparse
import inspect
import sys

import numpy as np

import variance_model as vm
from generality_variance import variance_components, BLOCK, N_BOOT


def show_source():
    for name in ("load_effects", "decompose", "block_var_of_mean", "exact_share_ci", "main"):
        fn = getattr(vm, name, None)
        if fn is None:
            continue
        print("=" * 70)
        print(f"{name}{inspect.signature(fn)}")
        print("-" * 70)
        try:
            print(inspect.getsource(fn))
        except OSError:
            print("(исходник недоступен)")


def as_dict(res) -> dict:
    """decompose может вернуть dict, namedtuple или кортеж — приводим к dict."""
    if isinstance(res, dict):
        return res
    if hasattr(res, "_asdict"):
        return dict(res._asdict())
    if isinstance(res, (tuple, list)):
        return {f"[{i}]": v for i, v in enumerate(res)}
    return {"result": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-source", action="store_true")
    ap.add_argument("--npz", nargs="+")
    ap.add_argument("--cell")
    ap.add_argument("--path-key")
    ap.add_argument("--house", help="какое домохозяйство брать (по умолчанию первое по алфавиту)")
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.show_source:
        show_source()
        return

    if not args.npz:
        sys.exit("укажите --npz (и, если нужно, --cell / --path-key), либо --show-source")

    kwargs = {}
    if args.cell is not None:
        kwargs["cell"] = args.cell
    if args.path_key is not None:
        kwargs["path_key"] = args.path_key

    paths = args.npz
    if hasattr(vm, "resolve_npz"):
        try:
            paths = vm.resolve_npz(paths)
        except Exception as e:                                   # noqa: BLE001
            print(f"[warn] resolve_npz не отработал ({e}), передаю пути как есть")

    d = vm.load_effects(paths, **kwargs)

    # load_effects возвращает словарь {household: matrix}
    if isinstance(d, dict):
        if not d:
            sys.exit("load_effects вернул пустой результат: проверьте --npz, --cell и --path-key "
                     "(список доступных значений печатает find_per_seed_npz.py)")
        print(f"load_effects вернул {len(d)} домохозяйств: {', '.join(sorted(d))}")
        if args.house:
            if args.house not in d:
                sys.exit(f"домохозяйство {args.house} не найдено")
            house = args.house
        else:
            house = sorted(d)[0]
        print(f"беру: {house}")
        d = d[house]
    d = np.asarray(d, dtype=float)
    if d.ndim != 2:
        sys.exit(f"ожидалась матрица (S, W), получено {d.shape}")
    S, W = d.shape
    print(f"матрица эффектов: S = {S} сидов, W = {W} окон\n")

    # ---------------- их реализация ----------------
    theirs = as_dict(vm.decompose(d, args.block, n_boot=N_BOOT, seed=args.seed))
    print("=== variance_model.decompose ===")
    for k, v in theirs.items():
        print(f"  {k:22s} {v}")

    # ---------------- моя реализация ----------------
    mine = variance_components(d)
    print("\n=== generality_variance.variance_components ===")
    for k in ("sigma_seed2", "sigma_win2", "V_win", "share_seed",
              "share_lo", "share_hi", "theta", "design_free_ratio", "mean_effect_w"):
        print(f"  {k:22s} {mine[k]}")

    # ---------------- сопоставление ----------------
    print("\n=== совпадения ===")
    hits = 0
    for k_mine, val_mine in mine.items():
        if not isinstance(val_mine, (int, float)) or isinstance(val_mine, bool):
            continue
        for k_th, val_th in theirs.items():
            if not isinstance(val_th, (int, float)) or isinstance(val_th, bool):
                continue
            if val_th != 0 and abs(val_mine - val_th) <= 1e-3 * max(1.0, abs(val_th)):
                print(f"  {k_mine:20s} = {k_th:20s} : {val_mine:.6f}")
                hits += 1
    if not hits:
        print("  ни одна пара величин не совпала — сравните вручную по таблицам выше")

    print("\nГлавное — доля сидовой дисперсии. Если она совпала до третьего знака, "
          "реализации согласованы и Таблицу 4 можно оставлять как есть.")
    print("Если разошлась только оценка дисперсии среднего по окнам, дело в настройках "
          f"бутстрэпа (у меня block={BLOCK}, B={N_BOOT}, seed фиксирован); это допустимо, "
          "но должно быть описано в §4.5.")
    print("Если разошлись компоненты дисперсии — это ошибка в одной из реализаций, "
          "и её надо устранить до подачи.")


if __name__ == "__main__":
    main()
