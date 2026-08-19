"""
find_per_seed_npz.py
====================
Ищет .npz с подсидовыми массивами (*_per_seed) и печатает, какие ячейки
факторного плана и какие пути (y_final / y_vmd) в них лежат — то есть готовые
значения для --npz, --cell и --path-key.

Запуск из корня проекта:

    python find_per_seed_npz.py
    python find_per_seed_npz.py --root .. --pattern "**/*.npz"
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--pattern", default="**/*.npz")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob(args.pattern))
    print(f"найдено .npz: {len(files)}\n")

    hits = 0
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as d:
                keys = list(d.files)
        except Exception as e:                                   # noqa: BLE001
            print(f"[!] {p}: не читается ({e})")
            continue

        per_seed = [k for k in keys if k.endswith("_per_seed")]
        if not per_seed:
            continue
        hits += 1

        cells = defaultdict(set)
        houses = set()
        for k in per_seed:
            parts = k.split("|")
            if len(parts) < 3:
                continue
            houses.add(parts[0])
            suffix = parts[2]
            path_key = "y_final" if suffix == "err_final_per_seed" else (
                "y_vmd" if suffix == "err_vmd_per_seed" else suffix)
            cell = parts[1].replace("_emb-on", "").replace("_emb-off", "")
            cells[cell].add(path_key)

        print("=" * 70)
        print(f"ФАЙЛ: {p}")
        print(f"  домохозяйств: {len(houses)}  ({', '.join(sorted(houses)[:5])}"
              f"{' ...' if len(houses) > 5 else ''})")
        # проверяем, что массивы двумерные (иначе load_effects их пропустит)
        with np.load(p, allow_pickle=True) as d:
            shapes = {d[k].ndim for k in per_seed[:20]}
        print(f"  размерность массивов: {sorted(shapes)}"
              f"{'  <-- нужна 2, иначе перезапустите factorial_aux_agg.py' if 2 not in shapes else '  OK'}")
        print("  доступные ячейки:")
        for cell in sorted(cells):
            for pk in sorted(cells[cell]):
                print(f"    --cell \"{cell}\" --path-key {pk}")

    if not hits:
        print("Ни в одном .npz нет массивов *_per_seed.\n"
              "Скорее всего подсидовые массивы писала другая (новая) версия "
              "factorial_aux_agg.py — поищите её вывод или перезапустите её.")


if __name__ == "__main__":
    main()
