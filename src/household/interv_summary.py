# -*- coding: utf-8 -*-
"""
interv_summary.py
=================
Сводка по ВСЕМ условиям интервенции из results/intervention_5seed/.
В статье (Таблица 5) отчитывается только zeroed; остальные условия описаны
в §5.3.2, но их числа нигде не приведены. Скрипт достаёт их все.

Запуск (ноутбук):
    python interv_summary.py
    python interv_summary.py --cell aux-sum_agg-convex --path y_vmd
"""

import argparse
import glob
import json
import numpy as np

EXCLUDE = {"ukdale_house_4"}


def val(x):
    """Из записи условия достаёт число: сама запись или её поле с MAE/дельтой."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ("MAE", "mae", "mae_w", "value", "delta", "delta_mae", "mean"):
            if k in x and isinstance(x[k], (int, float)):
                return float(x[k])
        nums = [v for v in x.values() if isinstance(v, (int, float))]
        if len(nums) == 1:
            return float(nums[0])
        raise SystemExit(f"Не понимаю запись условия: {list(x)}")
    raise SystemExit(f"Не понимаю тип записи условия: {type(x)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/intervention_5seed")
    ap.add_argument("--cell", default="aux-per_mode_agg-convex")
    ap.add_argument("--path", default="y_final")
    ap.add_argument("--keep_h4", action="store_true")
    args = ap.parse_args()

    units = {}
    for f in glob.glob(f"{args.dir}/*{args.cell}*.json"):
        for u, blob in json.load(open(f, encoding="utf-8")).items():
            if u in EXCLUDE and not args.keep_h4:
                continue
            units[u] = blob

    print(f"объектов: {len(units)}  ячейка: {args.cell}  выход: {args.path}")
    conds = sorted({k.split("|")[1] for b in units.values()
                    for k in b["conditions"] if k.startswith(args.path + "|")})
    print(f"условия: {conds}\n")

    base = {u: val(b["conditions"][f"{args.path}|identity"]) for u, b in units.items()}
    print(f"{'условие':24s} {'медиана, Вт':>12s} {'% от MAE':>10s} "
          f"{'хуже, из N':>11s} {'диапазон, Вт':>22s}")
    for c in conds:
        if c == "identity":
            continue
        d, rel = [], []
        for u, b in units.items():
            k = f"{args.path}|{c}"
            if k not in b["conditions"]:
                continue
            delta = val(b["conditions"][k]) - base[u]
            d.append(delta)
            rel.append(delta / base[u] * 100)
        d, rel = np.array(d), np.array(rel)
        print(f"{c:24s} {np.median(d):+12.3f} {np.median(rel):+10.2f} "
              f"{int((d > 0).sum()):6d}/{len(d):<4d} "
              f"[{d.min():+9.3f}, {d.max():+9.3f}]")

    print(f"\nдля справки, MAE под identity: медиана {np.median(list(base.values())):.2f} Вт")
    # дополнительные поля записи условия, если они есть
    sample = next(iter(units.values()))["conditions"].get(f"{args.path}|identity")
    if isinstance(sample, dict) and len(sample) > 1:
        print(f"в записи условия также: {[k for k in sample if k != 'MAE']} "
              f"(здесь используется MAE)")
    gates = [b["mean_gate"] for b in units.values() if "mean_gate" in b]
    if gates:
        print(f"средний вес гейта: медиана {np.median(gates):.3f}, "
              f"диапазон {min(gates):.3f}-{max(gates):.3f}")


if __name__ == "__main__":
    main()
