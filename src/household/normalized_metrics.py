"""
normalized_metrics.py
=====================
Масштабно-независимые метрики (замечание научрука №4): помимо ватт и отношения
к константному предиктору — MASE и нормированный MAE.

    MASE  = MAE / MAE_naive,  где naive — сезонный прогноз y(t) = y(t - 24)
            на ОБУЧАЮЩЕМ сегменте (стандартное определение Hyndman & Koehler);
    nMAE  = MAE / средняя нагрузка домохозяйства на обучающем сегменте.

Знаменатель MASE берётся из processed/house_<n>.npz (обучающий ряд, обратно
масштабированный в ватты), числитель — из factorial_<dataset>.json.
Обучения не требует.

Запуск:
    python normalized_metrics.py ^
        --root ukdale=processed --root refit=processed_refit --root sheerm=processed_sheerm ^
        --json results/factorial_2comp/factorial_ukdale.json ^
               results/factorial_2comp/factorial_refit.json ^
               results/factorial_2comp/factorial_sheerm.json ^
        --cells aux-per_mode_agg-convex_emb-on aux-sum_agg-convex_emb-on ^
        --exclude ukdale_house_4 --out results/normalized_metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def naive_and_mean(npz_path: Path, season: int = 24) -> tuple[float, float]:
    """Сезонная наивная MAE и средняя нагрузка на обучающем сегменте, в ваттах."""
    d = np.load(npz_path)
    lo, hi = float(d["scaler_lo"]), float(d["scaler_hi"])
    # Ytr: (N, T) окна целей с единичным шагом -> первый столбец даёт непрерывный ряд
    y = d["Ytr"][:, 0].astype(np.float64)
    y_w = y * (hi - lo) + lo
    if len(y_w) <= season:
        return float("nan"), float(np.mean(y_w))
    naive = float(np.mean(np.abs(y_w[season:] - y_w[:-season])))
    return naive, float(np.mean(y_w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="dataset=path")
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", default=["aux-per_mode_agg-convex_emb-on"])
    ap.add_argument("--season", type=int, default=24)
    ap.add_argument("--exclude", default="ukdale_house_4")
    ap.add_argument("--out", default="results/normalized_metrics.json")
    args = ap.parse_args()

    roots = {}
    for it in args.root:
        name, _, path = it.partition("=")
        roots[name] = Path(path)
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    data = {}
    for f in args.json:
        for h, rec in json.loads(Path(f).read_text(encoding="utf-8")).items():
            data.setdefault(h, {}).update(rec.get("cells", {}))
    houses = sorted(h for h in data if h not in exclude)

    rows = []
    for h in houses:
        m = re.match(r"([a-z\-]+)_house_(\d+)$", h)
        if not m or m.group(1) not in roots:
            print(f"[skip] {h}: нет корня для датасета"); continue
        npz = roots[m.group(1)] / f"house_{m.group(2)}.npz"
        if not npz.exists():
            print(f"[skip] {h}: нет {npz}"); continue
        naive, mean_w = naive_and_mean(npz, args.season)
        row = {"house": h, "naive_MAE": naive, "mean_load_W": mean_w}
        for c in args.cells:
            cell = data[h].get(c)
            if not cell:
                continue
            mae = cell["paths"]["y"]["MAE"]
            row[c] = {"MAE": mae, "MASE": mae / naive, "nMAE": mae / mean_w,
                      "MAE_vs_const": cell["paths"]["y"]["MAE_vs_const"]}
        rows.append(row)

    print(f"\n{'домохозяйство':20s}{'mean, W':>10}{'naive MAE':>11}", end="")
    for c in args.cells:
        print(f"{c[:18]:>20s}", end="")
    print()
    for r in rows:
        print(f"{r['house']:20s}{r['mean_load_W']:10.0f}{r['naive_MAE']:11.1f}", end="")
        for c in args.cells:
            v = r.get(c)
            print(f"{('MASE ' + format(v['MASE'], '.3f')) if v else '—':>20s}", end="")
        print()

    print("\nсводка по конфигурациям:")
    for c in args.cells:
        mase = np.array([r[c]["MASE"] for r in rows if c in r])
        nmae = np.array([r[c]["nMAE"] for r in rows if c in r])
        vs = np.array([r[c]["MAE_vs_const"] for r in rows if c in r])
        print(f"  {c}")
        print(f"    MASE:         медиана {np.median(mase):.3f}, "
              f"диапазон {mase.min():.3f}–{mase.max():.3f}, "
              f"хуже наивного (>1): {(mase > 1).sum()} из {len(mase)}")
        print(f"    nMAE:         медиана {np.median(nmae):.3f}, "
              f"диапазон {nmae.min():.3f}–{nmae.max():.3f}")
        print(f"    MAE/константа: медиана {np.median(vs):.3f} (для сверки с Таблицей 2)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"season": args.season, "cells": args.cells, "houses": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
