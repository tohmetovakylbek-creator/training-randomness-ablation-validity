# -*- coding: utf-8 -*-
"""
renormalize.py
==============
Перевод факториальных результатов на НОРМИРОВАННУЮ шкалу и на per-seed эстиманд.
Отвечает на критические замечания рецензии: несовпадение эстимандов, объединение
в ваттах, некорректная проверка практической эквивалентности.

Что делает:
  1. Делит все поокновые ошибки на MAE константного предиктора того же объекта.
     После этого эффект выражен в долях собственной ошибки объекта, порог
     практической значимости становится единой линией +-2 %, а взвешивание
     обратными дисперсиями перестаёт определяться маломощными рядами.
  2. Оставляет ТОЛЬКО per-seed массивы. Ансамблевые (err_final, err_vmd) в
     нормированный файл не переносятся: первичный эстиманд — средний per-seed
     эффект, ансамбль уходит во вторичный анализ.
  3. Пишет файлы в той же схеме ключей, что и раньше, поэтому variance_model.py
     и meta_analysis.py запускаются на них без единой правки.
  4. Собирает results_master.json — единый источник всех чисел для текста,
     чтобы расхождения вида "в R5 -0.31, а в 5.4.2 -0.28" больше не возникали.

Запуск (ноутбук):
    python renormalize.py --npz factorial_per_window_ukdale.npz \
                                 factorial_per_window_refit.npz \
                                 factorial_per_window_sheerm.npz \
           --mae_const mae_const_households.json \
           --map "processed :: house_1=ukdale_house_1" \
           --exclude ukdale_house_4 \
           --out factorial_per_window_norm.npz --json factorial_norm.json
Затем:
    python variance_model.py --npz factorial_per_window_norm.npz \
        --cell aux-per_mode_agg-convex --path y_final --block 3 --out results_norm/variance
    python meta_analysis.py --json factorial_norm.json --npz factorial_per_window_norm.npz \
        --cell aux-per_mode_agg-convex --path y_final --deltas 0.01,0.02,0.05 \
        --out results_norm/meta
(--deltas теперь в ДОЛЯХ ошибки объекта: 0.02 — это порог 2 %.)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

Z = 1.959963985
PRIMARY_CELL = "aux-per_mode_agg-convex"

# сопоставление ключей mae_const ("<папка> :: house_N") с объектами факториала
DEFAULT_DIR_MAP = {"processed_refit": "refit", "processed_sheerm": "sheerm",
                   "processed_ukdale": "ukdale", "processed_ecl": "ecl"}


def load_mae_const(path: Path, extra_map: dict[str, str]) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k in extra_map:                      # явное сопоставление из --map
            out[extra_map[k]] = float(v)
            continue
        m = re.match(r"\s*(\S+)\s*::\s*(\S+)\s*$", k)
        if not m:
            continue
        folder, name = m.group(1), m.group(2)
        prefix = DEFAULT_DIR_MAP.get(folder)
        if prefix is None:
            continue                            # неоднозначная папка — только через --map
        out[f"{prefix}_{name}"] = float(v)
    return out


def normalize(npz_paths, mae_const, exclude) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    missing, skipped = set(), set()
    for p in npz_paths:
        with np.load(p) as z:
            for k in z.files:
                unit = k.split("|")[0]
                if unit in exclude:
                    skipped.add(unit); continue
                if not k.endswith("_per_seed"):
                    continue                    # ансамблевые массивы не переносим
                c = mae_const.get(unit)
                if c is None:
                    missing.add(unit); continue
                merged[k] = z[k].astype(np.float64) / c
                merged[f"{unit}|mae_const"] = np.array(c)
    if missing:
        raise SystemExit(f"Нет константной MAE для: {sorted(missing)}. "
                         f"Добавьте их в --mae_const или сопоставьте через --map.")
    if skipped:
        print(f"исключены: {sorted(skipped)}")
    return merged


# ---------------------------------------------------------------------------
def moving_block_var_of_mean(x: np.ndarray, block: int = 3,
                             n_boot: int = 2000, seed: int = 0) -> float:
    """Дисперсия среднего по окнам с учётом зависимости соседних суток."""
    rng = np.random.default_rng(seed)
    W = len(x)
    nb = int(np.ceil(W / block))
    starts = np.arange(W - block + 1)
    idx = (rng.integers(0, len(starts), size=(n_boot, nb))[:, :, None]
           + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :W]
    return float(np.var(x[idx].mean(axis=1), ddof=1))


def decompose(d: np.ndarray, block: int = 3) -> dict:
    """d — матрица (сиды x окна) эффекта. Компоненты по суммам квадратов ANOVA."""
    S, W = d.shape
    seed_means = d.mean(axis=1)
    grand = d.mean()
    ms_between = W * np.sum((seed_means - grand) ** 2) / (S - 1)
    ms_within = np.sum((d - seed_means[:, None]) ** 2) / (S * (W - 1))
    sigma_seed2 = max(0.0, (ms_between - ms_within) / W)
    var_seed_of_mean = sigma_seed2 / S
    var_win_of_mean = moving_block_var_of_mean(d.mean(axis=0), block)
    share = var_seed_of_mean / (var_seed_of_mean + var_win_of_mean) \
        if (var_seed_of_mean + var_win_of_mean) > 0 else 0.0
    return {"effect": float(grand), "S": S, "W": W,
            "sigma_seed2": float(sigma_seed2),
            "var_seed_of_mean": float(var_seed_of_mean),
            "var_win_of_mean": float(var_win_of_mean),
            "v": float(var_seed_of_mean + var_win_of_mean),
            "share_seed": float(share),
            "seed_sd_of_effect": float(np.std(seed_means, ddof=1))}


def dersimonian_laird(y, v):
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y); w = 1.0 / v
    mu_fe = float((w * y).sum() / w.sum())
    Q = float((w * (y - mu_fe) ** 2).sum()); df = k - 1
    denom = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / denom) if denom > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float((ws * y).sum() / ws.sum()); se = float(np.sqrt(1.0 / ws.sum()))
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    from scipy import stats
    t = stats.t.ppf(0.975, k - 2) if k > 2 else np.nan
    half = t * np.sqrt(se ** 2 + tau2) if k > 2 else np.nan
    return {"k": k, "mu": mu, "se_mu": se, "tau2": tau2, "tau": float(np.sqrt(tau2)),
            "Q": Q, "df": df, "I2": I2,
            "p": float(2 * (1 - stats.norm.cdf(abs(mu / se)))) if se > 0 else np.nan,
            "pi_low": mu - half, "pi_high": mu + half}


def naive_stats(d: np.ndarray) -> dict:
    S, W = d.shape
    seed_means = d.mean(axis=1)
    se_b = float(np.std(seed_means, ddof=1) / np.sqrt(S))
    se_w = float(np.std(d.mean(axis=0), ddof=1) / np.sqrt(W))
    se = float(np.sqrt(se_b ** 2 + se_w ** 2)); mean = float(d.mean())
    return {"mean": mean, "se_total": se, "se_within": se_w, "se_between": se_b,
            "ci_low": mean - Z * se, "ci_high": mean + Z * se}


def build_json(arrays: dict) -> dict:
    units = sorted({k.split("|")[0] for k in arrays if not k.endswith("|mae_const")})
    cells = sorted({k.split("|")[1].rsplit("_emb-", 1)[0]
                    for k in arrays if "|" in k and "_emb-" in k})
    rep = {}
    for u in units:
        eff = {}
        for cell in cells:
            benefit = {}
            for tag, path in (("err_final_per_seed", "y_final"),
                              ("err_vmd_per_seed", "y_vmd")):
                on = arrays.get(f"{u}|{cell}_emb-on|{tag}")
                off = arrays.get(f"{u}|{cell}_emb-off|{tag}")
                if on is None or off is None:
                    continue
                benefit[path] = naive_stats(off - on)
            if benefit:
                eff[cell] = {"coherent": "per_mode" in cell,
                             "embeddings_benefit": benefit}
        rep[u] = {"embeddings_effect": eff,
                  "mae_const": float(arrays.get(f"{u}|mae_const", np.nan))}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--mae_const", required=True)
    ap.add_argument("--map", nargs="*", default=[],
                    help='явные пары "<ключ из mae_const>=<объект факториала>"')
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out", default="factorial_per_window_norm.npz")
    ap.add_argument("--json", default="factorial_norm.json")
    ap.add_argument("--master", default="results_master.json")
    ap.add_argument("--cell", default=PRIMARY_CELL)
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.02)
    args = ap.parse_args()

    extra = dict(m.split("=", 1) for m in args.map)
    mae_const = load_mae_const(Path(args.mae_const), extra)
    arrays = normalize([Path(p) for p in args.npz], mae_const, set(args.exclude))
    np.savez_compressed(args.out, **arrays)
    Path(args.json).write_text(json.dumps(build_json(arrays), ensure_ascii=False,
                                          indent=2), encoding="utf-8")

    units = sorted({k.split("|")[0] for k in arrays if not k.endswith("|mae_const")})
    rows = []
    for u in units:
        on = arrays.get(f"{u}|{args.cell}_emb-on|err_final_per_seed")
        off = arrays.get(f"{u}|{args.cell}_emb-off|err_final_per_seed")
        if on is None or off is None:
            continue
        r = decompose(off - on, args.block); r["unit"] = u
        rows.append(r)

    y = [r["effect"] for r in rows]; v = [r["v"] for r in rows]
    pooled = dersimonian_laird(y, v)
    shares = np.array([r["share_seed"] for r in rows])
    inside = (pooled["pi_low"] > -args.threshold) and (pooled["pi_high"] < args.threshold)

    master = {"cell": args.cell, "path": "y_final", "scale": "доли MAE константы",
              "estimand": "средний per-seed эффект (emb-off минус emb-on)",
              "threshold": args.threshold, "block": args.block,
              "units": rows, "pooled": pooled,
              "share_seed": {"median": float(np.median(shares)),
                             "mean": float(shares.mean()),
                             "above_half": int((shares > 0.5).sum()), "n": len(shares)},
              "equivalence_pi_within_threshold": bool(inside)}
    Path(args.master).write_text(json.dumps(master, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    print(f"объектов: {len(rows)}   шкала: доли MAE константы   "
          f"эстиманд: средний per-seed")
    print(f"{'объект':20s} {'эффект,%':>9s} {'sd сидов,%':>11s} {'доля сидов':>11s}")
    for r in sorted(rows, key=lambda r: r["effect"]):
        print(f"{r['unit']:20s} {r['effect']*100:+9.3f} "
              f"{r['seed_sd_of_effect']*100:11.3f} {r['share_seed']*100:10.0f}%")
    print(f"\nдоля сидовой дисперсии: медиана {np.median(shares)*100:.0f}%, "
          f"среднее {shares.mean()*100:.0f}%, выше половины у "
          f"{int((shares>0.5).sum())} из {len(shares)}")
    print(f"сводный эффект: {pooled['mu']*100:+.3f}% (SE {pooled['se_mu']*100:.3f}), "
          f"p = {pooled['p']:.3f}")
    print(f"tau = {pooled['tau']*100:.3f}%, I2 = {pooled['I2']:.1f}%, "
          f"Q({pooled['df']}) = {pooled['Q']:.1f}")
    print(f"интервал предсказания: [{pooled['pi_low']*100:+.3f}%, "
          f"{pooled['pi_high']*100:+.3f}%] при пороге ±{args.threshold*100:.0f}%")
    print(f"ПРАКТИЧЕСКАЯ ЭКВИВАЛЕНТНОСТЬ: "
          f"{'установлена' if inside else 'НЕ УСТАНОВЛЕНА'}")
    print(f"\nзаписано: {args.out}, {args.json}, {args.master}")


if __name__ == "__main__":
    main()
