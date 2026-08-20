"""
generality_variance.py
======================
Разложение дисперсии эффекта абляции для внешних архитектур (§5.2.6).
Считает ровно те же величины, что variance_model.py на основной модели:

    d(h,s,w) = |err|_off(h,s,w) - |err|_on(h,s,w)        (>0 => компонент помогает)
    sigma_interaction^2 = MS_interaction
    sigma_seed^2  = max(0, (MS_seed - MS_interaction) / W)
    V_win         = дисперсия среднего по окнам (moving block bootstrap, блок 3)
    share_seed    = (sigma_seed^2 / S) / (sigma_seed^2 / S + V_win)
    модельный 95% интервал для theta = sigma_seed^2 / sigma_interaction^2
    через F-распределение
    design-free ratio = sigma_seed^2 / V_win   (величина, сравнимая между дизайнами)

ПРОВЕРКА ПЕРЕД ИСПОЛЬЗОВАНИЕМ: прогоните этот файл на одном домохозяйстве
основного эксперимента (флаг --selftest с двумя .npy) и сверьте share_seed с
тем, что даёт variance_model.py. Числа должны совпасть до 3-го знака; если нет —
разошлись соглашения (ddof, длина блока, число бутстрэп-повторов), и дальше
идти нельзя.

Использование:
    python generality_variance.py --dir results_generality
    python generality_variance.py --dir results_generality --out table_5_2_6.csv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

BLOCK = 3
N_BOOT = 2000
ALPHA = 0.05
SEED = 0          # как в variance_model.block_var_of_mean
COMPONENT = {"patchtst": "RevIN (reversible instance normalisation)",
             "bilstm": "attention pooling over encoder states"}


# ---------------------------------------------------------------------------
def moving_block_bootstrap_var(x: np.ndarray, block: int = BLOCK,
                               n_boot: int = N_BOOT, seed: int = SEED) -> float:
    """Дисперсия среднего ряда x с учётом зависимости соседних окон.

    Генератор создаётся внутри вызова, поэтому результат не зависит от порядка
    обхода домохозяйств и воспроизводится независимо от остального прогона.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < block + 1:
        return float(np.var(x, ddof=1) / max(n, 1))
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    return float(means.var(ddof=1))


def variance_components(d: np.ndarray) -> dict:
    """Crossed seed × window decomposition for a balanced S × W design."""
    S, W = d.shape
    seed_means = d.mean(axis=1)
    window_means = d.mean(axis=0)
    grand = d.mean()
    residual = d - seed_means[:, None] - window_means[None, :] + grand
    ss_seed = W * float(((seed_means - grand) ** 2).sum())
    ss_window = S * float(((window_means - grand) ** 2).sum())
    ss_interaction = float((residual ** 2).sum())
    ms_seed = ss_seed / (S - 1)
    ms_window = ss_window / (W - 1)
    ms_interaction = ss_interaction / ((S - 1) * (W - 1))

    sigma_interaction2 = ms_interaction
    sigma_seed2 = max(0.0, (ms_seed - ms_interaction) / W)
    sigma_window2 = max(0.0, (ms_window - ms_interaction) / S)

    v_win = moving_block_bootstrap_var(d.mean(axis=0))
    share = (sigma_seed2 / S) / (sigma_seed2 / S + v_win) if (sigma_seed2 + v_win) > 0 else 0.0

    # Model-based F interval for theta = sigma_seed^2 / sigma_interaction^2.
    ratio = ms_seed / ms_interaction if ms_interaction > 0 else np.inf
    df1, df2 = S - 1, (S - 1) * (W - 1)
    f_hi = stats.f.ppf(1 - ALPHA / 2, df1, df2)
    f_lo = stats.f.ppf(ALPHA / 2, df1, df2)
    theta_lo = max(0.0, (ratio / f_hi - 1.0) / W)
    theta_hi = max(0.0, (ratio / f_lo - 1.0) / W)

    def share_of_theta(theta):
        s2 = theta * sigma_interaction2
        return (s2 / S) / (s2 / S + v_win) if (s2 + v_win) > 0 else 0.0

    return {
        "S": S, "W": W,
        "sigma_seed2": sigma_seed2, "sigma_window2": sigma_window2,
        "sigma_interaction2": sigma_interaction2, "V_win": v_win,
        "share_seed": share,
        "share_lo": share_of_theta(theta_lo), "share_hi": share_of_theta(theta_hi),
        "theta": sigma_seed2 / sigma_interaction2 if sigma_interaction2 > 0 else np.nan,
        "design_free_ratio": sigma_seed2 / v_win if v_win > 0 else np.nan,
        "truncated_at_zero": bool(sigma_seed2 == 0.0),
        "mean_effect_w": float(d.mean()),
    }


# ---------------------------------------------------------------------------
def collect(dirpath: Path) -> dict:
    pat = re.compile(r"err_(?P<arch>\w+?)_(?P<cfg>on|off)_(?P<ds>\w+?)_(?P<house>\d+)_seed(?P<seed>\d+)\.npy$")
    store = defaultdict(dict)
    for f in sorted(dirpath.glob("err_*.npy")):
        m = pat.match(f.name)
        if not m:
            print(f"  [!] не разобрано имя файла: {f.name}")
            continue
        g = m.groupdict()
        store[(g["arch"], f"{g['ds']}_{g['house']}")][(g["cfg"], int(g["seed"]))] = np.load(f)
    return store


def build_matrix(runs: dict) -> np.ndarray | None:
    seeds = sorted({s for (_, s) in runs})
    have = [s for s in seeds if ("on", s) in runs and ("off", s) in runs]
    if len(have) < 2:
        return None
    widths = {len(runs[("on", s)]) for s in have} | {len(runs[("off", s)]) for s in have}
    if len(widths) != 1:
        print("  [!] разное число тестовых окон между прогонами — дизайн не сбалансирован")
        return None
    return np.stack([runs[("off", s)] - runs[("on", s)] for s in have])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None, help="куда сохранить подробный CSV")
    ap.add_argument("--expected-seeds", default="42,7,13,99,2025",
                    help="точный набор сидов, обязательный для каждой пары on/off")
    args = ap.parse_args()

    d = Path(args.dir)
    expected_seeds = {int(x) for x in args.expected_seeds.split(",") if x.strip()}
    store = collect(d)
    if not store:
        raise SystemExit("не найдено ни одного err_*.npy")

    per_house, summary = [], defaultdict(list)
    for (arch, house), runs in sorted(store.items()):
        paired = {s for cfg, s in runs if cfg == "on" and ("off", s) in runs}
        if paired != expected_seeds:
            missing = sorted(expected_seeds - paired)
            extra = sorted(paired - expected_seeds)
            raise SystemExit(f"{arch} {house}: required paired seeds={sorted(expected_seeds)}; "
                             f"missing={missing}, extra={extra}")
        mat = build_matrix(runs)
        if mat is None:
            print(f"[skip] {arch} {house}: неполный набор сидов")
            continue
        r = variance_components(mat)
        r.update(arch=arch, house=house)
        per_house.append(r)
        summary[arch].append(r)

    # -------- подробная таблица по домохозяйствам
    lines = ["arch,house,S,W,mean_effect_w,share_seed,share_lo,share_hi,"
             "design_free_ratio,truncated_at_zero"]
    for r in per_house:
        lines.append(f"{r['arch']},{r['house']},{r['S']},{r['W']},{r['mean_effect_w']:.4f},"
                     f"{r['share_seed']:.4f},{r['share_lo']:.4f},{r['share_hi']:.4f},"
                     f"{r['design_free_ratio']:.3f},{int(r['truncated_at_zero'])}")
    csv_path = Path(args.out) if args.out else d / "variance_per_house.csv"
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    # -------- сводка для Таблицы §5.2.6
    print("\n=== Таблица для §5.2.6 ===")
    header = (f"{'architecture':<12} {'component':<34} {'S':>2} {'median':>7} "
              f"{'IQR':>13} {'med.lower':>10} {'>50%':>7} {'df ratio':>9} {'trunc':>6}")
    print(header)
    rows_md = ["| Architecture | Ablated component | S | Median seed share | IQR | "
               "Median lower limit | Households > 50% | Median design-free ratio |",
               "|---|---|---|---|---|---|---|---|"]
    for arch, rs in summary.items():
        sh = np.array([r["share_seed"] for r in rs])
        lo = np.array([r["share_lo"] for r in rs])
        dfr = np.array([r["design_free_ratio"] for r in rs])
        above = int((lo > 0.5).sum())
        trunc = int(sum(r["truncated_at_zero"] for r in rs))
        comp = COMPONENT.get(arch, "?")
        print(f"{arch:<12} {comp:<34} {rs[0]['S']:>2} {np.median(sh):>6.1%} "
              f"[{np.percentile(sh,25):.1%}, {np.percentile(sh,75):.1%}] "
              f"{np.median(lo):>9.1%} {above:>4}/{len(rs):<3} {np.nanmedian(dfr):>8.1f} {trunc:>5}")
        rows_md.append(f"| {arch} | {comp} | {rs[0]['S']} | {np.median(sh):.0%} | "
                       f"{np.percentile(sh,25):.0%}\u2013{np.percentile(sh,75):.0%} | "
                       f"{np.median(lo):.0%} | {above} of {len(rs)} | {np.nanmedian(dfr):.0f} |")

    (d / "table_5_2_6.md").write_text("\n".join(rows_md), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(per_house, ensure_ascii=False, indent=2,
                                               default=float), encoding="utf-8")
    print(f"\nПодробно: {csv_path}")
    print(f"Готовая таблица (markdown): {(d / 'table_5_2_6.md')}")
    print(f"\nНапоминание: при S = {len(expected_seeds)} интервалы компонентов дисперсии "
          "остаются широкими. В тексте заявлять агрегатные результаты и приводить "
          "интервалы, не делать уверенных выводов по отдельным домохозяйствам.")


if __name__ == "__main__":
    main()
