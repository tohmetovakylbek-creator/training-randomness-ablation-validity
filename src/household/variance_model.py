"""
variance_model.py
=================
Иерархическая модель разложения дисперсии эффекта абляции — на подсидовых
поокновых ошибках, сохранённых factorial_aux_agg.py. Обучения не требует.

ЗАЧЕМ. Нынешнее разложение — арифметика: SE^2 = se_within^2 + se_between^2.
Для методологической статьи этого мало: разложение и есть вклад, поэтому доля
сидовой дисперсии должна быть ОЦЕНКОЙ со своим доверительным интервалом, а не
одним числом. При пяти сидах эта оценка нестабильна, и величину нестабильности
надо показать, а не упомянуть.

МОДЕЛЬ. Для домохозяйства h эффект абляции по окну w и сиду s:

    d_{hsw} = mu_h + a_{hs} + e_{hsw},
        a_{hs} ~ N(0, sigma_seed^2)      случайный эффект сида
        e_{hsw} ~ N(0, sigma_win^2)      остаточная по окнам

Оценка компонент — по разложению сумм квадратов однофакторного ANOVA со
случайными эффектами (несбалансированности нет: у всех сидов одни и те же окна):

    sigma_win^2  = MS_within
    sigma_seed^2 = (MS_between - MS_within) / W        (усечение снизу нулём)

Величина, которая идёт в статью, — доля дисперсии СРЕДНЕГО эффекта, приходящаяся
на сиды:

    Var(mean) = sigma_seed^2 / S + sigma_win_eff^2 / S
    share_seed = (sigma_seed^2 / S) / Var(mean)

где sigma_win_eff^2 учитывает автокорреляцию соседних окон через moving block
bootstrap — иначе доля сидовой дисперсии окажется завышенной просто потому, что
окна не независимы.

ДВОЙНОЙ УЧЁТ. Компоненты оцениваются из ОДНОГО разложения сумм квадратов:
MS_between описывает разброс подсидовых средних, MS_within — разброс внутри сида
вокруг его собственного среднего. Они ортогональны по построению, пересечения
нет. Скрипт проверяет это численно: сумма компонент должна воспроизводить полную
дисперсию d_{hsw} (печатается невязка).

ЧУВСТВИТЕЛЬНОСТЬ:
  * доля по всем подмножествам сидов размера 2..S-1 — показывает, насколько
    оценка держится при меньшем числе прогонов;
  * bootstrap-интервал доли (ресемплинг сидов и блоков окон);
  * обе метрики (y_final и y_vmd) и обе ячейки.

Запуск:
    python variance_model.py --npz results/factorial_2comp/factorial_ukdale.npz \\
        results/factorial_2comp/factorial_refit.npz \\
        results/factorial_2comp/factorial_sheerm.npz \\
        --cell aux-per_mode_agg-convex --path y_final --block 3 \\
        --out results/variance
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

Z = 1.959963985


# ------------------------------------------------------------------ данные
def resolve_npz(paths):
    """Принимает файлы, папки или имена без префикса per_window_ и возвращает
    список существующих npz. Скрипт factorial_aux_agg.py пишет ДВА файла на
    датасет: factorial_<name>.json (отчёт) и factorial_per_window_<name>.npz
    (поокновые массивы). Нужен второй; частая ошибка — указать имя по образцу
    первого."""
    out = []
    for p in paths:
        q = Path(p)
        if q.is_dir():
            found = sorted(q.glob("factorial_per_window_*.npz"))
            if not found:
                print(f"  в {q} нет файлов factorial_per_window_*.npz")
            out.extend(found)
            continue
        if q.exists():
            out.append(q)
            continue
        # попытка вставить недостающий префикс
        alt = q.with_name(q.name.replace("factorial_", "factorial_per_window_", 1))
        if alt.exists():
            print(f"  {q.name} не найден, использую {alt.name}")
            out.append(alt)
            continue
        print(f"  НЕ НАЙДЕН: {q}")
        if q.parent.exists():
            have = sorted(x.name for x in q.parent.glob("*.npz"))
            print(f"    в {q.parent} есть: {', '.join(have) if have else '(нет .npz)'}")
    return out


def load_effects(npz_paths, cell, path_key, exclude=()):
    """Возвращает {дом: (S, W)} — поокновый эффект абляции по каждому сиду.

    Эффект = ошибка БЕЗ embeddings минус ошибка С embeddings, поэтому
    положительное значение означает, что embeddings помогают.
    """
    out = {}
    suffix = "err_final_per_seed" if path_key == "y_final" else "err_vmd_per_seed"
    for p in npz_paths:
        d = np.load(p)
        keys = list(d.keys())
        for k in keys:
            if not k.endswith(f"|{cell}_emb-on|{suffix}"):
                continue
            house = k.split("|")[0]
            if house in exclude:
                continue
            off = k.replace("_emb-on|", "_emb-off|")
            if off not in d:
                print(f"  пропуск {house}: нет пары emb-off")
                continue
            on_a, off_a = d[k], d[off]
            if on_a.ndim != 2:
                print(f"  пропуск {house}: нет подсидовых массивов "
                      f"(перезапустите factorial_aux_agg.py новой версией)")
                continue
            out[house] = off_a - on_a
    return out


# ------------------------------------------------- эффективная дисперсия окон
def block_var_of_mean(x, block, n_boot=2000, seed=0):
    """Дисперсия среднего ряда x с учётом автокорреляции — moving block bootstrap.
    Возвращает Var(mean), сопоставимую с sigma_win^2 / W при block = 1."""
    rng = np.random.default_rng(seed)
    n = len(x)
    block = int(min(max(1, block), n))
    nb = int(np.ceil(n / block))
    starts = n - block + 1
    m = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, starts, size=nb)
        m[b] = np.concatenate([x[s:s + block] for s in idx])[:n].mean()
    return float(m.var(ddof=1))


# ------------------------------------------------------------- разложение
def decompose(d, block, n_boot=2000, seed=0):
    """d: (S, W) поокновый эффект по сидам. Компоненты по ANOVA со случайными
    эффектами + доля сидовой дисперсии в дисперсии среднего эффекта."""
    S, W = d.shape
    grand = d.mean()
    seed_means = d.mean(axis=1)

    ss_between = W * ((seed_means - grand) ** 2).sum()
    ss_within = ((d - seed_means[:, None]) ** 2).sum()
    ms_between = ss_between / (S - 1) if S > 1 else np.nan
    ms_within = ss_within / (S * (W - 1))

    sigma_win2 = float(ms_within)
    sigma_seed2 = float(max(0.0, (ms_between - ms_within) / W)) if S > 1 else np.nan

    # дисперсия среднего по окнам, с поправкой на автокорреляцию
    var_win_mean = block_var_of_mean(d.mean(axis=0), block, n_boot, seed)
    var_seed_mean = sigma_seed2 / S if S > 1 else np.nan
    total = var_win_mean + var_seed_mean
    share = var_seed_mean / total if total > 0 else np.nan

    # ПРОВЕРКА НА ДВОЙНОЙ УЧЁТ. Корректная проверка ортогональности — тождество
    # разложения сумм квадратов: SS_total = SS_between + SS_within. Оно выполняется
    # точно, если компоненты не пересекаются. Сравнивать sigma_seed2 + sigma_win2
    # с наблюдаемой дисперсией НЕЛЬЗЯ: это разные величины (несмещённые оценки
    # компонент против выборочной дисперсии), их расхождение ожидаемо и ничего
    # не говорит о двойном учёте.
    ss_total = float(((d - grand) ** 2).sum())
    resid = (ss_between + ss_within - ss_total) / ss_total if ss_total > 0 else np.nan

    return {
        "n_seeds": S, "n_windows": W, "effect": float(grand),
        "sigma_seed2": sigma_seed2, "sigma_win2": sigma_win2,
        "icc_seed": float(sigma_seed2 / (sigma_seed2 + sigma_win2)) if S > 1 else np.nan,
        "var_seed_of_mean": var_seed_mean, "var_win_of_mean": var_win_mean,
        "se_total": float(np.sqrt(total)),
        "share_seed": float(share),
        "seed_effects": [float(x) for x in seed_means],
        "decomposition_residual": float(resid),
        "block": block,
    }


def exact_share_ci(d, block, var_win_mean, alpha=0.05, n_boot=2000, seed=0):
    """Точный доверительный интервал доли сидовой дисперсии через F-распределение.

    Для сбалансированной однофакторной модели со случайными эффектами
        MS_between / (sigma_win^2 + W*sigma_seed^2) ~ chi2_{S-1}/(S-1)
        MS_within  /  sigma_win^2                    ~ chi2_{S(W-1)}/(S(W-1))
    и они независимы, откуда для theta = sigma_seed^2 / sigma_win^2

        (MS_b/MS_w) / (1 + W*theta) ~ F(S-1, S(W-1))

    и границы theta выражаются в замкнутой форме через квантили F. Метод корректен
    при S = 5, в отличие от непараметрического бутстрапа: тот ресемплирует пять
    сидов с возвратом, повторы занижают MS_between, оценка компоненты усекается
    нулём в большой доле реплик, и нижняя граница садится на ноль почти везде.

    Границы theta переводятся в долю через
        share(theta) = theta*W / (theta*W + g*S),
    где g = var_win_mean * W / sigma_win^2 — коэффициент инфляции оконной
    дисперсии за счёт автокорреляции (оценивается block bootstrap отдельно).

    ДОПУЩЕНИЯ: нормальность сидовых эффектов и одинаковая оконная дисперсия по
    сидам. Их надо назвать в тексте статьи; взамен получается интервал, который
    при малом числе прогонов вообще определён.
    """
    S, W = d.shape
    if S < 2 or W < 2:
        return {"lo": float("nan"), "hi": float("nan"), "method": "exact_F"}
    grand = d.mean()
    seed_means = d.mean(axis=1)
    ms_b = float(W * ((seed_means - grand) ** 2).sum() / (S - 1))
    ms_w = float(((d - seed_means[:, None]) ** 2).sum() / (S * (W - 1)))
    if ms_w <= 0:
        return {"lo": float("nan"), "hi": float("nan"), "method": "exact_F"}

    d1, d2 = S - 1, S * (W - 1)
    f_obs = ms_b / ms_w
    lo_th = max(0.0, (f_obs / stats.f.ppf(1 - alpha / 2, d1, d2) - 1.0) / W)
    hi_th = max(0.0, (f_obs / stats.f.ppf(alpha / 2, d1, d2) - 1.0) / W)

    g = var_win_mean * W / ms_w                     # инфляция за автокорреляцию
    sh = lambda t: (t * W) / (t * W + g * S) if (t * W + g * S) > 0 else 0.0
    return {"lo": float(sh(lo_th)), "hi": float(sh(hi_th)),
            "theta_lo": float(lo_th), "theta_hi": float(hi_th),
            "F_obs": f_obs, "df1": d1, "df2": d2, "g_inflation": float(g),
            "method": "exact_F"}


def share_ci(d, block, n_boot=1000, seed=0, inner_boot=200):
    """Bootstrap-интервал доли сидовой дисперсии: ресемплинг СИДОВ (с возвратом)
    и блоков окон. При S=5 интервал будет широким — это и надо показать."""
    rng = np.random.default_rng(seed)
    S, W = d.shape
    vals = []
    for b in range(n_boot):
        si = rng.integers(0, S, size=S)
        r = decompose(d[si], block, n_boot=inner_boot, seed=int(rng.integers(1e6)))
        if np.isfinite(r["share_seed"]):
            vals.append(r["share_seed"])
    if len(vals) < 20:
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.quantile(vals, [0.025, 0.975]))


def subset_stability(d, block, min_k=2):
    """Доля сидовой дисперсии по всем подмножествам сидов размера k.
    Показывает, устоит ли оценка при меньшем числе прогонов."""
    S = d.shape[0]
    out = {}
    for k in range(min_k, S):
        vals = [decompose(d[list(c)], block, n_boot=300)["share_seed"]
                for c in itertools.combinations(range(S), k)]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[k] = {"n_subsets": len(vals), "median": float(np.median(vals)),
                      "min": float(np.min(vals)), "max": float(np.max(vals))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--cell", default="aux-per_mode_agg-convex")
    ap.add_argument("--path", default="y_final", choices=["y_final", "y_vmd"])
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--no_ci", action="store_true", help="пропустить bootstrap доли")
    ap.add_argument("--inner_boot", type=int, default=200,
                    help="итераций block bootstrap внутри каждой реплики (меньше = быстрее)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", default="",
                    help="домохозяйства через запятую, напр. ukdale_house_4")
    args = ap.parse_args()
    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())

    paths = resolve_npz(args.npz)
    eff = load_effects(paths, args.cell, args.path, exclude=exclude) if paths else {}
    if not eff:
        print("Ничего не найдено. Проверьте --cell и что npz содержит "
              "массивы *_per_seed.")
        return

    print(f"\n=== {args.cell} | {args.path} | блок {args.block} ===")
    print(f"{'домохозяйство':22s} {'S':>2s} {'эффект':>8s} {'sd сидов':>9s} "
          f"{'доля':>6s} {'точный CI (F)':>15s} {'бутстрап':>13s} {'ICC':>6s}")
    res, shares = {}, []
    for h in sorted(eff):
        r = decompose(eff[h], args.block, seed=0)
        r["exact_ci"] = exact_share_ci(eff[h], args.block, r["var_win_of_mean"])
        if not args.no_ci:
            r["share_ci"] = share_ci(eff[h], args.block, n_boot=args.n_boot,
                                     inner_boot=args.inner_boot)
            # доля усечённых реплик: если оценка компоненты часто упирается в ноль,
            # точечное значение занижено, и это надо показать
            r["share_ci_note"] = "усечение" if r["share_seed"] == 0 else ""
        r["subset_stability"] = subset_stability(eff[h], args.block)
        res[h] = r
        shares.append(r["share_seed"])
        ci = r.get("share_ci", (np.nan, np.nan))
        e = r["exact_ci"]
        bs = f"[{ci[0]:.0%},{ci[1]:.0%}]" if np.isfinite(ci[0]) else "—"
        ex = (f"[{e['lo']:.0%},{e['hi']:.0%}]" if np.isfinite(e.get("lo", np.nan))
              else "—")
        print(f"{h:22s} {r['n_seeds']:2d} {r['effect']:+8.2f} "
              f"{np.sqrt(r['sigma_seed2']):9.2f} "
              f"{r['share_seed']:5.0%} {ex:>15s} {bs:>13s} "
              f"{r['icc_seed']:6.3f}")

    shares = np.array([s for s in shares if np.isfinite(s)])
    print(f"\nдоля сидовой дисперсии по {len(shares)} домохозяйствам: "
          f"среднее {shares.mean():.0%}, медиана {np.median(shares):.0%}, "
          f"диапазон {shares.min():.0%}-{shares.max():.0%}, "
          f"выше 50%: {(shares > 0.5).sum()}, выше 80%: {(shares > 0.8).sum()}")

    lows = [r["exact_ci"]["lo"] for r in res.values()
            if np.isfinite(r["exact_ci"].get("lo", np.nan))]
    if lows:
        lows = np.array(lows)
        print(f"точный CI: медиана нижней границы {np.median(lows):.0%}, "
              f"домов с нижней границей выше 50%: {(lows > 0.5).sum()} из {len(lows)}")
        if not args.no_ci:
            bl = np.array([r["share_ci"][0] for r in res.values()
                           if np.isfinite(r.get("share_ci", (np.nan,))[0])])
            if len(bl):
                print(f"для сравнения, бутстрап: нижняя граница равна нулю "
                      f"у {(bl <= 1e-9).sum()} из {len(bl)} домов "
                      f"(ресемплинг {res[list(res)[0]]['n_seeds']} сидов с возвратом "
                      f"занижает MS_between)")

    mr = max(abs(r["decomposition_residual"]) for r in res.values())
    print(f"проверка на двойной учёт (SS_total = SS_seed + SS_win): "
          f"максимальная относительная невязка {mr:.2e} "
          f"{'OK, компоненты ортогональны' if mr < 1e-6 else '— РАЗОБРАТЬСЯ'}")

    ks = sorted({k for r in res.values() for k in r["subset_stability"]})
    if ks:
        print("\nустойчивость к числу сидов (медиана доли по подмножествам):")
        for k in ks:
            v = [r["subset_stability"][k]["median"] for r in res.values()
                 if k in r["subset_stability"]]
            print(f"   {k} сида: медиана по домам {np.median(v):.0%}")
        print(f"   {max(ks)+1} сидов (все): медиана {np.median(shares):.0%}")

    if args.out:
        o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
        f = o / f"variance_{args.cell}_{args.path}.json"
        f.write_text(json.dumps({"cell": args.cell, "path": args.path,
                                 "block": args.block, "houses": res},
                                ensure_ascii=False, indent=2, default=float))
        print(f"\nсохранено: {f}")


if __name__ == "__main__":
    main()
