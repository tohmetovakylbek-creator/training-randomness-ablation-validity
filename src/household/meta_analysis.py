"""
meta_analysis.py
================
Модель со случайным эффектом домохозяйства для эффекта mode embeddings.

Зачем. Прикидочный t-тест по семи домам показал, что среднее неотличимо от нуля,
а междомовое sd в 4-8 раз больше среднего. Для статьи этого мало: нужна оценка
самой компоненты дисперсии, показатель гетерогенности, интервал предсказания для
нового домохозяйства и расчёт мощности. Последний превращает критику в
инструмент: сколько домохозяйств нужно, чтобы архитектурное утверждение вообще
можно было проверить.

Модель DerSimonian-Laird: y_i = mu + u_i + e_i, где u_i ~ N(0, tau^2) —
эффект домохозяйства, e_i ~ N(0, v_i) — ошибка оценки внутри дома.

ВАЖНО о v_i. Раньше SE бралась из JSON (поле se_total), где межсидовая часть
считалась как sd(эффектов по сидам)/sqrt(S). Это НАИВНАЯ оценка: дисперсия
средних по сидам равна sigma_seed^2 + sigma_win^2/W, то есть содержит оконный
шум, и §5.2.1 статьи прямо её отвергает. Использование её здесь противоречило
разложению дисперсии: раздутая v_i занижает tau^2 и I^2.

Теперь по умолчанию v_i собирается из компонент ANOVA, тех же, что в
variance_model.py:
    v_i = V_win(i) + sigma_seed^2(i) / S
Для этого нужны per-seed массивы (--npz). Флаг --legacy_se возвращает прежнее
поведение и печатает обе оценки рядом, чтобы разница была видна.

Что печатается:
    mu            — сводный эффект со случайным эффектом дома
    tau^2, tau    — междомовая дисперсия и её корень (в ваттах)
    I^2, Q, p_Q   — гетерогенность
    PI            — интервал предсказания для НОВОГО домохозяйства; именно он
                    отвечает на вопрос «что будет, если я применю это у себя»
    k_required    — сколько домохозяйств нужно для 80% мощности при заданном
                    размере эффекта

Вход — JSON-файлы, которые пишет factorial_aux_agg.py (по одному на датасет).

Запуск:
    python meta_analysis.py --json results/factorial/factorial_ukdale.json \\
        results/factorial/factorial_refit.json results/factorial/factorial_sheerm.json \\
        --cell aux-per_mode_agg-convex --path y_final --out results/meta
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import variance_model as vm

Z = 1.959963985


def collect(json_paths, cell, path_key, exclude=()):
    """Собирает (метка_дома, эффект, se) по всем датасетам."""
    rows = []
    for p in json_paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for house_tag, blob in data.items():
            if house_tag in exclude:
                print(f"  исключено: {house_tag}")
                continue
            eff = blob.get("embeddings_effect", {})
            if cell not in eff:
                continue
            b = eff[cell]["embeddings_benefit"].get(path_key)
            if b is None:
                continue
            # se_total присутствует, если JSON создан версией с двухкомпонентной
            # неопределённостью; иначе восстанавливаем se из границ интервала
            se = b.get("se_total")
            if se is None:
                se = (b["ci_high"] - b["ci_low"]) / (2 * Z)
            if not np.isfinite(se) or se <= 0:
                print(f"  пропуск {house_tag}: непригодный SE ({se})")
                continue
            rows.append({"house": house_tag, "y": float(b["mean"]), "se": float(se),
                         "se_within": b.get("se_within"), "se_between": b.get("se_between"),
                         "coherent": bool(eff[cell].get("coherent", False))})
    return rows


def variances_from_components(npz_paths, cell, path_key, block, exclude, n_boot=2000, seed=0):
    """v_i = V_win + sigma_seed^2 / S — те же компоненты, что в variance_model.py."""
    paths = vm.resolve_npz(npz_paths) if hasattr(vm, "resolve_npz") else npz_paths
    eff = vm.load_effects(paths, cell, path_key, exclude=exclude)
    out = {}
    for h, d in eff.items():
        r = vm.decompose(d, block, n_boot=n_boot, seed=seed)
        out[h] = {"v": r["var_win_of_mean"] + r["sigma_seed2"] / r["n_seeds"],
                  "V_win": r["var_win_of_mean"],
                  "sigma_seed2": r["sigma_seed2"], "S": r["n_seeds"],
                  "effect": r["effect"]}
    return out


def dersimonian_laird(y, v):
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    w = 1.0 / v
    mu_fe = float((w * y).sum() / w.sum())
    Q = float((w * (y - mu_fe) ** 2).sum())
    df = k - 1
    denom = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / denom) if denom > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float((ws * y).sum() / ws.sum())
    se_mu = float(np.sqrt(1.0 / ws.sum()))
    z = mu / se_mu if se_mu > 0 else np.nan
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    p_Q = float(1 - stats.chi2.cdf(Q, df)) if df > 0 else np.nan
    # интервал предсказания для нового дома (Higgins-Thompson-Spiegelhalter)
    if k > 2:
        t = stats.t.ppf(0.975, k - 2)
        half = t * np.sqrt(se_mu ** 2 + tau2)
        pi = (mu - half, mu + half)
    else:
        pi = (np.nan, np.nan)
    return {"k": k, "mu_fixed": mu_fe, "mu": mu, "se_mu": se_mu, "z": z, "p": p,
            "tau2": tau2, "tau": float(np.sqrt(tau2)), "Q": Q, "df": df,
            "p_Q": p_Q, "I2": I2, "pi_low": pi[0], "pi_high": pi[1]}


def required_k(tau2, v_mean, deltas, power=0.80, alpha=0.05):
    """Сколько домохозяйств нужно, чтобы обнаружить эффект delta.
    SE сводной оценки ~ sqrt((v_mean + tau2) / k)."""
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    out = {}
    for d in deltas:
        if d <= 0:
            continue
        k = ((za + zb) ** 2) * (v_mean + tau2) / (d ** 2)
        out[d] = int(np.ceil(k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--npz", nargs="+", default=None,
                    help="per-seed массивы factorial_per_window_*.npz; без них "
                         "используется наивная SE из JSON (не рекомендуется)")
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--legacy_se", action="store_true",
                    help="считать по прежней (наивной) SE и показать обе оценки")
    ap.add_argument("--cell", default="aux-per_mode_agg-convex")
    ap.add_argument("--path", default="y_final", choices=["y_final", "y_vmd"])
    ap.add_argument("--deltas", default="1,2,5,10",
                    help="размеры эффекта в ваттах для расчёта мощности")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", default="",
                    help="домохозяйства через запятую, напр. ukdale_house_4")
    args = ap.parse_args()
    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())

    rows = collect(args.json, args.cell, args.path, exclude=exclude)

    comp = None
    if args.npz:
        comp = variances_from_components(args.npz, args.cell, args.path,
                                         args.block, exclude)
        kept = []
        for r in rows:
            c = comp.get(r["house"])
            if c is None:
                print(f"  пропуск {r['house']}: нет per-seed массивов")
                continue
            r["se_legacy"] = r["se"]
            r["v_components"] = float(c["v"])
            if not args.legacy_se:
                r["se"] = float(np.sqrt(c["v"]))
            kept.append(r)
        rows = kept
        src = "наивная SE из JSON" if args.legacy_se else "компоненты ANOVA (V_win + sigma_seed^2/S)"
        print(f"\nисточник дисперсии внутри дома: {src}")
        rat = np.array([r["se_legacy"] ** 2 / r["v_components"] for r in rows])
        print(f"наивная v_i завышена относительно компонентной в "
              f"{np.median(rat):.2f} раза по медиане (диапазон {rat.min():.2f}-{rat.max():.2f})")
    else:
        print("\n[!] --npz не задан: используется наивная SE из JSON, которую §5.2.1 "
              "отвергает. Для результатов статьи передайте --npz.")

    if len(rows) < 2:
        print(f"Найдено домохозяйств: {len(rows)} — недостаточно. "
              f"Проверьте --cell (должно совпадать с ключом в JSON).")
        return

    print(f"\n=== {args.cell} | {args.path} ===")
    print(f"{'домохозяйство':22s} {'эффект':>9s} {'SE':>7s}   95% CI")
    for r in sorted(rows, key=lambda r: r["y"]):
        lo, hi = r["y"] - Z * r["se"], r["y"] + Z * r["se"]
        star = "*" if lo * hi > 0 else " "
        print(f"{r['house']:22s} {r['y']:+9.2f} {r['se']:7.2f} {star} "
              f"[{lo:+.2f}, {hi:+.2f}]")

    y = [r["y"] for r in rows]
    v = [r["se"] ** 2 for r in rows]
    res = dersimonian_laird(y, v)
    v_mean = float(np.mean(v))
    n_pos = sum(1 for a in y if a > 0)

    sw = [r["se_within"] for r in rows if r.get("se_within") is not None]
    sb = [r["se_between"] for r in rows if r.get("se_between") is not None]
    if sw:
        share = np.mean([b ** 2 / (w ** 2 + b ** 2) for w, b in zip(sw, sb)])
        print(f"\nвнутридомовая SE: по окнам медиана {np.median(sw):.2f}, "
              f"по сидам медиана {np.median(sb):.2f}, "
              f"доля сидовой дисперсии в среднем {share:.0%}")
    print(f"\nдомохозяйств: {res['k']}   знаков  −{res['k']-n_pos} / +{n_pos}")
    print(f"сводный эффект (случайные эффекты): {res['mu']:+.2f} Вт "
          f"(SE {res['se_mu']:.2f}), p = {res['p']:.3f}")
    print(f"фиксированные эффекты (для сравнения): {res['mu_fixed']:+.2f} Вт")
    print(f"междомовая дисперсия: tau^2 = {res['tau2']:.2f}, tau = {res['tau']:.2f} Вт")
    print(f"гетерогенность: I^2 = {res['I2']:.1f}%, "
          f"Q({res['df']}) = {res['Q']:.1f}, p = {res['p_Q']:.4g}")
    print(f"интервал предсказания для нового домохозяйства: "
          f"[{res['pi_low']:+.2f}, {res['pi_high']:+.2f}] Вт")
    ratio = res["tau"] / abs(res["mu"]) if res["mu"] != 0 else np.inf
    print(f"tau / |сводный эффект| = {ratio:.1f}"
          f"  (>1 значит межддомовой разброс превышает сам эффект)"
          .replace("межддомовой", "междомовой"))

    deltas = [float(x) for x in args.deltas.split(",")]
    need = required_k(res["tau2"], v_mean, deltas)
    print(f"\nмощность 80% при alpha=0.05, средняя внутридомовая дисперсия "
          f"{v_mean:.2f}:")
    for d, k in need.items():
        print(f"   для эффекта {d:>5.1f} Вт нужно домохозяйств: {k}")

    if args.out:
        outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
        tag = "legacy" if (args.legacy_se or not args.npz) else "components"
        f = outp / f"meta_{args.cell}_{args.path}_{tag}.json"
        f.write_text(json.dumps({"cell": args.cell, "path": args.path,
                                 "houses": rows, "model": res,
                                 "v_mean": v_mean, "required_k": need,
                                 "variance_source": ("legacy_naive" if
                                                     (args.legacy_se or not args.npz)
                                                     else "anova_components")},
                                ensure_ascii=False, indent=2))
        print(f"\nсохранено: {f}")


if __name__ == "__main__":
    main()
