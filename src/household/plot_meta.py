"""
plot_meta.py
============
Перерисовка Фиг. 5 (форест-плот по домохозяйствам) и Фиг. 6 (требуемое число
домохозяйств) по ИСПРАВЛЕННЫМ компонентам дисперсии.

Важно: внутридомовая дисперсия берётся из компонент ANOVA, как в
variance_model.py и в исправленном meta_analysis.py:

    только окна : v_i = V_win(i)
    полная      : v_i = V_win(i) + sigma_seed^2(i) / S

Прежние версии рисунков строились по наивной оценке se_between = sd(эффектов
по сидам)/sqrt(S), которую §5.2.1 отвергает.

Фиг. 6 дополнительно показывает чувствительность расчёта мощности: средняя
внутридомовая дисперсия почти целиком определяется одним домохозяйством, и
пунктирные кривые показывают, что будет без него.

Обычный запуск (используются файлы из ``results/factorial_2comp``):
    python plot_meta.py

Запуск с явным указанием файлов:
    python plot_meta.py ^
        --npz results/factorial_2comp/factorial_per_window_ukdale.npz ^
              results/factorial_2comp/factorial_per_window_refit.npz ^
              results/factorial_2comp/factorial_per_window_sheerm.npz ^
        --json results/factorial_2comp/factorial_ukdale.json ^
               results/factorial_2comp/factorial_refit.json ^
               results/factorial_2comp/factorial_sheerm.json ^
        --cell aux-per_mode_agg-convex --path y_final ^
        --exclude ukdale_house_4 --dominant refit_house_10 ^
        --out results/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Springer line-art export: figures are inserted at approximately 6.1--6.5 in.
# Keeping the native width at 6.5 in and exporting at 600 dpi yields an image
# close to 3900 pixels wide without shrinking the physical font size.
plt.rcParams["savefig.dpi"] = 600

FIGURE_WIDTH_IN = 6.5
FIG5_HEIGHT_IN = 6.0
FIG6_HEIGHT_IN = 4.35
EXPORT_DPI = 600

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "factorial_2comp"


def default_input(name: str) -> str:
    """Возвращает стандартный файл проекта; локальный файл рядом со скриптом — запасной вариант."""
    project_path = RESULTS_DIR / name
    local_path = SCRIPT_DIR / name
    return str(project_path if project_path.exists() or not local_path.exists() else local_path)


DEFAULT_NPZ = [
    default_input("factorial_per_window_ukdale.npz"),
    default_input("factorial_per_window_refit.npz"),
    default_input("factorial_per_window_sheerm.npz"),
]
DEFAULT_JSON = [
    default_input("factorial_ukdale.json"),
    default_input("factorial_refit.json"),
    default_input("factorial_sheerm.json"),
]

import variance_model as vm

Z = 1.959963985
DARK, THIN, BAND = "0.15", "0.45", "#f2ded7"


def dl(y, v):
    """DerSimonian–Laird + интервал предсказания (Higgins–Thompson–Spiegelhalter)."""
    v = np.maximum(np.asarray(v, float), 1e-12)     # защита от нулевой дисперсии
    w = 1 / v
    mu_fe = (w * y).sum() / w.sum()
    Q = (w * (y - mu_fe) ** 2).sum()
    df = len(y) - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    t2 = max(0.0, (Q - df) / C)
    ws = 1 / (v + t2)
    mu = (ws * y).sum() / ws.sum()
    se = np.sqrt(1 / ws.sum())
    half = stats.t.ppf(0.975, max(len(y) - 2, 1)) * np.sqrt(se ** 2 + t2)
    return {"mu": mu, "se": se, "tau2": t2, "Q": Q, "df": df,
            "I2": max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0,
            "pi": (mu - half, mu + half)}


def required_k(v_mean, tau2, deltas, power=0.8, alpha=0.05):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return ((za + zb) ** 2) * (v_mean + tau2) / np.asarray(deltas) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        nargs="+",
        default=DEFAULT_NPZ,
        help="per-window NPZ; по умолчанию берутся UK-DALE, REFIT и SHEERM из results/factorial_2comp",
    )
    ap.add_argument(
        "--json",
        nargs="+",
        default=DEFAULT_JSON,
        help="factorial JSON; по умолчанию берутся UK-DALE, REFIT и SHEERM из results/factorial_2comp",
    )
    ap.add_argument("--cell", default="aux-per_mode_agg-convex")
    ap.add_argument("--path", default="y_final", choices=["y_final", "y_vmd"])
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--exclude", default="ukdale_house_4")
    ap.add_argument("--dominant", default="refit_house_10",
                    help="домохозяйство, доминирующее в средней дисперсии; "
                         "показывается пунктиром на Фиг. 6")
    ap.add_argument("--threshold_pct", type=float, default=2.0)
    ap.add_argument("--xlim", default=None,
                    help="пределы оси Фиг. 5, напр. \"-20,20\"; интервалы за краем "
                         "обрезаются и помечаются стрелкой")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "results" / "figures"))
    args = ap.parse_args()

    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    paths = vm.resolve_npz(args.npz) if hasattr(vm, "resolve_npz") else args.npz
    eff = vm.load_effects(paths, args.cell, args.path, exclude=exclude)

    mae = {}
    for f in args.json:
        for h, rec in json.loads(Path(f).read_text(encoding="utf-8")).items():
            c = rec.get("cells", {}).get(f"{args.cell}_emb-on")
            if c:
                mae[h] = c["paths"]["y" if args.path == "y_final" else "y_vmd"]["MAE"]

    rows = []
    for h in sorted(eff):
        r = vm.decompose(eff[h], args.block, seed=0)
        rows.append({"house": h, "y": r["effect"],
                     "v_win": r["var_win_of_mean"],
                     "v_full": r["var_win_of_mean"] + r["sigma_seed2"] / r["n_seeds"],
                     "mae": mae.get(h, np.nan)})
    rows.sort(key=lambda r: r["y"])
    y = np.array([r["y"] for r in rows])
    v_win = np.array([r["v_win"] for r in rows])
    v_full = np.array([r["v_full"] for r in rows])
    names = [r["house"].replace("_house_", " ").upper() for r in rows]
    band = np.array([args.threshold_pct / 100 * r["mae"] for r in rows])

    m_full, m_win = dl(y, v_full), dl(y, v_win)
    print(f"полная пропагация:  mu = {m_full['mu']:+.3f}, tau = {np.sqrt(m_full['tau2']):.3f}, "
          f"I2 = {m_full['I2']:.1f}%, PI = [{m_full['pi'][0]:+.2f}, {m_full['pi'][1]:+.2f}]")
    print(f"только окна:        mu = {m_win['mu']:+.3f}, tau = {np.sqrt(m_win['tau2']):.3f}, "
          f"I2 = {m_win['I2']:.1f}%")
    sig_win = int(sum(abs(a) > Z * np.sqrt(b) for a, b in zip(y, v_win)))
    sig_full = int(sum(abs(a) > Z * np.sqrt(b) for a, b in zip(y, v_full)))
    print(f"значимы: по окнам {sig_win} из {len(y)}, при полной пропагации {sig_full}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ Фиг. 5
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIG5_HEIGHT_IN))
    pos = np.arange(len(rows))[::-1]
    if args.xlim:
        x0, x1 = (float(t) for t in args.xlim.split(","))
    else:
        span = max(abs(y).max(), np.percentile(y + Z * np.sqrt(v_full), 90)) * 1.6
        x0, x1 = -span, span

    def seg(lo, hi, p, color, lw):
        """Рисует интервал с клиппингом; за краем ставит стрелку."""
        a, b = max(lo, x0), min(hi, x1)
        if a < b:
            ax.plot([a, b], [p, p], color=color, lw=lw, solid_capstyle="butt", zorder=3)
        if lo < x0:
            ax.plot(x0, p, marker="<", ms=4, color=color, zorder=4, clip_on=False)
        if hi > x1:
            ax.plot(x1, p, marker=">", ms=4, color=color, zorder=4, clip_on=False)

    for i, p in enumerate(pos):
        ax.barh(p, 2 * band[i], left=-band[i], height=0.75, color=BAND, zorder=0)
        seg(y[i] - Z * np.sqrt(v_full[i]), y[i] + Z * np.sqrt(v_full[i]), p, THIN, 0.9)
        seg(y[i] - Z * np.sqrt(v_win[i]), y[i] + Z * np.sqrt(v_win[i]), p, DARK, 2.6)
        ax.plot(np.clip(y[i], x0, x1), p, "o", ms=3.5, color=DARK, zorder=4)
    ax.set_xlim(x0, x1)
    ax.set_yticks(pos); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="0.2", lw=0.8)

    top = len(rows) + 0.8
    mu, pi = m_full["mu"], m_full["pi"]
    h = 0.34
    ax.fill([mu - Z * m_full["se"], mu, mu + Z * m_full["se"], mu],
            [top, top + h, top, top - h], color="white", ec=DARK, lw=1.1, zorder=5)
    ax.plot([pi[0], pi[1]], [top - 0.75, top - 0.75], color="#c0392b", lw=2.2, zorder=5)
    pad = (x1 - x0) * 0.02
    ax.annotate(f"pooled effect {mu:+.2f} W", (mu, top), textcoords="offset points",
                xytext=(26, 8), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
    ax.annotate(f"prediction interval [{pi[0]:+.2f}, {pi[1]:+.2f}] W",
                (pi[1], top - 0.75), textcoords="offset points", xytext=(10, -8),
                fontsize=7.5, color="#c0392b")
    ax.set_ylim(-1, top + 1.6)
    ax.set_xlabel("effect of removing the identity mechanism, W\n(positive = the mechanism helps)",
                  fontsize=9)
    handles = [plt.Line2D([], [], color=DARK, lw=2.6, label="95% CI, evaluation windows only"),
               plt.Line2D([], [], color=THIN, lw=0.9, label="95% CI, windows + training seeds"),
               plt.Rectangle((0, 0), 1, 1, color=BAND,
                             label=f"practical-significance band (\u00b1{args.threshold_pct:g}% of household MAE)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=1, frameon=False, fontsize=7.5)
    fig.tight_layout()
    f5 = out / "fig5_forest.png"
    fig.savefig(f5, dpi=EXPORT_DPI, bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------ Фиг. 6
    deltas = np.linspace(0.5, 8.0, 300)
    k_win = required_k(v_win.mean(), m_win["tau2"], deltas)
    k_full = required_k(v_full.mean(), m_full["tau2"], deltas)

    keep = [i for i, r in enumerate(rows) if r["house"] != args.dominant]
    m_full_d = dl(y[keep], v_full[keep]); m_win_d = dl(y[keep], v_win[keep])
    k_win_d = required_k(v_win[keep].mean(), m_win_d["tau2"], deltas)
    k_full_d = required_k(v_full[keep].mean(), m_full_d["tau2"], deltas)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIG6_HEIGHT_IN))
    ax.plot(deltas, k_win, color=DARK, lw=1.8, label="uncertainty from evaluation windows only")
    ax.plot(deltas, k_full, color="#c0392b", lw=1.8, label="uncertainty from windows + training seeds")
    excl = args.dominant.replace("_house_", " ")
    ax.plot(deltas, k_win_d, color=DARK, lw=1.0, ls="--",
            label=f"windows only, excluding {excl}")
    ax.plot(deltas, k_full_d, color="#c0392b", lw=1.0, ls="--",
            label=f"windows + seeds, excluding {excl}")
    for curve, col in ((k_win, DARK), (k_full, "#c0392b")):
        val = float(np.interp(2.0, deltas, curve))
        ax.plot(2.0, val, "o", color=col, ms=5)
        ax.annotate(f"{int(np.ceil(val))}", (2.0, val), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color=col)
    ax.plot(2.0, 5, "*", color="0.1", ms=13)
    ax.annotate("the original study:\n5 households", (2.0, 5),
                textcoords="offset points", xytext=(16, -14), fontsize=8)
    ax.axvline(2.0, color="0.6", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.axhline(1, color="0.8", lw=0.6)
    ax.set_xlabel("detectable mean effect, W", fontsize=9)
    ax.set_ylabel("households required (80% power, \u03b1 = 0.05)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    f6 = out / "fig6_power.png"
    fig.savefig(f6, dpi=EXPORT_DPI, bbox_inches="tight"); plt.close(fig)

    summary = {"cell": args.cell, "path": args.path,
               "full": {k: (list(v) if isinstance(v, tuple) else float(v))
                        for k, v in m_full.items()},
               "windows_only": {k: (list(v) if isinstance(v, tuple) else float(v))
                                for k, v in m_win.items()},
               "significant_windows_only": sig_win, "significant_full": sig_full,
               "v_mean_windows": float(v_win.mean()), "v_mean_full": float(v_full.mean()),
               "v_median_windows": float(np.median(v_win)), "v_median_full": float(np.median(v_full)),
               "required_at_2W": {"windows_only": int(np.ceil(np.interp(2.0, deltas, k_win))),
                                  "full": int(np.ceil(np.interp(2.0, deltas, k_full))),
                                  "windows_only_excl_dominant": int(np.ceil(np.interp(2.0, deltas, k_win_d))),
                                  "full_excl_dominant": int(np.ceil(np.interp(2.0, deltas, k_full_d)))},
               "required_at_1W": {"windows_only": int(np.ceil(np.interp(1.0, deltas, k_win))),
                                  "full": int(np.ceil(np.interp(1.0, deltas, k_full)))}}
    (out / "figure_data.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n{f5}\n{f6}\n{out / 'figure_data.json'}")
    print("\nчисла для подписей:", json.dumps(summary["required_at_2W"], ensure_ascii=False))


if __name__ == "__main__":
    main()
