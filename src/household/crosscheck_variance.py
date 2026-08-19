"""
crosscheck_variance.py
======================
Сверка generality_variance.py с variance_model.py на ОДНОМ домохозяйстве.
Цель — убедиться, что обе реализации считают одно и то же, прежде чем Таблица 4
пойдёт в статью. Расхождение обычно вызвано одним из четырёх соглашений:
ddof в суммах квадратов, длина блока бутстрэпа, число бутстрэп-повторов, и то,
берётся ли V_win из блочного бутстрэпа или как MS_within/W.

Кладётся в корень проекта, рядом с variance_model.py.

Три способа задать данные (нужен ровно один):

1) Матрица эффектов d размера (S, W) — если она у вас уже сохранена:
       python crosscheck_variance.py --d d_matrix.npy

2) Две матрицы по-оконных |ошибок| (S, W): без компонента и с ним:
       python crosscheck_variance.py --off off.npy --on on.npy

3) Домохозяйство из эксперимента §5.2.6 (проверка самосогласованности,
   не заменяет сверку с variance_model.py):
       python crosscheck_variance.py --from-generality results_generalit --arch patchtst --house sheerm_2

Флаг --inspect печатает публичные функции variance_model.py с сигнатурами —
по ним видно, какую из них звать на той же матрице d.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

from generality_variance import variance_components, BLOCK, N_BOOT


def load_generality(dirpath: str, arch: str, house: str) -> np.ndarray:
    d = Path(dirpath)
    on, off = {}, {}
    for f in d.glob(f"err_{arch}_*_{house}_seed*.npy"):
        cfg = f.stem.split("_")[2]
        seed = int(f.stem.split("seed")[-1])
        (on if cfg == "on" else off)[seed] = np.load(f)
    seeds = sorted(set(on) & set(off))
    if not seeds:
        sys.exit(f"не найдены пары on/off для {arch} {house} в {dirpath}")
    print(f"сиды: {seeds}, окон: {len(on[seeds[0]])}")
    return np.stack([off[s] - on[s] for s in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d")
    ap.add_argument("--on")
    ap.add_argument("--off")
    ap.add_argument("--from-generality")
    ap.add_argument("--arch", default="patchtst")
    ap.add_argument("--house")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    if args.inspect:
        try:
            import variance_model as vm
        except Exception as e:                                  # noqa: BLE001
            sys.exit(f"не удалось импортировать variance_model.py: {e}")
        print("Публичные функции variance_model.py:")
        for name, fn in inspect.getmembers(vm, inspect.isfunction):
            if not name.startswith("_"):
                print(f"  {name}{inspect.signature(fn)}")
        print("\nВызовите ту, что принимает матрицу (S, W) или два массива ошибок, "
              "на тех же данных и сравните share_seed с числом ниже.")
        return

    if args.d:
        d = np.load(args.d)
    elif args.on and args.off:
        d = np.load(args.off) - np.load(args.on)
    elif args.from_generality and args.house:
        d = load_generality(args.from_generality, args.arch, args.house)
    else:
        sys.exit("задайте --d, либо --on и --off, либо --from-generality с --house")

    d = np.asarray(d, dtype=float)
    if d.ndim != 2:
        sys.exit(f"ожидается матрица (S, W), получено {d.shape}")

    r = variance_components(d)

    S, W = d.shape
    seed_means = d.mean(axis=1)
    ms_between = W * float(((seed_means - d.mean()) ** 2).sum()) / (S - 1)
    ms_within = float(((d - seed_means[:, None]) ** 2).sum()) / (S * (W - 1))

    print("\n=== generality_variance.py ===")
    print(f"  S = {S}, W = {W}")
    print(f"  MS_between        = {ms_between:.6f}")
    print(f"  MS_within         = {ms_within:.6f}")
    print(f"  sigma_win^2       = {r['sigma_win2']:.6f}")
    print(f"  sigma_seed^2      = {r['sigma_seed2']:.6f}"
          f"{'   [обрезано в ноль]' if r['truncated_at_zero'] else ''}")
    print(f"  V_win (block bs)  = {r['V_win']:.6f}   (block={BLOCK}, B={N_BOOT})")
    print(f"  share_seed        = {r['share_seed']:.4f}")
    print(f"  95% CI для share  = [{r['share_lo']:.4f}, {r['share_hi']:.4f}]")
    print(f"  theta             = {r['theta']:.4f}")
    print(f"  design-free ratio = {r['design_free_ratio']:.4f}")
    print(f"  mean effect       = {r['mean_effect_w']:.4f} W")

    v_alt = ms_within / (S * W)   # дисперсия среднего по окнам при независимых окнах
    share_alt = (r["sigma_seed2"] / S) / (r["sigma_seed2"] / S + v_alt) if v_alt > 0 else float("nan")
    print("\n=== контрольные варианты (если variance_model.py даст другое число) ===")
    print(f"  V_win как MS_within/(S*W) = {v_alt:.6f}  ->  share = {share_alt:.4f}")
    print(f"  отношение V_win(bs)/V_win(iid) = {r['V_win'] / v_alt:.2f}"
          "   (>1 означает, что окна положительно зависимы)")

    print("\n=== что сверять с variance_model.py ===")
    print("  1. MS_between и MS_within — если расходятся, дело в ddof или в порядке осей (S, W).")
    print("  2. sigma_seed^2 — проверьте, обрезается ли отрицательная оценка в ноль там же.")
    print("  3. V_win — длина блока (у нас 3) и число повторов (у нас 2000);")
    print("     если variance_model.py использует iid-бутстрэп, сравнивайте со строкой выше.")
    print("  4. share_seed — итог. Совпадение до 3-го знака = обе реализации согласованы.")
    print("\nЕсли расходится только V_win, разница в бутстрэпе и её достаточно описать в §4.5;")
    print("если расходятся MS — это ошибка, и Таблицу 4 ставить нельзя до её устранения.")


if __name__ == "__main__":
    main()
