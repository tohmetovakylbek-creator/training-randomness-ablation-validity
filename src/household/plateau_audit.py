"""
plateau_audit.py
================
Локализует длинные плато (постоянные показания счётчика) по сплитам и считает,
сколько ТЕСТОВЫХ окон ими задето.

Повод: на REFIT house 9 максимальная серия постоянных значений в обучающих окнах
достигает 112 часов — почти пять суток. Интерполяция в to_hourly ограничена
шестью часами, значит это не достроенный пропуск, а реально записанная
константа: отключение, отъезд жильцов или залипший счётчик. Обрезка по 3-сигма
такое тоже не убирает — константа около среднего не выброс.

Почему это важно именно для теста. Окно, у которого целевой отрезок Y почти
целиком плоский, предсказывается тривиально: любая модель, продолжающая
последнее значение, получит на нём почти нулевую ошибку. Такие окна смещают MAE
и сжимают дисперсию поокновых ошибок, на которой строится block bootstrap.

Выводит по каждому дому и сплиту: максимальную серию, долю окон с плато длиннее
порога во входе X и отдельно в цели Y.

Запуск:
    python plateau_audit.py --processed "<...>\\processed" --label REFIT --min_run 24
"""
import argparse
from pathlib import Path
import numpy as np


def max_run(b):
    """Максимальная серия True в каждой строке (n, m) -> (n,)."""
    n = b.shape[0]
    run = np.zeros(n, dtype=np.int32)
    best = np.zeros(n, dtype=np.int32)
    for j in range(b.shape[1]):
        run = np.where(b[:, j], run + 1, 0)
        np.maximum(best, run, out=best)
    return best


def flat_runs(a, tol):
    """Серии подряд идущих равных значений: длина в отсчётах."""
    if a.shape[1] < 2:
        return np.zeros(len(a), dtype=np.int32)
    return max_run(np.abs(np.diff(a, axis=1)) < tol) + 1


ap = argparse.ArgumentParser()
ap.add_argument("--processed", required=True)
ap.add_argument("--label", default="dataset")
ap.add_argument("--pattern", default="house_*.npz")
ap.add_argument("--min_run", type=int, default=24, help="порог длины плато в часах")
ap.add_argument("--tol", type=float, default=1e-9)
a = ap.parse_args()

print(f"\n===== {a.label} =====  порог плато {a.min_run} ч")
print(f"{'файл':16s} {'сплит':5s} {'окон':>7s} {'макс.плато':>11s} "
      f"{'доля X с плато':>15s} {'доля Y с плато':>15s}")
worst = []
for f in sorted(Path(a.processed).glob(a.pattern)):
    d = np.load(f)
    for s in ("tr", "va", "te"):
        X, Y = d[f"X{s}"], d[f"Y{s}"]
        if len(X) == 0:
            continue
        rx, ry = flat_runs(X, a.tol), flat_runs(Y, a.tol)
        fx = float((rx >= a.min_run).mean())
        fy = float((ry >= a.min_run).mean())
        print(f"{f.name:16s} {s:5s} {len(X):7d} {int(rx.max()):>9d} ч "
              f"{fx:15.3f} {fy:15.3f}")
        if s == "te" and fy > 0.05:
            worst.append((f.name, fy, int(ry.max())))
    print()

if worst:
    print("ТЕСТОВЫЕ окна с плоской целью (доля > 5%) — их MAE не информативна:")
    for n, fy, m in sorted(worst, key=lambda t: -t[1]):
        print(f"   {n}: {fy:.1%} окон, максимум {m} ч подряд")
else:
    print("тестовых окон с плоской целью выше 5% нет")
