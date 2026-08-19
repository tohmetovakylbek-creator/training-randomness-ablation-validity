"""
flat_vs_interp.py
=================
Разделяет "линейные участки", которые находит gap_check, на два разных явления:

  плато        — первая разность нулевая: счётчик отдавал постоянное значение
                 (залипание, отключение, нижняя граница измерения);
  интерполяция — первая разность постоянна и не равна нулю: участок восстановлен
                 линейной интерполяцией в to_hourly (limit = 6 ч).

Различие существенно. NaN = 0 означает лишь, что после интерполяции пропусков не
осталось, но не говорит, какая доля обучающих данных измерена, а какая
достроена. На REFIT доля линейных участков доходит до 30%, и в Таблице 1 это
надо показывать явно.

Запуск:
    python flat_vs_interp.py --processed "<...>\processed" --label REFIT
"""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--processed", required=True)
ap.add_argument("--label", default="dataset")
ap.add_argument("--pattern", default="house_*.npz")
ap.add_argument("--tol", type=float, default=1e-9)
a = ap.parse_args()

print(f"\n===== {a.label} =====")
print(f"{'файл':16s} {'линейных':>9s} {'плато':>8s} {'интерп.':>9s} {'макс.плато':>11s}")
for f in sorted(Path(a.processed).glob(a.pattern)):
    x = np.load(f)["Xtr"]
    d1 = np.diff(x, axis=1)
    d2 = np.diff(x, n=2, axis=1)
    lin = np.abs(d2) < a.tol                      # (n, L-2)
    flat = lin & (np.abs(d1[:, :-1]) < a.tol)     # линейный И с нулевым наклоном
    interp = lin & ~flat
    # самая длинная серия плато внутри окна — грубая оценка длины залипания
    run = 0; best = 0
    for row in flat[: min(len(flat), 2000)]:
        c = 0
        for v in row:
            c = c + 1 if v else 0
            best = max(best, c)
    print(f"{f.name:16s} {lin.mean():9.4f} {flat.mean():8.4f} {interp.mean():9.4f} {best:>9d} ч")
print("\nплато — постоянные показания счётчика; интерп. — достроено интерполяцией "
      "до 6 ч в to_hourly")
