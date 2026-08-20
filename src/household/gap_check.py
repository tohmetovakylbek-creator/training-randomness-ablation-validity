"""
gap_check.py
============
Считает, сколько часов остаётся пропущенными ПОСЛЕ часового ресемплинга и
ограниченной интерполяции, по каждому дому. Нужно для оговорки в Threats to
Validity: после обрезки по 3-сигма идёт
    x = pd.Series(x).interpolate(limit_direction="both")
по всему ряду сразу, уже после вычисления индексов разбиения. Если остаточных
пропусков нет, заимствовать из соседнего сегмента нечего, и оговорка снимается
одной строкой. Если есть — интерполировать надо посегментно.

Скрипт не пересчитывает предобработку: он проверяет уже сохранённые окна на
NaN и на подозрительные плато (интерполяция даёт строго линейные участки).

Запуск:
    python gap_check.py --processed "<...>\processed" --label REFIT
"""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--processed", required=True)
ap.add_argument("--label", default="dataset")
ap.add_argument("--pattern", default="house_*.npz")
a = ap.parse_args()

print(f"\n===== {a.label} =====")
tot_nan = 0
for f in sorted(Path(a.processed).glob(a.pattern)):
    d = np.load(f)
    nan = sum(int(np.isnan(d[k]).sum()) for k in ("Xtr", "Xva", "Xte", "Ytr", "Yva", "Yte"))
    tot_nan += nan
    # линейные плато: три подряд точки с одинаковой второй разностью ~0
    x = d["Xtr"]
    dd = np.diff(x, n=2, axis=1)
    lin = float((np.abs(dd) < 1e-9).mean())
    print(f"{f.name:16s} NaN={nan:6d}   доля линейных участков в train={lin:.4f}")
print(f"\nвсего NaN по датасету: {tot_nan}")
print("если 0 — интерполяция после обрезки ничего не заполняла, "
      "и оговорка про заимствование из соседнего сегмента снимается")
