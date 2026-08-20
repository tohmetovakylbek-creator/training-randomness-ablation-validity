"""
compare_reweighting.py
======================
Сравнение трёх конфигураций для контроля на перевзвешивание (§5.1.3, §6.2.4):

    A. aux=sum,      agg=convex, lambda=0.2     — исходная («сломанная»)
    B. aux=per_mode, agg=convex, lambda=0.2     — исправленная
    C. aux=per_mode, agg=convex, lambda=0.2/K   — контроль: тот же эффективный
                                                  вес вспомогательного члена, что в A

Читает factorial_<dataset>.json, которые пишет factorial_aux_agg.py. Обучения не
требует. Первичная величина — отношение ошибки слитого прогноза к ошибке
константного предиктора (MAE_vs_const), та же, что в Таблице 2 и §5.1.3.

Запуск (без --cells печатает список доступных ячеек):

    python compare_reweighting.py --json results/factorial_2comp/factorial_ukdale.json ^
        results/factorial_2comp/factorial_refit.json ^
        results/factorial_2comp/factorial_sheerm.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = [
    ("y", "MAE_vs_const", "ошибка слитого прогноза / константа"),
    ("y_vmd", "std_ratio", "амплитуда ветви (sd ветви / sd цели)"),
    ("y_vmd", "MAE_vs_const", "ошибка ветви / константа"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--cells", nargs="*", default=None,
                   help="ячейки для сравнения, напр. aux-sum_agg-convex_emb-on "
                        "aux-per_mode_agg-convex_emb-on aux-per_mode_lam0.0333_agg-convex_emb-on")
    ap.add_argument("--exclude", default="", help="домохозяйства через запятую")
    args = ap.parse_args()

    data = {}
    for p in args.json:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for house, rec in d.items():
            data.setdefault(house, {}).update(rec.get("cells", {}))
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    data = {h: v for h, v in data.items() if h not in exclude}

    available = sorted({c for v in data.values() for c in v})
    if not args.cells:
        print(f"домохозяйств: {len(data)}\nдоступные ячейки:")
        for c in available:
            n = sum(1 for v in data.values() if c in v)
            print(f"   {c}   ({n} домохозяйств)")
        print("\nПередайте нужные через --cells (порядок важен: A B C).")
        return

    cells = args.cells
    for c in cells:
        if c not in available:
            raise SystemExit(f"ячейка не найдена: {c}")
    houses = sorted(h for h, v in data.items() if all(c in v for c in cells))
    print(f"домохозяйств с полным набором ячеек: {len(houses)} из {len(data)}")
    if len(houses) < 3:
        raise SystemExit("слишком мало домохозяйств для парных тестов")

    for path_key, metric, title in METRICS:
        vals = {c: np.array([data[h][c]["paths"][path_key][metric] for h in houses])
                for c in cells}
        print(f"\n{'='*78}\n{title}  [{path_key}.{metric}]\n{'='*78}")
        print(f"{'домохозяйство':22s}" + "".join(f"{c[:26]:>28s}" for c in cells))
        for i, h in enumerate(houses):
            print(f"{h:22s}" + "".join(f"{vals[c][i]:28.3f}" for c in cells))
        print(f"{'МЕДИАНА':22s}" + "".join(f"{np.median(vals[c]):28.3f}" for c in cells))

        if path_key == "y_vmd" and metric == "MAE_vs_const":
            print(f"{'хуже константы':22s}" +
                  "".join(f"{int((vals[c] > 1).sum()):>25d}/17" for c in cells))

        print("\n  парные сравнения (Wilcoxon signed-rank, n = %d):" % len(houses))
        for a, b in itertools.combinations(cells, 2):
            diff = vals[b] - vals[a]
            if np.allclose(diff, 0):
                print(f"    {a[:30]} vs {b[:30]}: идентичны")
                continue
            st, p = stats.wilcoxon(vals[a], vals[b])
            better = "вторая лучше" if np.median(diff) < 0 else "первая лучше"
            print(f"    {a[:30]:32s} vs {b[:30]:32s}  "
                  f"медиана разности {np.median(diff):+.4f}  p = {p:.4f}  ({better})")

    print("\n" + "=" * 78)
    print("Как читать (для §5.1.3):")
    print("  Если A против C по y.MAE_vs_const перестало быть значимым, а A против B было,")
    print("  инверсия объясняется весом вспомогательного члена, и формулировку надо смягчить.")
    print("  Если A против C значимо и в ту же сторону, альтернативное объяснение снято.")
    print("  Проверьте заодно, что в C ветвь всё ещё починена: sd ratio заметно выше 0.086")
    print("  и число домов, где ветвь хуже константы, ближе к 1, чем к 11 — иначе контроль")
    print("  сравнивает не то, что задумано.")


if __name__ == "__main__":
    main()
