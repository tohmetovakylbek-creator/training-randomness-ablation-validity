"""
ecl_select_clients.py
=====================
Пре-специфицированный отбор клиентов ECL как единиц репликации.

Отбор ДОЛЖЕН быть выполнен и зафиксирован ДО первого обучающего прогона:
вся статья — о том, что абляционные выводы заявляются без пре-спецификации,
поэтому выбор второго домена постфактум был бы прямым self-own.

Правило (одно, без исключений):
    1. Пригодность клиента:
         - доля нулевых часов в train  <= max_zero_frac_train (0.01)
         - самый длинный плоский прогон <= max_flat_run (48 ч)
         - ненулевая дисперсия в test
         - тестовых окон >= match_windows (120) при stride 24, L=168, T=24
    2. Из пригодного пула — случайная выборка n_clients (17) генератором
       numpy default_rng(select_seed=20260809), без замены.
Никакие метрики качества прогноза в отбор не входят.

Выход: ecl_selection.json (пул, выборка, конфиг, sha256 исходного CSV)
       и ecl_eligibility.csv (полная таблица диагностик по всем клиентам).
Обгоняет ~321 клиента за десятки секунд; повторный запуск с тем же CSV
и тем же сидом воспроизводит выборку побитово.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ecl_loader import ECLConfig, read_wide_csv, client_diagnostics, sha256_of


def _sort_key(name: str):
    """Естественный порядок: '2' раньше '10', 'OT' в конце."""
    return (0, int(name)) if str(name).isdigit() else (1, str(name))


def select(cfg: ECLConfig) -> dict:
    csv_path = Path(cfg.csv_path)
    df = read_wide_csv(csv_path)
    print(f"Файл: {csv_path}  строк={len(df)}  колонок-клиентов={df.shape[1]}")
    print(f"Период: {df.index.min()} .. {df.index.max()}")

    rows = []
    for name in df.columns:
        d = client_diagnostics(df[name], cfg)
        d["client"] = str(name)
        rows.append(d)
        if len(rows) % 50 == 0:
            print(f"  ... проверено {len(rows)} клиентов")

    table = pd.DataFrame(rows).set_index("client")
    cols = ["eligible", "n_hours", "n_test_windows", "zero_frac_train",
            "max_flat_run_h", "mean", "cv", "test_std", "reason"]
    table = table[[c for c in cols if c in table.columns]]
    table.to_csv("ecl_eligibility.csv", encoding="utf-8")

    pool = sorted(table.index[table["eligible"]].tolist(), key=_sort_key)
    print(f"\nПригодных клиентов: {len(pool)} из {df.shape[1]}")
    if len(pool) < cfg.n_clients:
        raise RuntimeError(
            f"Пригодных клиентов {len(pool)} < требуемых {cfg.n_clients}. "
            f"Ослаблять критерии постфактум НЕЛЬЗЯ — зафиксируйте новое правило "
            f"в pre-registration и перезапустите отбор.")

    rng = np.random.default_rng(cfg.select_seed)
    selected = sorted(rng.choice(np.array(pool, dtype=object),
                                 size=cfg.n_clients, replace=False).tolist(),
                      key=_sort_key)

    out = {
        "config": asdict(cfg),
        "csv_sha256": sha256_of(csv_path),
        "n_clients_total": int(df.shape[1]),
        "n_eligible": len(pool),
        "eligible_pool": pool,
        "selected": selected,
        "diagnostics_file": "ecl_eligibility.csv",
    }
    Path("ecl_selection.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Отобранные клиенты (пре-специфицированный сид "
          f"{cfg.select_seed}) ===")
    print(", ".join(selected))
    sub = table.loc[selected]
    print(f"\nCV: min={sub['cv'].min():.3f}  median={sub['cv'].median():.3f}  "
          f"max={sub['cv'].max():.3f}")
    print(f"Тестовых окон до усечения: min={int(sub['n_test_windows'].min())}  "
          f"max={int(sub['n_test_windows'].max())}")
    print("\nЗаписано: ecl_selection.json, ecl_eligibility.csv")
    print("Зафиксируйте оба файла в git ДО первого прогона обучения.")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help=r"путь к ECL.csv")
    p.add_argument("--n_clients", type=int, default=17)
    p.add_argument("--select_seed", type=int, default=20260809)
    p.add_argument("--match_windows", type=int, default=120)
    p.add_argument("--max_zero_frac_train", type=float, default=0.01)
    p.add_argument("--max_flat_run", type=int, default=48)
    args = p.parse_args()

    cfg = ECLConfig(
        csv_path=args.csv,
        dataset="ecl",
        n_clients=args.n_clients,
        select_seed=args.select_seed,
        match_windows=args.match_windows,
        max_zero_frac_train=args.max_zero_frac_train,
        max_flat_run=args.max_flat_run,
    )
    select(cfg)
