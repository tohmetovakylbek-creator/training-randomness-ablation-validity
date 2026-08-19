"""
refit_loader.py
================
Загрузка и предобработка REFIT (CLEAN_REFIT_081116) для того же протокола,
что уже используется на UK-DALE — с целью полностью устранить codebase/pipeline
effect как альтернативное объяснение non-replication (см. отчёт кафедры, п.2).

КРИТИЧНО ДЛЯ ЕДИНОГО PIPELINE: этот файл НЕ переписывает препроцессинг заново —
он импортирует те же функции (to_hourly, Scaler, chronological_split,
sigma_clip_fit, make_windows) напрямую из ukdale_loader.py. Единственное, что
отличается от UK-DALE — это парсинг сырого формата (REFIT CSV вместо
UK-DALE .dat), всё остальное (часовое усреднение, интерполяция, 3-sigma clip
train-fit, min-max train-fit, walk-forward split, окна L=168/T=24) —
БУКВАЛЬНО тот же код, что и на UK-DALE. Это даёт железную гарантию, что любая
разница REFIT vs UK-DALE в §5 — dataset effect, а не codebase effect.

Формат входных данных (официальный релиз CLEAN_REFIT_081116):
    <data_root>/CLEAN_House<N>.csv   (N = 1..21, house_14 отсутствует в релизе)
    Колонки (с заголовком): Time, Unix, Aggregate, Appliance1..Appliance9, Issues
        Time      — строка вида "2013-10-09 13:06:17"
        Unix      — unix epoch (сек)
        Aggregate — суммарная мощность дома, Вт
        Issues    — 0 = показание валидно, 1 = помечено REFIT как проблемное
                     (сбой связи и т.п.) — такие строки исключаются
    Номинальный интервал дискретизации ~8 с.

Модель, обучение, run_all.py, ablation.py, mechanistic.py, stats.py — ВСЕ
используются без единого изменения; на вход им нужен только совместимый
processed/house_<n>.npz, который этот загрузчик и производит — тот же формат,
что ukdale_loader.py (Xtr, Ytr, Xva, Yva, Xte, Yte, scaler_lo, scaler_hi).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# импортируем ГОТОВЫЕ, уже провалидированные на UK-DALE функции препроцессинга —
# намеренно не переписываем их заново, чтобы не рисковать тихим расхождением
from ukdale_loader import (
    to_hourly, Scaler, chronological_split, sigma_clip_fit, make_windows,
)


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
@dataclass
class REFITConfig:
    data_root: str                                  # папка с CLEAN_House<N>.csv
    house_ids: tuple[int, ...] = (6, 4, 10, 9, 2)    # те же 5 домов, что дали
                                                       # 233.44 / 252.84 Вт в §5.1
    resample: str = "1h"                              # идентично UK-DALE
    L: int = 168
    T: int = 24
    test_stride: int = 24
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    interp_limit_hours: int = 6
    sigma_clip: float = 3.0
    min_months: float = 4.0
    out_dir: str = "processed_refit"


# ---------------------------------------------------------------------------
# Парсинг CLEAN_REFIT CSV (единственная REFIT-специфичная часть)
# ---------------------------------------------------------------------------
def _find_refit_csv(root: Path, house: int) -> Path | None:
    """REFIT-релизы встречаются под слегка разными именами файлов в разных
    архивах; проверяем несколько известных вариантов."""
    candidates = [
        root / f"CLEAN_House{house}.csv",
        root / f"CLEAN_House{house:02d}.csv",
        root / f"House{house}.csv",
        root / "CLEAN_REFIT_081116" / f"CLEAN_House{house}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_refit_house_csv(path: Path, chunksize: int = 2_000_000) -> pd.Series:
    """Читает CLEAN_House<N>.csv по частям (файлы REFIT при ~8с интервале за
    ~2 года могут быть по несколько сотен МБ), фильтрует Issues != 0,
    возвращает pd.Series (Aggregate, Вт) с UTC DatetimeIndex по колонке Unix.

    Используем Unix (не строковый Time) как источник индекса — надёжнее и
    быстрее парсится, а также застраховано от локале-зависимых форматов дат
    в разных релизах REFIT."""
    parts = []
    n_total, n_dropped_issues = 0, 0
    for chunk in pd.read_csv(path, chunksize=chunksize,
                              usecols=lambda c: c in ("Unix", "Aggregate", "Issues")):
        n_total += len(chunk)
        if "Issues" in chunk.columns:
            bad = chunk["Issues"] != 0
            n_dropped_issues += int(bad.sum())
            chunk = chunk.loc[~bad]
        parts.append(chunk[["Unix", "Aggregate"]])
    df = pd.concat(parts, ignore_index=True)
    idx = pd.to_datetime(df["Unix"], unit="s", utc=True)
    s = pd.Series(df["Aggregate"].astype(np.float64).values, index=idx).sort_index()
    # могут быть дублирующиеся timestamps на стыках чанков/датасета — берём среднее
    s = s.groupby(level=0).mean()
    print(f"    [parse] {path.name}: {n_total} строк, отброшено по Issues={n_dropped_issues} "
          f"({100*n_dropped_issues/max(n_total,1):.1f}%)")
    return s


# ---------------------------------------------------------------------------
# Основной конвейер по одному дому — структурно идентичен process_house()
# из ukdale_loader.py, отличие только в источнике сырых данных
# ---------------------------------------------------------------------------
def process_house_refit(cfg: REFITConfig, house: int) -> dict | None:
    root = Path(cfg.data_root)
    csv_path = _find_refit_csv(root, house)
    if csv_path is None:
        print(f"[house {house}] CLEAN_House{house}.csv не найден в {root} — пропуск")
        return None
    print(f"[house {house}] источник: {csv_path.name} "
          f"({csv_path.stat().st_size / 1e6:.1f} МБ)")
    raw = parse_refit_house_csv(csv_path)

    # -------- дальше буквально то же самое, что ukdale_loader.process_house --------
    hourly, n_missing = to_hourly(raw, cfg.resample, cfg.interp_limit_hours)
    months = (hourly.index.max() - hourly.index.min()).days / 30.0
    if months < cfg.min_months:
        print(f"[house {house}] всего ~{months:.1f} мес (<{cfg.min_months}) — пропуск")
        return None

    x = hourly.values.astype(np.float64)
    n = len(x)
    split_result = chronological_split(n, cfg.split, cfg.L, cfg.T, cfg.test_stride)
    if split_result is None:
        print(f"[house {house}] даже train-часть короче L+T={cfg.L+cfg.T} ч — пропуск")
        return None
    tr, va, te = split_result

    lo_c, hi_c = sigma_clip_fit(x[tr], cfg.sigma_clip)
    x = np.clip(x, lo_c, hi_c)
    x = pd.Series(x).interpolate(limit_direction="both").values

    scaler = Scaler().fit(x[tr])
    xs = scaler.transform(x)

    cv = float(np.nanstd(hourly.values) / np.nanmean(hourly.values))
    print(f"[house {house}] N={n} ч (~{months:.1f} мес), пропусков={n_missing}, "
          f"mean={np.nanmean(hourly.values):.0f} Вт, CV={cv:.3f}")

    Xtr, Ytr = make_windows(xs[tr], cfg.L, cfg.T, stride=1)
    Xva, Yva = make_windows(xs[va], cfg.L, cfg.T, stride=1)
    Xte, Yte = make_windows(xs[te], cfg.L, cfg.T, stride=cfg.test_stride)

    reduced_sample = len(Xte) < 15
    tag = " [REDUCED SAMPLE]" if reduced_sample else ""
    print(f"[house {house}] окна: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}{tag}")

    return dict(
        house=house, cv=cv, mean_w=float(np.nanmean(hourly.values)),
        std_w=float(np.nanstd(hourly.values)), n_hours=n, months=round(months, 1),
        n_test_windows=int(len(Xte)), reduced_sample=bool(reduced_sample),
        scaler_lo=scaler.lo, scaler_hi=scaler.hi,
        Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte,
    )


def run(cfg: REFITConfig) -> None:
    out = Path(cfg.data_root) / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for h in cfg.house_ids:
        res = process_house_refit(cfg, h)
        if res is None:
            continue
        np.savez_compressed(
            out / f"house_{h}.npz",
            Xtr=res["Xtr"], Ytr=res["Ytr"], Xva=res["Xva"], Yva=res["Yva"],
            Xte=res["Xte"], Yte=res["Yte"],
            scaler_lo=res["scaler_lo"], scaler_hi=res["scaler_hi"],
        )
        summary.append({k: res[k] for k in
                        ("house", "cv", "mean_w", "std_w", "n_hours", "months",
                         "n_test_windows", "reduced_sample")})

    summary = sorted(summary, key=lambda d: d["cv"])
    (out / "summary.json").write_text(json.dumps(
        {"config": asdict(cfg), "houses": summary}, ensure_ascii=False, indent=2))
    print("\n=== Сводка по домам REFIT (отсортировано по CV) ===")
    for d in summary:
        tag = " [REDUCED SAMPLE]" if d["reduced_sample"] else ""
        print(f"  house {d['house']}: CV={d['cv']:.3f}, mean={d['mean_w']:.0f} Вт, "
              f"{d['n_hours']} ч (~{d['months']} мес), test_windows={d['n_test_windows']}{tag}")
    print(f"\nСохранено в: {out.resolve()}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help=r"папка с CLEAN_House<N>.csv (CLEAN_REFIT_081116)")
    p.add_argument("--houses", default="6,4,10,9,2",
                   help="дома через запятую (по умолч. те же 5, что дали 233.44/252.84 Вт в §5.1)")
    p.add_argument("--min_months", type=float, default=4.0)
    args = p.parse_args()
    cfg = REFITConfig(
        data_root=args.data_root,
        house_ids=tuple(int(x) for x in args.houses.split(",")),
        min_months=args.min_months,
    )
    run(cfg)
