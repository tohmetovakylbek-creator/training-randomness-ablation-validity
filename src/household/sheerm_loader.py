"""
sheerm_loader.py
=================
Загрузка и предобработка Load-рядов SHEERM (13 домохозяйств, Португалия) для
ТОГО ЖЕ протокола STLF, что уже используется на REFIT / UK-DALE / Pecan Street:

    - часовое разрешение (resample mean)
    - вход L = 168 ч, горизонт T = 24 ч
    - хронологическое разбиение 70 / 15 / 15 без перемешивания
    - walk-forward по тесту со сдвигом 24 ч
    - предобработка: линейная интерполяция коротких пропусков -> 3-sigma -> min-max
    - нормализация ТОЛЬКО на train (защита от утечки)

КРИТИЧНО ДЛЯ ЕДИНОГО PIPELINE (тот же принцип, что в refit_loader.py): этот файл
НЕ переписывает препроцессинг — импортирует to_hourly, Scaler, chronological_split,
sigma_clip_fit, make_windows напрямую из ukdale_loader.py. Единственное, что здесь
SHEERM-специфично — парсинг сырого формата (Date+Hour вместо Unix/строкового Time)
и автодетект колонки нагрузки. Это даёт ту же гарантию, что и на REFIT: любая
разница SHEERM vs REFIT/UK-DALE/Pecan Street в результатах — dataset effect,
а не codebase effect.

СОЗНАТЕЛЬНО НЕ ИСПОЛЬЗУЕТСЯ (см. отчёт по PV/Weather-циркулярности):
    - PV Generation House N.csv  — синтетическая величина, детерминированно
      выведенная из части признаков Weather (см. PV_calculations.ipynb),
      скрытая циркулярность с ковариатами; не участвует в forecasting-протоколе.
    - Weather House N.csv        — не нужен для чистого Load-forecasting;
      будет использован отдельно (не в этом loader'е) только для Фазы 3
      (корреляция силы mode-identity эффекта с волатильностью Price).

Формат входных данных (архив "Датасет SHEERM.zip"):
    <data_root>/Load House <N>.csv   (N = 1..13)
    Колонки: Date, Hour, + ровно ОДНА числовая колонка нагрузки (имя неизвестно
    заранее из выгрузки — см. _detect_load_column). Нативная гранулярность 15 мин
    (в Technical_validation.ipynb: one_year_points = 35040 = 365*24*4).
    Единица измерения — предположительно кВт (в валидационном ноутбуке авторов
    ось графика подписана "Load (kW)", ylim(0,5)) — проверяется эвристикой ниже
    и логируется явно, чтобы не тащить молчаливое расхождение единиц дальше
    в pipeline.

    Отдельно (НЕ per-house):
    <data_root>/Price Portugal.csv   — единый рыночный тариф на все 13 домов.
    Загружается отдельной вспомогательной функцией load_price_series() —
    не встроен в окна L/T, нужен только для Фазы 3 (см. load_price_series).

House 6 / House 7 в архиве заметно короче остальных (файлы ~477 КБ против
~2.3-3.7 МБ) — уже существующий min_months-гейт в общем pipeline должен
автоматически их исключить, если реального периода наблюдений недостаточно;
здесь это только логируется явно, специального кейса не требуется.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# импортируем ГОТОВЫЕ, уже провалидированные на UK-DALE/REFIT функции препроцессинга —
# намеренно не переписываем их заново (см. docstring выше)
from ukdale_loader import (
    to_hourly, Scaler, chronological_split, sigma_clip_fit, make_windows,
)


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
@dataclass
class SHEERMConfig:
    data_root: str                                    # папка с "Load House <N>.csv"
    house_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)  # House 6/7 excluded:
                                                         # only ~5.1 months of data (see
                                                         # summary.json run) — kept out of
                                                         # the main analysis by decision,
                                                         # not by the min_months gate
    resample: str = "1h"                                # идентично REFIT/UK-DALE/Pecan Street
    L: int = 168
    T: int = 24
    test_stride: int = 24
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    interp_limit_hours: int = 6
    sigma_clip: float = 3.0
    min_months: float = 4.0
    assume_native_unit: str = "kW"                      # "kW" или "W" — см. _to_watts
    out_dir: str = "processed_sheerm"


# ---------------------------------------------------------------------------
# Парсинг Load House <N>.csv (единственная SHEERM-специфичная часть)
# ---------------------------------------------------------------------------
def _find_sheerm_csv(root: Path, house: int) -> Path | None:
    candidates = [
        root / f"Load House {house}.csv",
        root / f"Load House {house:02d}.csv",
        root / "Load" / f"House {house}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _detect_load_column(columns: list[str]) -> str:
    """Явный (не угадывающий вслепую) выбор колонки нагрузки.

    1) Если после исключения Date/Hour остаётся РОВНО одна колонка — берём её
       (это ожидаемый случай по структуре, увиденной в Technical_validation.ipynb).
    2) Иначе ищем по имени (load/power/consumption/kw/w), регистронезависимо.
    3) Если ни то, ни другое не дало однозначного результата — падаем с явной
       ошибкой и списком колонок, а не молча берём первую попавшуюся числовую
       колонку (риск случайно схватить служебное поле).
    """
    remaining = [c for c in columns if c.lower() not in ("date", "hour")]
    if len(remaining) == 1:
        return remaining[0]

    pattern = re.compile(r"load|power|consumption|\bkw\b|\bw\b", re.IGNORECASE)
    matches = [c for c in remaining if pattern.search(c)]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Не удалось однозначно определить колонку нагрузки среди {remaining}. "
        f"Передайте имя явно через SHEERMConfig(load_column=...) "
        f"(добавьте это поле при необходимости) или переименуйте колонку в CSV."
    )


def _to_watts(x: np.ndarray, native_unit: str) -> np.ndarray:
    if native_unit.lower() == "kw":
        return x * 1000.0
    if native_unit.lower() == "w":
        return x
    raise ValueError(f"Неизвестная единица измерения: {native_unit!r} (ожидалось 'kW' или 'W')")


def parse_sheerm_house_csv(path: Path, native_unit: str = "kW") -> pd.Series:
    """Читает Load House <N>.csv -> pd.Series (мощность, Вт) с DatetimeIndex.

    Самостоятельная функция для быстрой инспекции/аудита одного файла вне
    основного пайплайна (process_house_sheerm ниже делает то же самое инлайн,
    как часть process_house_refit-подобного потока). В отличие от REFIT/UK-DALE,
    файлы SHEERM небольшие (15-минутная гранулярность за ~3 года — единицы МБ),
    чанкинг не нужен."""
    df = pd.read_csv(path)

    missing = [c for c in ("Date", "Hour") if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: ожидались колонки Date/Hour, отсутствуют {missing}. "
            f"Реальные колонки: {list(df.columns)}"
        )

    load_col = _detect_load_column(list(df.columns))

    timestamp = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Hour"].astype(str), errors="coerce"
    )
    timestamp = timestamp.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")  # см. process_house_sheerm
    n_bad_ts = int(timestamp.isna().sum())
    if n_bad_ts:
        print(f"    [parse] {path.name}: {n_bad_ts} строк с непарсящимся Date+Hour — отброшены")

    values_native = pd.to_numeric(df[load_col], errors="coerce")
    n_bad_val = int(values_native.isna().sum()) - n_bad_ts  # грубая оценка доп. потерь
    if n_bad_val > 0:
        print(f"    [parse] {path.name}: ещё {n_bad_val} строк с нечисловым '{load_col}' — отброшены")

    s_native = pd.Series(values_native.values, index=timestamp).dropna()
    s_native = s_native[~s_native.index.duplicated(keep="first")].sort_index()

    print(f"    [parse] {path.name}: колонка нагрузки='{load_col}', "
          f"{len(s_native)} валидных строк, диапазон нативных значений "
          f"[{s_native.min():.3f}, {s_native.max():.3f}] (unit={native_unit})")

    return pd.Series(_to_watts(s_native.values.astype(np.float64), native_unit), index=s_native.index)


# ---------------------------------------------------------------------------
# Основной конвейер по одному дому — структурно идентичен process_house_refit()
# ---------------------------------------------------------------------------
def process_house_sheerm(cfg: SHEERMConfig, house: int) -> dict | None:
    root = Path(cfg.data_root)
    csv_path = _find_sheerm_csv(root, house)
    if csv_path is None:
        print(f"[house {house}] Load House {house}.csv не найден в {root} — пропуск")
        return None
    print(f"[house {house}] источник: {csv_path.name} "
          f"({csv_path.stat().st_size / 1e6:.2f} МБ)")

    raw = pd.read_csv(csv_path)
    load_col = _detect_load_column(list(raw.columns))
    timestamp = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Hour"].astype(str), errors="coerce"
    )
    # КРИТИЧНО: to_hourly() (импортируется из ukdale_loader.py) строит
    # pd.date_range(..., tz="UTC") и делает h.reindex(full) — это требует, чтобы
    # входной индекс тоже был tz-aware UTC, иначе reindex молча не находит
    # совпадений и весь ряд превращается в NaN (длина индекса при этом остаётся
    # верной — из-за этого баг незаметен по n_hours/months, только по NaN в
    # cv/mean_w/std_w). Date+Hour не несёт информации о часовом поясе, поэтому
    # мы просто локализуем как UTC (тот же подход, что utc=True у REFIT/UK-DALE
    # при парсинге Unix-времени) — абсолютный сдвиг здесь не важен, важно только
    # чтобы весь pipeline был внутренне согласован по tz.
    timestamp = timestamp.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
    values_native = pd.to_numeric(raw[load_col], errors="coerce")
    values_w = _to_watts(values_native.values.astype(np.float64), cfg.assume_native_unit)

    s = pd.Series(values_w, index=timestamp).dropna()
    s = s[~s.index.duplicated(keep="first")].sort_index()

    # -------- дальше буквально то же самое, что ukdale_loader.process_house --------
    hourly, n_missing = to_hourly(s, cfg.resample, cfg.interp_limit_hours)
    months = (hourly.index.max() - hourly.index.min()).days / 30.0
    if months < cfg.min_months:
        print(f"[house {house}] всего ~{months:.1f} мес (<{cfg.min_months}) — "
              f"пропуск (ожидаемо для House 6/7, если это они)")
        return None

    x = hourly.values.astype(np.float64)
    n = len(x)
    split_result = chronological_split(n, cfg.split, cfg.L, cfg.T, cfg.test_stride)
    if split_result is None:
        print(f"[house {house}] даже train-часть короче L+T={cfg.L + cfg.T} ч — пропуск")
        return None
    tr, va, te = split_result

    lo_c, hi_c = sigma_clip_fit(x[tr], cfg.sigma_clip)
    x = np.clip(x, lo_c, hi_c)
    x = pd.Series(x).interpolate(limit_direction="both").values

    scaler = Scaler().fit(x[tr])
    xs = scaler.transform(x)

    cv = float(np.nanstd(hourly.values) / np.nanmean(hourly.values))
    print(f"[house {house}] N={n} ч (~{months:.1f} мес), пропусков={n_missing}, "
          f"mean={np.nanmean(hourly.values):.0f} Вт, CV={cv:.3f}, load_col='{load_col}'")

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
        load_col=load_col,
        scaler_lo=scaler.lo, scaler_hi=scaler.hi,
        Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte,
    )


def run(cfg: SHEERMConfig) -> None:
    out = Path(cfg.data_root) / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for h in cfg.house_ids:
        res = process_house_sheerm(cfg, h)
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
                         "n_test_windows", "reduced_sample", "load_col")})

    summary = sorted(summary, key=lambda d: d["cv"])
    (out / "summary.json").write_text(json.dumps(
        {"config": asdict(cfg), "houses": summary}, ensure_ascii=False, indent=2))

    print("\n=== Сводка по домам SHEERM (отсортировано по CV) ===")
    for d in summary:
        tag = " [REDUCED SAMPLE]" if d["reduced_sample"] else ""
        print(f"  house {d['house']}: CV={d['cv']:.3f}, mean={d['mean_w']:.0f} Вт, "
              f"{d['n_hours']} ч (~{d['months']} мес), test_windows={d['n_test_windows']}{tag}")

    n_excluded = len(cfg.house_ids) - len(summary)
    if n_excluded:
        print(f"\n[ВНИМАНИЕ] {n_excluded} дом(а) исключены (не найдены или короче "
              f"min_months={cfg.min_months}) — проверьте, что это действительно House 6/7, "
              f"а не неожиданная потеря данных по другим домам.")
    print(f"\nСохранено в: {out.resolve()}")


# ---------------------------------------------------------------------------
# Вспомогательная загрузка Price Portugal.csv — ДЛЯ ФАЗЫ 3 (не часть окон L/T)
# ---------------------------------------------------------------------------
def load_price_series(data_root: str, filename: str = "Price Portugal.csv") -> pd.Series:
    """Загружает единый рыночный тариф (общий на все 13 домов) как часовой Series.

    Используется отдельно от Load-пайплайна выше — для сопоставления силы
    mode-identity эффекта по дому с волатильностью цены в тот же период
    (см. Фазу 3 roadmap). Не участвует в построении окон X/Y для forecasting-модели."""
    path = Path(data_root) / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} не найден в {data_root}")

    df = pd.read_csv(path)
    ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    price_col = next((c for c in df.columns if "price" in c.lower()), None)
    if ts_col is None or price_col is None:
        raise ValueError(
            f"Не удалось определить колонки времени/цены в {filename}. "
            f"Колонки: {list(df.columns)}"
        )

    ts = pd.to_datetime(df[ts_col], errors="coerce")
    price = pd.to_numeric(df[price_col], errors="coerce")
    s = pd.Series(price.values, index=ts).dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    print(f"[price] {filename}: {len(s)} часовых точек, "
          f"диапазон [{s.min():.4f}, {s.max():.4f}], колонка='{price_col}'")
    return s


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help=r"папка с 'Load House <N>.csv' (распакованный Датасет SHEERM.zip)")
    p.add_argument("--houses", default="1,2,3,4,5,8,9,10,11,12,13",
                   help="дома через запятую (по умолч. 11 — House 6/7 исключены решением "
                        "из-за короткого периода наблюдений, см. SHEERMConfig.house_ids)")
    p.add_argument("--min_months", type=float, default=4.0)
    p.add_argument("--native_unit", default="kW", choices=["kW", "W"],
                   help="нативная единица Load House <N>.csv (см. docstring)")
    args = p.parse_args()
    cfg = SHEERMConfig(
        data_root=args.data_root,
        house_ids=tuple(int(x) for x in args.houses.split(",")),
        min_months=args.min_months,
        assume_native_unit=args.native_unit,
    )
    run(cfg)
