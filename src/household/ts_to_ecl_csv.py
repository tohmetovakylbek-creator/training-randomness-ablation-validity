"""
ts_to_ecl_csv.py
================
Конвертер electricity_hourly_dataset.ts (Monash Time Series Forecasting Archive)
в широкий CSV того же формата, что ожидает ecl_loader.py:

    date,T1,T2,...,T321
    2012-01-01 00:00:00,14.0,...

Зачем отдельный шаг, а не парсинг внутри загрузчика: формат .ts к энергетике
отношения не имеет, это контейнер sktime, и держать его разбор в ecl_loader.py
означало бы смешать протокол предобработки с транспортом данных. Конвертация
делается один раз, её результат хэшируется и фиксируется наравне с исходником.

ВАЖНО для пре-спецификации: Monash-версия — это те же 321 ряд, агрегированные
до часа СТОРОННИМ источником из исходного LD2011_2014 (15-минутные kW). То есть
условие §3 пре-спецификации соблюдено — почасовое агрегирование не является
решением авторов, — но название файла и происхождение в §3 надо заменить на
Monash до запуска отбора.

Формат .ts (Monash):
    строки-заголовки начинаются с '#' или '@', данные идут после '@data';
    каждая строка данных:  <имя>:<старт YYYY-MM-DD HH-MM-SS>:<v1,v2,...>
    пропуски обозначаются '?'.

Использование:
    python ts_to_ecl_csv.py --ts data/electricity_hourly_dataset.ts \
                            --out data/ECL_monash.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TS_FMT = "%Y-%m-%d %H-%M-%S"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ts(path: Path) -> tuple[dict[str, np.ndarray], dict[str, pd.Timestamp]]:
    series: dict[str, np.ndarray] = {}
    starts: dict[str, pd.Timestamp] = {}
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            parts = line.split(":")
            if len(parts) < 3:
                raise ValueError(
                    f"Строка {ln}: ожидалось '<имя>:<старт>:<значения>', получено "
                    f"{len(parts)} полей. Пришлите первые строки файла — допишу разбор.")
            name, start_raw, values_raw = parts[0], parts[1], ":".join(parts[2:])
            vals = np.array([np.nan if v.strip() in ("?", "", "NaN") else float(v)
                             for v in values_raw.split(",")], dtype=np.float64)
            series[name] = vals
            starts[name] = pd.to_datetime(start_raw.strip(), format=TS_FMT)
    if not series:
        raise ValueError("После '@data' не найдено ни одной строки данных.")
    return series, starts


def to_wide(series: dict[str, np.ndarray],
            starts: dict[str, pd.Timestamp]) -> pd.DataFrame:
    frames = []
    for name, vals in series.items():
        idx = pd.date_range(starts[name], periods=len(vals), freq="h")
        frames.append(pd.Series(vals, index=idx, name=name))
    df = pd.concat(frames, axis=1)          # выравнивание по общей временной оси
    df = df.sort_index()
    df.index.name = "date"
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ts", required=True, help="electricity_hourly_dataset.ts")
    p.add_argument("--out", required=True, help="выходной широкий CSV")
    args = p.parse_args()

    ts_path, out_path = Path(args.ts), Path(args.out)
    series, starts = parse_ts(ts_path)
    lengths = {len(v) for v in series.values()}
    uniq_starts = {str(s) for s in starts.values()}
    print(f"Рядов: {len(series)}")
    print(f"Длины рядов: {sorted(lengths)}")
    print(f"Стартовые метки: {sorted(uniq_starts)[:3]}"
          f"{' ...' if len(uniq_starts) > 3 else ''} (всего {len(uniq_starts)})")

    df = to_wide(series, starts)
    n_nan = int(df.isna().sum().sum())
    print(f"Итоговая таблица: {df.shape[0]} часов x {df.shape[1]} рядов, "
          f"период {df.index.min()} .. {df.index.max()}")
    print(f"Пропусков (включая выравнивание по оси): {n_nan} "
          f"({n_nan / df.size:.4%})")

    zero_prefix = int(sum(1 for c in df.columns
                          if float(np.nanmean(df[c].values[:2000] == 0)) > 0.5))
    print(f"Рядов с нулевым префиксом (>50 % нулей в первых 2000 ч): {zero_prefix} "
          f"— их отсеет ecl_select_clients.py")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S")
    print(f"\nЗаписано: {out_path.resolve()}")
    print(f"SHA-256 исходника .ts : {sha256_of(ts_path)}")
    print(f"SHA-256 полученного CSV: {sha256_of(out_path)}")
    print("Обе контрольные суммы внесите в §3 пре-спецификации до запуска отбора.")


if __name__ == "__main__":
    main()
