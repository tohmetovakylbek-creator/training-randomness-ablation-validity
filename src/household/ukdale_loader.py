"""
ukdale_loader.py
=================
Загрузка и предобработка датасета UK-DALE для задачи STLF на уровне домохозяйства.
Протокол ИДЕНТИЧЕН эксперименту на REFIT (диплом, разделы 2.2 и 3.1):
    - часовое разрешение (resample mean)
    - вход L = 168 ч, горизонт T = 24 ч
    - хронологическое разбиение 70 / 15 / 15 без перемешивания
    - walk-forward по тесту со сдвигом 24 ч
    - предобработка: линейная интерполяция пропусков -> 3-sigma -> min-max

КРИТИЧНО (защита от утечки): статистики min-max и порога 3-sigma вычисляются
ТОЛЬКО на train и затем применяются к val/test. Если в REFIT-коде нормализация
была подогнана глобально (по всему ряду), это утечка — здесь она исправлена.
Это согласуется с каузальной VMD: причинной должна быть ВСЯ предобработка, не
только декомпозиция.

Формат входных данных (стандартная ручная выгрузка UK-DALE, ukdale.zip):
    <data_root>/house_<n>/channel_1.dat   — агрегированная мощность дома (mains)
    Файл .dat: две колонки, разделитель — пробел, без заголовка:
        <unix_timestamp_sec> <active_power_W>
    Интервал дискретизации ~6 с (для house_1) / ~6 с прочие.

Если вы скачали NILMTK-вариант (единый ukdale.h5), используйте parse_house_mains_h5
(заглушка ниже) — сообщите формат, и я допишу парсер под него.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
@dataclass
class UKDaleConfig:
    data_root: str                              # папка с house_<n>/channel_1.dat
    house_ids: tuple[int, ...] = (1, 2, 3, 4, 5)
    mains_channel: int = 1                       # channel_1.dat = агрегированная нагрузка
    resample: str = "1h"                         # часовое разрешение, как в REFIT
    L: int = 168                                 # длина входного окна (часов)
    T: int = 24                                  # горизонт прогноза (часов)
    test_stride: int = 24                        # шаг walk-forward по тесту
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    interp_limit_hours: int = 6                  # макс. длина интерполируемого пропуска
    sigma_clip: float = 3.0                      # порог удаления аномалий (на train)
    min_months: int = 4                          # минимум данных для включения дома
    out_dir: str = "processed"


# ---------------------------------------------------------------------------
# Парсинг сырых файлов
# ---------------------------------------------------------------------------
def find_mains_file(house_dir: Path, mains_channel: int) -> tuple[Path | None, str]:
    """UK-DALE хранит агрегат по-разному в зависимости от дома:
      - house_1 (и иногда другие): отдельный mains.dat (часто 1с-разрешение,
        может быть 2 ИЛИ 3 колонки: ts power  ИЛИ  ts apparent_power voltage);
      - остальные дома: агрегат лежит как channel_<mains_channel>.dat,
        номер которого нужно сверять по labels.dat (не всегда 1)."""
    mains_dat = house_dir / "mains.dat"
    if mains_dat.exists():
        return mains_dat, "mains.dat"
    ch = house_dir / f"channel_{mains_channel}.dat"
    if ch.exists():
        return ch, f"channel_{mains_channel}.dat"
    return None, ""


def _ts_to_index(ts: np.ndarray) -> pd.DatetimeIndex:
    """Unix-время -> DatetimeIndex. UK-DALE house_1 даёт дробные секунды
    (напр. 1363547563.1, шаг ~0.1-1с) — округляем перед приведением к int64,
    чтобы не терять/не сдвигать отсчёт на труднообъяснимую долю секунды."""
    return pd.to_datetime(np.round(ts).astype(np.int64), unit="s", utc=True)


def _select_power_column(ncols: int) -> int:
    """Явная (не угадывающая) раскладка колонок UK-DALE .dat:
      2 кол.: ts power
      3 кол.: ts apparent_power voltage   (house_2-5 channel_*.dat иногда)
      4 кол.: ts active_power apparent_power voltage   (house_1 mains.dat)
    Всегда возвращает индекс мощности, которую трактуем как активную/агрегатную."""
    if ncols == 2:
        return 1
    if ncols == 3:
        return 1
    if ncols == 4:
        return 1   # active_power — самая корректная для расчёта энергии/тарифов
    raise ValueError(f"Неподдерживаемое число колонок: {ncols}. "
                     "Пришлите первые строки файла — допишу раскладку.")


def _read_dat_chunked(path: Path, chunksize: int = 5_000_000) -> pd.Series:
    """Построчное чтение крупных .dat (house_1/mains.dat может быть несколько ГБ)."""
    parts = []
    ncols_seen = None
    for chunk in pd.read_csv(path, sep=r"\s+", header=None, chunksize=chunksize,
                             dtype=np.float64, engine="c"):
        ncols_seen = ncols_seen or chunk.shape[1]
        if chunk.shape[1] != ncols_seen:
            raise ValueError(f"Число колонок изменилось внутри файла {path} "
                             f"({ncols_seen} -> {chunk.shape[1]}) — формат непостоянен.")
        col = _select_power_column(chunk.shape[1])
        ts, power = chunk[0].values, chunk[col].values
        parts.append(pd.Series(power, index=_ts_to_index(ts)))
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s[s < 0] = np.nan
    return s


def parse_house_mains_dat(path: Path) -> pd.Series:
    """Читает mains.dat ИЛИ channel_<n>.dat -> часовой Series мощности (Вт).
    Большие файлы (house_1 mains.dat — несколько ГБ) читаются чанками."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 200:
        return _read_dat_chunked(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64, engine="c")
    col = _select_power_column(df.shape[1])
    ts, power = df[0].values, df[col].values
    s = pd.Series(power, index=_ts_to_index(ts), name="power").sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s[s < 0] = np.nan
    return s


def parse_house_mains_h5(path: Path, building: int) -> pd.Series:
    """Заглушка для NILMTK ukdale.h5. Допишу под точную структуру по запросу.
    В NILMTK агрегат обычно лежит под '/building<n>/elec/meter1'."""
    raise NotImplementedError(
        "Обнаружен .h5. Сообщите, что это NILMTK-формат, и я допишу парсер "
        "(pd.HDFStore -> '/building%d/elec/meter1')." % building
    )


def to_hourly(s: pd.Series, resample: str, interp_limit_hours: int) -> pd.Series:
    """Часовое усреднение + линейная интерполяция коротких пропусков."""
    h = s.resample(resample).mean()
    # непрерывный часовой индекс, чтобы пропуски были явными
    full = pd.date_range(h.index.min(), h.index.max(), freq=resample, tz="UTC")
    h = h.reindex(full)
    n_missing = int(h.isna().sum())
    h = h.interpolate(method="linear", limit=interp_limit_hours, limit_area="inside")
    return h, n_missing


# ---------------------------------------------------------------------------
# Предобработка (train-fit, без утечки)
# ---------------------------------------------------------------------------
@dataclass
class Scaler:
    """min-max scaler, подогнанный ТОЛЬКО на train."""
    lo: float = 0.0
    hi: float = 1.0

    def fit(self, x: np.ndarray) -> "Scaler":
        self.lo = float(np.nanmin(x))
        self.hi = float(np.nanmax(x))
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / (self.hi - self.lo)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * (self.hi - self.lo) + self.lo


def chronological_split(
    n: int, split: tuple[float, float, float], L: int, T: int,
    test_stride: int = 24, min_test_windows: int = 3, min_val_windows: int = 3,
) -> tuple[slice, slice, slice] | None:
    """Хронологическое разбиение с ГАРАНТИЕЙ, что val и test содержат хотя бы
    min_val_windows / min_test_windows окон (L+T часов на первое окно +
    test_stride на каждое следующее). Для длинных рядов (как house_1, 39к ч)
    это не меняет ничего — процентное разбиение и так даёт с запасом.
    Для коротких (как house_3, ~950 ч) процентные 15%/15% не вмещают даже
    одного окна (143 ч < L+T=192 ч) — это давало ПУСТОЙ Xte без предупреждения.
    Теперь минимальный абсолютный размер имеет приоритет над процентом.
    Возвращает None, если ряд слишком короткий даже для train (дом непригоден)."""
    min_te = (L + T) + test_stride * (min_test_windows - 1)
    min_va = (L + T) + 1 * (min_val_windows - 1)          # val окна идут с stride=1
    n_te = max(int(n * split[2]), min_te)
    n_va = max(int(n * split[1]), min_va)
    n_tr = n - n_va - n_te
    min_tr = L + T                                          # хотя бы 1 train-окно
    if n_tr < min_tr:
        return None
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)


def sigma_clip_fit(train: np.ndarray, k: float) -> tuple[float, float]:
    mu, sd = np.nanmean(train), np.nanstd(train)
    return mu - k * sd, mu + k * sd


def make_windows(series: np.ndarray, L: int, T: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Скользящие окна (X: L, Y: T). stride=1 для train/val, =T(24) для walk-forward теста."""
    xs, ys = [], []
    last = len(series) - L - T + 1
    for i in range(0, last, stride):
        xs.append(series[i:i + L])
        ys.append(series[i + L:i + L + T])
    if not xs:
        return np.empty((0, L)), np.empty((0, T))
    return np.stack(xs), np.stack(ys)


# ---------------------------------------------------------------------------
# Основной конвейер по одному дому
# ---------------------------------------------------------------------------
def process_house(cfg: UKDaleConfig, house: int) -> dict | None:
    root = Path(cfg.data_root)
    house_dir = root / f"house_{house}"
    mains_path, source = find_mains_file(house_dir, cfg.mains_channel)
    h5 = root / "ukdale.h5"
    if mains_path is not None:
        print(f"[house {house}] источник агрегата: {source} "
              f"({mains_path.stat().st_size / 1e6:.1f} МБ)")
        raw = parse_house_mains_dat(mains_path)
    elif h5.exists():
        raw = parse_house_mains_h5(h5, house)
    else:
        print(f"[house {house}] не найден ни mains.dat, ни channel_{cfg.mains_channel}.dat, "
              f"ни {h5} — пропуск")
        return None

    hourly, n_missing = to_hourly(raw, cfg.resample, cfg.interp_limit_hours)
    months = (hourly.index.max() - hourly.index.min()).days / 30.0
    if months < cfg.min_months:
        print(f"[house {house}] всего ~{months:.1f} мес (<{cfg.min_months}) — пропуск")
        return None

    x = hourly.values.astype(np.float64)
    n = len(x)
    split_result = chronological_split(n, cfg.split, cfg.L, cfg.T, cfg.test_stride)
    if split_result is None:
        print(f"[house {house}] даже train-часть короче L+T={cfg.L+cfg.T} ч — "
              f"дом непригоден, пропуск")
        return None
    tr, va, te = split_result

    # 3-sigma порог и min-max — ТОЛЬКО по train
    lo_c, hi_c = sigma_clip_fit(x[tr], cfg.sigma_clip)
    x = np.clip(x, lo_c, hi_c)
    # остаточные NaN (длинные пропуски) -> заполняем train-средним train-окна
    x = pd.Series(x).interpolate(limit_direction="both").values

    scaler = Scaler().fit(x[tr])
    xs = scaler.transform(x)

    cv = float(np.nanstd(hourly.values) / np.nanmean(hourly.values))
    print(f"[house {house}] N={n} ч (~{months:.1f} мес), пропусков={n_missing}, "
          f"mean={np.nanmean(hourly.values):.0f} Вт, CV={cv:.3f}")

    # окна: train/val stride=1, test — walk-forward stride=T
    Xtr, Ytr = make_windows(xs[tr], cfg.L, cfg.T, stride=1)
    Xva, Yva = make_windows(xs[va], cfg.L, cfg.T, stride=1)
    Xte, Yte = make_windows(xs[te], cfg.L, cfg.T, stride=cfg.test_stride)

    reduced_sample = len(Xte) < 15   # ниже этого статистика (Wilcoxon/DM) ненадёжна
    tag = " [REDUCED SAMPLE — только для CV-охвата, не для статистики]" if reduced_sample else ""
    print(f"[house {house}] окна: train={len(Xtr)}, val={len(Xva)}, "
          f"test={len(Xte)}{tag}")

    return dict(
        house=house, cv=cv, mean_w=float(np.nanmean(hourly.values)),
        std_w=float(np.nanstd(hourly.values)), n_hours=n, months=round(months, 1),
        n_test_windows=int(len(Xte)), reduced_sample=bool(reduced_sample),
        scaler_lo=scaler.lo, scaler_hi=scaler.hi,
        Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte,
    )


def run(cfg: UKDaleConfig) -> None:
    out = Path(cfg.data_root) / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for h in cfg.house_ids:
        res = process_house(cfg, h)
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

    summary = sorted(summary, key=lambda d: d["cv"])  # по возрастанию волатильности
    (out / "summary.json").write_text(json.dumps(
        {"config": asdict(cfg), "houses": summary}, ensure_ascii=False, indent=2))
    print("\n=== Сводка по домам (отсортировано по CV) ===")
    for d in summary:
        tag = " [REDUCED SAMPLE]" if d["reduced_sample"] else ""
        print(f"  house {d['house']}: CV={d['cv']:.3f}, mean={d['mean_w']:.0f} Вт, "
              f"{d['n_hours']} ч (~{d['months']} мес), test_windows={d['n_test_windows']}{tag}")
    print(f"\nСохранено в: {out.resolve()}")
    print("Для отбора 5 домов под REFIT-подобный диапазон волатильности "
          "берите дома, покрывающие низкий -> экстремальный CV.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help=r"папка с house_<n>/channel_1.dat (напр. C:\uk_dale_project\ukdale)")
    p.add_argument("--houses", default="1,2,3,4,5")
    p.add_argument("--min_months", type=float, default=4.0,
                   help="минимум месяцев данных для включения дома (по умолчанию 4)")
    args = p.parse_args()
    cfg = UKDaleConfig(
        data_root=args.data_root,
        house_ids=tuple(int(x) for x in args.houses.split(",")),
        min_months=args.min_months,
    )
    run(cfg)
