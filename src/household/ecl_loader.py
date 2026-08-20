"""
ecl_loader.py
=============
Загрузка и предобработка ВТОРОГО домена (ECL — основной; ETTh1 — технический
sanity check) в точности тем же протоколом, что UK-DALE / REFIT / SHEERM.

Единственные функции предобработки берутся из ukdale_loader — это и есть
гарантия единого пайплайна, на которую ссылается §4 рукописи:
    to_hourly, Scaler, chronological_split, sigma_clip_fit, make_windows

ECL:    клиент = единица репликации (аналог домохозяйства).
        Ожидается файл ECL.csv из репозитория Informer/ETDataset:
        колонка `date` + 321 колонка клиентов (`0`...`319`, `OT`),
        26304 часовых отсчёта, значения — кВт·ч за час.
ETTh1:  один ряд (колонка OT). Единицы репликации нет, поэтому ETTh1
        НЕ входит в разложение дисперсии и мета-анализ; он нужен только
        чтобы убедиться, что пайплайн переносится (см. prereg, критерии S1-S3).

ВАЖНО (сопоставимость с основным исследованием):
    доля seed-дисперсии зависит от числа тестовых окон W механически
    (window-компонента делится на W, seed-компонента — только на S).
    У клиента ECL после сплита получается ~215 окон против 78-171 в основном
    исследовании. Поэтому по умолчанию test усекается до фиксированного
    W = --match_windows (120, медиана основного исследования), а полный
    вариант считается через --match_windows 0 как sensitivity.
    Усечение берёт САМУЮ РАННЮЮ часть теста (примыкающую к val), чтобы правило
    не зависело от данных.

Порядок работы:
    1) python ecl_select_clients.py --csv data/ECL.csv        -> ecl_selection.json
    2) python ecl_loader.py --csv data/ECL.csv --selection ecl_selection.json
    3) python ecl_loader.py --dataset etth1 --csv data/ETTh1.csv   (sanity check)

Выход: <out_dir>/client_<id>.npz с ключами Xtr,Ytr,Xva,Yva,Xte,Yte,scaler_lo,
scaler_hi — тот же формат, что house_<n>.npz, поэтому train.py/evaluate.py
править не нужно.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from ukdale_loader import (
    to_hourly, Scaler, chronological_split, sigma_clip_fit, make_windows,
)


# ---------------------------------------------------------------------------
# Конфигурация (значения по умолчанию = пре-спецификация, см. prereg_ecl.docx)
# ---------------------------------------------------------------------------
@dataclass
class ECLConfig:
    csv_path: str
    dataset: str = "ecl"                     # 'ecl' | 'etth1'
    clients: tuple[str, ...] = ()            # пусто -> берём из selection.json
    L: int = 168
    T: int = 24
    test_stride: int = 24
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    resample: str = "1h"
    interp_limit_hours: int = 6
    sigma_clip: float = 3.0
    match_windows: int = 120                 # 0 = не усекать (sensitivity)
    # --- пре-спецификация отбора клиентов (используется ecl_select_clients.py)
    max_zero_frac_train: float = 0.01        # доля нулей в train
    max_flat_run: int = 48                   # макс. длина плоского прогона, ч
    n_clients: int = 17                      # столько же, сколько домохозяйств
    select_seed: int = 20260809
    out_dir: str = "processed_ecl"


# ---------------------------------------------------------------------------
# Чтение широкого CSV
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wide_csv(csv_path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """ECL.csv / ETTh1.csv -> DataFrame с tz-aware (UTC) часовым индексом.

    tz-aware обязателен: ukdale_loader.to_hourly строит date_range(tz='UTC'),
    и при наивном индексе reindex дал бы сплошные NaN без единой ошибки."""
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"В {csv_path} нет колонки '{date_col}'. "
                         f"Первые колонки: {list(df.columns)[:5]}")
    idx = pd.to_datetime(df[date_col])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize("UTC")
    else:
        idx = idx.dt.tz_convert("UTC")
    df = df.drop(columns=[date_col])
    df.index = pd.DatetimeIndex(idx)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.astype(np.float64)


def hours_needed(L: int, T: int, stride: int, n_windows: int) -> int:
    """Сколько часов теста нужно ровно для n_windows окон при данном шаге."""
    return L + T + stride * (n_windows - 1)


def max_flat_run(x: np.ndarray) -> int:
    """Длина самого длинного прогона идентичных соседних значений (в отсчётах).
    Прогон из m одинаковых значений даёт m-1 нулевых разностей -> возвращаем m."""
    if len(x) < 2:
        return len(x)
    d = np.diff(x)
    flat = (d == 0)
    best = cur = 0
    for f in flat:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best + 1 if best else 1


# ---------------------------------------------------------------------------
# Диагностика пригодности клиента (используется и отбором, и загрузчиком)
# ---------------------------------------------------------------------------
def client_diagnostics(series: pd.Series, cfg: ECLConfig) -> dict:
    hourly, n_missing = to_hourly(series, cfg.resample, cfg.interp_limit_hours)
    x = hourly.values.astype(np.float64)
    n = len(x)

    sp = chronological_split(n, cfg.split, cfg.L, cfg.T, cfg.test_stride)
    if sp is None:
        return dict(n_hours=n, eligible=False, reason="ряд короче train-окна")
    tr, va, te = sp

    x_filled = pd.Series(x).interpolate(limit_direction="both").values
    n_te_hours = te.stop - te.start
    n_te_windows = max(0, (n_te_hours - cfg.L - cfg.T) // cfg.test_stride + 1)

    zero_frac_train = float(np.mean(x_filled[tr] == 0.0))
    flat = int(max_flat_run(x_filled))
    test_std = float(np.std(x_filled[te]))
    mean_val = float(np.mean(x_filled))
    cv = float(np.std(x_filled) / mean_val) if mean_val > 0 else float("inf")

    need = cfg.match_windows if cfg.match_windows > 0 else 1
    reasons = []
    if zero_frac_train > cfg.max_zero_frac_train:
        reasons.append(f"нулей в train {zero_frac_train:.3%} > {cfg.max_zero_frac_train:.1%}")
    if flat > cfg.max_flat_run:
        reasons.append(f"плоский прогон {flat} ч > {cfg.max_flat_run} ч")
    if test_std <= 0:
        reasons.append("нулевая дисперсия в test")
    if n_te_windows < need:
        reasons.append(f"тестовых окон {n_te_windows} < {need}")
    if not np.isfinite(cv):
        reasons.append("неопределённый CV")

    return dict(
        n_hours=int(n), n_missing=int(n_missing), n_test_windows=int(n_te_windows),
        zero_frac_train=round(zero_frac_train, 6), max_flat_run_h=flat,
        mean=round(mean_val, 4), cv=round(cv, 4), test_std=round(test_std, 6),
        eligible=len(reasons) == 0, reason="; ".join(reasons),
    )


# ---------------------------------------------------------------------------
# Основной конвейер по одному клиенту
# ---------------------------------------------------------------------------
def process_client(series: pd.Series, name: str, cfg: ECLConfig) -> dict | None:
    hourly, n_missing = to_hourly(series, cfg.resample, cfg.interp_limit_hours)
    x = hourly.values.astype(np.float64)
    n = len(x)

    sp = chronological_split(n, cfg.split, cfg.L, cfg.T, cfg.test_stride)
    if sp is None:
        print(f"[{name}] ряд короче train-окна — пропуск")
        return None
    tr, va, te = sp

    # 3-sigma порог и min-max — ТОЛЬКО по train (как в ukdale_loader)
    lo_c, hi_c = sigma_clip_fit(x[tr], cfg.sigma_clip)
    x = np.clip(x, lo_c, hi_c)
    x = pd.Series(x).interpolate(limit_direction="both").values

    scaler = Scaler().fit(x[tr])
    xs = scaler.transform(x)

    # --- выравнивание числа тестовых окон -----------------------------------
    te_full_hours = te.stop - te.start
    truncated = False
    if cfg.match_windows > 0:
        need = hours_needed(cfg.L, cfg.T, cfg.test_stride, cfg.match_windows)
        if te_full_hours < need:
            print(f"[{name}] теста {te_full_hours} ч < {need} ч для "
                  f"W={cfg.match_windows} — пропуск")
            return None
        te = slice(te.start, te.start + need)   # самая ранняя часть теста
        truncated = te_full_hours > need

    Xtr, Ytr = make_windows(xs[tr], cfg.L, cfg.T, stride=1)
    Xva, Yva = make_windows(xs[va], cfg.L, cfg.T, stride=1)
    Xte, Yte = make_windows(xs[te], cfg.L, cfg.T, stride=cfg.test_stride)

    # --- базовые прогнозы в исходных единицах -------------------------------
    span = scaler.hi - scaler.lo
    if len(Xte):
        last = Xte[:, -1][:, None]                       # persistence
        snaive = Xte[:, -cfg.T:]                         # сезонный naive (сутки)
        mae_const = float(np.mean(np.abs(Yte - last))) * span
        mae_snaive = float(np.mean(np.abs(Yte - snaive))) * span
    else:
        mae_const = mae_snaive = float("nan")
    xt = x[tr]
    mase_scale = float(np.mean(np.abs(xt[cfg.T:] - xt[:-cfg.T]))) if len(xt) > cfg.T else float("nan")

    cv = float(np.std(x) / np.mean(x)) if np.mean(x) > 0 else float("inf")
    print(f"[{name}] N={n} ч, пропусков={n_missing}, mean={np.mean(x):.3f}, "
          f"CV={cv:.3f} | окна: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}"
          f"{' (усечён)' if truncated else ''} | MAE_const={mae_const:.4f}, "
          f"MAE_snaive={mae_snaive:.4f}, MASE_scale={mase_scale:.4f}")

    return dict(
        client=name, cv=cv, mean=float(np.mean(x)), std=float(np.std(x)),
        n_hours=n, n_missing=int(n_missing),
        n_test_windows=int(len(Xte)), test_truncated=bool(truncated),
        test_windows_available=int(max(0, (te_full_hours - cfg.L - cfg.T)
                                       // cfg.test_stride + 1)),
        scaler_lo=scaler.lo, scaler_hi=scaler.hi,
        mae_const=mae_const, mae_snaive=mae_snaive, mase_scale=mase_scale,
        Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte,
    )


# ---------------------------------------------------------------------------
def run(cfg: ECLConfig, selection: dict | None = None) -> None:
    csv_path = Path(cfg.csv_path)
    df = read_wide_csv(csv_path)

    if cfg.dataset == "etth1":
        if "OT" not in df.columns:
            raise ValueError("В ETTh1.csv нет колонки OT.")
        names = ["OT"]
    elif cfg.clients:
        names = list(cfg.clients)
    elif selection is not None:
        names = list(selection["selected"])
    else:
        raise ValueError("Не заданы клиенты: укажите --selection или --clients.")

    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"Нет таких колонок в CSV: {missing[:10]}")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for name in names:
        res = process_client(df[name], name, cfg)
        if res is None:
            continue
        np.savez_compressed(
            out / f"client_{name}.npz",
            Xtr=res["Xtr"], Ytr=res["Ytr"], Xva=res["Xva"], Yva=res["Yva"],
            Xte=res["Xte"], Yte=res["Yte"],
            scaler_lo=res["scaler_lo"], scaler_hi=res["scaler_hi"],
        )
        summary.append({k: res[k] for k in
                        ("client", "cv", "mean", "std", "n_hours", "n_missing",
                         "n_test_windows", "test_truncated", "test_windows_available",
                         "mae_const", "mae_snaive", "mase_scale")})

    summary = sorted(summary, key=lambda d: d["cv"])
    meta = {
        "config": asdict(cfg),
        "csv_sha256": sha256_of(csv_path),
        "selection_file": selection.get("_path") if selection else None,
        "clients": summary,
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\n=== Сводка ({cfg.dataset}, отсортировано по CV) ===")
    for d in summary:
        print(f"  {d['client']}: CV={d['cv']:.3f}, mean={d['mean']:.3f}, "
              f"test_windows={d['n_test_windows']}"
              f"{' [усечён с ' + str(d['test_windows_available']) + ']' if d['test_truncated'] else ''}")
    ws = {d["n_test_windows"] for d in summary}
    print(f"\nЧисло тестовых окон одинаково у всех клиентов: {len(ws) == 1} {sorted(ws)}")
    print(f"Обработано клиентов: {len(summary)} из {len(names)}")
    print(f"Сохранено в: {out.resolve()}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help=r"путь к ECL.csv или ETTh1.csv")
    p.add_argument("--dataset", default="ecl", choices=["ecl", "etth1"])
    p.add_argument("--selection", default=None,
                   help="ecl_selection.json из ecl_select_clients.py")
    p.add_argument("--clients", default="",
                   help="список колонок через запятую (в обход selection.json)")
    p.add_argument("--match_windows", type=int, default=120,
                   help="фиксированное число тестовых окон; 0 = не усекать")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    sel = None
    if args.selection:
        sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        sel["_path"] = args.selection

    out_dir = args.out_dir or ("processed_etth1" if args.dataset == "etth1"
                               else "processed_ecl")
    cfg = ECLConfig(
        csv_path=args.csv,
        dataset=args.dataset,
        clients=tuple(c.strip() for c in args.clients.split(",") if c.strip()),
        match_windows=(0 if args.dataset == "etth1" else args.match_windows),
        out_dir=out_dir,
    )
    run(cfg, selection=sel)
