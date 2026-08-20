"""
evaluate.py
===========
Метрики качества прогноза в Ваттах (после inverse-transform из [0,1]).
Основные метрики работы — MAE и RMSE (MAPE/sMAPE приведены, но MAPE на данных
домохозяйств — артефактен из-за периодов низкого потребления; см. диплом 3.1).
"""
from __future__ import annotations
import numpy as np


def inverse(x01: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x01 * (hi - lo) + lo


def per_window_abs_error(pred_w: np.ndarray, true_w: np.ndarray) -> np.ndarray:
    """Средняя |ошибка| по каждому тестовому окну (для парных тестов): (n_windows,)."""
    return np.abs(pred_w - true_w).mean(axis=1)


def metrics(pred_w: np.ndarray, true_w: np.ndarray) -> dict:
    err = pred_w - true_w
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true_w - true_w.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denom = np.clip(np.abs(true_w), 1e-6, None)
    mape = float((np.abs(err) / denom).mean() * 100)
    smape = float((2 * np.abs(err) / np.clip(np.abs(pred_w) + np.abs(true_w), 1e-6, None)).mean() * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "sMAPE": smape}
