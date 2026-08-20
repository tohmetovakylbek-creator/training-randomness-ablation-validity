"""
models/vmd.py
=============
Каузальная вариационная декомпозиция мод (VMD), применяемая ТОЛЬКО к входному
скользящему окну (L=168 ч), а не ко всему ряду — защита от утечки времени
(Feng et al., 2025). Параметры по диплому: K=6, alpha=2000.

VMD считается на CPU (vmdpy) при подготовке батча; результат кэшируется.
"""
from __future__ import annotations

import numpy as np
from vmdpy import VMD

# Параметры по диплому (раздел 2.2)
K_MODES = 6
ALPHA = 2000.0
TAU = 0.0          # шумовая толерантность (0 = строгое восстановление)
DC = 0             # без принудительной DC-компоненты
INIT = 1           # равномерная инициализация центральных частот
TOL = 1e-7


def vmd_decompose(window: np.ndarray, K: int = K_MODES, alpha: float = ALPHA) -> np.ndarray:
    """Раскладывает 1D окно длины L на K мод. Возвращает массив (K, L).

    vmdpy требует чётную длину сигнала; при нечётной — дублируем последний
    отсчёт и затем обрезаем, чтобы не вносить фазовый сдвиг."""
    x = np.asarray(window, dtype=np.float64)
    L = len(x)
    pad = (L % 2 == 1)
    if pad:
        x = np.concatenate([x, x[-1:]])
    u, _, _ = VMD(x, alpha, TAU, K, DC, INIT, TOL)  # u: (K, len)
    if pad:
        u = u[:, :L]
    # сумма мод может слегка отличаться от сигнала -> относим остаток к ВЧ-моде,
    # чтобы сохранить точное восстановление Σu_k = x (важно для агрегации)
    resid = x[:L] - u.sum(axis=0)
    u[-1] = u[-1] + resid
    return u.astype(np.float32)


def vmd_batch(windows: np.ndarray, K: int = K_MODES, alpha: float = ALPHA) -> np.ndarray:
    """Батч окон (B, L) -> (B, K, L). Тяжёлая CPU-операция, кэшируйте результат."""
    out = np.empty((windows.shape[0], K, windows.shape[1]), dtype=np.float32)
    for i in range(windows.shape[0]):
        out[i] = vmd_decompose(windows[i], K, alpha)
    return out
