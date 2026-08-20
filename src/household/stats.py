"""
stats.py
========
Статистика значимости на ФИНАЛЬНЫХ прогнозах (исправляет «несвежую» таблицу
Вилкоксона из диплома). Тесты на парных поошибочных значениях по тестовым окнам:
  * Wilcoxon signed-rank (как в дипломе);
  * Diebold-Mariano (стандарт для сравнения прогнозов);
  * размер эффекта (rank-biserial для Вилкоксона + относительное улучшение).
Считается и по каждому дому, и пулингом (с пометкой, что пулинг раздувает n).
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def wilcoxon_effect(err_model: np.ndarray, err_base: np.ndarray) -> dict:
    """err_*: средняя |ошибка| по окну (n_windows,). Меньше = лучше."""
    diff = err_base - err_model            # >0 -> модель лучше
    try:
        w, p = stats.wilcoxon(err_model, err_base)
    except ValueError:
        w, p = float("nan"), 1.0
    n = len(diff)
    # rank-biserial effect size
    pos = (diff > 0).sum(); neg = (diff < 0).sum()
    rb = (pos - neg) / n if n else 0.0
    return {"W": float(w), "p_value": float(p), "rank_biserial": float(rb),
            "median_improvement_w": float(np.median(diff)),
            "rel_improvement_pct": float(diff.mean() / err_base.mean() * 100)}


def diebold_mariano(err_model: np.ndarray, err_base: np.ndarray, power: int = 1) -> dict:
    """DM-тест на равную точность. loss = |err|^power. H0: одинаковая точность."""
    d = np.abs(err_model) ** power - np.abs(err_base) ** power
    mean_d, n = d.mean(), len(d)
    # дисперсия с поправкой на автокорреляцию (h=1: только lag0 для walk-forward с непересек. окнами)
    var_d = d.var(ddof=1) / n
    dm = mean_d / np.sqrt(var_d) if var_d > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return {"DM_stat": float(dm), "p_value": float(p), "favors": "model" if mean_d < 0 else "baseline"}


def compare(model_err_by_house: dict, base_err_by_house: dict, base_name: str) -> dict:
    """Сравнение модели с одним бейзлайном по домам + пулинг."""
    out = {"per_house": {}, "baseline": base_name}
    all_m, all_b = [], []
    for h in model_err_by_house:
        em, eb = model_err_by_house[h], base_err_by_house[h]
        out["per_house"][h] = {"wilcoxon": wilcoxon_effect(em, eb),
                               "dm": diebold_mariano(em, eb)}
        all_m.append(em); all_b.append(eb)
    all_m, all_b = np.concatenate(all_m), np.concatenate(all_b)
    out["pooled"] = {"wilcoxon": wilcoxon_effect(all_m, all_b),
                     "dm": diebold_mariano(all_m, all_b),
                     "note": "пулинг по домам раздувает n; смотрите также per_house и размеры эффекта"}
    return out
