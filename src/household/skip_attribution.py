"""
skip_attribution.py
====================
Замечание руководителя №3: «Почему skip-connection работает лучше identity-
механизма, не объяснено механистически. Является ли это защитой от ошибок
декомпозиции или альтернативным путём обучения?»

Подход (gradient/feature attribution в широком смысле — здесь используется
поведенческая атрибуция через gate g и per-window ошибки, что интерпретируемее
и дешевле, чем integrated gradients, при этом отвечает на тот же вопрос):

Для каждого тестового окна извлекаются:
  - g       — значение gate (§3.4): g→1 доверяет VMD-пути, g→0 доверяет skip-пути
  - err_vmd = |y_vmd - y_true|.mean(), err_skip = |y_skip - y_true|.mean()
  - skip_advantage = err_vmd - err_skip  (>0 => skip-путь для этого окна лучше)
  - window_volatility = std(raw window x)  — локальная волатильность окна
  - window_cv = std(x)/mean(x)             — коэф. вариации окна (не путать с CV дома)
  - anomaly_score = max |z-score| внутри окна — есть ли резкий выброс
  - imf6_energy_frac = var(IMF6) / sum_k var(IMF_k) — доля «шумовой» моды
    (прокси того, насколько VMD «свалила» энергию в высокочастотный остаток —
    ближайшая доступная замена «качеству декомпозиции», так как каузальная VMD
    в этой реализации восстанавливает x точно по построению, Σu_k=x всегда,
    поэтому reconstruction error как признак недоступен — см. models/vmd.py)

Затем считаются корреляции (Spearman, устойчивее к выбросам, чем Pearson):
  g                vs window_volatility, anomaly_score, imf6_energy_frac
  skip_advantage   vs window_volatility, anomaly_score, imf6_energy_frac

Интерпретация:
  - Если g положительно коррелирует с волатильностью/аномальностью/imf6-долей
    => skip-путь ведёт себя как "защита от плохой декомпозиции/аномалий"
    (гипотеза A из замечания №3).
  - Если корреляции слабые/отсутствуют => skip-путь работает равномерно,
    похоже на альтернативный путь обучения, а не защитный механизм (гипотеза B).

Результат: JSON с корреляциями + scatter-график (2x3) для включения в статью.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

from train import load_house, SEEDS as DEFAULT_SEEDS
from models.vmd_patchtst_aswa import VMDPatchTSTASWA
from evaluate import inverse


def train_for_attribution(house_data, seed: int, device: str = "cpu",
                          epochs: int = 60, batch_size: int = 64):
    """Обучает full-модель (как train_one из train.py), но возвращает саму
    модель (не только метрики) — нужна для forward(..., return_gate=True)
    на тестовых окнах после обучения."""
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F

    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Mtr, Ytr = house_data["tr"]
    Xva, Mva, Yva = house_data["va"]

    tr_dl = DataLoader(TensorDataset(torch.tensor(Mtr), torch.tensor(Xtr), torch.tensor(Ytr)),
                       batch_size=batch_size, shuffle=True)
    model = VMDPatchTSTASWA().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    lo, hi = house_data["scaler"]
    from evaluate import metrics as _metrics
    Yva_w = inverse(Yva, lo, hi)
    best_val, best_state = float("inf"), None
    Mva_t, Xva_t = torch.tensor(Mva).to(device), torch.tensor(Xva).to(device)

    for ep in range(epochs):
        model.train()
        for mb, xb, yb in tr_dl:
            mb, xb, yb = mb.to(device), xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(mb, xb)
            loss, _ = model.loss(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Mva_t, Xva_t)["y"].cpu().numpy()
        val_mae = _metrics(inverse(pv, lo, hi), Yva_w)["MAE"]
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


def compute_window_features(Xte: np.ndarray, Mte: np.ndarray) -> dict:
    """Xte: (N, L) исходные окна (нормализованные [0,1]). Mte: (N, K, L) VMD-моды."""
    volatility = Xte.std(axis=1)                                  # (N,)
    mean_ = Xte.mean(axis=1)
    cv = volatility / np.clip(np.abs(mean_), 1e-6, None)           # (N,)
    z = (Xte - Xte.mean(axis=1, keepdims=True)) / np.clip(Xte.std(axis=1, keepdims=True), 1e-6, None)
    anomaly_score = np.abs(z).max(axis=1)                          # (N,)

    mode_var = Mte.var(axis=2)                                     # (N, K)
    imf6_frac = mode_var[:, -1] / np.clip(mode_var.sum(axis=1), 1e-9, None)  # (N,)

    return {"window_volatility": volatility, "window_cv": cv,
            "anomaly_score": anomaly_score, "imf6_energy_frac": imf6_frac}


def run_attribution(house_data, house_id: int, device: str, epochs: int,
                    seeds, out_dir: Path):
    Xte, Mte, Yte = house_data["te"]
    lo, hi = house_data["scaler"]
    true_w = inverse(Yte, lo, hi)
    n = Xte.shape[0]

    gates, errs_vmd, errs_skip = [], [], []
    for s in seeds:
        model = train_for_attribution(house_data, s, device, epochs)
        with torch.no_grad():
            out = model(torch.tensor(Mte).to(device), torch.tensor(Xte).to(device),
                       return_gate=True)
        g = out["gate"].cpu().numpy().reshape(-1)                  # (N,)
        y_vmd_w = inverse(out["y_vmd"].cpu().numpy(), lo, hi)
        y_skip_w = inverse(out["y_skip"].cpu().numpy(), lo, hi)
        err_vmd = np.abs(y_vmd_w - true_w).mean(axis=1)            # (N,)
        err_skip = np.abs(y_skip_w - true_w).mean(axis=1)
        gates.append(g); errs_vmd.append(err_vmd); errs_skip.append(err_skip)
        print(f"  [house {house_id}] seed {s}: mean gate={g.mean():.3f} "
              f"(1=trusts VMD-path, 0=trusts skip-path)")

    g_mean = np.mean(gates, axis=0)                                 # (N,) усреднено по сидам
    err_vmd_mean = np.mean(errs_vmd, axis=0)
    err_skip_mean = np.mean(errs_skip, axis=0)
    skip_advantage = err_vmd_mean - err_skip_mean                   # >0 => skip лучше на этом окне

    feats = compute_window_features(Xte, Mte)

    corrs = {}
    for target_name, target in (("gate", g_mean), ("skip_advantage", skip_advantage)):
        corrs[target_name] = {}
        for feat_name, feat in feats.items():
            rho, p = sstats.spearmanr(target, feat)
            corrs[target_name][feat_name] = {"spearman_rho": float(rho), "p_value": float(p)}
            print(f"  [house {house_id}] {target_name} vs {feat_name}: "
                  f"rho={rho:.3f}, p={p:.4f}")

    out_dir.mkdir(exist_ok=True, parents=True)
    result = {
        "house": house_id, "n_test_windows": int(n),
        "gate_mean": float(g_mean.mean()), "gate_std": float(g_mean.std()),
        "correlations": corrs,
    }
    (out_dir / f"skip_attribution_house{house_id}.json").write_text(json.dumps(result, indent=2))

    # scatter-график 2x3 (gate/skip_advantage x 3 признака)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=150)
        targets = {"gate": g_mean, "skip_advantage": skip_advantage}
        feat_items = list(feats.items())
        for row, (tname, tval) in enumerate(targets.items()):
            for col, (fname, fval) in enumerate(feat_items):
                ax = axes[row, col]
                ax.scatter(fval, tval, s=18, alpha=0.6, color="#4C72B0")
                rho = corrs[tname][fname]["spearman_rho"]
                ax.set_title(f"{tname} vs {fname}\n(Spearman \u03c1={rho:.2f})", fontsize=9)
                ax.set_xlabel(fname, fontsize=8)
                ax.set_ylabel(tname, fontsize=8)
                ax.tick_params(labelsize=7)
        plt.tight_layout()
        fig_path = out_dir / f"skip_attribution_house{house_id}.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {fig_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    print(f"  Saved: {out_dir / f'skip_attribution_house{house_id}.json'}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed", required=True)
    p.add_argument("--houses", default="1,2,4,5", help="дома для анализа (через запятую)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", default="42,7,13,99,2025")
    p.add_argument("--out", default="results_skip_attribution")
    args = p.parse_args()

    proc = Path(args.processed)
    cache = proc / "vmd_cache"
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out)

    for h in (int(x) for x in args.houses.split(",")):
        print(f"\n===== SKIP ATTRIBUTION: house {h} =====")
        hd = load_house(proc / f"house_{h}.npz", cache)
        run_attribution(hd, h, args.device, args.epochs, seeds, out_dir)
