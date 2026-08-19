from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.vmd import vmd_batch, K_MODES
from models.vmd_patchtst_aswa import VMDPatchTSTASWA, count_params
from evaluate import inverse, metrics, per_window_abs_error

SEEDS = (42, 7, 13, 99, 2025)

def _cache_vmd(X, cache_path, K):
    """Кэширует VMD-разложение X (форма (n_windows, L)) в cache_path.
    ПРОВЕРЯЕТ, что кэш соответствует ТЕКУЩЕМУ X (число окон и длина окна) —
    если .npz дома был пересоздан с другим числом окон (напр. после смены
    параметров разбиения, как было с house_3), старый кэш больше не подходит.
    Без этой проверки get() тихо вернул бы несовместимый по форме массив,
    что уже один раз привело к RuntimeError/AssertionError на house_2/house_3."""
    if cache_path.exists():
        cached = np.load(cache_path)["modes"]
        if cached.shape[0] == X.shape[0] and cached.shape[-1] == X.shape[-1]:
            return cached
        print(f"    [VMD-кэш] {cache_path.name}: форма не совпадает "
              f"(кэш {cached.shape} vs текущие данные {X.shape}) — пересчитываю")
    modes = vmd_batch(X.astype(np.float32), K=K)
    np.savez_compressed(cache_path, modes=modes)
    return modes

def load_house(npz_path, cache_dir, K=K_MODES):
    d = np.load(npz_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    out = {}
    for split in ("tr", "va", "te"):
        X, Y = d[f"X{split}"], d[f"Y{split}"]
        modes = _cache_vmd(X, cache_dir / f"{stem}_{split}_vmd.npz", K)
        out[split] = (X.astype(np.float32), modes, Y.astype(np.float32))
    out["scaler"] = (float(d["scaler_lo"]), float(d["scaler_hi"]))
    return out

def train_one(house_data, seed, device="cpu", epochs=60, batch_size=64, model_kwargs=None,
              return_repr=False, ckpt_path=None, return_model=False):
    """ДОБАВЛЕНО (ревизия EAAI): ckpt_path — если файл есть, веса грузятся вместо
    обучения; иначе модель обучается и лучшие веса сохраняются туда. Логика
    обучения, оптимизатор, расписание и критерий отбора best_state не изменены,
    поэтому при том же seed воспроизводятся прежние числа. return_model — вернуть
    модель для интервенционных экспериментов без переобучения."""
    torch.manual_seed(seed); np.random.seed(seed)
    model_kwargs = model_kwargs or {}
    Xtr, Mtr, Ytr = house_data["tr"]
    Xva, Mva, Yva = house_data["va"]
    Xte, Mte, Yte = house_data["te"]

    def loader(X, M, Y, shuffle):
        ds = TensorDataset(torch.tensor(M), torch.tensor(X), torch.tensor(Y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_dl = loader(Xtr, Mtr, Ytr, True)
    model = VMDPatchTSTASWA(**model_kwargs).to(device)

    lo, hi = house_data["scaler"]
    Yva_w = inverse(Yva, lo, hi)

    ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
    loaded = False
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        best_val = float(state.get("val_mae", float("nan")))
        loaded = True

    if not loaded:
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
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
            val_mae = metrics(inverse(pv, lo, hi), Yva_w)["MAE"]
            if val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": best_state, "val_mae": best_val, "seed": seed,
                        "model_kwargs": model_kwargs, "scaler": [lo, hi]}, ckpt_path)

    model.eval()
    with torch.no_grad():
        if return_repr:
            o = model(torch.tensor(Mte).to(device), torch.tensor(Xte).to(device), return_repr=True)
            pte, repr_ = o["y"].cpu().numpy(), o["repr"].cpu().numpy()
        else:
            pte = model(torch.tensor(Mte).to(device), torch.tensor(Xte).to(device))["y"].cpu().numpy()
            repr_ = None
    pred_w, true_w = inverse(pte, lo, hi), inverse(Yte, lo, hi)
    res = {"pred_w": pred_w, "true_w": true_w, "val_mae": best_val,
           "n_params": count_params(model), "loaded_from_ckpt": loaded}
    if return_repr:
        res["repr"], res["modes_te"] = repr_, Mte
    if return_model:
        res["model"] = model
    return res

def train_ensemble(house_data, device="cpu", epochs=60, model_kwargs=None, seeds=SEEDS,
                   ckpt_dir=None, ckpt_tag="full"):
    preds, val_maes = [], []
    true_w = None
    for s in seeds:
        ckpt = None if ckpt_dir is None else Path(ckpt_dir) / f"{ckpt_tag}_seed{s}.pt"
        r = train_one(house_data, s, device, epochs, model_kwargs=model_kwargs, ckpt_path=ckpt)
        preds.append(r["pred_w"]); val_maes.append(r["val_mae"]); true_w = r["true_w"]
    preds = np.stack(preds)
    ens = preds.mean(axis=0)
    return {
        "per_seed": preds, "ensemble": ens, "true_w": true_w,
        "per_seed_metrics": [metrics(p, true_w) for p in preds],
        "ensemble_metrics": metrics(ens, true_w),
        "ensemble_window_err": per_window_abs_error(ens, true_w),
    }
