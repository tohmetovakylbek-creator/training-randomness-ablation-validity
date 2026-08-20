"""
intervention_legacy.py
======================
Тот же интервенционный эксперимент, что и в intervention.py, но на ИСТОРИЧЕСКОЙ
реализации (train_vmd_patchtst_ukdale.py) — той, что породила исходный результат.

Зачем. Если перестановка mode embeddings ломает прогноз в исторической
реализации и НЕ ломает в unified pipeline, диссоциация локализуется до
конкретного различия между пайплайнами, а не остаётся необъяснённой. Главный
подозреваемый — форма вспомогательного лосса: исторический требует, чтобы
КАЖДАЯ мода по отдельности восстанавливала полную цель (тогда идентичность моды
необходима), unified — чтобы цель восстанавливала СУММА мод (тогда она избыточна).

Переобучение НЕ требуется: исторический скрипт уже сохранял веса в
    <output_dir>/vmd_patchtst_aswa_house{H}_seed{S}.pt
Адаптер их читает. Если чекпойнтов не осталось — их придётся получить повторным
прогоном исходного скрипта, этот адаптер сам не обучает.

Отличия, которые адаптер обязан воспроизвести точно (иначе загруженные веса
будут применяться не к тем входам, что при обучении):
  * моды повторно нормируются: (modes - scaler_lo) / (scaler_hi - scaler_lo);
  * моды берутся из <processed>/vmd_precomputed/house{H}_{split}_modes.npy,
    а не из VMD-кэша unified pipeline (это РАЗНЫЕ разложения: в unified остаток
    сносится в последнюю моду ради точного Sum(u_k) = x);
  * порядок аргументов forward: (x_raw, modes), а не (modes, raw).

Запуск:
    python intervention_legacy.py \
        --legacy_script <путь>/train_vmd_patchtst_ukdale.py \
        --processed <путь>/processed \
        --ckpt_dir results/ukdale_vmd_patchtst_aswa \
        --houses 1,2,4 --dataset ukdale_legacy --device cuda \
        --out results/intervention_legacy
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluate import inverse, metrics, per_window_abs_error
from intervention import (
    cyclic_perm, random_perms, swapped_embeddings,
    probe_accuracy, moving_block_bootstrap_ci,
)

SEEDS = (42, 7, 13, 99, 2025)


# ------------------------------------------------------- загрузка legacy-модуля
def load_legacy_module(path: Path):
    """Импортирует исторический скрипт как модуль, не запуская main()."""
    spec = importlib.util.spec_from_file_location("legacy_vmd", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["legacy_vmd"] = mod
    spec.loader.exec_module(mod)
    for cls in ("VMDPatchTSTASWA", "PatchTSTEncoder", "ASWAModule"):
        if not hasattr(mod, cls):
            raise AttributeError(f"В {path.name} нет класса {cls} — не тот файл?")
    return mod


# --------------------------------------------------------------- прокси-обёртка
class _EncoderProxy:
    """Даёт исторический encoder под тем же именем атрибута, что ждёт
    swapped_embeddings из intervention.py (vmd_encoder.mode_embed)."""

    def __init__(self, enc):
        object.__setattr__(self, "_enc", enc)

    @property
    def mode_embed(self):
        return self._enc.mode_embeddings


class LegacyWrapper(nn.Module):
    """Приводит историческую модель к интерфейсу unified-модели:

        forward(modes, raw, return_gate=..., return_repr=...) -> dict
        .vmd_encoder.mode_embed

    Внутри вызывает ровно те же операции, что исторический forward, поэтому
    числа совпадают с исходным скриптом бит в бит.
    """

    def __init__(self, legacy_model):
        super().__init__()
        self.legacy = legacy_model
        self.K = legacy_model.K

    @property
    def vmd_encoder(self):
        return _EncoderProxy(self.legacy.encoder)

    def _encode_modes(self, modes, return_repr=False):
        """Повторяет цикл по модам из исторического forward.
        Дополнительно (для probing) возвращает mean-pool представлений энкодера,
        аналогично return_repr в unified PatchTSTEncoder."""
        enc = self.legacy.encoder
        preds, reprs = [], []
        for k in range(self.K):
            x = modes[:, k, :]
            B = x.shape[0]
            z = x.view(B, enc.n_patches, enc.patch_size)
            z = enc.patch_embedding(z)
            z = z + enc.pos_embedding + enc.mode_embeddings[k].unsqueeze(0).unsqueeze(0)
            h = enc.encoder(z)                      # (B, n_patches, d_model)
            preds.append(enc.output_proj(h.reshape(B, -1)))
            if return_repr:
                reprs.append(h.mean(dim=1))         # (B, d_model)
        u_hat = torch.stack(preds, dim=1)           # (B, K, T)
        repr_ = torch.stack(reprs, dim=1) if return_repr else None
        return u_hat, repr_

    def forward(self, modes, raw, return_gate=False, return_repr=False):
        u_hat, repr_ = self._encode_modes(modes, return_repr=return_repr)
        y_vmd = self.legacy.aswa(u_hat)
        y_skip = self.legacy.skip(raw)
        gate = torch.sigmoid(self.legacy.gate_layer(torch.cat([y_vmd, y_skip], dim=-1)))
        y = gate * y_vmd + (1 - gate) * y_skip
        out = {"y": y, "y_vmd": y_vmd, "y_skip": y_skip, "u_hat": u_hat}
        if return_gate:
            out["gate"] = gate
        if return_repr:
            out["repr"] = repr_
        return out


# ------------------------------------------------------------------- данные
def load_house_legacy(processed: Path, house: int):
    """Данные в исторической раскладке: processed/house_{H}.npz +
    processed/vmd_precomputed/house{H}_{split}_modes.npy."""
    d = np.load(processed / f"house_{house}.npz")
    lo, hi = float(d["scaler_lo"]), float(d["scaler_hi"])
    vmd_dir = processed / "vmd_precomputed"
    out = {"scaler": (lo, hi)}
    for split in ("tr", "va", "te"):
        p = vmd_dir / f"house{house}_{split}_modes.npy"
        if not p.exists():
            raise FileNotFoundError(f"Не найдены исторические моды: {p}")
        modes = np.load(p).astype(np.float32)
        # ТА ЖЕ повторная нормировка, что в train_vmd_patchtst_ukdale.py
        modes = (modes - lo) / (hi - lo)
        out[split] = (d[f"X{split}"].astype(np.float32), modes, d[f"Y{split}"].astype(np.float32))
    return out


@torch.no_grad()
def forward_test(model, Mte, Xte, device, batch=256, return_repr=False):
    ys, yv, gs, rs = [], [], [], []
    for i in range(0, len(Mte), batch):
        mb = torch.tensor(Mte[i:i + batch]).to(device)
        xb = torch.tensor(Xte[i:i + batch]).to(device)
        o = model(mb, xb, return_gate=True, return_repr=return_repr)
        ys.append(o["y"].cpu().numpy())
        yv.append(o["y_vmd"].cpu().numpy())
        gs.append(o["gate"].cpu().numpy())
        if return_repr:
            rs.append(o["repr"].cpu().numpy())
    r = np.concatenate(rs) if return_repr else None
    return np.concatenate(ys), np.concatenate(yv), np.concatenate(gs), r


@torch.no_grad()
def forward_test_per_window_perm(model, Mte, Xte, device, rng, batch=256):
    K = Mte.shape[1]
    ys, yv = [], []
    for i in range(0, len(Mte), batch):
        m = Mte[i:i + batch].copy()
        for j in range(len(m)):
            p = rng.permutation(K)
            while tuple(p) == tuple(range(K)):
                p = rng.permutation(K)
            m[j] = m[j][p]
        o = model(torch.tensor(m).to(device), torch.tensor(Xte[i:i + batch]).to(device))
        ys.append(o["y"].cpu().numpy())
        yv.append(o["y_vmd"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(yv)


# ------------------------------------------------------------------- эксперимент
def run_house(legacy_mod, house_data, house, ckpt_dir, device="cpu", seeds=SEEDS,
              n_perms=50, block=24, seed_rng=0, K=6, L=168, T=24):
    Xte, Mte, Yte = house_data["te"]
    lo, hi = house_data["scaler"]
    true_w = inverse(Yte, lo, hi)
    rng = np.random.default_rng(seed_rng)

    conds = ["identity", "cyclic", "random", "zeroed", "shuffled_per_window"]
    acc_final = {c: [] for c in conds}
    acc_vmd = {c: [] for c in conds}
    gates, probes, used = [], [], []

    for s in seeds:
        ckpt = Path(ckpt_dir) / f"vmd_patchtst_aswa_house{house}_seed{s}.pt"
        if not ckpt.exists():
            print(f"    [seed {s}] чекпойнт не найден, пропускаю: {ckpt.name}")
            continue
        base = legacy_mod.VMDPatchTSTASWA(K=K, L=L, T=T)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = state.get("model", state)          # поддержка обоих форматов
        base.load_state_dict(state)
        model = LegacyWrapper(base).to(device).eval()
        used.append(s)

        orig = model.vmd_encoder.mode_embed.detach().clone()

        yf, yv, g, repr_ = forward_test(model, Mte, Xte, device, return_repr=True)
        acc_final["identity"].append(yf); acc_vmd["identity"].append(yv)
        gates.append(float(g.mean()))
        probes.append(probe_accuracy(repr_, seed=s))

        with swapped_embeddings(model, orig[cyclic_perm(K, 1)]):
            yf, yv, _, _ = forward_test(model, Mte, Xte, device)
        acc_final["cyclic"].append(yf); acc_vmd["cyclic"].append(yv)

        pf, pv = [], []
        for p in random_perms(K, n_perms, rng):
            with swapped_embeddings(model, orig[p]):
                a, b, _, _ = forward_test(model, Mte, Xte, device)
            pf.append(a); pv.append(b)
        acc_final["random"].append(np.mean(pf, axis=0))
        acc_vmd["random"].append(np.mean(pv, axis=0))

        with swapped_embeddings(model, torch.zeros_like(orig)):
            yf, yv, _, _ = forward_test(model, Mte, Xte, device)
        acc_final["zeroed"].append(yf); acc_vmd["zeroed"].append(yv)

        yf, yv = forward_test_per_window_perm(model, Mte, Xte, device, rng)
        acc_final["shuffled_per_window"].append(yf)
        acc_vmd["shuffled_per_window"].append(yv)

        del model, base
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not used:
        raise FileNotFoundError(f"Для house {house} не найдено ни одного чекпойнта в {ckpt_dir}")

    out = {"house": house, "K": K, "seeds_used": used,
           "n_test_windows": int(len(Yte)), "mean_gate": float(np.mean(gates)),
           "probe": {k: float(np.mean([p[k] for p in probes])) for k in probes[0]},
           "conditions": {}, "per_window": {}}

    base_err = {}
    for path_name, store in (("y_final", acc_final), ("y_vmd", acc_vmd)):
        for c in conds:
            ens = inverse(np.mean(store[c], axis=0), lo, hi)
            err = per_window_abs_error(ens, true_w)
            out["per_window"][f"{path_name}|{c}"] = err
            m = metrics(ens, true_w)
            out["conditions"][f"{path_name}|{c}"] = {"MAE": m["MAE"], "RMSE": m["RMSE"]}
            if c == "identity":
                base_err[path_name] = err

    for path_name in ("y_final", "y_vmd"):
        for c in conds:
            if c == "identity":
                continue
            d = out["per_window"][f"{path_name}|{c}"] - base_err[path_name]
            out["conditions"][f"{path_name}|{c}"]["delta_MAE_vs_identity"] = \
                moving_block_bootstrap_ci(d, block=block, seed=seed_rng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy_script", required=True, help="путь к train_vmd_patchtst_ukdale.py")
    ap.add_argument("--processed", required=True)
    ap.add_argument("--ckpt_dir", required=True,
                    help="папка с vmd_patchtst_aswa_house{H}_seed{S}.pt")
    ap.add_argument("--houses", default="1,2,4")
    ap.add_argument("--dataset", default="ukdale_legacy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_perms", type=int, default=50)
    ap.add_argument("--block", type=int, default=24)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--out", default="results/intervention_legacy")
    args = ap.parse_args()

    legacy_mod = load_legacy_module(Path(args.legacy_script))
    proc = Path(args.processed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    all_res, npz_store = {}, {}
    for h in [int(x) for x in args.houses.split(",")]:
        print(f"\n===== legacy house {h} =====")
        hd = load_house_legacy(proc, h)
        res = run_house(legacy_mod, hd, h, args.ckpt_dir, device=args.device,
                        n_perms=args.n_perms, block=args.block, K=args.K)
        tag = f"{args.dataset}_house_{h}"
        for k, v in res.pop("per_window").items():
            npz_store[f"{tag}|{k}"] = v
        all_res[tag] = res
        print(f"  probe_acc={res['probe']['probe_accuracy']:.3f} "
              f"(control={res['probe']['control_task_accuracy']:.3f})  "
              f"mean_gate={res['mean_gate']:.3f}  seeds={res['seeds_used']}")
        for c in ("cyclic", "random", "zeroed", "shuffled_per_window"):
            dv = res["conditions"][f"y_vmd|{c}"]["delta_MAE_vs_identity"]
            df = res["conditions"][f"y_final|{c}"]["delta_MAE_vs_identity"]
            print(f"  {c:20s} dMAE y_vmd={dv['mean']:+7.2f} "
                  f"[{dv['ci_low']:+.2f},{dv['ci_high']:+.2f}]  "
                  f"y_final={df['mean']:+7.2f} [{df['ci_low']:+.2f},{df['ci_high']:+.2f}]")
        (outdir / f"intervention_{args.dataset}.json").write_text(
            json.dumps(all_res, ensure_ascii=False, indent=2))
        np.savez_compressed(outdir / f"per_window_{args.dataset}.npz", **npz_store)

    print(f"\nСохранено: {outdir}/intervention_{args.dataset}.json")
    print("Сравнивайте с unified-прогоном: интерес представляет РАЗНИЦА "
          "dMAE(y_vmd) между двумя пайплайнами при сопоставимом probe accuracy.")


if __name__ == "__main__":
    main()
