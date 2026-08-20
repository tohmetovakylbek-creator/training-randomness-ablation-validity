"""
aggregation_check.py
====================
Проверяет конкретную гипотезу о причине схлопывания VMD-ветви.

Наблюдение: в unified-модели вспомогательный лосс требует, чтобы СУММА
помодовых прогнозов восстанавливала цель:

    l_aux = huber( u_hat.sum(dim=1), target )

а ASWA агрегирует их ВЫПУКЛОЙ комбинацией — softmax по модам, веса в сумме
дают единицу:

    y_vmd = sum_k w_k * u_hat_k ,   sum_k w_k = 1

Если обучение выполняет вспомогательное требование, то sum_k u_hat_k ~ target,
и тогда взвешенное среднее тех же слагаемых систематически меньше цели примерно
в число мод раз. То есть масштаб теряется не из-за плохого обучения, а
структурно: там, где нужна сумма, стоит среднее.

Скрипт сравнивает на тесте четыре агрегации помодовых прогнозов и обе ветви:
    sum_u    = sum_k u_hat_k        (то, что оптимизирует вспомогательный лосс)
    mean_u   = (1/K) sum_k u_hat_k
    y_vmd    = ASWA (softmax-веса)  (то, что идёт в fusion)
    K*y_vmd  = ASWA, умноженный на K (грубая проверка на масштабный множитель)
    y_skip, y_final

Если sum_u оказывается заметно лучше y_vmd, гипотеза подтверждена, и весь
разговор про mode identity надо вести уже поверх этого факта.

Обучения нет — читает те же чекпойнты, что intervention.py.

Запуск:
    python aggregation_check.py --processed "<...>\\processed" --houses 1,4 \\
        --dataset ukdale --ckpt_dir checkpoints --device cuda --out results/agg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import load_house, train_one, SEEDS
from evaluate import inverse, metrics


def stats(pred_w, true_w, const_mae):
    m = metrics(pred_w, true_w)
    return {"MAE": m["MAE"], "RMSE": m["RMSE"],
            "std_ratio": float(pred_w.std() / true_w.std()),
            "bias": float((pred_w - true_w).mean()),
            "corr": float(np.corrcoef(pred_w.ravel(), true_w.ravel())[0, 1]),
            "MAE_vs_const": float(m["MAE"] / const_mae)}


def run_house(house_data, house_tag, ckpt_dir, device="cpu", epochs=60, seeds=SEEDS):
    Xte, Mte, Yte = house_data["te"]
    Ytr = house_data["tr"][2]
    lo, hi = house_data["scaler"]
    true_w = inverse(Yte, lo, hi)
    const = float(inverse(Ytr, lo, hi).mean())
    const_mae = float(np.abs(true_w - const).mean())

    acc = {k: [] for k in ("sum_u", "mean_u", "y_vmd", "y_skip", "y_final")}
    wmats, per_mode = [], []
    K = None

    for s in seeds:
        ckpt = Path(ckpt_dir) / f"{house_tag}_full_seed{s}.pt"
        if not ckpt.exists():
            print(f"    чекпойнт не найден, пропуск: {ckpt.name}")
            continue
        r = train_one(house_data, s, device=device, epochs=epochs,
                      model_kwargs={}, ckpt_path=ckpt, return_model=True)
        model = r["model"]
        with torch.no_grad():
            wmats.append(model.aswa.weights().cpu().numpy())      # (T, K)
            buf = {k: [] for k in acc}
            um = []
            for i in range(0, len(Mte), 256):
                o = model(torch.tensor(Mte[i:i + 256]).to(device),
                          torch.tensor(Xte[i:i + 256]).to(device))
                u = o["u_hat"].cpu().numpy()                       # (B, K, T)
                um.append(u)
                buf["sum_u"].append(u.sum(axis=1))
                buf["mean_u"].append(u.mean(axis=1))
                buf["y_vmd"].append(o["y_vmd"].cpu().numpy())
                buf["y_skip"].append(o["y_skip"].cpu().numpy())
                buf["y_final"].append(o["y"].cpu().numpy())
        for k in acc:
            acc[k].append(np.concatenate(buf[k]))
        u_all = np.concatenate(um, axis=0)
        K = u_all.shape[1]
        per_mode.append(u_all.mean(axis=(0, 2)))                   # средний уровень моды k
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    out = {"house": house_tag, "K": K, "n_test_windows": int(len(Yte)),
           "target_std": float(true_w.std()), "target_mean": float(true_w.mean()),
           "const_baseline_MAE": const_mae, "paths": {}}

    preds = {k: inverse(np.mean(v, axis=0), lo, hi) for k, v in acc.items()}
    preds["K_times_y_vmd"] = inverse(np.mean(acc["y_vmd"], axis=0) * K, lo, hi)
    for k, p in preds.items():
        out["paths"][k] = stats(p, true_w, const_mae)

    w = np.mean(wmats, axis=0)                                     # (T, K)
    out["aswa_weights"] = {"mean_per_mode": w.mean(axis=0).tolist(),
                           "max_per_mode": w.max(axis=0).tolist(),
                           "row_sum": float(w.sum(axis=1).mean()),
                           "entropy_mean": float(
                               (-(w * np.log(w + 1e-12)).sum(axis=1)).mean()),
                           "entropy_uniform": float(np.log(K))}
    out["mode_mean_level_normalized"] = np.mean(per_mode, axis=0).tolist()

    best = min(out["paths"], key=lambda k: out["paths"][k]["MAE"])
    out["best_aggregation"] = best
    out["sum_beats_aswa"] = bool(out["paths"]["sum_u"]["MAE"] < out["paths"]["y_vmd"]["MAE"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--houses", default="1")
    ap.add_argument("--dataset", default="ukdale")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results/agg")
    args = ap.parse_args()

    proc = Path(args.processed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) / args.dataset
    cache = proc / "vmd_cache"

    allr = {}
    for h in [x.strip() for x in args.houses.split(",")]:
        tag = f"{args.dataset}_house_{h}"
        print(f"\n===== {tag} =====")
        hd = load_house(proc / f"house_{h}.npz", cache)
        r = run_house(hd, tag, ckpt_dir, device=args.device, epochs=args.epochs)
        allr[tag] = r
        print(f"  цель: std={r['target_std']:.1f} mean={r['target_mean']:.1f}  "
              f"константа MAE={r['const_baseline_MAE']:.1f}")
        for k in ("sum_u", "mean_u", "y_vmd", "K_times_y_vmd", "y_skip", "y_final"):
            d = r["paths"][k]
            print(f"  {k:14s} MAE={d['MAE']:7.1f}  std_ratio={d['std_ratio']:.3f}  "
                  f"bias={d['bias']:+8.1f}  corr={d['corr']:+.3f}  "
                  f"MAE/const={d['MAE_vs_const']:.3f}")
        a = r["aswa_weights"]
        print(f"  веса ASWA: сумма по модам={a['row_sum']:.3f}  "
              f"энтропия={a['entropy_mean']:.3f} (равномерная={a['entropy_uniform']:.3f})")
        print(f"  средние веса по модам: "
              f"{['%.3f' % x for x in a['mean_per_mode']]}")
        print(f"  лучшая агрегация: {r['best_aggregation']}   "
              f"сумма лучше ASWA: {r['sum_beats_aswa']}")
        (outdir / f"agg_{args.dataset}.json").write_text(
            json.dumps(allr, ensure_ascii=False, indent=2))
    print(f"\nСохранено: {outdir}/agg_{args.dataset}.json")


if __name__ == "__main__":
    main()
