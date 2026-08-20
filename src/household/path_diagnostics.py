"""
path_diagnostics.py
===================
Характеризует три выхода модели — y_vmd, y_skip и итоговый y — на тесте.

Зачем. В прогоне house 4 у VMD-пути обнаружилось pred_std = 29.5 Вт при
bias = -250 Вт, то есть путь выдаёт почти константу далеко ниже цели, а его MAE
(253) почти равен модулю смещения. Если это так, интервенции над mode embeddings
применяются к уже вырожденной ветви, и нулевой эффект объясняется не
диссоциацией представления и полезности, а тем, что менять там нечего. Это
первая альтернативная гипотеза, которую выдвинет рецензент, и её нужно закрыть
данными, а не рассуждением.

Скрипт ничего не обучает — читает те же чекпойнты, что и intervention.py.

Ключевые величины:
  std_ratio      = std(прогноз) / std(цель). Около 0 — путь выродился в константу.
  MAE_vs_const   = MAE пути, делённый на MAE тривиального предсказания
                   средним обучающей цели. >= 1 означает, что путь не лучше
                   константы.
  corr           = корреляция прогноза с целью по всем точкам.
  err_corr       = корреляция ошибок VMD- и skip-путей. Сильная отрицательная
                   означает, что итог хорош не потому, что хороши пути, а потому
                   что их смещения гасят друг друга.

Запуск:
    python path_diagnostics.py --processed "<...>\\processed" --houses 1,4 \\
        --dataset ukdale --ckpt_dir checkpoints --device cuda --out results/paths
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import load_house, train_one, SEEDS
from evaluate import inverse, metrics
from intervention import forward_test, ckpt_for


def path_stats(pred_w, true_w, const_mae):
    m = metrics(pred_w, true_w)
    return {
        "MAE": m["MAE"], "RMSE": m["RMSE"],
        "std": float(pred_w.std()), "bias": float((pred_w - true_w).mean()),
        "std_ratio": float(pred_w.std() / true_w.std()),
        "corr": float(np.corrcoef(pred_w.ravel(), true_w.ravel())[0, 1]),
        "MAE_vs_const": float(m["MAE"] / const_mae),
    }


def run_house(house_data, house_tag, ckpt_dir, device="cpu", epochs=60, seeds=SEEDS,
              aux_mode="sum", aggregation="convex"):
    Xte, Mte, Yte = house_data["te"]
    Ytr = house_data["tr"][2]
    lo, hi = house_data["scaler"]
    true_w = inverse(Yte, lo, hi)

    # тривиальный ориентир: константа = среднее обучающей цели
    const = float(inverse(Ytr, lo, hi).mean())
    const_mae = float(np.abs(true_w - const).mean())

    pf, pv, ps, gg = [], [], [], []
    for s in seeds:
        ckpt = ckpt_for(ckpt_dir, house_tag, aux_mode, aggregation, s)
        if not ckpt.exists():
            print(f"    чекпойнт не найден, пропуск: {ckpt.name}")
            continue
        r = train_one(house_data, s, device=device, epochs=epochs,
                      model_kwargs={"aux_mode": aux_mode, "aggregation": aggregation},
                      ckpt_path=ckpt, return_model=True)
        model = r["model"]
        with torch.no_grad():
            ys, yv, ysk, gs = [], [], [], []
            for i in range(0, len(Mte), 256):
                o = model(torch.tensor(Mte[i:i + 256]).to(device),
                          torch.tensor(Xte[i:i + 256]).to(device), return_gate=True)
                ys.append(o["y"].cpu().numpy()); yv.append(o["y_vmd"].cpu().numpy())
                ysk.append(o["y_skip"].cpu().numpy()); gs.append(o["gate"].cpu().numpy())
        pf.append(np.concatenate(ys)); pv.append(np.concatenate(yv))
        ps.append(np.concatenate(ysk)); gg.append(np.concatenate(gs))
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    y_fin = inverse(np.mean(pf, axis=0), lo, hi)
    y_vmd = inverse(np.mean(pv, axis=0), lo, hi)
    y_skp = inverse(np.mean(ps, axis=0), lo, hi)
    gate = np.mean(gg, axis=0).ravel()

    e_v, e_s = (y_vmd - true_w).ravel(), (y_skp - true_w).ravel()
    out = {
        "house": house_tag,
        "aux_mode": aux_mode, "aggregation": aggregation,
        "n_test_windows": int(len(Yte)),
        "target_std": float(true_w.std()), "target_mean": float(true_w.mean()),
        "const_baseline_MAE": const_mae,
        "gate": {"mean": float(gate.mean()), "std": float(gate.std()),
                 "q05": float(np.quantile(gate, 0.05)),
                 "q50": float(np.quantile(gate, 0.50)),
                 "q95": float(np.quantile(gate, 0.95))},
        "y_vmd": path_stats(y_vmd, true_w, const_mae),
        "y_skip": path_stats(y_skp, true_w, const_mae),
        "y_final": path_stats(y_fin, true_w, const_mae),
        "err_corr_vmd_skip": float(np.corrcoef(e_v, e_s)[0, 1]),
    }
    v = out["y_vmd"]
    out["vmd_path_collapsed"] = bool(v["std_ratio"] < 0.25 or v["MAE_vs_const"] >= 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--houses", default="1")
    ap.add_argument("--dataset", default="ukdale")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="results/paths")
    ap.add_argument("--aux_mode", default="sum", choices=["sum", "per_mode"])
    ap.add_argument("--aggregation", default="convex", choices=["convex", "sum"])
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    cell = f"aux-{args.aux_mode}_agg-{args.aggregation}"

    proc = Path(args.processed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) / args.dataset
    cache = proc / "vmd_cache"

    allr = {}
    for h in [x.strip() for x in args.houses.split(",")]:
        tag = f"{args.dataset}_house_{h}"
        print(f"\n===== {tag} | {cell} =====")
        hd = load_house(proc / f"house_{h}.npz", cache)
        r = run_house(hd, tag, ckpt_dir, device=args.device, epochs=args.epochs,
                      seeds=seeds, aux_mode=args.aux_mode, aggregation=args.aggregation)
        allr[tag] = r
        print(f"  цель: std={r['target_std']:.1f} mean={r['target_mean']:.1f}  "
              f"константный ориентир MAE={r['const_baseline_MAE']:.1f}")
        g = r["gate"]
        print(f"  gate: mean={g['mean']:.3f} sd={g['std']:.3f} "
              f"[q05={g['q05']:.3f} q50={g['q50']:.3f} q95={g['q95']:.3f}]")
        for p in ("y_vmd", "y_skip", "y_final"):
            d = r[p]
            print(f"  {p:8s} MAE={d['MAE']:7.1f}  std_ratio={d['std_ratio']:.3f}  "
                  f"bias={d['bias']:+8.1f}  corr={d['corr']:+.3f}  "
                  f"MAE/const={d['MAE_vs_const']:.3f}")
        print(f"  корреляция ошибок VMD и skip: {r['err_corr_vmd_skip']:+.3f}")
        print(f"  VMD-путь выродился: {r['vmd_path_collapsed']}")
        (outdir / f"paths_{args.dataset}_{cell}.json").write_text(
            json.dumps(allr, ensure_ascii=False, indent=2))

    # сводка по всем домам: сколько ветвей хуже константного прогноза
    vs = [r["y_vmd"]["MAE_vs_const"] for r in allr.values()]
    ss = [r["y_skip"]["MAE_vs_const"] for r in allr.values()]
    fs = [r["y_final"]["MAE_vs_const"] for r in allr.values()]
    sr = [r["y_vmd"]["std_ratio"] for r in allr.values()]
    n = len(vs)
    print(f"\nпо {n} домохозяйствам ({cell}):")
    print(f"  VMD-путь хуже константы: {sum(1 for x in vs if x >= 1)}/{n}  "
          f"(медиана MAE/const {np.median(vs):.2f}, std_ratio {np.median(sr):.3f})")
    print(f"  skip-путь хуже константы: {sum(1 for x in ss if x >= 1)}/{n}  "
          f"(медиана {np.median(ss):.2f})")
    print(f"  итог хуже константы: {sum(1 for x in fs if x >= 1)}/{n}  "
          f"(медиана {np.median(fs):.2f})")
    print(f"\nСохранено: {outdir}/paths_{args.dataset}_{cell}.json")


if __name__ == "__main__":
    main()
