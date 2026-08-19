"""
factorial_aux_agg.py
====================
Факторный эксперимент 2x2x2: согласованность вспомогательной цели с агрегацией,
и внутри каждой ячейки — проверка гипотезы о mode identity.

    aux_mode      = sum | per_mode
    aggregation   = convex | sum
    embeddings    = on | off

Диагональ (per_mode+convex и sum+sum) согласована: там, где aux требует, чтобы
каждая мода предсказывала полную цель, выпуклая комбинация K оценок одной
величины — корректный ансамбль; там, где aux требует Σ_k u_hat_k = y, агрегация
тоже должна суммировать. Внедиагональные ячейки рассогласованы; нынешняя
реализация — это sum+convex.

Главный вопрос: воспроизводится ли эффект mode embeddings в СОГЛАСОВАННЫХ
ячейках. Если да — эффект реален, но требует согласованности; если нет — H1
опровергнута уже на исправленной архитектуре.

Скрипт для каждой конфигурации обучает ансамбль (с чекпойнтами, повторный запуск
не переобучает), считает метрики, диагностику путей и энтропию весов ASWA, затем
внутри каждой ячейки (aux, agg) сравнивает embeddings on/off по поокновым
ошибкам с moving block bootstrap.

Запуск:
    python factorial_aux_agg.py --processed "<...>\\processed" --houses 1,4 \\
        --dataset ukdale --ckpt_dir checkpoints_factorial --device cuda \\
        --epochs 60 --seeds 42,7,13 --block 3 --out results/factorial

Совет: сначала прогнать с --seeds 42,7,13 (3 сида) — это разведка. Полные 5
сидов имеет смысл добирать только в тех ячейках, где что-то нашлось.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from train import load_house, train_one
from evaluate import inverse, metrics, per_window_abs_error
from intervention import moving_block_bootstrap_ci

AUX = ("sum", "per_mode")
AGG = ("convex", "sum")
EMB = (True, False)


def parse_cells(spec: str):
    """'all' -> все 4 пары (aux, agg); иначе 'sum:convex,per_mode:convex'."""
    if spec.strip().lower() == "all":
        return [(a, g) for a in AUX for g in AGG]
    pairs = []
    for tok in spec.split(","):
        aux, _, agg = tok.strip().partition(":")
        assert aux in AUX and agg in AGG, f"неизвестная ячейка: {tok}"
        pairs.append((aux, agg))
    return pairs


def cfg_tag(aux, agg, emb, lam=0.2, no_gate=False):
    """Метка ячейки. Вес вспомогательного члена и отключение гейта попадают в
    метку ТОЛЬКО если отличаются от исторических значений — иначе старые ключи
    и чекпойнты поехали бы."""
    lam_tag = "" if abs(lam - 0.2) < 1e-9 else f"_lam{lam:g}"
    gate_tag = "_nogate" if no_gate else ""
    return f"aux-{aux}{lam_tag}_agg-{agg}{gate_tag}_emb-{'on' if emb else 'off'}"


def eval_config(house_data, house_tag, ckpt_dir, aux, agg, emb,
                device="cpu", epochs=60, seeds=(42, 7, 13), lam=0.2, no_gate=False):
    Xte, Mte, Yte = house_data["te"]
    Ytr = house_data["tr"][2]
    lo, hi = house_data["scaler"]
    true_w = inverse(Yte, lo, hi)
    const = float(inverse(Ytr, lo, hi).mean())
    const_mae = float(np.abs(true_w - const).mean())

    mk = {"aux_mode": aux, "aggregation": agg, "use_mode_embeddings": emb,
          "lambda_aux": lam, "disable_skip": no_gate}
    acc = {k: [] for k in ("y", "y_vmd", "y_skip", "sum_u")}
    ent = []

    for s in seeds:
        ckpt = Path(ckpt_dir) / f"{house_tag}_{cfg_tag(aux, agg, emb, lam, no_gate)}_seed{s}.pt"
        r = train_one(house_data, s, device=device, epochs=epochs,
                      model_kwargs=mk, ckpt_path=ckpt, return_model=True)
        model = r["model"]
        with torch.no_grad():
            w = model.aswa.weights().cpu().numpy()
            ent.append(float((-(w * np.log(w + 1e-12)).sum(axis=1)).mean()))
            buf = {k: [] for k in acc}
            for i in range(0, len(Mte), 256):
                o = model(torch.tensor(Mte[i:i + 256]).to(device),
                          torch.tensor(Xte[i:i + 256]).to(device))
                buf["y"].append(o["y"].cpu().numpy())
                buf["y_vmd"].append(o["y_vmd"].cpu().numpy())
                buf["y_skip"].append(o["y_skip"].cpu().numpy())
                buf["sum_u"].append(o["u_hat"].cpu().numpy().sum(axis=1))
        for k in acc:
            acc[k].append(np.concatenate(buf[k]))
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    preds = {k: inverse(np.mean(v, axis=0), lo, hi) for k, v in acc.items()}
    out = {"aux_mode": aux, "aggregation": agg, "embeddings": emb, "lambda_aux": lam,
           "no_gate": no_gate,
           "aswa_entropy": float(np.mean(ent)),
           "aswa_entropy_uniform": float(np.log(Mte.shape[1])),
           "const_baseline_MAE": const_mae, "paths": {}}
    for k, p in preds.items():
        m = metrics(p, true_w)
        out["paths"][k] = {"MAE": m["MAE"], "RMSE": m["RMSE"],
                           "std_ratio": float(p.std() / true_w.std()),
                           "bias": float((p - true_w).mean()),
                           "corr": float(np.corrcoef(p.ravel(), true_w.ravel())[0, 1]),
                           "MAE_vs_const": float(m["MAE"] / const_mae)}
    out["_err_final"] = per_window_abs_error(preds["y"], true_w)
    out["_err_vmd"] = per_window_abs_error(preds["y_vmd"], true_w)
    # ДОБАВЛЕНО: поокновые ошибки отдельно по каждому сиду. Ансамблевые оценки
    # выше усредняют предсказания и потому не содержат дисперсии инициализации;
    # межсидовая компонента считается из этих массивов.
    out["_err_final_per_seed"] = np.stack(
        [per_window_abs_error(inverse(p, lo, hi), true_w) for p in acc["y"]])
    out["_err_vmd_per_seed"] = np.stack(
        [per_window_abs_error(inverse(p, lo, hi), true_w) for p in acc["y_vmd"]])
    out["seeds"] = list(seeds)
    return out


def two_component_effect(err_off, err_on, block, seeds, n_boot=5000, seed_rng=0):
    """Эффект embeddings с ДВУМЯ источниками неопределённости.

    err_off, err_on — (n_seeds, n_windows) поокновые ошибки.

    within  — дисперсия по тестовым окнам при фиксированном наборе сидов
              (то, что даёт moving block bootstrap; так считалось раньше);
    between — дисперсия оценки эффекта по сидам, se = sd(effect_s)/sqrt(S).
              Сиды спарены: сид s конфигурации off сравнивается с сидом s
              конфигурации on, что убирает общую компоненту.

    Итоговая SE = sqrt(se_within^2 + se_between^2). Именно её надо публиковать:
    интервал только по окнам занижает неопределённость архитектурного сравнения,
    потому что вообще не видит дисперсии инициализации.
    """
    S = err_off.shape[0]
    ens = err_off.mean(axis=0) - err_on.mean(axis=0)
    ci = moving_block_bootstrap_ci(ens, block=block, n_boot=n_boot, seed=seed_rng)
    se_within = (ci["ci_high"] - ci["ci_low"]) / (2 * 1.959963985)

    per_seed = (err_off - err_on).mean(axis=1)          # (n_seeds,)
    if S > 1:
        sd_seed = float(per_seed.std(ddof=1))
        se_between = sd_seed / np.sqrt(S)
    else:
        sd_seed, se_between = float("nan"), float("nan")
    se_total = float(np.sqrt(se_within ** 2 + se_between ** 2)) if S > 1 else se_within

    mean = float(ens.mean())
    z = 1.959963985
    return {
        "mean": mean,
        "ci_low": mean - z * se_total, "ci_high": mean + z * se_total,
        "se_total": se_total, "se_within": float(se_within),
        "se_between": float(se_between), "sd_across_seeds": sd_seed,
        "n_seeds": S, "per_seed_effect": [float(x) for x in per_seed],
        "seed_signs_negative": int((per_seed < 0).sum()),
        "share_between": float(se_between ** 2 / (se_within ** 2 + se_between ** 2))
                          if S > 1 else float("nan"),
        "ci_within_only": {"ci_low": ci["ci_low"], "ci_high": ci["ci_high"]},
        "block": block, "n_boot": n_boot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--houses", default="1,4")
    ap.add_argument("--dataset", default="ukdale")
    ap.add_argument("--ckpt_dir", default="checkpoints_factorial")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="42,7,13")
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--out", default="results/factorial")
    ap.add_argument("--lambda_aux", type=float, default=0.2,
                    help="вес вспомогательного члена; 0.2 — исторический, "
                         "0.0333 = 0.2/K — контроль на перевзвешивание")
    ap.add_argument("--no_gate", action="store_true",
                    help="отключить прямой путь и слияние (disable_skip=True): "
                         "выход модели = выход декомпозиционной ветви")
    ap.add_argument("--emb", default="both", choices=["both", "on", "off"],
                    help="какие настройки embeddings гнать (контролю достаточно on)")
    ap.add_argument("--cells", default="all",
                    help="'all' или список через запятую, напр. sum:convex,per_mode:convex")
    args = ap.parse_args()
    pairs = parse_cells(args.cells)

    seeds = tuple(int(x) for x in args.seeds.split(","))
    lam = args.lambda_aux
    no_gate = args.no_gate
    emb_list = {"both": EMB, "on": (True,), "off": (False,)}[args.emb]
    proc = Path(args.processed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache = proc / "vmd_cache"

    # СЛИЯНИЕ, А НЕ ПЕРЕЗАПИСЬ: иначе прогон одной ячейки затрёт все остальные,
    # на которых стоят §5.1 и §5.2.
    npz_path = outdir / f"factorial_per_window_{args.dataset}.npz"
    json_path = outdir / f"factorial_{args.dataset}.json"
    npz_store = {}
    if npz_path.exists():
        with np.load(npz_path) as old:
            npz_store = {k: old[k] for k in old.files}
        print(f"загружено ранее сохранённых массивов: {len(npz_store)}")
    allr = json.loads(json_path.read_text()) if json_path.exists() else {}
    if allr:
        print(f"загружено ранее сохранённых домохозяйств в JSON: {len(allr)}")
    for h in [x.strip() for x in args.houses.split(",")]:
        tag = f"{args.dataset}_house_{h}"
        print(f"\n{'='*66}\n{tag}\n{'='*66}")
        hd = load_house(proc / f"house_{h}.npz", cache)
        cells = dict(allr.get(tag, {}).get("cells", {}))

        for (aux, agg), emb in itertools.product(pairs, emb_list):
            name = cfg_tag(aux, agg, emb, lam, no_gate)
            print(f"\n-- {name}")
            r = eval_config(hd, tag, ckpt_dir, aux, agg, emb,
                            device=args.device, epochs=args.epochs, seeds=seeds,
                            lam=lam, no_gate=no_gate)
            npz_store[f"{tag}|{name}|err_final"] = r.pop("_err_final")
            npz_store[f"{tag}|{name}|err_vmd"] = r.pop("_err_vmd")
            npz_store[f"{tag}|{name}|err_final_per_seed"] = r.pop("_err_final_per_seed")
            npz_store[f"{tag}|{name}|err_vmd_per_seed"] = r.pop("_err_vmd_per_seed")
            cells[name] = r
            p = r["paths"]
            print(f"   y_final MAE={p['y']['MAE']:7.1f} (MAE/const={p['y']['MAE_vs_const']:.3f})"
                  f"  y_vmd MAE={p['y_vmd']['MAE']:7.1f} std_ratio={p['y_vmd']['std_ratio']:.3f}"
                  f"  энтропия ASWA={r['aswa_entropy']:.3f}")

        # эффект embeddings внутри каждой ячейки (aux, agg)
        eff = dict(allr.get(tag, {}).get("embeddings_effect", {}))
        for aux, agg in pairs:
            on = f"{cfg_tag(aux, agg, True, lam, no_gate)}"
            off = f"{cfg_tag(aux, agg, False, lam, no_gate)}"
            if any(f"{tag}|{c}|err_final_per_seed" not in npz_store for c in (on, off)):
                print(f"   (эффект embeddings для {on} не считается: "
                      f"нет пары on/off — так и задумано при --emb on)")
                continue
            d = {}
            for path_key, suffix in (("y_final", "err_final"), ("y_vmd", "err_vmd")):
                # >0 означает: без embeddings ХУЖЕ, то есть embeddings помогают
                d[path_key] = two_component_effect(
                    npz_store[f"{tag}|{off}|{suffix}_per_seed"],
                    npz_store[f"{tag}|{on}|{suffix}_per_seed"],
                    block=args.block, seeds=seeds, seed_rng=0)
            eff[cfg_tag(aux, agg, True, lam, no_gate).replace("_emb-on", "")] = {
                "coherent": bool((aux == "per_mode" and agg == "convex")
                                 or (aux == "sum" and agg == "sum")),
                "embeddings_benefit": d,
            }
        allr[tag] = {"cells": cells, "embeddings_effect": eff}

        print(f"\n  Эффект embeddings (>0 значит embeddings помогают), {tag}:")
        for k, v in eff.items():
            b = v["embeddings_benefit"]["y_final"]
            bv = v["embeddings_benefit"]["y_vmd"]
            mark = "согласована" if v["coherent"] else "рассогласована"
            print(f"   {k:26s} [{mark:14s}] y_final={b['mean']:+7.2f} "
                  f"[{b['ci_low']:+.2f},{b['ci_high']:+.2f}]  "
                  f"y_vmd={bv['mean']:+7.2f} [{bv['ci_low']:+.2f},{bv['ci_high']:+.2f}]")
            print(f"   {'':26s}  SE: окна={b['se_within']:.2f} сиды={b['se_between']:.2f} "
                  f"итог={b['se_total']:.2f} (доля сидов {b['share_between']:.0%}); "
                  f"интервал только по окнам был "
                  f"[{b['ci_within_only']['ci_low']:+.2f},"
                  f"{b['ci_within_only']['ci_high']:+.2f}]; "
                  f"знаков по сидам -{b['seed_signs_negative']}/"
                  f"+{b['n_seeds']-b['seed_signs_negative']}")

        json_path.write_text(json.dumps(allr, ensure_ascii=False, indent=2))
        np.savez_compressed(npz_path, **npz_store)

    print(f"\nСохранено: {outdir}/factorial_{args.dataset}.json")


if __name__ == "__main__":
    main()
