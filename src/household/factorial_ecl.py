"""
factorial_ecl.py  (версия 2)
============================
Факториал на втором домене (ECL): 17 клиентов x 3 конфигурации x 5 сидов
= 255 прогонов. Соответствует prereg_ecl.docx v1.2 (разделы 7-8).

ЧТО ИСПРАВЛЕНО ОТНОСИТЕЛЬНО ВЕРСИИ 1. Версия 1 считала контраст МЕЖДУ ЯЧЕЙКАМИ
(per_mode+convex минус sum+convex). Но variance_model.py и meta_analysis.py
раскладывают эффект ЭМБЕДДИНГОВ ВНУТРИ ячейки: load_effects() берёт пару
'..._emb-on|err_*_per_seed' / '..._emb-off|...' и считает off - on. Именно этот
эстиманд даёт числа основного исследования (доля 83 %, отношение 24), и именно
его надо реплицировать. Контраст между ячейками сохранён отдельно — он нужен
для R3 (маскирование), но это другая величина.

Конфигурации:
    aux-per_mode_agg-convex, emb-on    -> R1, R2, R4 (пара для эффекта эмбеддингов)
    aux-per_mode_agg-convex, emb-off   -> та же пара
    aux-sum_agg-convex,      emb-on    -> R3 (маскирование между ячейками)
Везде disable_skip=False, lambda_aux — дефолт модели. Гиперпараметры на ECL
НЕ настраиваются (§7 пре-спецификации).

ЕДИНИЦЫ. Поокновые ошибки делятся на MAE persistence-прогноза того же клиента
ДО сохранения. Средние потребления клиентов различаются на три порядка, и в
абсолютных единицах мета-анализ определялся бы двумя клиентами из семнадцати.
Нормировка проходит через всю цепочку сама: R1 и R2 безразмерны и не меняются,
R4 становится сопоставимым. Множитель mae_const сохраняется в npz, поэтому
возврат к ваттам возможен в любой момент.

ВЫХОД (схема ключей — как у factorial_aux_agg.py, анализаторы правки не требуют):
    results_ecl/factorial_per_window_ecl.npz
        ключи вида  ecl_T148|aux-per_mode_agg-convex_emb-on|err_final_per_seed
        значения    массивы (сиды x окна)
        плюс        ecl_T148|mae_const  (скаляр, множитель возврата к ваттам)
    results_ecl/factorial_ecl.json
        {house_tag: {"embeddings_effect": {cell: {"coherent":..,
                     "embeddings_benefit": {"y_final": {...}, "y_vmd": {...}}}}}}
    results_ecl/per_client/<client>.npz — промежуточные, для возобновления

Возобновляемость двойная: чекпойнты в --ckpt_dir переиспользует train_one,
а готовые клиенты пропускаются по наличию per_client/<client>.npz (--force
пересчитывает). Прерывание ничего не теряет.

Запуск:
    python factorial_ecl.py --device cuda
    python factorial_ecl.py --device cuda --clients T68,T274,T148     # партиями
Анализ после:
    python variance_model.py --npz results_ecl/factorial_per_window_ecl.npz \\
        --cell aux-per_mode_agg-convex --path y_final --block 3 --out results_ecl/variance
    python meta_analysis.py --json results_ecl/factorial_ecl.json \\
        --npz results_ecl/factorial_per_window_ecl.npz \\
        --cell aux-per_mode_agg-convex --path y_final \\
        --deltas 0.001,0.002,0.005,0.01 --out results_ecl/meta
(--deltas в ДОЛЯХ MAE persistence, не в ваттах: единицы теперь относительные.)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train import load_house, train_one
from evaluate import inverse

Z = 1.959963985

# (ячейка, метка эмбеддингов) -> model_kwargs
CONFIGS: dict[tuple[str, str], dict] = {
    ("aux-per_mode_agg-convex", "emb-on"): {
        "aux_mode": "per_mode", "aggregation": "convex",
        "use_mode_embeddings": True, "disable_skip": False},
    ("aux-per_mode_agg-convex", "emb-off"): {
        "aux_mode": "per_mode", "aggregation": "convex",
        "use_mode_embeddings": False, "disable_skip": False},
    ("aux-sum_agg-convex", "emb-on"): {
        "aux_mode": "sum", "aggregation": "convex",
        "use_mode_embeddings": True, "disable_skip": False},
}
PRIMARY_CELL = "aux-per_mode_agg-convex"     # ячейка с парой emb-on/emb-off
MASKING_CELL = "aux-sum_agg-convex"          # ячейка для R3
COHERENT_CELLS = {"aux-per_mode_agg-convex"}
SEEDS = (42, 7, 13, 99, 2025)


# ---------------------------------------------------------------------------
def window_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """MAE по каждому окну: (W, T) -> (W,)."""
    return np.mean(np.abs(pred - true), axis=1)


def persistence_mae(npz_path: Path) -> float:
    """MAE persistence-прогноза клиента в исходных единицах — знаменатель §8."""
    d = np.load(npz_path)
    Xte, Yte = d["Xte"], d["Yte"]
    span = float(d["scaler_hi"]) - float(d["scaler_lo"])
    return float(np.mean(np.abs(Yte - Xte[:, -1][:, None]))) * span


def forward_outputs(model, data, device: str) -> dict[str, np.ndarray]:
    """Снимает все выходы модели формы (окна x горизонт) в исходных единицах.
    Имена ветвей не угадываются: берётся то, что модель реально вернула."""
    Xte, Mte, Yte = data["te"]
    lo, hi = data["scaler"]
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Mte).to(device), torch.tensor(Xte).to(device))
    res = {}
    if isinstance(out, dict):
        for k, v in out.items():
            if torch.is_tensor(v) and v.ndim == 2 and tuple(v.shape) == Yte.shape:
                res[k] = inverse(v.detach().cpu().numpy(), lo, hi)
    return res


def naive_stats(d: np.ndarray) -> dict:
    """Сводка эффекта по матрице (сиды x окна). SE здесь НАИВНАЯ и служит только
    запасным вариантом: meta_analysis.py пересчитает v_i по компонентам ANOVA,
    если ему передать --npz (см. его docstring, §5.2.1 статьи)."""
    seed_means = d.mean(axis=1)
    S = len(seed_means)
    se_between = float(np.std(seed_means, ddof=1) / np.sqrt(S)) if S > 1 else float("nan")
    se_within = float(np.std(d.mean(axis=0), ddof=1) / np.sqrt(d.shape[1]))
    se = float(np.sqrt(se_between ** 2 + se_within ** 2))
    mean = float(d.mean())
    return {"mean": mean, "se_total": se, "se_within": se_within,
            "se_between": se_between,
            "ci_low": mean - Z * se, "ci_high": mean + Z * se,
            "n_seeds": int(S), "n_windows": int(d.shape[1])}


# ---------------------------------------------------------------------------
def run_client(name: str, npz_path: Path, args) -> dict:
    data = load_house(npz_path, Path(args.cache_dir) / name)
    mae_const = persistence_mae(npz_path)
    scale = mae_const if args.relative else 1.0
    ckpt_dir = Path(args.ckpt_dir) / name
    errs: dict[tuple[str, str, str], list] = {}

    for (cell, emb), mk in CONFIGS.items():
        for seed in args.seeds:
            t0 = time.time()
            r = train_one(data, seed, device=args.device, epochs=args.epochs,
                          model_kwargs=mk,
                          ckpt_path=ckpt_dir / f"{cell}_{emb}_seed{seed}.pt",
                          return_model=True)
            true_w = r["true_w"]
            outs = forward_outputs(r["model"], data, args.device)
            outs.setdefault("y", r["pred_w"])
            for out_key, tag in (("y", "err_final_per_seed"),
                                 ("y_vmd", "err_vmd_per_seed"),
                                 ("y_skip", "err_skip_per_seed")):
                if out_key in outs:
                    errs.setdefault((cell, emb, tag), []).append(
                        window_mae(outs[out_key], true_w) / scale)
            del r
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"    {cell} {emb} seed={seed}: "
                  f"MAE/const={np.mean(window_mae(outs['y'], true_w)) / mae_const:.4f} "
                  f"({time.time() - t0:.0f} c)")

    house = f"ecl_{name}"
    payload = {f"{house}|{cell}_{emb}|{tag}": np.stack(v)
               for (cell, emb, tag), v in errs.items()}
    payload[f"{house}|mae_const"] = np.array(mae_const)
    out_npz = Path(args.out_dir) / "per_client" / f"{name}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)
    return payload


# ---------------------------------------------------------------------------
def build_json(all_arrays: dict, out_dir: Path) -> dict:
    """Отчёт в схеме, которую читает meta_analysis.collect()."""
    houses: dict[str, dict] = {}
    for key in all_arrays:
        if "|" not in key or key.endswith("|mae_const"):
            continue
        house = key.split("|")[0]
        houses.setdefault(house, {})
    report: dict[str, dict] = {}
    for house in sorted(houses):
        eff_blob: dict[str, dict] = {}
        for cell in {c for c, _ in CONFIGS}:
            benefit = {}
            for tag, path_key in (("err_final_per_seed", "y_final"),
                                  ("err_vmd_per_seed", "y_vmd")):
                on = all_arrays.get(f"{house}|{cell}_emb-on|{tag}")
                off = all_arrays.get(f"{house}|{cell}_emb-off|{tag}")
                if on is None or off is None:
                    continue
                benefit[path_key] = naive_stats(off - on)   # + значит эмбеддинги помогают
            if benefit:
                eff_blob[cell] = {"coherent": cell in COHERENT_CELLS,
                                  "embeddings_benefit": benefit}
        blob: dict = {"embeddings_effect": eff_blob,
                      "mae_const": float(all_arrays.get(f"{house}|mae_const", np.nan))}
        # R3: маскирование между ячейками при emb-on
        mask = {}
        for tag, path_key in (("err_final_per_seed", "y_final"),
                              ("err_vmd_per_seed", "y_vmd"),
                              ("err_skip_per_seed", "y_skip")):
            a = all_arrays.get(f"{house}|{PRIMARY_CELL}_emb-on|{tag}")
            b = all_arrays.get(f"{house}|{MASKING_CELL}_emb-on|{tag}")
            if a is None or b is None:
                continue
            mask[path_key] = {"coherent_mae_over_const": float(a.mean()),
                              "broken_mae_over_const": float(b.mean()),
                              "ratio": float(b.mean() / a.mean()) if a.mean() else float("nan"),
                              "cell_contrast": naive_stats(a - b)}
        if mask:
            blob["masking_between_cells"] = mask
        report[house] = blob
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="processed_ecl")
    p.add_argument("--selection", default="ecl_selection.json")
    p.add_argument("--cache_dir", default="cache_ecl")
    p.add_argument("--ckpt_dir", default="ckpt_ecl")
    p.add_argument("--out_dir", default="results_ecl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--clients", default="", help="подмножество через запятую")
    p.add_argument("--absolute", dest="relative", action="store_false",
                   help="сохранять ошибки в исходных единицах (по умолчанию — "
                        "относительные, делённые на MAE persistence клиента)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(relative=True)
    args = p.parse_args()
    args.seeds = tuple(int(s) for s in args.seeds.split(","))

    if args.clients:
        names = [c.strip() for c in args.clients.split(",") if c.strip()]
    else:
        sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        names = list(sel["selected"])

    total = len(names) * len(CONFIGS) * len(args.seeds)
    print(f"Клиентов: {len(names)}, конфигураций: {len(CONFIGS)}, "
          f"сидов: {len(args.seeds)} -> {total} прогонов")
    print(f"Единицы ошибок: {'относительные (÷ MAE persistence)' if args.relative else 'исходные'}")
    for (cell, emb), mk in CONFIGS.items():
        print(f"  {cell} {emb}: {json.dumps(mk, ensure_ascii=False)}")
    print()

    out_dir = Path(args.out_dir)
    (out_dir / "per_client").mkdir(parents=True, exist_ok=True)
    all_arrays: dict[str, np.ndarray] = {}
    t_start = time.time()

    for i, name in enumerate(names, 1):
        npz_path = Path(args.data_dir) / f"client_{name}.npz"
        if not npz_path.exists():
            print(f"[{i}/{len(names)}] {name}: нет {npz_path} — пропуск")
            continue
        cached = out_dir / "per_client" / f"{name}.npz"
        if cached.exists() and not args.force:
            print(f"[{i}/{len(names)}] {name}: уже посчитан — загружаю")
            with np.load(cached) as z:
                all_arrays.update({k: z[k] for k in z.files})
            continue
        print(f"[{i}/{len(names)}] клиент {name}")
        all_arrays.update(run_client(name, npz_path, args))
        el = time.time() - t_start
        print(f"    готово; прошло {el/60:.1f} мин, оценка до конца "
              f"{el/i*(len(names)-i)/60:.1f} мин\n")

    if not all_arrays:
        print("Нет результатов.")
        return

    merged = out_dir / "factorial_per_window_ecl.npz"
    np.savez_compressed(merged, **all_arrays)
    report = build_json(all_arrays, out_dir)
    (out_dir / "factorial_ecl.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Эффект эмбеддингов, {PRIMARY_CELL}, y_final "
          f"(в долях MAE persistence) ===")
    vals = []
    for h, blob in sorted(report.items()):
        b = blob["embeddings_effect"].get(PRIMARY_CELL, {}).get(
            "embeddings_benefit", {}).get("y_final")
        if b:
            vals.append(b["mean"])
            print(f"  {h:14s} {b['mean']:+.5f}  (наивная SE {b['se_total']:.5f})")
    if vals:
        v = np.array(vals)
        print(f"  клиентов {len(v)}, медиана {np.median(v):+.5f}, "
              f"знаки {int((v < 0).sum())}-/{int((v > 0).sum())}+")

    ratios = [blob["masking_between_cells"]["y_vmd"]["ratio"]
              for blob in report.values() if "masking_between_cells" in blob
              and "y_vmd" in blob["masking_between_cells"]]
    if ratios:
        print(f"\nR3, отношение ошибки VMD-ветви (sum / per_mode): "
              f"медиана {np.median(ratios):.1f}x по {len(ratios)} клиентам")

    print(f"\nЗаписано: {merged.resolve()}")
    print(f"          {(out_dir / 'factorial_ecl.json').resolve()}")
    print("\nДальше (единицы относительные, --deltas в долях MAE persistence):")
    print(f"  python variance_model.py --npz {merged} "
          f"--cell {PRIMARY_CELL} --path y_final --block 3 --out {out_dir}/variance")
    print(f"  python meta_analysis.py --json {out_dir}/factorial_ecl.json "
          f"--npz {merged} --cell {PRIMARY_CELL} --path y_final "
          f"--deltas 0.001,0.002,0.005,0.01 --out {out_dir}/meta")


if __name__ == "__main__":
    main()
