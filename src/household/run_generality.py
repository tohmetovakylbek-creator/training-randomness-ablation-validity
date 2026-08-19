"""
run_generality.py
=================
Прогон эксперимента §5.2.6: две внешние архитектуры x две конфигурации
(компонент включён / выключен) x 5 сидов x те же 17 домохозяйств.

Читает ТОЛЬКО processed/house_<n>.npz (Xtr,Ytr,Xva,Yva,Xte,Yte,scaler_lo,scaler_hi) —
то есть ровно те файлы, которые уже созданы ukdale_loader / refit_loader /
sheerm_loader. Новый препроцессинг не выполняется, окна и разбиения те же.

Пример (Windows / PowerShell, одна строка):

  python run_generality.py ^
    --root ukdale=C:\\Users\\User\\PycharmProjects\\uk_dale_project\\processed ^
    --root refit=C:\\Users\\User\\PycharmProjects\\uk_dale_project\\processed_refit ^
    --root sheerm=C:\\Users\\User\\PycharmProjects\\uk_dale_project\\processed_sheerm ^
    --houses ukdale:1 refit:2,4,6,9,10 sheerm:1,2,3,4,5,8,9,10,11,12,13 ^
    --device cuda --epochs 60 --out results_generality

Крашоустойчиво: каждый прогон пишется отдельным .npy сразу после обучения,
повторный запуск пропускает уже посчитанное. Прервать и продолжить можно в любой момент.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from generality_models import CONFIGS, COMPONENT, train_one

SEEDS = (42, 7, 13, 99, 2025)  # те же пять сидов, что в основном эксперименте


def parse_roots(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--root ожидает вид name=path, получено: {it}")
        name, path = it.split("=", 1)
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"папка не найдена: {p}")
        out[name] = p
    return out


def parse_houses(items: list[str]) -> list[tuple[str, int]]:
    out = []
    for it in items:
        ds, lst = it.split(":", 1)
        for h in lst.split(","):
            out.append((ds, int(h)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True,
                    help="dataset=path к папке с house_<n>.npz (можно несколько раз)")
    ap.add_argument("--houses", nargs="+", required=True,
                    help="список вида ukdale:1 refit:2,4,6,9,10 sheerm:1,2,...")
    ap.add_argument("--arch", default="patchtst,bilstm",
                    help="какие архитектуры гнать (по умолчанию обе)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results_generality")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()

    roots = parse_roots(args.root)
    houses = parse_houses(args.houses)
    archs = [a.strip() for a in args.arch.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("runs", {})
    manifest["config"] = {"seeds": seeds, "epochs": args.epochs, "archs": archs,
                          "houses": [f"{d}_{h}" for d, h in houses],
                          "component": {a: COMPONENT[a] for a in archs}}

    total = len(houses) * len(archs) * 2 * len(seeds)
    done = 0
    t_start = time.time()

    for ds, h in houses:
        npz_path = roots[ds] / f"house_{h}.npz"
        if not npz_path.exists():
            print(f"[SKIP] нет файла {npz_path}")
            done += len(archs) * 2 * len(seeds)
            continue
        npz = np.load(npz_path)
        print(f"\n=== {ds} house {h}: test windows = {npz['Xte'].shape[0]}")

        for arch in archs:
            for cfg in ("on", "off"):
                for seed in seeds:
                    key = f"{arch}_{cfg}_{ds}_{h}_seed{seed}"
                    fpath = out / f"err_{key}.npy"
                    done += 1
                    if fpath.exists():
                        print(f"  [{done}/{total}] {key}: уже посчитано, пропуск")
                        continue
                    t0 = time.time()
                    r = train_one(CONFIGS[(arch, cfg)], npz, seed,
                                  device=args.device, epochs=args.epochs)
                    np.save(fpath, r["window_err"].astype(np.float64))
                    manifest["runs"][key] = {"mae_w": r["mae"], "val_mae": r["val_mae"],
                                             "n_windows": int(len(r["window_err"])),
                                             "seconds": round(time.time() - t0, 1)}
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
                    elapsed = time.time() - t_start
                    eta = elapsed / max(done, 1) * (total - done) / 60
                    print(f"  [{done}/{total}] {key}: MAE={r['mae']:.2f} Вт, "
                          f"{time.time()-t0:.0f} с, ETA ~{eta:.0f} мин")

    print(f"\nГотово. Результаты в {out.resolve()}")
    print("Дальше: python generality_variance.py --dir", args.out)


if __name__ == "__main__":
    main()
