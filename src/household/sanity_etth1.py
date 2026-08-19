"""
sanity_etth1.py
===============
Технический sanity check переноса пайплайна на ETTh1 (критерии S1 и S3
пре-спецификации). ETTh1 в разложение дисперсии и мета-анализ НЕ входит.

S1: MAE модели меньше MAE persistence И меньше MAE сезонного наивного прогноза.
    Внимание: на OT (температура масла) persistence СИЛЬНЕЕ сезонного наивного
    (1.36 против 1.67), потому что ряд инерционный и суточной периодичности
    почти нет. Порогом служит меньшая из двух величин, то есть 1.36.
S3: два запуска с одним и тем же сидом дают побитово идентичные предсказания.
    Если нет — в обучении остался недетерминизм, и он испортит именно оценку
    seed-компоненты, то есть ядро статьи.

S2 (кривые обучения не расходятся) требует истории валидации, которой train_one
не возвращает; см. примечание в конце вывода.

Запуск (из корня проекта, рядом с train.py и evaluate.py) — когерентная ячейка
per_mode + convex, эмбеддинги включены, гейт работает:

    python sanity_etth1.py --npz processed_etth1/client_OT.npz --device cuda

Именно эти значения стоят по умолчанию, поэтому в типичном случае никаких
флагов конфигурации передавать не нужно. Если понадобится другая ячейка:

    --aux_mode sum --aggregation sum --no_mode_embeddings --disable_skip
    --lambda_aux 0.05

JSON в командной строке убран намеренно: PowerShell 5.1 портит внутренние
кавычки при передаче нативному процессу, и на этом уже потеряно несколько
попыток запуска.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train import load_house, train_one


def baselines(npz_path: Path, T: int = 24) -> dict:
    d = np.load(npz_path)
    Xte, Yte = d["Xte"], d["Yte"]
    lo, hi = float(d["scaler_lo"]), float(d["scaler_hi"])
    span = hi - lo
    last = Xte[:, -1][:, None]
    snaive = Xte[:, -T:]
    return {
        "mae_const": float(np.mean(np.abs(Yte - last))) * span,
        "mae_snaive": float(np.mean(np.abs(Yte - snaive))) * span,
        "n_test_windows": int(len(Xte)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="processed_etth1/client_OT.npz")
    p.add_argument("--cache_dir", default="cache_etth1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aux_mode", default="per_mode", choices=["per_mode", "sum"])
    p.add_argument("--aggregation", default="convex", choices=["convex", "sum"])
    p.add_argument("--no_mode_embeddings", action="store_true",
                   help="use_mode_embeddings=False (по умолчанию эмбеддинги включены)")
    p.add_argument("--disable_skip", action="store_true",
                   help="отключить skip-ветвь (эксперимент §5.2.7)")
    p.add_argument("--lambda_aux", type=float, default=None,
                   help="не задавать — будет взят дефолт VMDPatchTSTASWA, "
                        "как в factorial_aux_agg.py")
    args = p.parse_args()

    npz_path = Path(args.npz)
    mk = {
        "aux_mode": args.aux_mode,
        "aggregation": args.aggregation,
        "use_mode_embeddings": not args.no_mode_embeddings,
        "disable_skip": bool(args.disable_skip),
    }
    if args.lambda_aux is not None:
        mk["lambda_aux"] = args.lambda_aux
    print(f"Конфигурация модели: {json.dumps(mk, ensure_ascii=False)}")
    base = baselines(npz_path)
    print(f"Базовые прогнозы ({base['n_test_windows']} тестовых окон): "
          f"persistence MAE={base['mae_const']:.4f}, "
          f"сезонный наивный MAE={base['mae_snaive']:.4f}")
    threshold = min(base["mae_const"], base["mae_snaive"])
    print(f"Порог S1 (меньшая из двух): {threshold:.4f}\n")

    data = load_house(npz_path, Path(args.cache_dir))

    print(f"[прогон 1] seed={args.seed}, device={args.device}, epochs={args.epochs}")
    r1 = train_one(data, args.seed, device=args.device, epochs=args.epochs,
                   model_kwargs=mk)
    mae1 = float(np.mean(np.abs(r1["pred_w"] - r1["true_w"])))
    print(f"  MAE теста = {mae1:.4f}, лучшая валидационная MAE = {r1['val_mae']:.4f}, "
          f"параметров = {r1['n_params']}")

    print(f"[прогон 2] тот же seed={args.seed} — проверка детерминизма")
    r2 = train_one(data, args.seed, device=args.device, epochs=args.epochs,
                   model_kwargs=mk)
    mae2 = float(np.mean(np.abs(r2["pred_w"] - r2["true_w"])))
    identical = bool(np.array_equal(r1["pred_w"], r2["pred_w"]))
    max_dev = float(np.max(np.abs(r1["pred_w"] - r2["pred_w"])))
    print(f"  MAE теста = {mae2:.4f}, макс. расхождение предсказаний = {max_dev:.3e}")

    print("\n=== Критерии пре-спецификации ===")
    s1 = mae1 < threshold
    print(f"S1 (MAE < min(persistence, сезонный наивный)): "
          f"{'ПРОЙДЕН' if s1 else 'НЕ ПРОЙДЕН'} — {mae1:.4f} против {threshold:.4f}")
    print(f"S3 (побитовая воспроизводимость при одном сиде): "
          f"{'ПРОЙДЕН' if identical else 'НЕ ПРОЙДЕН'}")
    if not identical:
        print("     Недетерминизм: попробуйте torch.use_deterministic_algorithms(True) "
              "и CUBLAS_WORKSPACE_CONFIG=:4096:8. Если расхождение остаётся, это надо "
              "внести в §10 пре-спецификации как известное свойство среды, потому что "
              "оно входит в измеряемую seed-компоненту.")
    print("S2: требует истории валидации по эпохам. Минимальная правка train_one — "
          "накапливать val_mae в список и возвращать его в res['val_history']; "
          "критерий: итоговая валидационная MAE не хуже лучшей более чем на 10 %.")

    out = {
        "baselines": base, "threshold_S1": threshold,
        "mae_run1": mae1, "mae_run2": mae2, "val_mae": r1["val_mae"],
        "max_abs_deviation": max_dev,
        "S1_passed": s1, "S3_passed": identical,
        "model_kwargs": mk, "epochs": args.epochs, "seed": args.seed,
    }
    Path("sanity_etth1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("\nЗаписано: sanity_etth1.json")


if __name__ == "__main__":
    main()
