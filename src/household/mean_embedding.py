# -*- coding: utf-8 -*-
"""
mean_embedding.py
=================
Контроль выхода за пределы обучающего распределения для интервенции §5.3.2.

Претензия рецензии: зануление эмбеддингов мод ставит модель в состояние,
которого она при обучении не видела, поэтому ухудшение прогноза может отражать
шок от нетипичного входа, а не изъятие информации о моде.

Контроль: заменить каждый эмбеддинг на СРЕДНЕЕ по модам. Норма и типичное
направление вектора сохраняются, различающая моды составляющая исчезает.
Если эффект такой же, как при занулении, интервенция валидна; если зануление
бьёт заметно сильнее, часть измеренного эффекта — артефакт.

Сравниваются три состояния одной и той же обученной модели, без переобучения:
    intact  — как есть
    mean    — все строки таблицы эмбеддингов заменены на их среднее
    zero    — все строки обнулены (прежняя интервенция)

Таблица эмбеддингов ищется автоматически: параметр формы (K, d), где K — число
мод. Если кандидатов несколько или ни одного, скрипт скажет об этом и покажет
подходящие имена — тогда задайте --emb_param явно.

Запуск (настольный):
    python mean_embedding.py --npz_dir processed_sheerm --prefix house_ \\
        --ckpt_dir checkpoints_factorial/sheerm --no_unit_subdir \\
        --ckpt "sheerm_house_{unit}_aux-per_mode_agg-convex_emb-on_seed{seed}.pt" \\
        --units 1,2,3,4,5,8,9,10,11,12,13 --cache_dir cache_probe_sheerm \\
        --device cuda --out meanemb_sheerm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import load_house, train_one
from evaluate import inverse


def find_embedding_param(model, K: int, explicit: str | None):
    cands = [(n, p) for n, p in model.named_parameters()
             if p.ndim == 2 and p.shape[0] == K]
    if explicit:
        for n, p in model.named_parameters():
            if n == explicit:
                return n, p
        raise SystemExit(f"Нет параметра {explicit}. Есть: "
                         f"{[n for n,_ in model.named_parameters()]}")
    if len(cands) == 1:
        return cands[0]
    raise SystemExit(
        f"Кандидатов на таблицу эмбеддингов: {len(cands)} "
        f"({[n for n,_ in cands]}). Задайте --emb_param явно.")


def predict(model, data, device: str) -> np.ndarray:
    Xte, Mte, _ = data["te"]
    lo, hi = data["scaler"]
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Mte).to(device), torch.tensor(Xte).to(device))
    y = out["y"] if isinstance(out, dict) else out
    return inverse(y.detach().cpu().numpy(), lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--prefix", default="house_")
    ap.add_argument("--units", default="")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--ckpt", default="{unit}_aux-per_mode_agg-convex_emb-on_seed{seed}.pt")
    ap.add_argument("--no_unit_subdir", action="store_true")
    ap.add_argument("--cache_dir", default="cache_meanemb")
    ap.add_argument("--seeds", default="42,7,13,99,2025")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--emb_param", default=None)
    ap.add_argument("--perms", type=int, default=50,
                    help="число неединичных перестановок строк таблицы эмбеддингов "
                         "для норм-сохраняющего контроля (условие random)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="mean_embedding.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    units = ([u.strip() for u in args.units.split(",") if u.strip()] or
             sorted(q.stem[len(args.prefix):] for q in
                    Path(args.npz_dir).glob(f"{args.prefix}*.npz")))
    MK = {"aux_mode": "per_mode", "aggregation": "convex",
          "use_mode_embeddings": True, "disable_skip": False}

    rows = []
    for u in units:
        npz = Path(args.npz_dir) / f"{args.prefix}{u}.npz"
        if not npz.exists():
            print(f"{u}: нет {npz} — пропуск"); continue
        cdir = Path(args.ckpt_dir) if args.no_unit_subdir else Path(args.ckpt_dir) / u
        for s in seeds:
            ck = cdir / args.ckpt.format(unit=u, seed=s)
            data = load_house(npz, Path(args.cache_dir) / u)
            r = train_one(data, s, device=args.device, epochs=0, model_kwargs=MK,
                          ckpt_path=ck, return_model=True)
            if not r.get("loaded_from_ckpt", False):
                raise SystemExit(f"Веса не загрузились из {ck} — прерываю, "
                                 f"чтобы не измерять необученную модель.")
            model, true_w = r["model"], r["true_w"]
            name, param = find_embedding_param(model, args.K, args.emb_param)
            orig = param.detach().clone()

            p_intact = predict(model, data, args.device)
            with torch.no_grad():
                param.copy_(orig.mean(dim=0, keepdim=True).expand_as(orig))
            p_mean = predict(model, data, args.device)
            with torch.no_grad():
                param.zero_()
            p_zero = predict(model, data, args.device)

            # норм-сохраняющий контроль: перестановка строк, усреднение ОШИБОК
            # (не предсказаний), чтобы усреднение не работало как ансамбль
            rng = np.random.default_rng(s)
            K = orig.shape[0]
            errs = []
            for _ in range(args.perms):
                while True:
                    perm = rng.permutation(K)
                    if not np.array_equal(perm, np.arange(K)):
                        break
                with torch.no_grad():
                    param.copy_(orig[torch.as_tensor(perm, device=orig.device)])
                errs.append(np.abs(predict(model, data, args.device) - true_w))
            mae_perm = float(np.mean(np.stack(errs)))
            with torch.no_grad():
                param.copy_(orig)

            mae = lambda p: float(np.mean(np.abs(p - true_w)))
            m_i, m_m, m_z = mae(p_intact), mae(p_mean), mae(p_zero)
            rows.append({"unit": u, "seed": s, "emb_param": name,
                         "mae_intact": m_i, "mae_mean": m_m, "mae_zero": m_z,
                         "mae_perm": mae_perm,
                         "delta_mean": m_m - m_i, "delta_zero": m_z - m_i,
                         "delta_perm": mae_perm - m_i, "n_perms": args.perms,
                         "emb_norm_orig": float(orig.norm(dim=1).mean()),
                         "emb_norm_mean": float(orig.mean(dim=0).norm())})
            print(f"{u:6s} seed={s:5d}  intact={m_i:9.3f}  "
                  f"mean={m_m:9.3f} ({m_m-m_i:+7.3f})  "
                  f"perm={mae_perm:9.3f} ({mae_perm-m_i:+7.3f})  "
                  f"zero={m_z:9.3f} ({m_z-m_i:+7.3f})")
            del r, model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if not rows:
        print("нет результатов"); return
    dm = np.array([r["delta_mean"] for r in rows])
    dz = np.array([r["delta_zero"] for r in rows])
    dp = np.array([r["delta_perm"] for r in rows])
    mi = np.array([r["mae_intact"] for r in rows])
    print("\n=== Итог по парам объект-сид ===")
    for lbl, d in (("перестановка (норма сохранена)", dp),
                   ("замена на среднее", dm),
                   ("зануление", dz)):
        print(f"  {lbl:32s} медиана {np.median(d):+8.3f} Вт  "
              f"{np.median(d / mi) * 100:+6.2f} % от MAE")
    from scipy import stats
    print(f"\nУилкоксон zero против perm: p = {stats.wilcoxon(dz, dp).pvalue:.3g}")
    print(f"Уилкоксон mean против perm: p = {stats.wilcoxon(dm, dp).pvalue:.3g}")
    print("\nЧитается так: перестановка сохраняет норму и распределение векторов, "
          "разрушая только соответствие моде. Если её эффект близок к занулению — "
          "дело в информации, и прежние числа §5.3.3 в силе. Если близок к нулю, "
          "а зануление и среднее дают больше — модель реагирует на изменение "
          "нормы, и первичным надо делать эффект перестановки.")
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"записано: {args.out}")


if __name__ == "__main__":
    main()
