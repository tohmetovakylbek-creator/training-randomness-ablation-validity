# -*- coding: utf-8 -*-
"""
probe_control.py
================
Контроль тривиальности линейного probe (замечание рецензии).

Претензия: probe восстанавливает идентичность моды с высокой точностью потому,
что эмбеддинг моды прибавляется прямо к представлению — то есть probe читает
собственный вход механизма, а не выученную структуру. Проверка простая:
тот же probe на моделях, обученных БЕЗ эмбеддингов. Если точность там падает
до случайной, механизм действительно вносит информацию; если остаётся высокой,
probe тривиален и утверждение §5.3 надо снимать.

Обучения нет: используются сохранённые веса, только прямой проход.

Что печатается на объект и сид:
    acc_on    — точность probe на модели с эмбеддингами
    acc_off   — то же на модели без эмбеддингов (КОНТРОЛЬ)
    acc_shuf  — на модели с эмбеддингами при перемешанных метках (нижняя граница)
    chance    — 1/K

Запуск (настольный; нужен GPU только для прямого прохода, на CPU тоже пойдёт):
    python probe_control.py --npz_dir processed_ecl --units_from ecl_selection.json \
        --ckpt_dir ckpt_ecl --prefix client_ --device cuda \
        --ckpt_on  "aux-per_mode_agg-convex_emb-on_seed{seed}.pt" \
        --ckpt_off "aux-per_mode_agg-convex_emb-off_seed{seed}.pt"

Для домохозяйств подставьте своё имя чекпойнтов из factorial_aux_agg.py:
если не знаете — запустите с --list_ckpt, скрипт покажет, что лежит в каталоге.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import load_house, train_one


def get_repr(npz_path: Path, cache_dir: Path, ckpt: Path, seed: int,
             mk: dict, device: str):
    """Прямой проход по тесту с сохранённых весов -> (representation, modes)."""
    data = load_house(npz_path, cache_dir)
    r = train_one(data, seed, device=device, epochs=0, model_kwargs=mk,
                  ckpt_path=ckpt, return_repr=True)
    if not r.get("loaded_from_ckpt", False):
        raise SystemExit(f"Веса не загрузились из {ckpt} — обучение запускать нельзя, "
                         f"проверьте путь и шаблон имени.")
    return np.asarray(r["repr"]), np.asarray(r["modes_te"])


def to_mode_matrix(rep: np.ndarray, K: int):
    """Приводит representation к (n_примеров, d) с метками моды.
    Ожидается форма (окна, K, d); при (окна, K*d) делится на K."""
    if rep.ndim == 3:
        B, k, d = rep.shape
        if k != K:
            raise SystemExit(f"Ожидалось K={K} мод, в representation {k}.")
        X = rep.reshape(B * k, d)
        y = np.tile(np.arange(k), B)
        return X, y
    if rep.ndim == 2 and rep.shape[1] % K == 0:
        B = rep.shape[0]; d = rep.shape[1] // K
        X = rep.reshape(B * K, d)
        y = np.tile(np.arange(K), B)
        return X, y
    raise SystemExit(f"Не понимаю форму representation {rep.shape}: probe по "
                     f"идентичности моды на ней не ставится. Пришлите форму — подстрою.")


def linear_probe(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Мультиномиальная логистическая регрессия, разбиение по окнам пополам,
    чтобы probe не оценивался на тех же окнах, на которых подобран."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    K = int(y.max()) + 1
    n_win = len(y) // K
    cut = n_win // 2
    idx = np.arange(len(y))
    win = idx // K
    tr, te = win < cut, win >= cut
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.transform(X[tr]), y[tr])
    return float(clf.score(sc.transform(X[te]), y[te]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_dir", required=True)
    p.add_argument("--prefix", default="house_", help="client_ для ECL")
    p.add_argument("--units", default="", help="список через запятую")
    p.add_argument("--units_from", default="", help="ecl_selection.json")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_on", default="aux-per_mode_agg-convex_emb-on_seed{seed}.pt",
                   help="в шаблоне доступны {seed} и {unit}")
    p.add_argument("--ckpt_off", default="aux-per_mode_agg-convex_emb-off_seed{seed}.pt")
    p.add_argument("--no_unit_subdir", action="store_true",
                   help="чекпойнты лежат прямо в ckpt_dir, а не в ckpt_dir/<объект>/")
    p.add_argument("--cache_dir", default="cache_probe")
    p.add_argument("--seeds", default="42,7,13,99,2025")
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--list_ckpt", action="store_true")
    p.add_argument("--out", default="probe_control.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.list_ckpt:
        for f in sorted(Path(args.ckpt_dir).rglob("*.pt"))[:40]:
            print(f)
        return

    if args.units:
        units = [u.strip() for u in args.units.split(",") if u.strip()]
    elif args.units_from:
        units = list(json.loads(Path(args.units_from).read_text(encoding="utf-8"))["selected"])
    else:
        units = sorted(q.stem[len(args.prefix):] for q in
                       Path(args.npz_dir).glob(f"{args.prefix}*.npz"))

    MK_ON = {"aux_mode": "per_mode", "aggregation": "convex",
             "use_mode_embeddings": True, "disable_skip": False}
    MK_OFF = dict(MK_ON, use_mode_embeddings=False)

    rows = []
    for u in units:
        npz = Path(args.npz_dir) / f"{args.prefix}{u}.npz"
        if not npz.exists():
            print(f"{u}: нет {npz} — пропуск"); continue
        cdir = Path(args.ckpt_dir) if args.no_unit_subdir else Path(args.ckpt_dir) / u
        for s in seeds:
            try:
                rep_on, _ = get_repr(npz, Path(args.cache_dir) / u,
                                     cdir / args.ckpt_on.format(seed=s, unit=u), s, MK_ON, args.device)
                rep_off, _ = get_repr(npz, Path(args.cache_dir) / u,
                                      cdir / args.ckpt_off.format(seed=s, unit=u), s, MK_OFF, args.device)
            except SystemExit as e:
                print(e); return
            Xon, yon = to_mode_matrix(rep_on, args.K)
            Xoff, yoff = to_mode_matrix(rep_off, args.K)
            rng = np.random.default_rng(s)
            acc_on = linear_probe(Xon, yon)
            acc_off = linear_probe(Xoff, yoff)
            acc_shuf = linear_probe(Xon, rng.permutation(yon))
            rows.append({"unit": u, "seed": s, "acc_on": acc_on,
                         "acc_off": acc_off, "acc_shuf": acc_shuf,
                         "chance": 1.0 / args.K})
            print(f"{u:12s} seed={s:5d}  on={acc_on:.3f}  off={acc_off:.3f}  "
                  f"shuffled={acc_shuf:.3f}  chance={1/args.K:.3f}")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if not rows:
        print("нет результатов"); return
    on = np.array([r["acc_on"] for r in rows]); off = np.array([r["acc_off"] for r in rows])
    sh = np.array([r["acc_shuf"] for r in rows])
    print("\n=== Итог ===")
    print(f"с эмбеддингами:  медиана {np.median(on):.3f}  диапазон {on.min():.3f}-{on.max():.3f}")
    print(f"БЕЗ эмбеддингов: медиана {np.median(off):.3f}  диапазон {off.min():.3f}-{off.max():.3f}")
    print(f"перемешанные метки: медиана {np.median(sh):.3f}   случайный уровень {1/args.K:.3f}")
    from scipy import stats
    w = stats.wilcoxon(on, off)
    print(f"парный критерий Уилкоксона (on против off): p = {w.pvalue:.3g}")
    print("\nЧитается так: если off близко к случайному уровню, probe нетривиален "
          "и вывод §5.3 остаётся. Если off близко к on, probe читает вход механизма, "
          "и утверждение надо снимать.")
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"записано: {args.out}")


if __name__ == "__main__":
    main()
