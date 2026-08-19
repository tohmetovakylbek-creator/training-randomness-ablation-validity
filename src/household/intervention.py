"""
intervention.py
===============
Каузальная интервенция над mode identity (ревизия под EAAI, замечание семинара
«probing доказывает декодируемость, но не использование»).

Идея: взять ОБУЧЕННУЮ full-модель и, ничего не переобучая, подменить на inference
соответствие «мода k -> embedding e_k». Если голова прогноза действительно
использует идентичность моды, перестановка должна ухудшить прогноз. Если MAE не
меняется при probe accuracy ~0.99 — представление кодирует идентичность, но
прогноз её не использует (representation-utility dissociation в интервенционной,
а не наблюдательной форме).

Условия:
    identity   — исходное соответствие (контроль);
    cyclic     — мода k получает embedding моды (k+1) mod K (детерминированно);
    random     — случайные нетождественные перестановки (n_perms штук, усреднение);
    zeroed     — embeddings обнулены на inference (отделяет «использует
                 идентичность» от «использует вектор как произвольный сдвиг»);
    shuffled_per_window — перестановка своя для каждого тестового окна
                 (разрушает идентичность, сохраняя маргинальное распределение
                 добавки; контроль на «модель просто любит любой сдвиг»).

КРИТИЧНО — две метрики, не одна:
    y_vmd   — выход VMD-пути ДО gated fusion. Только он может зависеть от
              mode embeddings. Здесь измеряется, использует ли VMD-путь
              идентичность.
    y_final — итоговый прогноз после fusion с skip-путём. Здесь измеряется,
              доходит ли эффект до пользователя.
Если сообщать только y_final, эффект окажется размыт gated skip (который, по
результатам абляции, доминирует), и нулевой результат будет артефактом
архитектуры, а не свойством механизма. Рецензент EAAI это поймает.

Запуск:
    python intervention.py --processed <path>/processed --houses 1,2,4,5 \
        --dataset ukdale --ckpt_dir checkpoints --device cuda --out results/intervention

Чекпойнты: если их нет, модели обучаются один раз и сохраняются; при повторных
запусках обучение не повторяется.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from train import load_house, train_one, SEEDS
from evaluate import inverse, metrics, per_window_abs_error


# --------------------------------------------------------------- перестановки
def cyclic_perm(K: int, shift: int = 1) -> np.ndarray:
    return np.roll(np.arange(K), shift)


def random_perms(K: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """n случайных перестановок без неподвижной тождественной."""
    out, seen = [], set()
    identity = tuple(range(K))
    guard = 0
    while len(out) < n and guard < 100 * n:
        guard += 1
        p = rng.permutation(K)
        t = tuple(p)
        if t == identity or t in seen:
            continue
        seen.add(t)
        out.append(p)
    return out


class swapped_embeddings:
    """Контекст-менеджер: временно подменяет mode_embed в vmd_encoder.

    Работает и для nn.Parameter (learnable), и для buffer (fixed-режимы),
    потому что пишет через .data и восстанавливает исходный тензор на выходе.
    """

    def __init__(self, model, new_embed: torch.Tensor):
        self.enc = model.vmd_encoder
        self.new = new_embed
        self.orig = None

    def __enter__(self):
        self.orig = self.enc.mode_embed.detach().clone()
        self.enc.mode_embed.data.copy_(self.new.to(self.enc.mode_embed.device))
        return self.enc

    def __exit__(self, *exc):
        self.enc.mode_embed.data.copy_(self.orig)
        return False


class swapped_aswa:
    """Временно переставляет ПОМОДОВЫЕ веса ASWA той же перестановкой.

    weights() = softmax(fc2(act(fc1(horizon_embed))) + horizon_bias), колонка k
    отвечает моде k. Чтобы колонка k стала колонкой perm[k], переставляем строки
    fc2.weight, элементы fc2.bias и колонки horizon_bias.
    """

    def __init__(self, model, perm):
        self.a = model.aswa
        self.perm = torch.as_tensor(np.asarray(perm), dtype=torch.long)
        self.orig = None

    def __enter__(self):
        a = self.a
        p = self.perm.to(a.fc2.weight.device)
        self.orig = (a.fc2.weight.detach().clone(), a.fc2.bias.detach().clone(),
                     a.horizon_bias.detach().clone())
        a.fc2.weight.data.copy_(self.orig[0][p])
        a.fc2.bias.data.copy_(self.orig[1][p])
        a.horizon_bias.data.copy_(self.orig[2][:, p])
        return a

    def __exit__(self, *exc):
        a = self.a
        a.fc2.weight.data.copy_(self.orig[0])
        a.fc2.bias.data.copy_(self.orig[1])
        a.horizon_bias.data.copy_(self.orig[2])
        return False


# ------------------------------------------------------------------- прогоны
@torch.no_grad()
def forward_test(model, Mte, Xte, device, batch=256):
    """Возвращает (y_final, y_vmd, gate) в нормированной шкале."""
    ys, yv, gs = [], [], []
    for i in range(0, len(Mte), batch):
        mb = torch.tensor(Mte[i:i + batch]).to(device)
        xb = torch.tensor(Xte[i:i + batch]).to(device)
        o = model(mb, xb, return_gate=True)
        ys.append(o["y"].cpu().numpy())
        yv.append(o["y_vmd"].cpu().numpy())
        gs.append(o["gate"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(yv), np.concatenate(gs)


@torch.no_grad()
def forward_test_per_window_perm(model, Mte, Xte, device, rng, batch=256):
    """Условие shuffled_per_window: для каждого окна своя перестановка мод.

    Реализуется перестановкой самих мод во входе (эквивалентно перестановке
    соответствия мода<->embedding, но позволяет менять её пооконно).
    """
    K = Mte.shape[1]
    ys, yv = [], []
    for i in range(0, len(Mte), batch):
        m = Mte[i:i + batch].copy()
        for j in range(len(m)):
            p = rng.permutation(K)
            while tuple(p) == tuple(range(K)):
                p = rng.permutation(K)
            m[j] = m[j][p]
        mb = torch.tensor(m).to(device)
        xb = torch.tensor(Xte[i:i + batch]).to(device)
        o = model(mb, xb)
        ys.append(o["y"].cpu().numpy())
        yv.append(o["y_vmd"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(yv)


# ------------------------------------------------------------------- пробинг
def probe_accuracy(repr_te: np.ndarray, seed: int = 0) -> dict:
    """Linear probe + КОНТРОЛЬ со случайными метками (замечание семинара
    о probe controls / selectivity). Selectivity = acc - acc_random_label."""
    n, K, d = repr_te.shape
    X = repr_te.reshape(n * K, d)
    y = np.tile(np.arange(K), n)
    clf = LogisticRegression(max_iter=1000)
    acc = float(cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean())
    rng = np.random.default_rng(seed)
    y_rand = rng.permutation(y)
    acc_ctrl = float(cross_val_score(LogisticRegression(max_iter=1000), X, y_rand,
                                     cv=3, scoring="accuracy").mean())
    return {"probe_accuracy": acc, "control_task_accuracy": acc_ctrl,
            "selectivity": acc - acc_ctrl, "chance": 1.0 / K}


# --------------------------------------------------------------- block bootstrap
def moving_block_bootstrap_ci(diff: np.ndarray, block: int = 24, n_boot: int = 5000,
                              alpha: float = 0.05, seed: int = 0) -> dict:
    """CI среднего для ряда попарных разностей с временной зависимостью
    (перекрывающиеся walk-forward окна). block ~ длина горизонта в окнах."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    block = int(min(max(1, block), n))
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([diff[s:s + block] for s in idx])[:n]
        means[b] = sample.mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "block": block, "n_boot": n_boot}


# ------------------------------------------------------------------- эксперимент
def ckpt_for(ckpt_dir, house_tag, aux, agg, seed):
    """Имя чекпойнта в раскладке factorial_aux_agg.py; если такого нет —
    падаем на старое имя intervention.py (конфигурация по умолчанию)."""
    p = Path(ckpt_dir) / f"{house_tag}_aux-{aux}_agg-{agg}_emb-on_seed{seed}.pt"
    if p.exists():
        return p
    legacy = Path(ckpt_dir) / f"{house_tag}_full_seed{seed}.pt"
    return legacy if legacy.exists() else p


def run_house(house_data, house_tag, ckpt_dir, device="cpu", epochs=60,
              seeds=SEEDS, n_perms=50, block=24, seed_rng=0,
              aux_mode="sum", aggregation="convex"):
    Xte, Mte, Yte = house_data["te"]
    lo, hi = house_data["scaler"]
    true_w = inverse(Yte, lo, hi)
    K = Mte.shape[1]
    rng = np.random.default_rng(seed_rng)
    perms = random_perms(K, n_perms, rng)

    # условия с одним прогоном на seed
    single = ["identity", "cyclic", "zeroed", "aswa_perm", "shuffled_per_window", "relabel_all"]
    acc_final = {c: [] for c in single}
    acc_vmd = {c: [] for c in single}
    # random: (n_perms, n_seeds, n_windows, T) — усредняем по seed, НЕ по перестановкам
    rnd_final = [[] for _ in perms]
    rnd_vmd = [[] for _ in perms]
    # диагностика: усреднение предсказаний и по перестановкам тоже
    ens_final, ens_vmd = [], []
    gates, probes = [], []

    cyc = cyclic_perm(K, 1)

    for s in seeds:
        ckpt = ckpt_for(ckpt_dir, house_tag, aux_mode, aggregation, s)
        r = train_one(house_data, s, device=device, epochs=epochs,
                      model_kwargs={"aux_mode": aux_mode, "aggregation": aggregation},
                      ckpt_path=ckpt, return_model=True, return_repr=True)
        model = r["model"]
        probes.append(probe_accuracy(r["repr"], seed=s))
        orig = model.vmd_encoder.mode_embed.detach().clone()

        yf, yv, g = forward_test(model, Mte, Xte, device)
        acc_final["identity"].append(yf); acc_vmd["identity"].append(yv)
        gates.append(float(g.mean()))

        with swapped_embeddings(model, orig[cyc]):
            yf, yv, _ = forward_test(model, Mte, Xte, device)
        acc_final["cyclic"].append(yf); acc_vmd["cyclic"].append(yv)

        with swapped_embeddings(model, torch.zeros_like(orig)):
            yf, yv, _ = forward_test(model, Mte, Xte, device)
        acc_final["zeroed"].append(yf); acc_vmd["zeroed"].append(yv)

        # только веса ASWA, embeddings нетронуты
        with swapped_aswa(model, cyc):
            yf, yv, _ = forward_test(model, Mte, Xte, device)
        acc_final["aswa_perm"].append(yf); acc_vmd["aswa_perm"].append(yv)

        # пооконная перестановка самих мод (ломает и identity, и привязку к ASWA)
        yf, yv = forward_test_per_window_perm(model, Mte, Xte, device, rng)
        acc_final["shuffled_per_window"].append(yf)
        acc_vmd["shuffled_per_window"].append(yv)

        # НУЛЕВОЙ КОНТРОЛЬ: моды, embeddings и веса ASWA переставлены СОГЛАСОВАННО.
        # Это чистое переобозначение, выход обязан совпасть с identity бит в бит.
        with swapped_embeddings(model, orig[cyc]), swapped_aswa(model, cyc):
            yf, yv, _ = forward_test(model, Mte[:, cyc, :], Xte, device)
        acc_final["relabel_all"].append(yf); acc_vmd["relabel_all"].append(yv)

        # глобальные случайные перестановки embeddings
        pf, pv = [], []
        for i, p in enumerate(perms):
            with swapped_embeddings(model, orig[p]):
                a, b, _ = forward_test(model, Mte, Xte, device)
            rnd_final[i].append(a); rnd_vmd[i].append(b)
            pf.append(a); pv.append(b)
        ens_final.append(np.mean(pf, axis=0)); ens_vmd.append(np.mean(pv, axis=0))

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    out = {"house": house_tag, "K": K, "n_test_windows": int(len(Yte)),
           "aux_mode": aux_mode, "aggregation": aggregation,
           "n_perms": len(perms), "mean_gate": float(np.mean(gates)),
           "probe": {k: float(np.mean([p[k] for p in probes])) for k in probes[0]},
           "conditions": {}, "per_window": {}, "pred": {}}

    base_err = {}
    for path_name, store in (("y_final", acc_final), ("y_vmd", acc_vmd)):
        for c in single:
            ens = inverse(np.mean(store[c], axis=0), lo, hi)
            err = per_window_abs_error(ens, true_w)
            out["per_window"][f"{path_name}|{c}"] = err
            out["pred"][f"{path_name}|{c}"] = ens
            m = metrics(ens, true_w)
            out["conditions"][f"{path_name}|{c}"] = {
                "MAE": m["MAE"], "RMSE": m["RMSE"],
                "pred_std": float(ens.std()), "bias": float((ens - true_w).mean())}
            if c == "identity":
                base_err[path_name] = err

    # random: усредняем ОШИБКИ по перестановкам, а не предсказания
    for path_name, store, ens_store in (("y_final", rnd_final, ens_final),
                                        ("y_vmd", rnd_vmd, ens_vmd)):
        per_perm_err = np.stack([
            per_window_abs_error(inverse(np.mean(store[i], axis=0), lo, hi), true_w)
            for i in range(len(perms))])            # (n_perms, n_windows)
        err = per_perm_err.mean(axis=0)
        out["per_window"][f"{path_name}|random"] = err
        per_perm_pred = np.stack([inverse(np.mean(store[i], axis=0), lo, hi)
                                  for i in range(len(perms))])
        out["conditions"][f"{path_name}|random"] = {
            "MAE": float(err.mean()),
            "MAE_sd_across_perms": float(per_perm_err.mean(axis=1).std(ddof=1)),
            "RMSE": float(np.mean([metrics(p, true_w)["RMSE"] for p in per_perm_pred])),
            "pred_std": float(np.mean([p.std() for p in per_perm_pred])),
            "bias": float(np.mean([(p - true_w).mean() for p in per_perm_pred])),
        }
        # диагностика: тот же прогон, но с усреднением предсказаний по перестановкам
        ens = inverse(np.mean(ens_store, axis=0), lo, hi)
        err_e = per_window_abs_error(ens, true_w)
        out["per_window"][f"{path_name}|random_pred_averaged"] = err_e
        out["pred"][f"{path_name}|random_pred_averaged"] = ens
        out["conditions"][f"{path_name}|random_pred_averaged"] = {
            "MAE": float(err_e.mean()), "RMSE": float(metrics(ens, true_w)["RMSE"]),
            "pred_std": float(ens.std()), "bias": float((ens - true_w).mean())}

    all_conds = [c for c in single if c != "identity"] + ["random", "random_pred_averaged"]
    for path_name in ("y_final", "y_vmd"):
        for c in all_conds:
            d = out["per_window"][f"{path_name}|{c}"] - base_err[path_name]
            out["conditions"][f"{path_name}|{c}"]["delta_MAE_vs_identity"] = \
                moving_block_bootstrap_ci(d, block=block, seed=seed_rng)

    # проверка нулевого контроля
    z = out["conditions"]["y_vmd|relabel_all"]["delta_MAE_vs_identity"]["mean"]
    out["relabel_all_check"] = {"delta_y_vmd": z, "passes": bool(abs(z) < 1e-4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--houses", default="1,2,4,5")
    ap.add_argument("--dataset", default="ukdale", help="метка датасета для имён файлов")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_perms", type=int, default=50)
    ap.add_argument("--block", type=int, default=24, help="длина блока в окнах для bootstrap")
    ap.add_argument("--out", default="results/intervention")
    ap.add_argument("--aux_mode", default="sum", choices=["sum", "per_mode"])
    ap.add_argument("--aggregation", default="convex", choices=["convex", "sum"])
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                    help="должны совпадать с сидами, которыми обучались чекпойнты")
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    cell = f"aux-{args.aux_mode}_agg-{args.aggregation}"

    proc = Path(args.processed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache = proc / "vmd_cache"

    all_res, npz_store = {}, {}
    for h in [x.strip() for x in args.houses.split(",")]:
        tag = f"{args.dataset}_house_{h}"
        print(f"\n===== {tag} | {cell} =====")
        hd = load_house(proc / f"house_{h}.npz", cache)
        res = run_house(hd, tag, ckpt_dir, device=args.device, epochs=args.epochs,
                        seeds=seeds, n_perms=args.n_perms, block=args.block,
                        aux_mode=args.aux_mode, aggregation=args.aggregation)
        for k, v in res.pop("per_window").items():
            npz_store[f"{tag}|err|{k}"] = v
        for k, v in res.pop("pred").items():
            npz_store[f"{tag}|pred|{k}"] = v
        npz_store[f"{tag}|true"] = inverse(hd["te"][2], *hd["scaler"])
        all_res[tag] = res
        print(f"  probe_acc={res['probe']['probe_accuracy']:.3f} "
              f"(control={res['probe']['control_task_accuracy']:.3f})  "
              f"mean_gate={res['mean_gate']:.3f}")
        chk = res["relabel_all_check"]
        print(f"  нулевой контроль relabel_all: dMAE={chk['delta_y_vmd']:+.2e} "
              f"{'OK' if chk['passes'] else 'ВНИМАНИЕ: должно быть ~0'}")
        for c in ("cyclic", "random", "random_pred_averaged", "zeroed",
                  "aswa_perm", "shuffled_per_window"):
            dv = res["conditions"][f"y_vmd|{c}"]["delta_MAE_vs_identity"]
            df = res["conditions"][f"y_final|{c}"]["delta_MAE_vs_identity"]
            print(f"  {c:20s} dMAE y_vmd={dv['mean']:+7.2f} "
                  f"[{dv['ci_low']:+.2f},{dv['ci_high']:+.2f}]  "
                  f"y_final={df['mean']:+7.2f} [{df['ci_low']:+.2f},{df['ci_high']:+.2f}]")
        (outdir / f"intervention_{args.dataset}_{cell}.json").write_text(
            json.dumps(all_res, ensure_ascii=False, indent=2))
        np.savez_compressed(outdir / f"per_window_{args.dataset}_{cell}.npz", **npz_store)

    print(f"\nСохранено: {outdir}/intervention_{args.dataset}_{cell}.json")


if __name__ == "__main__":
    main()
