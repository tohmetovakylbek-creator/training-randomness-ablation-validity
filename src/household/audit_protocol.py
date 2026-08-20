"""
audit_protocol.py
=================
Восстанавливает протокол предобработки из готовых .npz, ничего не пересчитывая.

Зачем. Репозиторий vmd-lean-load-forecasting содержит только data/ukdale_loader.py;
загрузчика REFIT в нём нет. В докстринге сказано, что протокол «ИДЕНТИЧЕН
эксперименту на REFIT», но идентичность нигде не проверена — REFIT-эксперимент
живёт в отдельном проекте со своей предобработкой. Прежде чем тратить часы GPU на
факторный прогон, надо убедиться, что данные двух датасетов действительно
получены одним протоколом. Иначе кросс-датасетное утверждение статьи опирается на
две разные предобработки, и любое расхождение результатов смешано с расхождением
данных.

Что делает скрипт. По числу окон в train/val/test и по L, T он восстанавливает:
    * длину исходного часового ряда N,
    * фактические пропорции хронологического разбиения,
    * шаг окон на тесте (test_stride),
и сверяет их с эталонными 70/15/15 и stride=1/1/24. Дополнительно печатает
диапазон min-max скейлера — резко разные диапазоны у train и всего ряда выдают
глобальную нормировку (утечку).

Проверка работает так: N оценивается из n_tr, затем из N предсказывается n_va и
сверяется с фактическим. Совпадение подтверждает, что train/val нарезаны с
шагом 1 и пропорции те, что заявлены.

Запуск:
    python audit_protocol.py --processed "<...>\\uk-dale-disaggregated\\processed" --label UK-DALE
    python audit_protocol.py --processed "<...>\\REFIT\\processed" --label REFIT

Оба вывода сравнить построчно.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REF_SPLIT = (0.70, 0.15, 0.15)
REF_STRIDES = (1, 1, 24)


def audit_house(npz_path: Path, L: int, T: int, split=REF_SPLIT):
    d = np.load(npz_path)
    keys = set(d.keys())
    need = {"Xtr", "Ytr", "Xva", "Yva", "Xte", "Yte", "scaler_lo", "scaler_hi"}
    missing = need - keys
    out = {"file": npz_path.name, "missing_keys": sorted(missing)}
    if missing:
        return out

    n_tr, n_va, n_te = len(d["Xtr"]), len(d["Xva"]), len(d["Xte"])
    L_act, T_act = d["Xtr"].shape[1], d["Ytr"].shape[1]
    out.update({"n_tr": n_tr, "n_va": n_va, "n_te": n_te,
                "L": int(L_act), "T": int(T_act),
                "scaler_lo": float(d["scaler_lo"]), "scaler_hi": float(d["scaler_hi"])})
    if (L_act, T_act) != (L, T):
        out["warn_LT"] = f"L/T = {L_act}/{T_act}, ожидалось {L}/{T}"

    # N из n_tr при шаге 1 на train
    N = (n_tr + L_act + T_act - 1) / split[0]
    out["implied_series_hours"] = round(N, 1)
    out["implied_series_days"] = round(N / 24, 1)

    # предсказание n_va при шаге 1 -> проверка пропорций
    n_va_pred = split[1] * N - L_act - T_act + 1
    out["n_va_predicted"] = round(n_va_pred, 1)
    out["val_matches_protocol"] = bool(abs(n_va_pred - n_va) <= max(2, 0.01 * n_va))

    # шаг окон на тесте
    usable_te = split[2] * N - L_act - T_act
    out["implied_test_stride"] = round(usable_te / (n_te - 1), 2) if n_te > 1 else None

    # фактические доли, если считать, что все три сегмента нарезаны с шагом 1
    tot = n_tr + n_va + n_te
    out["naive_window_shares"] = [round(x / tot, 3) for x in (n_tr, n_va, n_te)]

    out["test_days"] = round(n_te * (out["implied_test_stride"] or 1) / 24, 1) \
        if out["implied_test_stride"] else None
    out["test_underpowered"] = bool(n_te < 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--label", default="dataset")
    ap.add_argument("--L", type=int, default=168)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--pattern", default="house_*.npz",
                    help="в REFIT файлы могут называться иначе, напр. House*.npz")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    proc = Path(args.processed)
    if not proc.exists():
        print(f"Папка не найдена: {proc}")
        return
    files = sorted(proc.glob(args.pattern))
    if not files:
        print(f"По шаблону {args.pattern} в {proc} ничего нет. Содержимое:")
        for p in sorted(proc.iterdir())[:25]:
            print("   ", p.name)
        return

    print(f"\n===== {args.label}: {proc} =====")
    print(f"эталон: разбиение {REF_SPLIT}, шаги окон {REF_STRIDES}, L/T = {args.L}/{args.T}\n")
    res = []
    for f in files:
        r = audit_house(f, args.L, args.T)
        res.append(r)
        if r.get("missing_keys"):
            print(f"{f.name}: отсутствуют ключи {r['missing_keys']}")
            continue
        flag_v = "OK" if r["val_matches_protocol"] else "НЕ СХОДИТСЯ"
        s = r["implied_test_stride"]
        flag_s = "OK" if s and abs(s - REF_STRIDES[2]) < 1.0 else "ОТЛИЧАЕТСЯ"
        pw = "  МАЛО ДЛЯ СТАТИСТИКИ" if r["test_underpowered"] else ""
        print(f"{f.name}")
        print(f"   окна  tr={r['n_tr']:>7d}  va={r['n_va']:>6d}  te={r['n_te']:>5d}"
              f"   L/T={r['L']}/{r['T']}")
        print(f"   ряд   ~{r['implied_series_days']} сут   "
              f"val предсказан {r['n_va_predicted']} -> {flag_v}")
        print(f"   test_stride ~{s} -> {flag_s}   тест ~{r['test_days']} сут{pw}")
        print(f"   скейлер [{r['scaler_lo']:.1f}, {r['scaler_hi']:.1f}]")
        if "warn_LT" in r:
            print(f"   ВНИМАНИЕ: {r['warn_LT']}")
        print()

    ok_v = sum(1 for r in res if r.get("val_matches_protocol"))
    ok_s = sum(1 for r in res if r.get("implied_test_stride")
               and abs(r["implied_test_stride"] - REF_STRIDES[2]) < 1.0)
    weak = [r["file"] for r in res if r.get("test_underpowered")]
    print(f"итог: пропорции сходятся у {ok_v}/{len(res)}, "
          f"test_stride совпадает у {ok_s}/{len(res)}")
    if weak:
        print(f"тест меньше 100 окон: {', '.join(weak)}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"label": args.label, "processed": str(proc), "houses": res},
            ensure_ascii=False, indent=2))
        print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
