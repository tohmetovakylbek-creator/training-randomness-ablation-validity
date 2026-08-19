"""
patch_model.py
==============
Добавляет в модель два флага, нужных для факторного эксперимента:

    aggregation = "convex" | "sum"
        convex — как сейчас: softmax по модам, веса в сумме дают 1.
        sum    — те же softmax-веса, умноженные на K, то есть в сумме дают K.
                 При равномерных весах это в точности Σ_k u_hat_k. Адаптивность
                 ASWA сохраняется, чинится только масштаб.

    aux_mode = "sum" | "per_mode"
        sum      — как сейчас: Huber(Σ_k u_hat_k, y).
        per_mode — как в исторической реализации: Σ_k Huber(u_hat_k, y),
                   каждая мода по отдельности обязана предсказать полную цель.

ЗАМЕЧАНИЕ. В докстринге vmd_patchtst_aswa.py сказано, что покомпонентных целей
без утечки не существует. Это неверно: историческая реализация подавала каждой
моде в качестве цели тот же самый y, а не будущее её собственной моды. Утечки
там нет — раскладывать будущее не требуется. Похоже, именно это недоразумение и
привело к замене вспомогательной цели.

Запуск из папки репозитория:
    python patch_model.py

Оригиналы сохраняются как models/aswa_original.py и
models/vmd_patchtst_aswa_original.py.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ASWA_OLD_INIT = '''    def __init__(self, n_modes=6, horizon=24, d_emb=16, hidden=32, equal_weights=False):
        super().__init__()
        self.n_modes, self.T = n_modes, horizon
        self.equal_weights = equal_weights'''

ASWA_NEW_INIT = '''    def __init__(self, n_modes=6, horizon=24, d_emb=16, hidden=32, equal_weights=False,
                 aggregation="convex"):
        super().__init__()
        self.n_modes, self.T = n_modes, horizon
        self.equal_weights = equal_weights
        assert aggregation in ("convex", "sum"), aggregation
        self.aggregation = aggregation'''

ASWA_OLD_FWD = '''    def forward(self, mode_forecasts):
        w = self.weights().transpose(0, 1).unsqueeze(0)
        return (mode_forecasts * w).sum(dim=1)'''

ASWA_NEW_FWD = '''    def forward(self, mode_forecasts):
        w = self.weights()
        if self.aggregation == "sum":
            # веса в сумме дают K, а не 1: при равномерных весах это Σ_k u_hat_k.
            # Согласовано со вспомогательной целью вида Huber(Σ_k u_hat_k, y).
            w = w * self.n_modes
        w = w.transpose(0, 1).unsqueeze(0)
        return (mode_forecasts * w).sum(dim=1)'''

M_OLD_SIG = '''        lambda_aux: float = 0.2, lambda_div: float = 0.0,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.lambda_aux, self.lambda_div = lambda_aux, lambda_div
        self.huber_delta = huber_delta'''

M_NEW_SIG = '''        lambda_aux: float = 0.2, lambda_div: float = 0.0,
        huber_delta: float = 1.0,
        # факторный эксперимент: согласованность aux-цели и агрегации
        aux_mode: str = "sum", aggregation: str = "convex",
    ):
        super().__init__()
        self.lambda_aux, self.lambda_div = lambda_aux, lambda_div
        self.huber_delta = huber_delta
        assert aux_mode in ("sum", "per_mode"), aux_mode
        self.aux_mode = aux_mode
        self.aggregation = aggregation'''

M_OLD_ASWA = '''        self.aswa = ASWA(n_modes=n_modes, horizon=T, equal_weights=equal_weights)'''
M_NEW_ASWA = '''        self.aswa = ASWA(n_modes=n_modes, horizon=T, equal_weights=equal_weights,
                         aggregation=aggregation)'''

M_OLD_LOSS = '''        # leakage-free агрегатный aux: Σ_k u_hat_k должен реконструировать y
        agg = out["u_hat"].sum(dim=1)                       # (B, T)
        l_aux = F.huber_loss(agg, target, delta=d)
        total = l_main + self.lambda_aux * l_aux'''

M_NEW_LOSS = '''        if self.aux_mode == "sum":
            # агрегатный aux: Σ_k u_hat_k должен реконструировать y
            agg = out["u_hat"].sum(dim=1)                   # (B, T)
            l_aux = F.huber_loss(agg, target, delta=d)
        else:
            # исторический aux: КАЖДАЯ мода по отдельности предсказывает полную
            # цель; утечки нет, целью служит тот же y
            u = out["u_hat"]                                # (B, K, T)
            l_aux = sum(F.huber_loss(u[:, k, :], target, delta=d)
                        for k in range(u.shape[1]))
        total = l_main + self.lambda_aux * l_aux'''


def apply(path: Path, edits, backup_name):
    src = path.read_text(encoding="utf-8")
    bad = [(n, src.count(o)) for n, o, _ in edits if src.count(o) != 1]
    if bad:
        print(f"\n{path.name}: не удалось применить правку")
        for n, c in bad:
            print(f"  {n}: вхождений {c}, ожидалось 1")
        return False
    for _, o, n in edits:
        src = src.replace(o, n)
    b = path.with_name(backup_name)
    if not b.exists():
        shutil.copy2(path, b)
        print(f"  оригинал сохранён: {b.name}")
    path.write_text(src, encoding="utf-8")
    return True


def main():
    root = Path("models")
    if not (root / "aswa.py").exists():
        print("Запускайте из папки репозитория (там, где лежит models/).")
        sys.exit(1)

    ok1 = apply(root / "aswa.py",
                [("ASWA.__init__", ASWA_OLD_INIT, ASWA_NEW_INIT),
                 ("ASWA.forward", ASWA_OLD_FWD, ASWA_NEW_FWD)],
                "aswa_original.py")
    ok2 = apply(root / "vmd_patchtst_aswa.py",
                [("сигнатура модели", M_OLD_SIG, M_NEW_SIG),
                 ("создание ASWA", M_OLD_ASWA, M_NEW_ASWA),
                 ("вспомогательный лосс", M_OLD_LOSS, M_NEW_LOSS)],
                "vmd_patchtst_aswa_original.py")
    if not (ok1 and ok2):
        sys.exit(2)

    for m in [k for k in list(sys.modules) if k.startswith("models")]:
        del sys.modules[m]
    from models.vmd_patchtst_aswa import VMDPatchTSTASWA
    import inspect
    p = inspect.signature(VMDPatchTSTASWA.__init__).parameters
    print(f"\nГотово. aux_mode={'aux_mode' in p}  aggregation={'aggregation' in p}")
    print("Умолчания не изменились (aux_mode='sum', aggregation='convex'), "
          "поэтому прежние прогоны воспроизводятся без изменений.")


if __name__ == "__main__":
    main()
