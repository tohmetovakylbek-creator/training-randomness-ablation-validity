"""
models/vmd_patchtst_aswa.py
===========================
Полная гибридная модель VMD-PatchTST-ASWA (диплом, рисунок 4) + флаги ablation.

Поток (вход уже декомпозирован VMD вне модели и подаётся как modes (B,K,L);
исходный сигнал raw (B,L) для skip-пути):

    modes --(PatchTST + mode embeddings)--> u_hat (B,K,T)
    u_hat --(ASWA)--------------------------> y_vmd (B,T)
    raw   --(lite PatchTST, K=1)------------> y_skip (B,T)
    (y_vmd, y_skip) --(gated fusion)--------> y_final (B,T)

Составная функция потерь (диплом, формула 6, с поправкой против утечки):
    L = L_main + lambda_aux · L_aux  [+ lambda_div · L_div]
    L_main = Huber(y_final, y)
    L_aux  = Huber(Σ_k u_hat_k, y)   <-- агрегатная supervised-цель.

ПРИМЕЧАНИЕ ПО L_aux: в дипломе сказано «сумма Huber Loss отдельных мод».
Покомпонентных целей без утечки не существует (для них пришлось бы разложить
будущее). Поэтому реализован единственный leakage-free вариант: невзвешенная
сумма прогнозов мод должна реконструировать целевой сигнал — это даёт градиент
каждому энкодеру независимо от веса моды в ASWA. Подтвердите, что это
соответствует исходному замыслу студентки.

Флаги ablation:
    use_mode_embeddings, mode_embed_mode  -> P2 (embeddings) и P5 (контроль)
    equal_weights (ASWA->mean)            -> вклад ASWA
    disable_skip                          -> вклад skip connection
    lambda_aux, lambda_div                -> настройка/проверка функции потерь
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchtst import PatchTSTEncoder
from .aswa import ASWA, GatedSkip


class VMDPatchTSTASWA(nn.Module):
    def __init__(
        self,
        n_modes: int = 6, L: int = 168, T: int = 24, patch_len: int = 24,
        # mode embeddings (центральный компонент)
        use_mode_embeddings: bool = True, mode_embed_mode: str = "learnable",
        # ASWA / skip
        equal_weights: bool = False, disable_skip: bool = False,
        # потери
        lambda_aux: float = 0.2, lambda_div: float = 0.0,
        huber_delta: float = 1.0,
        # факторный эксперимент: согласованность aux-цели и агрегации
        aux_mode: str = "sum", aggregation: str = "convex",
    ):
        super().__init__()
        self.lambda_aux, self.lambda_div = lambda_aux, lambda_div
        self.huber_delta = huber_delta
        assert aux_mode in ("sum", "per_mode"), aux_mode
        self.aux_mode = aux_mode
        self.aggregation = aggregation

        # VMD-путь: shared PatchTST с mode embeddings
        self.vmd_encoder = PatchTSTEncoder(
            n_modes=n_modes, L=L, T=T, patch_len=patch_len,
            d_model=128, n_heads=8, n_layers=3, d_ff=256,
            use_mode_embeddings=use_mode_embeddings, mode_embed_mode=mode_embed_mode,
        )
        self.aswa = ASWA(n_modes=n_modes, horizon=T, equal_weights=equal_weights,
                         aggregation=aggregation)

        # skip-путь: облегчённый PatchTST по исходному сигналу (1 «мода»)
        self.skip_encoder = PatchTSTEncoder(
            n_modes=1, L=L, T=T, patch_len=patch_len,
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            use_mode_embeddings=False,
        )
        self.fusion = GatedSkip(horizon=T, disabled=disable_skip)

    def forward(self, modes: torch.Tensor, raw: torch.Tensor,
                return_repr: bool = False, return_gate: bool = False):
        if return_repr:
            u_hat, repr_pooled = self.vmd_encoder(modes, return_repr=True)
        else:
            u_hat = self.vmd_encoder(modes)                  # (B, K, T)
        y_vmd = self.aswa(u_hat)                             # (B, T)
        y_skip = self.skip_encoder(raw.unsqueeze(1)).squeeze(1)  # (B, T)
        if return_gate:
            y_final, gate = self.fusion(y_vmd, y_skip, return_gate=True)
        else:
            y_final = self.fusion(y_vmd, y_skip)             # (B, T)
        out = {"y": y_final, "y_vmd": y_vmd, "y_skip": y_skip, "u_hat": u_hat}
        if return_repr:
            out["repr"] = repr_pooled
        if return_gate:
            out["gate"] = gate                                # (B, 1) in [0,1]
        return out

    def identity_parameters(self):
        """Параметры identity-механизма (§3.6): mode embeddings + вся ASWA.
        Используется для дифференцированного LR/WD в tune_identity_lr.py (замечание №2)."""
        params = []
        if hasattr(self.vmd_encoder, "mode_embed") and self.vmd_encoder.use_mode_embeddings \
           and self.vmd_encoder.mode_embed_mode == "learnable":
            params.append(self.vmd_encoder.mode_embed)
        params += list(self.aswa.parameters())
        return params

    def other_parameters(self):
        """Все параметры, кроме identity-механизма (для раздельного optimizer param_group)."""
        identity_ids = {id(p) for p in self.identity_parameters()}
        return [p for p in self.parameters() if id(p) not in identity_ids]


    def loss(self, out: dict, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        d = self.huber_delta
        l_main = F.huber_loss(out["y"], target, delta=d)
        if self.aux_mode == "sum":
            # агрегатный aux: Σ_k u_hat_k должен реконструировать y
            agg = out["u_hat"].sum(dim=1)                   # (B, T)
            l_aux = F.huber_loss(agg, target, delta=d)
        else:
            # исторический aux: КАЖДАЯ мода по отдельности предсказывает полную
            # цель; утечки нет, целью служит тот же y
            u = out["u_hat"]                                # (B, K, T)
            l_aux = sum(F.huber_loss(u[:, k, :], target, delta=d)
                        for k in range(u.shape[1]))
        total = l_main + self.lambda_aux * l_aux
        parts = {"main": l_main.item(), "aux": l_aux.item()}
        if self.lambda_div > 0:
            # регуляризация разнообразия через косинусную близость прогнозов мод
            u = F.normalize(out["u_hat"], dim=-1)            # (B,K,T)
            sim = torch.einsum("bkt,bjt->bkj", u, u)         # попарные косинусы
            K = u.shape[1]
            off = sim.sum(dim=(1, 2)) - K                    # минус диагональ
            l_div = (off / (K * (K - 1))).mean()
            total = total + self.lambda_div * l_div
            parts["div"] = l_div.item()
        return total, parts


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
