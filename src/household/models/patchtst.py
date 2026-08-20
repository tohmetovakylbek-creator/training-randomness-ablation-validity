"""
models/patchtst.py
==================
PatchTST-энкодер (channel-independent, веса разделяемые между модами) с
ОБУЧАЕМЫМИ MODE EMBEDDINGS — центральный вклад работы.

Параметры по диплому: d_model=128, 8 голов, 3 слоя, d_ff=256, патч P=24,
N=7 патчей на окно L=168, горизонт T=24.

Mode embeddings: каждой моде k присваивается обучаемый вектор e_k in R^{d_model},
добавляемый к патч-эмбеддингам ДО энкодера. Это возвращает shared-энкодеру
частотную идентичность моды (см. §4.3, гипотеза H1). Флаг use_mode_embeddings
и режим mode_embed_mode используются в ablation (P2) и контроле P5.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchTSTEncoder(nn.Module):
    """Channel-independent PatchTST. Вход (B, K, L) -> прогноз по модам (B, K, T).

    Один и тот же энкодер применяется к каждой моде (веса разделяемые).
    Mode embedding инъектируется добавлением к патч-эмбеддингам.
    """

    def __init__(
        self,
        n_modes: int = 6,
        L: int = 168,
        T: int = 24,
        patch_len: int = 24,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_mode_embeddings: bool = True,
        mode_embed_mode: str = "learnable",   # learnable | random_fixed | orthogonal_fixed
    ):
        super().__init__()
        assert L % patch_len == 0, "L должно делиться на patch_len (168 / 24 = 7)"
        self.n_modes = n_modes
        self.L, self.T, self.patch_len = L, T, patch_len
        self.n_patches = L // patch_len           # N = 7
        self.d_model = d_model
        self.use_mode_embeddings = use_mode_embeddings
        self.mode_embed_mode = mode_embed_mode

        # патч-проекция и позиционные эмбеддинги (разделяемые между модами)
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # mode embeddings e_k in R^{d_model}
        if use_mode_embeddings:
            emb = self._init_mode_embeddings(n_modes, d_model, mode_embed_mode)
            if mode_embed_mode == "learnable":
                self.mode_embed = nn.Parameter(emb)              # обучаемые
            else:
                self.register_buffer("mode_embed", emb)          # фиксированные (контроль P5)
        else:
            self.register_buffer("mode_embed", torch.zeros(n_modes, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, T)

    @staticmethod
    def _init_mode_embeddings(n_modes, d_model, mode):
        if mode == "orthogonal_fixed":
            m = torch.empty(n_modes, d_model)
            nn.init.orthogonal_(m)              # ортогональные, фиксированные
            return m
        # learnable инициализируется так же, как random_fixed
        return torch.randn(n_modes, d_model) * 0.02

    def _embed(self, modes: torch.Tensor) -> torch.Tensor:
        """(B, K, L) -> токены (B*K, N, d_model) с добавленным mode embedding."""
        B, K, L = modes.shape
        # разбиение на патчи: (B, K, N, P)
        patches = modes.unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        z = self.patch_proj(patches)                       # (B, K, N, d_model)
        z = z + self.pos_embed.unsqueeze(0)                # позиционные
        z = z + self.mode_embed.view(1, K, 1, self.d_model)  # MODE EMBEDDING
        return z.reshape(B * K, self.n_patches, self.d_model)

    def forward(self, modes: torch.Tensor, return_repr: bool = False):
        """modes: (B, K, L). Возвращает прогнозы по модам (B, K, T).
        return_repr=True -> также пуллинг представлений энкодера (B, K, d_model)
        для mechanistic-анализа (P3)."""
        B, K, _ = modes.shape
        z = self._embed(modes)                  # (B*K, N, d_model)
        h = self.encoder(z)                     # (B*K, N, d_model)
        y = self.head(h.reshape(B * K, -1))     # (B*K, T)
        y = y.reshape(B, K, self.T)
        if return_repr:
            repr_pooled = h.mean(dim=1).reshape(B, K, self.d_model)  # mean-pool по патчам
            return y, repr_pooled
        return y
