"""
Compact FT-Transformer (Gorishniy et al., 2021, "Revisiting Deep Learning
Models for Tabular Data") with the ablated component made explicit.

Gorishniy et al. define the feature tokenizer as
    T_j(x_j) = b_j + f_j(x_j)
where b_j is a per-feature bias ("identity" term, independent of the
feature's value) and f_j is value-dependent (a linear map for numerical
features, an embedding lookup for categorical ones). USE_IDENTITY below
toggles b_j on/off; f_j is untouched. This is the tabular counterpart of
the mode-identity / client-identity mechanism used elsewhere in the
paper: does the shared encoder get told *which* feature it is looking
at, on top of the feature's value.

This is a compact reimplementation for auditability of the ablation, not
a byte-for-byte port of the official codebase. If exact fidelity to
published FT-Transformer numbers matters more than ablation
transparency, fork https://github.com/yandex-research/rtdl instead and
zero out its analogous bias parameter -- note that its public API does
not expose a ready-made switch for this, so you would still be editing
library internals either way.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class FTConfig:
    n_num: int
    cat_cardinalities: list          # e.g. [5, 12, 3] for 3 categorical cols
    n_classes: int
    d_token: int = 64
    n_layers: int = 3
    n_heads: int = 8
    dropout: float = 0.1
    use_identity: bool = True        # the ablated switch


class FeatureTokenizer(nn.Module):
    def __init__(self, cfg: FTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_token

        if cfg.n_num > 0:
            self.num_weight = nn.Parameter(torch.empty(cfg.n_num, d))
            self.num_bias = nn.Parameter(torch.empty(cfg.n_num, d))
            nn.init.uniform_(self.num_weight, -1 / d ** 0.5, 1 / d ** 0.5)
            nn.init.uniform_(self.num_bias, -1 / d ** 0.5, 1 / d ** 0.5)
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d) for card in cfg.cat_cardinalities]
        )
        if cfg.cat_cardinalities:
            self.cat_bias = nn.Parameter(torch.empty(len(cfg.cat_cardinalities), d))
            nn.init.uniform_(self.cat_bias, -1 / d ** 0.5, 1 / d ** 0.5)
        else:
            self.register_parameter("cat_bias", None)

        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.uniform_(self.cls, -1 / d ** 0.5, 1 / d ** 0.5)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        B = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        tokens = [self.cls.expand(B, -1, -1)]

        if self.cfg.n_num > 0:
            num_tok = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0)   # (B, n_num, d)
            if self.cfg.use_identity:
                num_tok = num_tok + self.num_bias.unsqueeze(0)
            tokens.append(num_tok)

        if self.cfg.cat_cardinalities:
            cat_tok = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
            )                                                              # (B, n_cat, d)
            if self.cfg.use_identity:
                cat_tok = cat_tok + self.cat_bias.unsqueeze(0)
            tokens.append(cat_tok)

        return torch.cat(tokens, dim=1)                                    # (B, 1+n_num+n_cat, d)


class FTTransformer(nn.Module):
    def __init__(self, cfg: FTConfig):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_token, nhead=cfg.n_heads, dim_feedforward=cfg.d_token * 4,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_token),
            nn.Linear(cfg.d_token, cfg.n_classes),
        )

    def forward(self, x_num, x_cat):
        tokens = self.tokenizer(x_num, x_cat)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out)


def build_model(n_num, cat_cardinalities, n_classes, use_identity, seed):
    torch.manual_seed(seed)
    cfg = FTConfig(n_num=n_num, cat_cardinalities=cat_cardinalities,
                    n_classes=n_classes, use_identity=use_identity)
    return FTTransformer(cfg)
