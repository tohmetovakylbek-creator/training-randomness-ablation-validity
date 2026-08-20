from __future__ import annotations
import torch
import torch.nn as nn

class ASWA(nn.Module):
    def __init__(self, n_modes=6, horizon=24, d_emb=16, hidden=32, equal_weights=False,
                 aggregation="convex"):
        super().__init__()
        self.n_modes, self.T = n_modes, horizon
        self.equal_weights = equal_weights
        assert aggregation in ("convex", "sum"), aggregation
        self.aggregation = aggregation
        self.horizon_embed = nn.Parameter(torch.randn(horizon, d_emb) * 0.02)
        self.fc1 = nn.Linear(d_emb, hidden)
        self.fc2 = nn.Linear(hidden, n_modes)
        self.act = nn.GELU()
        self.horizon_bias = nn.Parameter(torch.zeros(horizon, n_modes))

    def weights(self):
        if self.equal_weights:
            return torch.full((self.T, self.n_modes), 1.0 / self.n_modes, device=self.horizon_embed.device)
        logits = self.fc2(self.act(self.fc1(self.horizon_embed))) + self.horizon_bias
        return torch.softmax(logits, dim=-1)

    def forward(self, mode_forecasts):
        w = self.weights()
        if self.aggregation == "sum":
            # веса в сумме дают K, а не 1: при равномерных весах это Σ_k u_hat_k.
            # Согласовано со вспомогательной целью вида Huber(Σ_k u_hat_k, y).
            w = w * self.n_modes
        w = w.transpose(0, 1).unsqueeze(0)
        return (mode_forecasts * w).sum(dim=1)

class GatedSkip(nn.Module):
    def __init__(self, horizon=24, disabled=False):
        super().__init__()
        self.disabled = disabled
        self.gate = nn.Linear(2 * horizon, 1)

    def forward(self, y_vmd, y_skip, return_gate=False):
        if self.disabled:
            g = torch.ones(y_vmd.shape[0], 1, device=y_vmd.device)
            out = y_vmd
        else:
            g = torch.sigmoid(self.gate(torch.cat([y_vmd, y_skip], dim=-1)))
            out = g * y_vmd + (1.0 - g) * y_skip
        return (out, g) if return_gate else out
