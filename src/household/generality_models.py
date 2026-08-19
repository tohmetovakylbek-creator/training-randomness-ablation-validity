"""
generality_models.py
====================
Две ВНЕШНИЕ архитектуры для §5.2.6 — обе БЕЗ декомпозиции и БЕЗ gated fusion,
у каждой ровно один бинарный переключаемый компонент. Это и есть контроль на
вопрос «не является ли доминирование сидовой дисперсии свойством именно вашей
гибридной модели?».

  A. PatchTST (на исходном сигнале) с RevIN и без него.
     RevIN (reversible instance normalisation) — компонент, чью полезность в
     литературе устанавливают именно абляцией. Энкодер переиспользуется из
     models/patchtst.py, поэтому гиперпараметры буквально те же, что в статье.

  B. BiLSTM с attention-пуллингом и без него (последнее скрытое состояние).
     Не трансформер вообще — показывает, что эффект не про self-attention.

Файл кладётся в КОРЕНЬ проекта (рядом с train.py), чтобы работали импорты
`from models.patchtst import PatchTSTEncoder` и `from evaluate import ...`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate import inverse, metrics, per_window_abs_error
from models.patchtst import PatchTSTEncoder


# ---------------------------------------------------------------------------
# A. PatchTST +/- RevIN
# ---------------------------------------------------------------------------
class RevIN(nn.Module):
    """Обратимая нормализация по каждому окну (Kim et al., 2022)."""

    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def norm(self, x):                       # x: (B, L)
        self.mu = x.mean(dim=1, keepdim=True).detach()
        self.sd = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        z = (x - self.mu) / self.sd
        if self.affine:
            z = z * self.weight + self.bias
        return z

    def denorm(self, y):                     # y: (B, T)
        if self.affine:
            y = (y - self.bias) / (self.weight + self.eps)
        return y * self.sd + self.mu


class PatchTSTBaseline(nn.Module):
    """PatchTST на недекомпозированном сигнале. Единственный переключатель — RevIN."""

    def __init__(self, L: int = 168, T: int = 24, use_revin: bool = True):
        super().__init__()
        self.use_revin = use_revin
        self.revin = RevIN() if use_revin else None
        self.enc = PatchTSTEncoder(
            n_modes=1, L=L, T=T, patch_len=24,
            d_model=128, n_heads=8, n_layers=3, d_ff=256,
            use_mode_embeddings=False,
        )

    def forward(self, x):                    # x: (B, L)
        if self.use_revin:
            x = self.revin.norm(x)
        y = self.enc(x.unsqueeze(1)).squeeze(1)   # (B, T)
        if self.use_revin:
            y = self.revin.denorm(y)
        return y


# ---------------------------------------------------------------------------
# B. BiLSTM +/- attention
# ---------------------------------------------------------------------------
class BiLSTMBaseline(nn.Module):
    """BiLSTM-энкодер. Переключатель — attention-пуллинг по скрытым состояниям
    против использования только последнего состояния (как в baselines/deep_baselines.py)."""

    def __init__(self, L: int = 168, T: int = 24, hidden: int = 128, layers: int = 2,
                 use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=0.1)
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(2 * hidden, 2 * hidden), nn.Tanh(), nn.Linear(2 * hidden, 1)
            )
        self.head = nn.Linear(2 * hidden, T)

    def forward(self, x):                    # x: (B, L)
        h, _ = self.lstm(x.unsqueeze(-1))    # (B, L, 2H)
        if self.use_attention:
            a = torch.softmax(self.attn(h).squeeze(-1), dim=1)     # (B, L)
            ctx = (h * a.unsqueeze(-1)).sum(dim=1)                 # (B, 2H)
        else:
            ctx = h[:, -1]
        return self.head(ctx)


# ---------------------------------------------------------------------------
# Каталог конфигураций: (архитектура, компонент) -> фабрика модели
# ---------------------------------------------------------------------------
CONFIGS = {
    ("patchtst", "on"):  lambda: PatchTSTBaseline(use_revin=True),
    ("patchtst", "off"): lambda: PatchTSTBaseline(use_revin=False),
    ("bilstm", "on"):    lambda: BiLSTMBaseline(use_attention=True),
    ("bilstm", "off"):   lambda: BiLSTMBaseline(use_attention=False),
}

COMPONENT = {"patchtst": "RevIN (reversible instance normalisation)",
             "bilstm": "attention pooling over encoder states"}


# ---------------------------------------------------------------------------
# Обучение — протокол ИДЕНТИЧЕН train.py / baselines/deep_baselines.py
# ---------------------------------------------------------------------------
def train_one(model_fn, npz, seed: int, device: str = "cuda",
              epochs: int = 60, batch_size: int = 64) -> dict:
    """Возвращает по-оконные |ошибки| в Ваттах для одного (дом, конфиг, сид)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, Ytr = npz["Xtr"].astype(np.float32), npz["Ytr"].astype(np.float32)
    Xva, Yva = npz["Xva"].astype(np.float32), npz["Yva"].astype(np.float32)
    Xte, Yte = npz["Xte"].astype(np.float32), npz["Yte"].astype(np.float32)
    lo, hi = float(npz["scaler_lo"]), float(npz["scaler_hi"])

    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                    batch_size=batch_size, shuffle=True)
    model = model_fn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    Yva_w = inverse(Yva, lo, hi)
    Xva_t = torch.tensor(Xva).to(device)
    best, best_state = float("inf"), None

    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.huber_loss(model(xb), yb, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy()
        vm = metrics(inverse(pv, lo, hi), Yva_w)["MAE"]
        if vm < best:
            best = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = model(torch.tensor(Xte).to(device)).cpu().numpy()

    pred_w, true_w = inverse(pte, lo, hi), inverse(Yte, lo, hi)
    return {
        "window_err": per_window_abs_error(pred_w, true_w),   # (n_windows,) в Ваттах
        "mae": float(metrics(pred_w, true_w)["MAE"]),
        "val_mae": float(best),
    }
