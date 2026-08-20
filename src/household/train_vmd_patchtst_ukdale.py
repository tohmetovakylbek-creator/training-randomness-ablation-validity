"""
train_vmd_patchtst_ukdale.py
=============================
Обучение VMD-PatchTST-ASWA на UK-DALE с предвычисленными VMD модами.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================
def load_house_with_vmd(processed_dir: Path, vmd_dir: Path, house_id: int) -> dict:
    """Загружает предобработанные данные + предвычисленные VMD моды."""
    npz_path = processed_dir / f"house_{house_id}.npz"
    data = np.load(npz_path)
    
    result = {
        "Xtr": data["Xtr"], "Ytr": data["Ytr"],
        "Xva": data["Xva"], "Yva": data["Yva"],
        "Xte": data["Xte"], "Yte": data["Yte"],
        "scaler_lo": float(data["scaler_lo"]),
        "scaler_hi": float(data["scaler_hi"]),
    }
    
    # Загружаем предвычисленные VMD моды
    for split in ['tr', 'va', 'te']:
        modes_path = vmd_dir / f"house{house_id}_{split}_modes.npy"
        if not modes_path.exists():
            raise FileNotFoundError(f"VMD моды не найдены: {modes_path}")
        result[f"modes_{split}"] = np.load(modes_path)  # (N, K, L)
    
    return result

def denormalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x * (hi - lo) + lo

# ============================================================================
# 2. ASWA MODULE (из диплома)
# ============================================================================
class ASWAModule(nn.Module):
    """Adaptive Scale-Weighted Aggregation из диплома."""
    def __init__(self, K: int = 6, pred_len: int = 24, d_emb: int = 16):
        super().__init__()
        self.K = K
        self.pred_len = pred_len
        self.horizon_emb = nn.Embedding(pred_len, d_emb)
        self.W1 = nn.Linear(d_emb, 32)
        self.W2 = nn.Linear(32, K)
        self.horizon_bias = nn.Parameter(torch.zeros(pred_len, K))
    
    def forward(self, mode_forecasts: torch.Tensor) -> torch.Tensor:
        """
        Args: mode_forecasts: (B, K, T)
        Returns: (B, T)
        """
        B, K, T = mode_forecasts.shape
        h_idx = torch.arange(T, device=mode_forecasts.device)
        e_h = self.horizon_emb(h_idx)  # (T, d_emb)
        weights = self.W2(F.gelu(self.W1(e_h)))  # (T, K)
        weights = weights + self.horizon_bias
        weights = F.softmax(weights, dim=-1)  # (T, K)
        weights = weights.unsqueeze(0)  # (1, T, K)
        preds = mode_forecasts.permute(0, 2, 1)  # (B, T, K)
        forecast = (preds * weights).sum(dim=-1)  # (B, T)
        return forecast

# ============================================================================
# 3. PATCHTST ENCODER (общий для всех мод + mode embeddings)
# ============================================================================
class PatchTSTEncoder(nn.Module):
    """Shared PatchTST encoder с mode embeddings."""
    def __init__(self, L: int = 168, T: int = 24, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 3, patch_size: int = 24,
                 K: int = 6, dropout: float = 0.1):
        super().__init__()
        self.L = L
        self.T = T
        self.patch_size = patch_size
        self.n_patches = L // patch_size
        self.K = K
        
        # Mode embeddings (КРИТИЧЕСКИЙ КОМПОНЕНТ из диплома!)
        self.mode_embeddings = nn.Parameter(torch.randn(K, d_model) * 0.02)
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        
        # Transformer encoder (shared)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model * self.n_patches, T)
    
    def forward(self, x: torch.Tensor, mode_idx: int) -> torch.Tensor:
        """
        Args:
            x: (B, L) - одна VMD мода
            mode_idx: индекс моды (0..K-1)
        Returns:
            (B, T) - прогноз для этой моды
        """
        B = x.shape[0]
        
        # Patching
        x = x.view(B, self.n_patches, self.patch_size)
        x = self.patch_embedding(x)  # (B, n_patches, d_model)
        
        # Add positional + mode embedding
        x = x + self.pos_embedding + self.mode_embeddings[mode_idx].unsqueeze(0).unsqueeze(0)
        
        # Transformer
        x = self.encoder(x)
        x = x.reshape(B, -1)
        x = self.output_proj(x)
        return x

# ============================================================================
# 4. SKIP CONNECTION PATCHTST (облегчённый)
# ============================================================================
class SkipPatchTST(nn.Module):
    """Облегчённый PatchTST для skip connection."""
    def __init__(self, L: int = 168, T: int = 24, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, patch_size: int = 24):
        super().__init__()
        self.L = L
        self.T = T
        self.patch_size = patch_size
        self.n_patches = L // patch_size
        
        self.patch_embedding = nn.Linear(patch_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model * self.n_patches, T)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, self.n_patches, self.patch_size)
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x.reshape(B, -1)
        return self.output_proj(x)

# ============================================================================
# 5. ПОЛНАЯ МОДЕЛЬ VMD-PatchTST-ASWA
# ============================================================================
class VMDPatchTSTASWA(nn.Module):
    def __init__(self, K=6, L=168, T=24):
        super().__init__()
        self.K = K
        self.L = L
        self.T = T
        
        # Shared PatchTST encoder с mode embeddings
        self.encoder = PatchTSTEncoder(L=L, T=T, K=K)
        
        # ASWA
        self.aswa = ASWAModule(K=K, pred_len=T)
        
        # Skip connection
        self.skip = SkipPatchTST(L=L, T=T)
        
        # Gate для fusion
        self.gate_layer = nn.Linear(2 * T, 1)
    
    def forward(self, x_raw: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: (B, L) - исходный сигнал
            modes: (B, K, L) - K VMD мод
        Returns:
            (B, T) - финальный прогноз
        """
        B = x_raw.shape[0]
        
        # 1. Прогноз каждой моды через shared encoder с mode embeddings
        mode_preds = []
        for k in range(self.K):
            pred_k = self.encoder(modes[:, k, :], mode_idx=k)  # (B, T)
            mode_preds.append(pred_k)
        mode_preds = torch.stack(mode_preds, dim=1)  # (B, K, T)
        
        # 2. ASWA агрегация
        vmd_pred = self.aswa(mode_preds)  # (B, T)
        
        # 3. Skip connection
        skip_pred = self.skip(x_raw)  # (B, T)
        
        # 4. Gated fusion
        gate_input = torch.cat([vmd_pred, skip_pred], dim=-1)  # (B, 2T)
        gate = torch.sigmoid(self.gate_layer(gate_input))  # (B, 1)
        
        final = gate * vmd_pred + (1 - gate) * skip_pred  # (B, T)
        return final

# ============================================================================
# 6. TRAINING
# ============================================================================
def train_model(model, Xtr, modes_tr, Ytr, Xva, modes_va, Yva,
                scaler_lo, scaler_hi, epochs=50, batch_size=64, lr=5e-4, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    criterion_main = nn.HuberLoss(delta=1.0)
    criterion_aux = nn.HuberLoss(delta=1.0)
    
    # Convert to tensors (нормализованные)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    modes_tr_t = torch.tensor(modes_tr, dtype=torch.float32)
    # Нормализуем моды в [0,1] используя тот же scaler
    modes_tr_t = (modes_tr_t - scaler_lo) / (scaler_hi - scaler_lo)
    
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    Yva_t = torch.tensor(Yva, dtype=torch.float32)
    modes_va_t = torch.tensor(modes_va, dtype=torch.float32)
    modes_va_t = (modes_va_t - scaler_lo) / (scaler_hi - scaler_lo)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(Xtr))
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(Xtr), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_b = Xtr_t[batch_idx].to(device)
            Y_b = Ytr_t[batch_idx].to(device)
            modes_b = modes_tr_t[batch_idx].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: получаем прогноз и промежуточные результаты
            B = X_b.shape[0]
            mode_preds = []
            for k in range(model.K):
                pred_k = model.encoder(modes_b[:, k, :], mode_idx=k)
                mode_preds.append(pred_k)
            mode_preds = torch.stack(mode_preds, dim=1)  # (B, K, T)
            
            vmd_pred = model.aswa(mode_preds)
            skip_pred = model.skip(X_b)
            gate_input = torch.cat([vmd_pred, skip_pred], dim=-1)
            gate = torch.sigmoid(model.gate_layer(gate_input))
            final_pred = gate * vmd_pred + (1 - gate) * skip_pred
            
            # Composite loss (формула 6 из диплома)
            loss_main = criterion_main(final_pred, Y_b)
            
            # Aux loss: сумма Huber Loss по каждой моде
            # Вспомогательный сигнал для обучения encoder для каждой моды
            # Используем ту же target Y (грубо, но работает)
            loss_aux = sum(criterion_aux(mode_preds[:, k, :], Y_b) for k in range(model.K))
            
            loss = loss_main + 0.2 * loss_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t.to(device), modes_va_t.to(device))
            val_loss = criterion_main(val_pred, Yva_t.to(device)).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train: {epoch_loss/n_batches:.4f}, Val: {val_loss:.4f}", flush=True)
    
    model.load_state_dict(best_state)
    return model

# ============================================================================
# 7. EVALUATION
# ============================================================================
def evaluate_model(model, Xte, modes_te, Yte, scaler_lo, scaler_hi, device="cpu"):
    model = model.to(device)
    model.eval()
    
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)
    modes_te_t = torch.tensor(modes_te, dtype=torch.float32).to(device)
    modes_te_t = (modes_te_t - scaler_lo) / (scaler_hi - scaler_lo)
    
    with torch.no_grad():
        pred_norm = model(Xte_t, modes_te_t).cpu().numpy()
    
    pred = denormalize(pred_norm, scaler_lo, scaler_hi)
    true = denormalize(Yte, scaler_lo, scaler_hi)
    
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred)**2))
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    
    return {"MAE_W": float(mae), "RMSE_W": float(rmse), "R2": float(r2)}

# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str,
                        default=r"C:\Users\User\PycharmProjects\uk_dale_project\uk-dale-disaggregated\processed")
    parser.add_argument("--houses", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 99, 2025])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/ukdale_vmd_patchtst_aswa")
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    vmd_dir = processed_dir / "vmd_precomputed"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("TRAINING VMD-PatchTST-ASWA ON UK-DALE")
    print("=" * 70)
    
    all_results = {}
    
    for house_id in args.houses:
        print(f"\n{'='*60}")
        print(f"House {house_id}")
        print(f"{'='*60}")
        
        try:
            data = load_house_with_vmd(processed_dir, vmd_dir, house_id)
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            continue
        
        print(f"  Train: {len(data['Xtr'])}, Val: {len(data['Xva'])}, Test: {len(data['Xte'])}")
        
        seed_results = []
        for seed in args.seeds:
            print(f"\n  -> Seed {seed}:", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = VMDPatchTSTASWA(K=6, L=168, T=24)
            
            model = train_model(
                model,
                data["Xtr"], data["modes_tr"], data["Ytr"],
                data["Xva"], data["modes_va"], data["Yva"],
                data["scaler_lo"], data["scaler_hi"],
                epochs=args.epochs, device=args.device
            )
            
            # Save
            model_path = output_dir / f"vmd_patchtst_aswa_house{house_id}_seed{seed}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Evaluate
            metrics = evaluate_model(
                model, data["Xte"], data["modes_te"], data["Yte"],
                data["scaler_lo"], data["scaler_hi"], device=args.device
            )
            
            print(f"    MAE={metrics['MAE_W']:.2f}W, RMSE={metrics['RMSE_W']:.2f}W, R²={metrics['R2']:.4f}", flush=True)
            seed_results.append(metrics)
        
        # Ensemble
        ensemble_mae = np.mean([r["MAE_W"] for r in seed_results])
        ensemble_rmse = np.mean([r["RMSE_W"] for r in seed_results])
        ensemble_r2 = np.mean([r["R2"] for r in seed_results])
        
        all_results[f"House_{house_id}"] = {
            "seeds": seed_results,
            "ensemble": {
                "MAE_W": float(ensemble_mae),
                "RMSE_W": float(ensemble_rmse),
                "R2": float(ensemble_r2)
            }
        }
        
        print(f"\n  [Ensemble] MAE={ensemble_mae:.2f}W, RMSE={ensemble_rmse:.2f}W, R²={ensemble_r2:.4f}", flush=True)
    
    # Save
    results_path = output_dir / "vmd_patchtst_aswa_ukdale_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Результаты сохранены: {results_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()