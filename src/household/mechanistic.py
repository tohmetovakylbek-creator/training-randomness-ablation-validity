"""
mechanistic.py
==============
Mechanistic-анализ гипотезы H1 (статья, §6.3). Превращает ablation-число в
доказательство механизма «mode identity».

P3 — разделимость представлений мод в энкодере С и БЕЗ mode embeddings:
   * linear-probe accuracy классификации индекса моды по пуллингу представлений
     (без embeddings -> представления мод запутаны -> низкая accuracy);
   * CKA-сходство между представлениями разных мод (без embeddings -> высокое
     перекрытие; с embeddings -> ниже, моды разнесены);
   * t-SNE-проекция для рисунка.

P5 — identity, не capacity: learnable vs random_fixed vs orthogonal_fixed e_k.
   Если фиксированные ортогональные embeddings уже дают почти весь выигрыш —
   решает именно идентичность, а не дополнительные обучаемые параметры.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from train import train_one, train_ensemble


# ---------------------------------------------------------------- CKA
def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear CKA между матрицами представлений A,B (n_samples, dim)."""
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    hsic = np.linalg.norm(B.T @ A, "fro") ** 2
    norm = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
    return float(hsic / norm) if norm > 0 else 0.0


def mode_separability(repr_te: np.ndarray):
    """repr_te: (n_windows, K, d_model). Возвращает probe-accuracy и среднюю
    попарную CKA между модами."""
    n, K, d = repr_te.shape
    # linear probe: предсказать индекс моды по её представлению.
    # reshape (n,K,d)->(n*K,d) идёт построчно, поэтому метка строки i*K+k равна k:
    X = repr_te.reshape(n * K, d)
    y = np.tile(np.arange(K), n)
    clf = LogisticRegression(max_iter=1000)
    acc = float(cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean())
    # средняя попарная CKA между модами
    ckas = []
    for a in range(K):
        for b in range(a + 1, K):
            ckas.append(linear_cka(repr_te[:, a, :], repr_te[:, b, :]))
    return {"probe_accuracy": acc, "mean_pairwise_cka": float(np.mean(ckas)),
            "chance_accuracy": 1.0 / K}


def run_P3(house_data, device="cpu", epochs=60, seed=42):
    """Сравнивает разделимость представлений с/без mode embeddings."""
    out = {}
    for tag, kw in {"with_embed": {}, "without_embed": {"use_mode_embeddings": False}}.items():
        r = train_one(house_data, seed=seed, device=device, epochs=epochs,
                      model_kwargs=kw, return_repr=True)
        sep = mode_separability(r["repr"])
        sep["test_MAE"] = float(np.abs(r["pred_w"] - r["true_w"]).mean())
        out[tag] = sep
        print(f"  [P3 {tag:14s}] probe_acc={sep['probe_accuracy']:.3f} "
              f"(chance={sep['chance_accuracy']:.3f}) CKA={sep['mean_pairwise_cka']:.3f} "
              f"MAE={sep['test_MAE']:.2f}")
    out["repr_for_tsne"] = "вызвать tsne_plot(repr) отдельно для рисунка"
    return out


def tsne_plot(repr_te: np.ndarray, path: str = "tsne_modes.png"):
    """t-SNE проекция представлений мод (для рисунка §6.3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    n, K, d = repr_te.shape
    X = repr_te.reshape(n * K, d)
    lab = np.array([k for _ in range(n) for k in range(K)])
    emb = TSNE(n_components=2, perplexity=30, init="pca").fit_transform(X)
    plt.figure(figsize=(6, 5))
    for k in range(K):
        m = lab == k
        plt.scatter(emb[m, 0], emb[m, 1], s=6, label=f"IMF {k+1}")
    plt.legend(markerscale=2); plt.title("Encoder representations of VMD modes")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


def run_P5(house_data, device="cpu", epochs=60, seeds=(42, 7, 13, 99, 2025)):
    """Контроль identity-vs-capacity: learnable / random_fixed / orthogonal_fixed."""
    out = {}
    for mode in ("learnable", "random_fixed", "orthogonal_fixed"):
        r = train_ensemble(house_data, device=device, epochs=epochs, seeds=seeds,
                           model_kwargs={"use_mode_embeddings": True, "mode_embed_mode": mode})
        out[mode] = r["ensemble_metrics"]
        print(f"  [P5 {mode:16s}] MAE={out[mode]['MAE']:.2f}")
    # baseline без embeddings для контраста
    r0 = train_ensemble(house_data, device=device, epochs=epochs, seeds=seeds,
                        model_kwargs={"use_mode_embeddings": False})
    out["no_embed"] = r0["ensemble_metrics"]
    print(f"  [P5 {'no_embed':16s}] MAE={out['no_embed']['MAE']:.2f}")
    return out
