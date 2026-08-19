# desktop (GPU)
python -c "
import numpy as np
for cfg in ('on','off'):
    for seed in (42,7,13,99,2025):
        a = np.load(f'results_generality/err_bilstm_{cfg}_refit_10_seed{seed}.npy')
        print(f'{cfg} seed{seed}: mean={a.mean():.2f}  std={a.std():.2f}  n={len(a)}  finite={np.isfinite(a).all()}')
"