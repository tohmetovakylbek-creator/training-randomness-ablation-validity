# check_shapes.py
import numpy as np
from pathlib import Path
import sys

proc = Path(sys.argv[1])
vmd = proc / "vmd_precomputed"
for npz in sorted(proc.glob("house_*.npz")):
    h = npz.stem.split("_")[1]
    d = np.load(npz)
    print(f"\nhouse {h}")
    for s in ("tr", "va", "te"):
        X = d[f"X{s}"]
        p = vmd / f"house{h}_{s}_modes.npy"
        if p.exists():
            m = np.load(p, mmap_mode="r")
            flag = "OK" if m.shape[0] == X.shape[0] else "РАСХОЖДЕНИЕ"
            print(f"  {s}: npz X={X.shape}  modes={m.shape}  {flag}")
        else:
            print(f"  {s}: npz X={X.shape}  modes: файла нет")