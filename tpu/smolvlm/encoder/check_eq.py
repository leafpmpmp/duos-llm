"""Compare a block's simulator output with float. Run in the block folder.

Raw cosine says little here: a few huge values dominate it. delta-cos looks only at
what the block adds, and typical-cos leaves out values of magnitude 20 or more.

Usage: python3 check_eq.py NAME
"""
import sys

import numpy as np

x = np.load("test_in.npz")["x"].astype(np.float64)
ref = np.load("top_outputs.npz")["y_Add"].astype(np.float64)
host = np.load("host_eq.npz")
y = host[[k for k in host.files if k.startswith("y")][-1]].astype(np.float64)


def cos(a, b):
    return (a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b)


typical = np.abs(ref) < 20
print(sys.argv[1], "delta-cos", round(cos(y - x, ref - x), 4), "typical-cos", round(cos(y[typical], ref[typical]), 4),
      "range", round(float(y.min()), 1), round(float(y.max()), 1), "float range", round(float(ref.min()), 1), round(float(ref.max()), 1))
