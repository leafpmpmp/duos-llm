# SPDX-License-Identifier: MIT
# Run a layer cvimodel in the TPU-MLIR simulator on the held-out inputs and
# compare with the float reference. h_out is judged by the layer's own
# contribution (h_out - h): the residual stream alone would look similar.
# Usage (tpu-mlir container, cwd = layer dir): python3 ../evaluate.py MODEL.cvimodel
import glob
import os
import subprocess
import sys

import numpy as np

model = sys.argv[1]


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def find(d, name):
    for k in d.files:
        if k.startswith(name):
            return d[k].astype(np.float32).ravel()
    raise KeyError(f"{name} not in {d.files}")


rows = []
for tin in sorted(glob.glob("test_in_pos*.npz")):
    pos = tin[len("test_in_pos"):-4]
    out = f"/tmp/eval_{os.path.basename(model)}_{pos}.npz"
    subprocess.run(["model_runner.py", "--input", tin, "--model", model, "--output", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    got, ref, x = np.load(out), np.load(f"test_ref_pos{pos}.npz"), np.load(tin)
    h = x["h"].ravel()
    if "dh" in ref.files:
        dh_got, dh_ref = find(got, "dh"), ref["dh"].ravel()
    else:
        dh_got, dh_ref = find(got, "h_out") - h, ref["h_out"].ravel() - h
    rows.append((pos, cos(dh_got, dh_ref),
                 cos(find(got, "k_new"), ref["k_new"]), cos(find(got, "v_new"), ref["v_new"])))
print(f"{model}: cos(delta h)  cos(k_new)  cos(v_new)")
for r in rows:
    print(f"  pos {r[0]}: {r[1]:.4f}  {r[2]:.4f}  {r[3]:.4f}")
