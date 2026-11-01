# SPDX-License-Identifier: MIT
# Run the LM-head cvimodel in the simulator on the held-out inputs:
# cosine of the logits and whether the top-1 token matches the float reference.
# Usage (tpu-mlir container, cwd = head dir): python3 ../../evaluate_head.py MODEL.cvimodel
import glob
import subprocess
import sys

import numpy as np

model = sys.argv[1]
for tin in sorted(glob.glob("test_in_pos*.npz")):
    pos = tin[len("test_in_pos"):-4]
    out = f"/tmp/head_{pos}.npz"
    subprocess.run(["model_runner.py", "--input", tin, "--model", model, "--output", out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = np.load(out)
    got = d[[k for k in d.files if k.startswith("logits")][0]].astype(np.float64).ravel()
    ref = np.load(f"test_ref_pos{pos}.npz")["logits"].astype(np.float64).ravel()
    c = got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref))
    top_ref = np.argsort(-ref)[:5]
    print(f"pos {pos}: cos {c:.4f}, top1 {'same' if got.argmax() == ref.argmax() else 'DIFFERENT'}, "
          f"ref top5 in got top5: {len(set(top_ref) & set(np.argsort(-got)[:5]))}/5")
