#!/bin/bash
# Compile the patch and post models in BF16 for cv181x, in WSL, and check them on the host.
# Usage: bash compile_pp.sh NAME SHAPE   e.g. compile_pp.sh patch "[[1,3,512,512]]"
set -u
N=$1; S=$2
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
W=~/duosight-tpu/work/pp/$N
mkdir -p $W && cp /mnt/d/duosight-eval/fast/$N/* $W/ && cd $W
model_transform.py --model_name $N --model_def model.onnx --input_shapes "$S" --test_input test_in.npz \
	--test_result top_outputs.npz --mlir model.mlir > c.log 2>&1 || { echo "$N transform FAILED"; tail -3 c.log; exit 1; }
model_deploy.py --mlir model.mlir --quantize BF16 --processor cv181x --test_input test_in.npz \
	--test_reference top_outputs.npz --tolerance 0.0,-1.0 --model ${N}_bf16.cvimodel > d.log 2>&1 ||
	{ echo "$N deploy FAILED"; grep -iE "error" d.log | tail -4 | cut -c1-200; exit 1; }
model_runner.py --input test_in.npz --model ${N}_bf16.cvimodel --output host.npz > r.log 2>&1
python3 - $N <<'PY'
import sys, numpy as np
last = lambda f: (lambda z: z[[k for k in z.files if k.startswith("y")][-1]])(np.load(f))
r, h = last("top_outputs.npz"), last("host.npz")
c = float(r.ravel() @ h.ravel() / np.linalg.norm(r) / np.linalg.norm(h))
print(f"{sys.argv[1]} BF16: cosine to float {c:.6f}, max difference {np.abs(r - h).max():.3g}, float range {r.min():.1f} to {r.max():.1f}")
PY
model_tool --info ${N}_bf16.cvimodel | grep -E "ION Memory|shared_gmem|private_gmem|cmdbuf|  weight " | tr -s ' ' | head -5
mkdir -p /mnt/d/duosight-eval/fast/out && cp ${N}_bf16.cvimodel /mnt/d/duosight-eval/fast/out/
