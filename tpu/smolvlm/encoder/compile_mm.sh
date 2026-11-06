#!/bin/bash
# Compile the MatMul block for cv181x in WSL, and compare it with float on the host.
# Usage: bash compile_mm.sh DIR N STEP
#   STEP bf16: all BF16, no calibration
#   STEP int8: calibration, then all INT8 and mix-a (the conv block's mix-a: BF16 norms,
#              attention and residual adds; INT8 weight layers and GELU)
set -u
D=$1; N=$2; STEP=$3
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
W=~/duosight-tpu/work/$D
[ -d $W ] || cp -r /mnt/d/duosight-eval/tpu/$D $W
cd $W
if [ ! -s block.mlir ]; then
	model_transform.py --model_name block --model_def block.onnx --input_shapes "[[$N,768]]" \
		--test_input test_in.npz --test_result top_outputs.npz --mlir block.mlir > transform.log 2>&1
	echo "transform exit=$?"
fi

deploy() {  # NAME, then model_deploy options
	local name=$1; shift
	model_deploy.py --mlir block.mlir "$@" --processor cv181x --test_input test_in.npz \
		--test_reference top_outputs.npz --tolerance 0.0,-1.0 --model block_$name.cvimodel > deploy_$name.log 2>&1
	echo "$name deploy exit=$?"
	model_runner.py --input test_in.npz --model block_$name.cvimodel --output host_$name.npz > host_$name.log 2>&1
	python3 - "$name" <<'PY'
import sys, numpy as np
name = sys.argv[1]
r = np.load("top_outputs.npz"); r = r[[k for k in r.files if k.startswith("y")][-1]]
h = np.load(f"host_{name}.npz"); y = h[[k for k in h.files if k.startswith("y")][-1]]
a, b = y.ravel().astype(np.float64), r.ravel().astype(np.float64)
m = np.abs(b) < 20
cos = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)
print(name, "cos vs float", round(cos(a, b), 6), "typical-values cos", round(cos(a[m], b[m]), 6),
      "range", round(float(y.min()), 1), round(float(y.max()), 1))
PY
	mkdir -p /mnt/d/duosight-eval/tpu/out/mm${N}
	cp block_$name.cvimodel /mnt/d/duosight-eval/tpu/out/mm${N}/
	ls -la block_$name.cvimodel | awk '{print $5, $9}'
}

case $STEP in
bf16)
	deploy bf16 --quantize BF16 ;;
int8)
	[ -s cali_table ] || run_calibration.py block.mlir --dataset calib --input_num 32 -o cali_table > cali.log 2>&1
	echo "calibration exit=$? lines=$(wc -l < cali_table)"
	deploy int8 --quantize INT8 --calibration_table cali_table
	python3 - <<'PY'
import re
text = open("block.mlir").read()
locs = dict(re.findall(r'^#(loc\d+) = loc\("([^"]+)"\)', text, re.M))
ops = [(k, locs.get(l, l)) for _, k, l in re.findall(r'^\s*(%\d+) = "top\.(\w+)"\([^)]*\).*?loc\(#(loc\d+)\)\s*$', text, re.M)
       if k not in ("Weight", "None", "Input")]
core = lambda n: n.startswith(("/norm1/", "/norm2/", "/attn/")) or n in ("/Add_output_0_Add", "y_Add")
bf = [n for _, n in ops if core(n)]
open("qt_a.txt", "w").write("".join(f"{n} BF16\n" for n in bf))
print("mix-a table:", len(bf), "BF16 of", len(ops), "ops; INT8:", sorted({k for k, n in ops if not core(n)}))
PY
	deploy mixa --quantize INT8 --calibration_table cali_table --quantize_table qt_a.txt ;;
esac
