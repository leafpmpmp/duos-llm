#!/bin/bash
# Compile one block with grouped attention, with the final build's tables, in WSL.
# Usage: bash compile_g.sh DIR LAYER
#   DIR: export under D:/duosight-eval/tpu (export_grouped.py), LAYER: l00 to l11
# The calibration table and the quantize table come from ~/duosight-tpu/work/e512/LAYER.
# The new attention ops (Slice, MatMul, Softmax, Concat) and the permute and reshape after
# them run in BF16, so they need no thresholds.
set -u
D=$1; L=$2
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
E=~/duosight-tpu/work/e512/$L
W=~/duosight-tpu/work/$D/$L
mkdir -p ~/duosight-tpu/work/$D
[ -d $W ] || cp -r /mnt/d/duosight-eval/tpu/$D/$L $W
cd $W
if [ ! -s block.mlir ]; then
	model_transform.py --model_name block --model_def block.onnx --input_shapes "[[1,768,1,1024]]" \
		--test_input test_in.npz --test_result top_outputs.npz --mlir block.mlir > c.log 2>&1 ||
		{ echo "$D $L transform FAILED"; tail -3 c.log | cut -c1-160; exit 1; }
fi
cp $E/cali_table cali_table
QT=$E/qt_eq.txt
[ $L = l05 ] && QT=$E/qt_c.txt
python3 - "$QT" <<'PY'
import re, sys
text = open("block.mlir").read()
locs = dict(re.findall(r'^#(loc\d+) = loc\("([^"]+)"\)', text, re.M))
ops = {}
for var, kind, args, loc in re.findall(r'^\s*(%\d+) = "top\.(\w+)"\(([^)]*)\).*?loc\(#(loc\d+)\)\s*$', text, re.M):
    ops[var] = (kind, locs.get(loc, loc), [a.strip() for a in args.split(",") if a.strip()])
names = {n for k, n, _ in ops.values()}
bf = [l.split()[0] for l in open(sys.argv[1]) if l.strip()]
keep = [n for n in bf if n in names]
new = [n for k, n, _ in ops.values() if k in ("Slice", "MatMul", "Softmax", "Concat") and not n.startswith("/norm")]
# Walk back from the o conv input to the Concat.
o = next(v for v, (k, n, a) in ops.items() if n == "/o/Conv_output_0_Conv")
tail, v = [], ops[o][2][0]
while ops[v][0] in ("Reshape", "Permute"):
    tail.append(ops[v][1]); v = ops[v][2][0]
assert ops[v][0] == "Concat", ops[v]
out = list(dict.fromkeys(keep + new + tail))
open("qt_g.txt", "w").write("".join(f"{n} BF16\n" for n in out))
print(f"qtable: {len(keep)} of {len(bf)} old names found, {len(new)} attention ops, tail {tail}")
PY
model_deploy.py --mlir block.mlir --quantize INT8 --calibration_table cali_table --quantize_table qt_g.txt \
	--processor cv181x --test_input test_in.npz --test_reference top_outputs.npz --tolerance 0.0,-1.0 \
	--model block_g.cvimodel > d.log 2>&1 || { echo "$D $L deploy FAILED"; grep -iE "error|threshold" d.log | tail -4 | cut -c1-200; exit 1; }
model_runner.py --input test_in.npz --model block_g.cvimodel --output host_g.npz > r.log 2>&1
python3 - "$D" "$L" "$E" <<'PY'
import sys, numpy as np
D, L, E = sys.argv[1:4]
last = lambda f: (lambda z: z[[k for k in z.files if k.startswith("y")][-1]])(np.load(f))
r, g, e = last("top_outputs.npz"), last("host_g.npz"), last(f"{E}/host_eq.npz")
cos = lambda u, v: float(u.ravel() @ v.ravel() / np.linalg.norm(u) / np.linalg.norm(v))
print(f"{D} {L}: cos to float {cos(g, r):.6f} (final build {cos(e, r):.6f}), cos to final build {cos(g, e):.6f}")
PY
model_tool --info block_g.cvimodel | grep -E "ION Memory|shared_gmem|private_gmem|cmdbuf|  weight " | tr -s ' ' | head -6
mkdir -p /mnt/d/duosight-eval/tpu/out/$D
cp block_g.cvimodel /mnt/d/duosight-eval/tpu/out/$D/$L.cvimodel
