#!/bin/bash
# Compile a block from export_split.py for cv181x in WSL, and compare it with float and with
# the grouped final build (compile_g.sh output).
# Usage: bash compile_s.sh DIR LAYER   (DIR under D:/duosight-eval/tpu, e.g. split512 l05)
# Calibration is KL, then fc1_main and the GELU output of the main branch get their true
# maximum as threshold, as fix_one.sh did: KL clips their large values. The branch of
# large channels, the norms, attention, residual adds and the fc2 gain run in BF16.
set -u
D=$1; L=$2
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
W=~/duosight-tpu/work/$D/$L
mkdir -p ~/duosight-tpu/work/$D
[ -d $W ] || cp -r /mnt/d/duosight-eval/tpu/$D/$L $W
cd $W
if [ ! -s block.mlir ]; then
	model_transform.py --model_name block --model_def block.onnx --input_shapes "[[1,768,1,1024]]" \
		--test_input test_in.npz --test_result top_outputs.npz --mlir block.mlir > c.log 2>&1 ||
		{ echo "$D $L transform FAILED"; tail -3 c.log | cut -c1-160; exit 1; }
fi
if [ ! -s cali_table.kl ]; then
	run_calibration.py block.mlir --dataset calib --input_num 32 -o cali_table.kl > cal.log 2>&1 ||
		{ echo "$D $L calibration FAILED"; tail -3 cal.log | cut -c1-160; exit 1; }
fi
python3 - <<'PY'
import re
wide = ("/mlp/fc1_main/Conv_output_0_Conv", "/mlp/gelu_main/Mul_5_output_0_Mul")
out = []
for line in open("cali_table.kl"):
    p = line.split()
    if not line.startswith("#") and len(p) >= 4 and p[0] in wide:
        line = f"{p[0]} {max(abs(float(p[2])), abs(float(p[3])))} {p[2]} {p[3]}\n"
        print("widened", p[0], p[1], "->", line.split()[1])
    out.append(line)
open("cali_table", "w").writelines(out)

text = open("block.mlir").read()
locs = dict(re.findall(r'^#(loc\d+) = loc\("([^"]+)"\)', text, re.M))
ops = {}
for var, kind, args, loc in re.findall(r'^\s*(%\d+) = "top\.(\w+)"\(([^)]*)\).*?loc\(#(loc\d+)\)\s*$', text, re.M):
    ops[var] = (kind, locs.get(loc, loc), [a.strip() for a in args.split(",") if a.strip()])
keep = ("/norm1/", "/norm2/", "/gain/", "/mlp/fc1_big/", "/mlp/gelu_big/", "/mlp/fc2_big/")
bf = [n for k, n, _ in ops.values() if n.startswith(keep) or n in ("/mlp/Add_output_0_Add", "/Add_output_0_Add", "y_Add")
      or (k in ("Slice", "MatMul", "Softmax", "Concat") and not n.startswith(("/norm", "/mlp/")))]
o = next(v for v, (k, n, a) in ops.items() if n == "/o/Conv_output_0_Conv")
v = ops[o][2][0]
while ops[v][0] in ("Reshape", "Permute"):
    bf.append(ops[v][1]); v = ops[v][2][0]
assert ops[v][0] == "Concat", ops[v]
open("qt_s.txt", "w").write("".join(f"{n} BF16\n" for n in dict.fromkeys(bf)))
print(f"qtable: {len(set(bf))} BF16 of {len(ops)} ops")
PY
model_deploy.py --mlir block.mlir --quantize INT8 --calibration_table cali_table --quantize_table qt_s.txt \
	--processor cv181x --test_input test_in.npz --test_reference top_outputs.npz --tolerance 0.0,-1.0 \
	--model block_s.cvimodel > d.log 2>&1 || { echo "$D $L deploy FAILED"; grep -iE "error|threshold" d.log | tail -4 | cut -c1-200; exit 1; }
model_runner.py --input test_in.npz --model block_s.cvimodel --output host_s.npz > r.log 2>&1
python3 - "$D" "$L" <<'PY'
import sys, numpy as np
D, L = sys.argv[1:3]
last = lambda f: (lambda z: z[[k for k in z.files if k.startswith("y")][-1]])(np.load(f))
r, s, g = last("top_outputs.npz"), last("host_s.npz"), last(f"../../g512/{L}/host_g.npz")
cos = lambda u, v: float(u.ravel() @ v.ravel() / np.linalg.norm(u) / np.linalg.norm(v))
m = np.abs(r) < 20
print(f"{D} {L}: cos to float {cos(s, r):.6f} (final build {cos(g, r):.6f}); typical values {cos(s[m], r[m]):.6f} (final build {cos(g[m], r[m]):.6f})")
PY
model_tool --info block_s.cvimodel | grep -E "ION Memory|shared_gmem|cmdbuf|  weight " | tr -s ' ' | head -4
mkdir -p /mnt/d/duosight-eval/tpu/out/$D && cp block_s.cvimodel /mnt/d/duosight-eval/tpu/out/$D/$L.cvimodel
