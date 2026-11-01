# SPDX-License-Identifier: MIT
# Make a MIX2 quantize table from a top-level MLIR: every op BF16 except the
# weight matmuls (weight > 64x64), SiLU and the SiLU * up product, which stay
# INT8. Matches qtable_attn.txt for the 1-token layers.
# Usage (tpu-mlir container or host): python3 gen_qtable.py MODEL.mlir > qtable.txt
import re
import sys

text = open(sys.argv[1]).read()
locs = dict(re.findall(r'^#(loc\d+) = loc\("([^"]+)"\)', text, re.M))
ops = re.findall(r'^\s*(%\d+) = "top\.(\w+)"\(([^)]*)\).*?loc\(#(loc\d+)\)\s*$', text, re.M)
weights = {}
for var, kind, args, loc in ops:
    if kind == "Weight":
        m = re.search(rf'^\s*{re.escape(var)} = "top\.Weight".*?tensor<([\dx]+)xf32>', text, re.M)
        n = 1
        for d in m.group(1).split("x"):
            n *= int(d)
        weights[var] = n
silu = {var for var, kind, _, _ in ops if kind == "SiLU"}
for var, kind, args, loc in ops:
    if kind in ("Input", "Weight", "None"):
        continue
    operands = [a.strip() for a in args.split(",") if a.strip()]
    if kind == "MatMul" and any(weights.get(a, 0) > 64 * 64 for a in operands):
        continue
    if kind == "SiLU" or (kind == "Mul" and any(a in silu for a in operands)):
        continue
    print(f"{locs[loc]} BF16")
