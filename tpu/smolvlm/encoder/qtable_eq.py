"""Print the BF16 table for an equalized block. Run in the block folder.

Norms, attention, the residual adds and the fc2 gain stay in BF16. The weight convs
stay in INT8. With the argument "gelu", GELU moves to BF16 too.

Usage: python3 qtable_eq.py [gelu] > qt_eq.txt
"""
import re
import sys

text = open("block.mlir").read()
locs = dict(re.findall(r'^#(loc\d+) = loc\("([^"]+)"\)', text, re.M))
pattern = r'^\s*(%\d+) = "top\.(\w+)"\([^)]*\).*?loc\(#(loc\d+)\)\s*$'
ops = [locs.get(l, l) for _, k, l in re.findall(pattern, text, re.M) if k not in ("Weight", "None", "Input")]
CORE = ("/MatMul_output_0_MatMul", "/Softmax_output_0_Softmax", "/MatMul_1_output_0_MatMul",
        "/Add_output_0_Add", "y_Add")
GELU = ("/Mul_output_0_Mul", "/Mul_1_output_0_Mul", "/Mul_2_output_0_Mul", "/Mul_3_output_0_Mul",
        "/Add_1_output_0_Add", "/Mul_4_output_0_Mul", "/Tanh_output_0_Tanh", "/Add_2_output_0_Add",
        "/Mul_5_output_0_Mul")
gelu = len(sys.argv) > 1 and sys.argv[1] == "gelu"
for name in ops:
    if name.startswith(("/norm1/", "/norm2/", "/gain/")) or name in CORE or (gelu and name in GELU):
        print(f"{name} BF16")
