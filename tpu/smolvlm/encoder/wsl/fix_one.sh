# Rebuild one block with an edited calibration table, without running calibration.
# fc1 and the GELU output get their true maximum as threshold, because KL clips their
# large values. A block with no table takes the one from the pre-equalization build.
# Usage: bash fix_one.sh LAYER
L=$1
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
cd ~/duosight-tpu/work/e512/"$L" || exit 1
if [ ! -s cali_table ]; then
	rm -f block.mlir block_origin.mlir top_outputs.npz
	cp ~/duosight-tpu/work/s512/"$L"/cali_table cali_table
fi
[ -f cali_table.kl ] || cp cali_table cali_table.kl
python3 - <<'PY'
import re
wide = ("/fc1/Conv_output_0_Conv", "/Mul_5_output_0_Mul")
out, seen = [], set()
for line in open("cali_table.kl"):
    p = line.split()
    if line.startswith("#") or len(p) < 4:
        out.append(line)
        continue
    name, th, lo, hi = p[0], float(p[1]), float(p[2]), float(p[3])
    seen.add(name)
    if name == "/fc2/Conv_output_0_Conv" and max(abs(lo), abs(hi)) > 1.5:
        # This table predates equalization: fc2 now spans +-1, and the old range moves to gain.
        out.append(f"/gain/Conv_output_0_Conv {th} {lo} {hi}\n")
        th, lo, hi = 1.0, -1.0, 1.0
    elif name in wide:
        th = max(abs(lo), abs(hi))
    out.append(f"{name} {th} {lo} {hi}\n")
open("cali_table", "w").writelines(out)
PY
rm -f block_eq.cvimodel
bash /mnt/d/duosight-eval/tpu/compile_eq.sh e512 1024 "$L" gelu
