# Compile one equalized block. Usage: bash one_eq.sh LAYER [gelu]
L=$1
cd ~/duosight-tpu/work/e512 || exit 1
if [ -s "$L/block_eq.cvimodel" ]; then
	echo "$L already built"
	exit 0
fi
# A stopped run may have left these incomplete.
if [ ! -s "$L/cali_table" ]; then
	rm -f "$L"/block.mlir "$L"/block_origin.mlir "$L"/top_outputs.npz
fi
bash /mnt/d/duosight-eval/tpu/compile_eq.sh e512 1024 "$L" ${2:-}
