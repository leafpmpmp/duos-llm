#!/bin/bash
# Compile one equalized block for cv181x in WSL, then check it against float.
# Usage: bash compile_eq.sh DIR N LAYER [gelu]
#   DIR: folder under ~/duosight-tpu/work
#   N: patches (1024 for 512 px)
#   gelu: keep GELU in BF16 (blocks 5, 10, 11, where x^3 outgrows INT8)
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
HERE=$(dirname "$(readlink -f "$0")")
D=$1
N=$2
L=$3
G=${4:-}
cd ~/duosight-tpu/work/"$D"/"$L" || exit 1
if [ ! -f block.mlir ]; then
	model_transform.py --model_name block --model_def block.onnx --input_shapes "[[1,768,1,$N]]" \
		--test_input test_in.npz --test_result top_outputs.npz --mlir block.mlir > c.log 2>&1 ||
		{ echo "$L transform FAILED"; tail -3 c.log | cut -c1-140; exit 1; }
fi
if [ ! -f cali_table ]; then
	run_calibration.py block.mlir --dataset calib --input_num 32 -o cali_table >> c.log 2>&1 ||
		{ echo "$L calibration FAILED"; tail -3 c.log | cut -c1-140; exit 1; }
fi
python3 "$HERE/qtable_eq.py" "$G" > qt_eq.txt
model_deploy.py --mlir block.mlir --quantize INT8 --calibration_table cali_table \
	--quantize_table qt_eq.txt --processor cv181x --test_input test_in.npz \
	--test_reference top_outputs.npz --tolerance 0.0,-1.0 --model block_eq.cvimodel >> c.log 2>&1 ||
	{ echo "$L deploy FAILED"; tail -3 c.log | cut -c1-140; exit 1; }
model_runner.py --input test_in.npz --model block_eq.cvimodel --output host_eq.npz > r.log 2>&1
python3 "$HERE/check_eq.py" "$L"
mkdir -p /mnt/d/duosight-eval/tpu/out/"$D"
cp block_eq.cvimodel /mnt/d/duosight-eval/tpu/out/"$D"/"$L".cvimodel
