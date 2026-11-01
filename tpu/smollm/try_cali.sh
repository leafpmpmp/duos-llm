#!/bin/sh
# SPDX-License-Identifier: MIT
# Try calibration variants on one MIX layer; print simulator accuracy.
# Usage (tpu-mlir container, cwd = layer dir): sh ../../try_cali.sh NAME
set -eu
N=$1
for v in "kl" "sq --sq" "mse --cali_method use_mse"; do
	set -- $v
	tag=$1
	shift
	[ -f "cali_$tag" ] || run_calibration.py "$N.mlir" --dataset calib --input_num 32 "$@" -o "cali_$tag" > "cali_$tag.log" 2>&1
	model_deploy.py --mlir "$N.mlir" --quantize INT8 --calibration_table "cali_$tag" \
		--quantize_table ../../qtable_base.txt --quant_input --quant_input_list 2,3,4,5 \
		--quant_output --quant_output_list 2,3 --processor cv181x \
		--model "${N}_mix_$tag.cvimodel" > "deploy_$tag.log" 2>&1
	python3 ../../evaluate.py "${N}_mix_$tag.cvimodel" | sed "s/^/$tag /"
done
