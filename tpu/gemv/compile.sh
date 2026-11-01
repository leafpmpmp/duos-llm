#!/bin/sh
# SPDX-License-Identifier: MIT
# Compile one GEMV test graph to an INT8 cvimodel for cv181x (SG2000).
# int8 input and output, so the timing covers only the TPU.
# Usage (in the tpu-mlir container, cwd = this directory): sh compile.sh NAME K
set -eu
N=$1
K=$2
mkdir -p "$N.build" && cd "$N.build"
model_transform.py --model_name "$N" --model_def "../$N.onnx" \
	--input_shapes "[[1,$K]]" --test_input "../calib_$K/in0.npz" \
	--test_result top_outputs.npz --mlir "$N.mlir"
run_calibration.py "$N.mlir" --dataset "../calib_$K" --input_num 8 -o cali_table
model_deploy.py --mlir "$N.mlir" --quantize INT8 --calibration_table cali_table \
	--processor cv181x --quant_input --quant_output \
	--test_input "../calib_$K/in0.npz" --test_reference top_outputs.npz \
	--tolerance 0.9,0.6 --model "../${N}_int8.cvimodel"
