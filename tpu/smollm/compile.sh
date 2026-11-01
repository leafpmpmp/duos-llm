#!/bin/sh
# SPDX-License-Identifier: MIT
# Compile an exported decode layer or the LM head for cv181x (SG2000).
# Usage (tpu-mlir container, cwd = model dir): sh ../compile.sh NAME S MODE [QTABLE]
#   NAME: lNN (decode layer) or head (final norm + LM head)
#   MODE: int8 | bf16 | mix | mix2 (INT8 with the ops in QTABLE as BF16;
#         mix2 only names the output, for a second table)
#   QTABLE: default ./qtable.txt (made by gen_qtable.py for T-token models)
# int8 layers use int8 I/O (the KV cache stays int8 in the input buffers).
# T-token bf16 layers take the RoPE tables, cache and mask as BF16.
# mix layers keep h and dh fp32 (the residual stream has activations up to
# ~26000) and quantize only the RoPE tables, k_cache/v_cache and k_new/v_new.
# The head keeps fp32 logits for sampling.
# T-token exports (export_t.py) bring their own ./shapes and ./quant_lists.
set -eu
N=$1
S=$2
MODE=$3
QT=${4:-qtable.txt}
OUTQ=--quant_output
if [ -f shapes ]; then
	SHAPES=$(cat shapes)
	QLISTS=$(cat quant_lists)
	[ "$N" = head ] && OUTQ=
elif [ "$N" = head ]; then
	SHAPES="[[1,1,576]]"
	QLISTS=
	OUTQ=
else
	SHAPES="[[1,1,576],[1,1,1,64],[1,1,1,64],[1,3,$S,64],[1,3,$S,64],[1,3,3,$((S + 1))]]"
	QLISTS="--quant_input_list 2,3,4,5 --quant_output_list 2,3"
fi
TEST=$(ls test_in_pos*.npz | sed -n 3p)

if [ ! -f "$N.mlir" ]; then
	model_transform.py --model_name "$N" --model_def "$N.onnx" --input_shapes "$SHAPES" \
		--test_input "$TEST" --test_result top_outputs.npz --mlir "$N.mlir"
fi
if [ "$MODE" != bf16 ] && [ ! -f cali_table ]; then
	run_calibration.py "$N.mlir" --dataset calib --input_num 32 -o cali_table
fi
case $MODE in
int8)	set -- --quantize INT8 --calibration_table cali_table --quant_input $OUTQ ;;
bf16)	set -- --quantize BF16
	[ -n "$QLISTS" ] && [ "$N" != head ] && set -- "$@" --quant_input $QLISTS --quant_output ;;
mix|mix2)
	[ -f "$QT" ] || python3 ../../gen_qtable.py "$N.mlir" > "$QT"
	if [ "$N" = head ]; then # fp32 h in, fp32 logits out
		set -- --quantize INT8 --calibration_table cali_table --quantize_table "$QT"
	else
		set -- --quantize INT8 --calibration_table cali_table --quantize_table "$QT" \
			--quant_input $QLISTS --quant_output
	fi ;;
esac
model_deploy.py --mlir "$N.mlir" "$@" --processor cv181x --test_input "$TEST" \
	--test_reference top_outputs.npz --tolerance 0.0,-1.0 --model "${N}_$MODE.cvimodel"
