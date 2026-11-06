#!/bin/sh
# Combine the T=16 and the T=1 build of each decoder layer into one cvimodel: two
# programs over one weight section. Run in the tpu-mlir container.
# Usage: sh combine_t1.sh T16_DIR T1_DIR OUT_DIR
#   T16_DIR, T1_DIR: compiled builds (lNN/lNN_mix2.cvimodel), T1_DIR from export_pad.py
set -eu
mkdir -p "$3"
for i in $(seq -f '%02g' 0 29); do
	a=$1/l$i/l${i}_mix2.cvimodel
	b=$2/l$i/l${i}_mix2.cvimodel
	wa=$(model_tool --info "$a" | awk '$2 == "weight" {print $6}')
	wb=$(model_tool --info "$b" | awk '$2 == "weight" {print $6}')
	[ "$wa" = "$wb" ] || { echo "l$i: the weights differ"; exit 1; }
	model_tool --combine "$a" "$b" -o "$3/l$i.cvimodel" > /dev/null
done
echo "30 layers in $3"
