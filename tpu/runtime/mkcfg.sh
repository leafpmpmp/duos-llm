#!/bin/sh
# SPDX-License-Identifier: MIT
# Make a model directory for llm from per-layer variants (symlinks).
# Usage (on the board, in the models dir): [HEAD=variant] sh mkcfg.sh NAME DEFAULT [BF16_LAYERS...]
#   HEAD: head variant (default int8)
#   DEFAULT: int8 | mix | bf16 (variant for layers not listed)
#   e.g. sh mkcfg.sh mixbf mix 1 2 11   -> layers 1, 2, 11 BF16, the rest MIX
set -eu
name=$1
def=$2
shift 2
rm -rf "$name" && mkdir "$name"
for i in $(seq 0 29); do
	l=$(printf l%02d "$i")
	v=$def
	for b in "$@"; do [ "$b" -eq "$i" ] && v=bf16; done
	ln -s "../${l}_$v.cvimodel" "$name/$l.cvimodel"
done
ln -s "../head_${HEAD:-int8}.cvimodel" "$name/head.cvimodel"
ln -s ../embed.f16 "$name/embed.f16"
echo "$name: default $def, BF16 layers: ${*:-none}"
