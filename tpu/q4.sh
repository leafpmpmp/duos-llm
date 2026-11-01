#!/bin/sh
# SPDX-License-Identifier: MIT
# Build config q4 for one cache size, or pack it for the board.
# q4: T = 4 tokens per call. The layers below are BF16. The other layers and
# the head are MIX2.
# Usage: tpu/q4.sh build S
#        tpu/q4.sh package S | ssh root@BOARD 'mkdir -p /root/llm/tpu && tar xf - -C /root/llm/tpu'
#   S: KV cache slots, 512 or 1024
set -eu
cmd=${1:-}
S=${2:-}
case $S in
512)	name=smollm135s512
	bf16="1 2 3 4 5 7 8 9 10 11 12 13 14 17 19 21 23 27 28 29" ;;
1024)	name=smollm135s1k
	bf16="1 2 3 4 5 7 8 9 10 11 12 17 21 27 28 29" ;;
*)	echo "usage: $0 build|package 512|1024" >&2
	exit 1 ;;
esac
top=$(cd "$(dirname "$0")/.." && pwd)
sm=$top/tpu/smollm
B=build_s$S
mix2=$(for i in $(seq 0 29); do
	case " $bf16 " in *" $i "*) ;; *) echo "$i" ;; esac
done)

names() { # layer numbers to lNN names
	for i in "$@"; do printf 'l%02d ' "$i"; done
}

build() {
	cd "$sm"
	py=../../.venv/bin/python
	hf=../../hf/SmolLM2-135M-Instruct
	text=../../data/wiki-20k.txt
	# The head has no cache: all cache sizes share it.
	if [ ! -f build_t4/head/head_mix2.cvimodel ]; then
		$py export_t.py $hf $text build_t4 4 head
		BUILD=build_t4 sh compile_all.sh mix2 qtable_head.txt head
		$py make_prefix.py $hf build_t4/prefix-imstart.kv --ids 1
	fi
	S=$S $py export_t.py $hf $text $B 4 0-29
	BUILD=$B sh compile_all.sh bf16 "" $(names $bf16)
	BUILD=$B sh compile_all.sh mix2 "" $(names $mix2)
	PREFIX_ID=1 $py ref_ppl.py $hf ../../data/wiki-eval.txt $B/wiki-eval-$S-p1.i32 "$S"
}

package() {
	stage=$top/deploy/tpu
	rm -rf "$stage"
	mkdir -p "$stage/shared" "$stage/$name"
	cp -l "$top/tpu/runtime/llm-arm64" "$stage/"
	cp -l "$sm/build_t4/head/head_mix2.cvimodel" "$sm/build_t4/embed.f16" \
		"$sm/build_t4/prefix-imstart.kv" "$top/tpu/runtime/system.txt" \
		"$top/tpu/runtime/mkcfg.sh" "$stage/shared/"
	for i in $bf16; do
		l=$(printf l%02d "$i")
		cp -l "$sm/$B/$l/${l}_bf16.cvimodel" "$stage/$name/"
	done
	for i in $mix2; do
		l=$(printf l%02d "$i")
		cp -l "$sm/$B/$l/${l}_mix2.cvimodel" "$stage/$name/"
	done
	cp -l "$sm/$B/wiki-eval-$S-p1.i32" "$stage/$name/"
	cd "$stage/$name"
	for f in ../shared/*; do ln -s "$f" .; done
	ln -s ../llm-arm64 .
	HEAD=mix2 sh mkcfg.sh q4 mix2 $bf16 >&2
	tar cf - -C "$stage" llm-arm64 shared "$name"
}

case $cmd in
build) build ;;
package) package ;;
*) echo "usage: $0 build|package 512|1024" >&2
	exit 1 ;;
esac
