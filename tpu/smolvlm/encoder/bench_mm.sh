#!/bin/bash
# Time the 1x1 conv block and the MatMul block on the board, in one session, alternating.
# Block 6 at 512 px, each in all INT8, mix-a and all BF16.
# Usage: bash bench_mm.sh [ROUNDS]   (KNOWN: known_hosts file for the board)
ROUNDS=${1:-2}
BOARD=root@192.168.100.108
SSHO="-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o UserKnownHostsFile=$KNOWN"
T=/d/duosight-eval/tpu/out
ssh $SSHO $BOARD 'mkdir -p /tmp/mmb/conv /tmp/mmb/mm; echo "busy: $(pgrep -c duos-detect) detect, $(pgrep -c vlm) vlm, $(pgrep -c chain) chain"' || exit 1
scp -q $SSHO $T/512/block_{int8,mixa,bf16}.cvimodel $BOARD:/tmp/mmb/conv/ || exit 1
scp -q $SSHO $T/mm1024/block_{int8,mixa,bf16}.cvimodel $BOARD:/tmp/mmb/mm/ || exit 1
for r in $(seq 1 $ROUNDS); do
	for p in int8 mixa bf16; do
		for w in conv mm; do
			ssh $SSHO $BOARD "cd /tmp/mmb && LD_LIBRARY_PATH=/root/tpu/lib /root/llm/tpu/tpubench-arm64 $w/block_$p.cvimodel 20" |
				awk -v r=$r -v w=$w -v p=$p '/^forward/ {print "round", r, w, p, $0}'
		done
	done
done
