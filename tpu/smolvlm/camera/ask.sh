#!/bin/bash
# SPDX-License-Identifier: MIT
# Ask questions about one camera frame, with every step on the board.
# The encoder and the decoder are separate programs, so they swap in CMA.
# Usage: bash ask.sh QUESTION...
#   RAW: use this raw frame instead of opening the camera. watch.py sets it,
#        because the detector holds the camera.
#   VBLANK, GAIN: sensor settings when this script takes the frame itself
V=/root/vlm
T=/tmp/ask
SD=/dev/v4l-subdev2
VBLANK=${VBLANK:-14000}
GAIN=${GAIN:-940}
export LD_LIBRARY_PATH=/root/tpu/lib
mkdir -p $T

last=$(date +%s.%N)
start=$last
step() {
	now=$(date +%s.%N)
	printf "%-16s %6.2f s\n" "$1" "$(echo "$now - $last" | bc)"
	last=$now
}

if [ -z "$RAW" ]; then
	RAW=$T/f.raw
	cd /root/tpu && sh camera-setup.sh 1728 1296 0 >/dev/null 2>&1
	v4l2-ctl -d $SD -c vertical_blanking=$VBLANK -c analogue_gain=$GAIN
	E=$(v4l2-ctl -d $SD -l | sed -n 's/.*exposure.*max=\([0-9]*\).*/\1/p')
	v4l2-ctl -d $SD -c exposure=$E
	v4l2-ctl -d /dev/video0 --stream-mmap=4 --stream-skip=3 --stream-count=1 --stream-to=$RAW >/dev/null 2>&1 || exit 1
	v4l2-ctl -d $SD -c vertical_blanking=40 -c analogue_gain=0 -c exposure=1288
	step "capture"
fi
cd $V
./frame $RAW 1728 1296 $T/f.rgb >/dev/null || exit 1
step "raw to canvas"
./pre w $T/f.rgb $T/in.bin || exit 1
step "patch embedding"
(cd /root/enc && ./chain2 e512 1024 < $T/in.bin > $T/out.bin 2> $T/chain.log) || { cat $T/chain.log; exit 1; }
step "12 blocks"
./post w $T/out.bin $T/rows.bin || exit 1
step "connector"
python3 tok.py encode w/vocab.json $T/rows.bin $T/prompts.bin "$@" || exit 1
step "tokenize"
./vlm m m/prefix-imstart.kv 49190 24 49279 < $T/prompts.bin > $T/ans.txt 2> $T/vlm.log || { cat $T/vlm.log; exit 1; }
step "decoder, all"
python3 tok.py decode w/vocab.json $T/ans.txt > $T/ans_text.txt
step "detokenize"
printf "%-16s %6.2f s\n" "total" "$(echo "$(date +%s.%N) - $start" | bc)"
grep -E '^loaded|encoder mean' $T/chain.log $T/vlm.log | cut -d: -f2- | cut -c1-70
i=1
while read -r line; do
	p=$(sed -n "${i}p" $T/ans.txt | awk '{print $2 " ms prefill, " $3 " ms decode"}')
	echo "Q: ${!i}"
	echo "A: $line   ($p)"
	i=$((i + 1))
done < $T/ans_text.txt
