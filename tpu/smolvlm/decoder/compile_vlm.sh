#!/bin/bash
# Compile one SmolVLM decoder model for cv181x in WSL, then check it against float.
# Uses compile.sh from duos-llm. The layer list is the q4 config for S=512.
# Usage: bash compile_vlm.sh NAME   (NAME: l00..l29 or head)
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
N=$1
FROM=/mnt/d/duosight-eval/decoder
UP=/mnt/d/duosight-eval/duos-llm/tpu/smollm
W=$HOME/duosight-tpu/work/smollm
BF16=" 1 2 3 4 5 7 8 9 10 11 12 13 14 17 19 21 23 27 28 29 "

case $N in
head)	MODE=mix2
	QT=../../qtable_head.txt ;;
*)	i=$((10#${N#l}))
	case "$BF16" in
	*" $i "*) MODE=bf16 ;;
	*) MODE=mix2 ;;
	esac
	QT= ;;
esac

if [ -s "$FROM/out2/${N}_$MODE.cvimodel" ]; then
	echo "$N already built"
	exit 0
fi
mkdir -p "$W/build_vlm2" "$FROM/out2"
cp "$UP/compile.sh" "$UP/gen_qtable.py" "$UP/qtable_head.txt" "$UP/evaluate.py" "$UP/evaluate_head.py" "$W/"
# The clone on Windows has CRLF line ends.
sed -i 's/\r$//' "$W/compile.sh" "$W/gen_qtable.py" "$W/qtable_head.txt" "$W/evaluate.py" "$W/evaluate_head.py"
[ -d "$W/build_vlm2/$N" ] || cp -r "$FROM/build_vlm/$N" "$W/build_vlm2/" || exit 1
cd "$W/build_vlm2/$N" || exit 1

sh ../../compile.sh "$N" 512 "$MODE" $QT > compile.log 2>&1 ||
	{ echo "$N compile FAILED"; tail -4 compile.log | cut -c1-160; exit 1; }
if [ "$N" = head ]; then
	python3 ../../evaluate_head.py "${N}_$MODE.cvimodel" | cut -c1-100
else
	python3 ../../evaluate.py "${N}_$MODE.cvimodel" | cut -c1-100
fi
cp "${N}_$MODE.cvimodel" "$FROM/out2/"
echo "$N $MODE ok"
