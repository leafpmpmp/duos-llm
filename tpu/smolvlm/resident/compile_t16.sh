#!/bin/bash
# Compile one decoder model of the resident build (T=16, S=128) for cv181x in WSL, then
# check it against float. Uses compile.sh from duos-llm.
# Usage: bash compile_t16.sh NAME [MODE] [BUILD]
#   NAME: l00..l29 or head; MODE: mix2 (default) or bf16
#   BUILD: export folder under D:/duosight-eval/decoder (default build_t16); models go to out_t16...
export PATH=$HOME/duosight-tpu/venv/bin:$PATH
N=$1
MODE=${2:-mix2}
B=${3:-build_t16}
O=out_${B#build_}
FROM=/mnt/d/duosight-eval/decoder
UP=/mnt/d/duosight-eval/duos-llm/tpu/smollm
W=$HOME/duosight-tpu/work/smollm
QT=
[ "$N" = head ] && QT=../../qtable_head.txt

if [ -s "$FROM/$O/${N}_$MODE.cvimodel" ]; then
	echo "$N $MODE already built"
	exit 0
fi
mkdir -p "$W/$B" "$FROM/$O"
cp "$UP/compile.sh" "$UP/gen_qtable.py" "$UP/qtable_head.txt" "$UP/evaluate.py" "$UP/evaluate_head.py" "$W/"
# The clone on Windows has CRLF line ends.
sed -i 's/\r$//' "$W/compile.sh" "$W/gen_qtable.py" "$W/qtable_head.txt" "$W/evaluate.py" "$W/evaluate_head.py"
[ -d "$W/$B/$N" ] || cp -r "$FROM/$B/$N" "$W/$B/" || exit 1
cd "$W/$B/$N" || exit 1

sh ../../compile.sh "$N" 128 "$MODE" $QT > compile_$MODE.log 2>&1 ||
	{ echo "$N $MODE compile FAILED"; tail -4 compile_$MODE.log | cut -c1-160; exit 1; }
if [ "$N" = head ]; then
	python3 ../../evaluate_head.py "${N}_$MODE.cvimodel" | cut -c1-100
else
	python3 ../../evaluate.py "${N}_$MODE.cvimodel" | cut -c1-100
fi
cp "${N}_$MODE.cvimodel" "$FROM/$O/"
echo "$N $MODE ok"
