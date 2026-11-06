#!/bin/bash
# Run a WSL script once per argument string, with WSL restarted between runs. A run starts
# only with START_MB of free commit, and stops if free commit falls under LIMIT_MB.
# Usage: bash guard_run.sh SCRIPT "ARGS" "ARGS"...   (SCRIPT: a path, or a file next to this one)
LIMIT_MB=10240
START_MB=14336
TRIES=3
T=/d/duosight-eval/tpu
LOG=$T/run.log
free_mb() {
	powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)" | tr -d '\r'
}
SRC=$1; shift
[ -f "$SRC" ] || SRC="$(dirname "$0")/$SRC"
S=$(basename "$SRC")
tr -d '\r' < "$SRC" > $T/$S
tr -d '\r' < "$(dirname "$0")/stop_mm.sh" > $T/stop_mm.sh
MIN=999999
run_one() {
	local try F P stopped
	for try in $(seq 1 $TRIES); do
		while :; do
			F=$(free_mb)
			[ -n "$F" ] && [ "$F" -ge "$START_MB" ] && break
			sleep 60
		done
		echo "$(date +%T) start $S $* try $try free_mb=$F" >> $LOG
		MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/duosight-eval/tpu/$S "$@" 2>&1 | tr -d '\000' | tee -a $LOG &
		P=$!
		stopped=0
		while kill -0 $P 2>/dev/null; do
			F=$(free_mb)
			[ -n "$F" ] && [ "$F" -lt "$MIN" ] && MIN=$F
			if [ -n "$F" ] && [ "$F" -lt "$LIMIT_MB" ]; then
				echo "$(date +%T) STOP in $S $*: free commit $F MB" | tee -a $LOG
				MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/duosight-eval/tpu/stop_mm.sh >> $LOG 2>&1
				stopped=1
				break
			fi
			sleep 5
		done
		wsl --shutdown
		sleep 6
		[ $stopped = 0 ] && return 0
		sleep 90
	done
	return 1
}
for A in "$@"; do
	run_one $A || break
done
echo "guard done: lowest free commit $MIN MB"
