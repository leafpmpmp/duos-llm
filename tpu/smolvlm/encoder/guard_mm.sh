#!/bin/bash
# Run compile_mm.sh one step at a time, with WSL restarted between steps. A step starts
# only with START_MB of free commit, and stops if free commit falls under LIMIT_MB.
# Usage: bash guard_mm.sh DIR N STEP...
LIMIT_MB=10240
START_MB=14336
TRIES=3
T=/d/duosight-eval/tpu
LOG=$T/mm.log
free_mb() {
	powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)" | tr -d '\r'
}
D=$1; N=$2; shift 2
tr -d '\r' < "$(dirname "$0")/compile_mm.sh" > $T/compile_mm.sh
tr -d '\r' < "$(dirname "$0")/stop_mm.sh" > $T/stop_mm.sh
MIN=999999
run_one() {
	local S=$1 try F P stopped
	for try in $(seq 1 $TRIES); do
		while :; do
			F=$(free_mb)
			[ -n "$F" ] && [ "$F" -ge "$START_MB" ] && break
			sleep 60
		done
		echo "$(date +%T) start $S try $try free_mb=$F" | tee -a $LOG
		MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/duosight-eval/tpu/compile_mm.sh $D $N $S >> $LOG 2>&1 &
		P=$!
		stopped=0
		while kill -0 $P 2>/dev/null; do
			F=$(free_mb)
			[ -n "$F" ] && [ "$F" -lt "$MIN" ] && MIN=$F
			if [ -n "$F" ] && [ "$F" -lt "$LIMIT_MB" ]; then
				echo "$(date +%T) STOP in $S: free commit $F MB" | tee -a $LOG
				MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/duosight-eval/tpu/stop_mm.sh >> $LOG 2>&1
				stopped=1
				break
			fi
			sleep 10
		done
		wsl --shutdown
		sleep 6
		[ $stopped = 0 ] && return 0
		sleep 90
	done
	return 1
}
for S in "$@"; do
	run_one "$S" || break
done
echo "guard done: lowest free commit $MIN MB"
ls -la $T/out/mm$N 2>/dev/null
