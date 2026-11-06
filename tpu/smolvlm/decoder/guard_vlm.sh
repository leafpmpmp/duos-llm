#!/bin/bash
# Compile decoder models one at a time, with WSL restarted between them. A model starts
# only with START_MB of free commit, and stops if free commit falls under LIMIT_MB.
# Usage: bash guard_vlm.sh NAME...
LIMIT_MB=10240
START_MB=14336
TRIES=3
T=/d/duosight-eval/decoder
free_mb() {
	powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)" | tr -d '\r'
}
MIN=999999
run_one() {
	local N=$1 try F P stopped
	for try in $(seq 1 $TRIES); do
		while :; do
			F=$(free_mb)
			[ -n "$F" ] && [ "$F" -ge "$START_MB" ] && break
			sleep 60
		done
		echo "$(date +%T) start $N try $try free_mb=$F" | tee -a $T/compile.log
		MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/duosight-eval/decoder/compile_vlm.sh "$N" >> $T/compile.log 2>&1 &
		P=$!
		stopped=0
		while kill -0 $P 2>/dev/null; do
			F=$(free_mb)
			[ -n "$F" ] && [ "$F" -lt "$MIN" ] && MIN=$F
			if [ -n "$F" ] && [ "$F" -lt "$LIMIT_MB" ]; then
				echo "$(date +%T) STOP in $N: free commit $F MB" | tee -a $T/compile.log
				MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/duosight-eval/decoder/stop_vlm.sh >> $T/compile.log 2>&1
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
for N in "$@"; do
	run_one "$N" || break
done
echo "guard done: lowest free commit $MIN MB, built $(ls $T/out2 | grep -c cvimodel) models"
