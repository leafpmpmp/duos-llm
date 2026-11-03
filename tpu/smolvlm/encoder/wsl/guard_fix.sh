#!/bin/bash
# Run fix_one.sh for each block, with WSL restarted between blocks and a commit guard.
LIMIT_MB=10240
START_MB=14336
TRIES=4
T=/d/duosight-eval/tpu
free_mb() {
	powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)" | tr -d '\r'
}
MIN=999999
: > $T/fix.log
run_one() {
	local L=$1 try F P waited stopped
	for try in $(seq 1 $TRIES); do
		waited=0
		while [ "$(free_mb)" -lt "$START_MB" ]; do
			waited=$((waited + 30))
			if [ $waited -gt 1800 ]; then
				echo "$(date +%T) gave up on $L: free commit stayed under $START_MB MB" | tee -a $T/fix.log
				return 1
			fi
			sleep 30
		done
		echo "$(date +%T) start $L try $try free_mb=$(free_mb)" | tee -a $T/fix.log
		MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/duosight-eval/tpu/${SCRIPT:-fix_one.sh} $L >> $T/fix.log 2>&1 &
		P=$!
		stopped=0
		while kill -0 $P 2>/dev/null; do
			F=$(free_mb)
			[ -n "$F" ] && [ "$F" -lt "$MIN" ] && MIN=$F
			if [ -n "$F" ] && [ "$F" -lt "$LIMIT_MB" ]; then
				echo "$(date +%T) STOP in $L: free commit $F MB" | tee -a $T/fix.log
				MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/duosight-eval/tpu/stop_eq.sh >> $T/fix.log 2>&1
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
for L in "$@"; do
	run_one "$L" || break
done
echo "guard done: lowest free commit $MIN MB"
tr -d '\000' < $T/fix.log | tail -14
ls $T/out/e512 | wc -l
