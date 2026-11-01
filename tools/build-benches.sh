#!/bin/sh
# SPDX-License-Identifier: MIT
# Cross-build the benchmarks.
# Usage: tools/build-benches.sh arm64|riscv64
# Output: build/<arch>-tools
# arm64 also builds the TPU benchmarks. They need tools/build-cviruntime.sh.
set -eu
top=$(cd "$(dirname "$0")/.." && pwd)
ARCH=${1:?usage: $0 arm64|riscv64}
RT=${RT:-$top/build/cviruntime/arm64/install}
IMG=duos-llm-build:$ARCH
docker image inspect "$IMG" >/dev/null 2>&1 ||
	docker build -t "$IMG" --build-arg ARCH="$ARCH" "$top/docker"
mkdir -p "$top/build/$ARCH-tools"

case $ARCH in
arm64)
	[ -d "$RT/lib" ] || { echo "no TPU runtime in $RT" >&2; exit 1; }
	docker run --rm -u "$(id -u):$(id -g)" -v "$top:/w" -v "$RT:/rt:ro" -w /w "$IMG" sh -euc '
CC="aarch64-linux-gnu-gcc -O2 -std=gnu11 -Wall -Wextra -mtune=cortex-a53"
O=build/arm64-tools
TPU="-I/rt/include -L/rt/lib -lcviruntime -lcvikernel -lpthread -lm -ldl -lstdc++"
$CC -o $O/bw tools/bw.c
$CC -o $O/tpubench tpu/tpubench.c $TPU
$CC -o $O/cmacopy tpu/cmacopy.c $TPU
' ;;
riscv64)
	docker run --rm -u "$(id -u):$(id -g)" -v "$top:/w" -w /w "$IMG" sh -euc '
CC="riscv64-linux-gnu-gcc -O2 -Wall -Wextra -mabi=lp64d"
O=build/riscv64-tools
$CC -o $O/bw tools/bw.c
$CC -fno-math-errno -march=rv64gc_zfh -o $O/zfh tools/zfh.c
$CC -march=rv64gc_xtheadvector -o $O/dotbench tools/dotbench.c
' ;;
*)
	echo "usage: $0 arm64|riscv64" >&2
	exit 1 ;;
esac
