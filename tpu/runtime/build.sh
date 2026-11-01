#!/bin/sh
# SPDX-License-Identifier: MIT
# Cross-build the TPU runtime for the A53 (arm64).
# Needs ../../build/arm64 (./build-arm64.sh) for the llama.cpp vocabulary code
# and tools/build-cviruntime.sh for the TPU runtime.
# Output: llm-arm64
set -eu
here=$(cd "$(dirname "$0")" && pwd)
top=$(cd "$here/../.." && pwd)
RT=${RT:-$top/build/cviruntime/arm64/install}
[ -d "$RT/lib" ] || { echo "no TPU runtime in $RT" >&2; exit 1; }
IMG=duos-llm-build:arm64
docker image inspect "$IMG" >/dev/null 2>&1 ||
	docker build -t "$IMG" --build-arg ARCH=arm64 "$top/docker"

docker run --rm -u "$(id -u):$(id -g)" -v "$top:/w" -v "$RT:/rt:ro" -w /w/tpu/runtime \
	"$IMG" sh -euc '
B=/w/build/arm64
aarch64-linux-gnu-gcc -O2 -std=gnu11 -Wall -Wextra -mtune=cortex-a53 \
	-I/rt/include -I/w/llama.cpp/include -I/w/llama.cpp/ggml/include -c -o llm.o llm.c
aarch64-linux-gnu-g++ -o llm-arm64 llm.o $B/src/libllama.a $B/ggml/src/libggml.a \
	$B/ggml/src/libggml-cpu.a $B/ggml/src/libggml-base.a \
	-L/rt/lib -lcviruntime -lcvikernel -lpthread -lm -ldl
aarch64-linux-gnu-strip llm-arm64
rm llm.o
'
ls -l "$here/llm-arm64"
