#!/bin/sh
# SPDX-License-Identifier: MIT
# Cross-build llama.cpp for the Duo S Cortex-A53 (arm64).
# Usage: ./build-arm64.sh
# Output: build/arm64/bin
set -eu
here=$(cd "$(dirname "$0")" && pwd)
IMG=duos-llm-build:arm64
docker image inspect "$IMG" >/dev/null 2>&1 ||
	docker build -t "$IMG" --build-arg ARCH=arm64 "$here/docker"

# The A53 is ARMv8.0-A: NEON, but no dotprod, i8mm or FP16 arithmetic.
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$here:/w" -w /w "$IMG" sh -euc "
cat > /tmp/arm64.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF
cmake -S llama.cpp -B build/arm64 -DCMAKE_TOOLCHAIN_FILE=/tmp/arm64.cmake \
	-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
	-DGGML_CPU_ARM_ARCH=armv8-a -DCMAKE_C_FLAGS=-mtune=cortex-a53 -DCMAKE_CXX_FLAGS=-mtune=cortex-a53 \
	-DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build/arm64 -j\$(nproc) --target llama-bench llama-cli llama-completion llama-perplexity llama-simple test-quantize-fns
"
