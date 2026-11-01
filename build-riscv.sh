#!/bin/sh
# SPDX-License-Identifier: MIT
# Cross-build llama.cpp for the Duo S C906 (riscv64).
# Usage: [ZFH=1] ./build-riscv.sh scalar|xtv
#   ZFH=1: use Zfh fp16 conversions (the C906 runs them; the kernel ISA string omits zfh)
# Output: build/riscv64-<variant>[-zfh]/bin
set -eu
variant=${1:?scalar or xtv}
here=$(cd "$(dirname "$0")" && pwd)
IMG=duos-llm-build:riscv64
docker image inspect "$IMG" >/dev/null 2>&1 ||
	docker build -t "$IMG" --build-arg ARCH=riscv64 "$here/docker"

case $variant in
xtv)	vec="-DGGML_XTHEADVECTOR=ON" ;;
scalar)	vec="-DGGML_XTHEADVECTOR=OFF" ;;
*)	echo "unknown variant $variant" >&2; exit 1 ;;
esac
zfh=OFF
if [ "${ZFH:-0}" = 1 ]; then zfh=ON; variant=$variant-zfh; fi

# The C906 has no V 1.0, Zvfh, Zicbop or Zihintpause. GGML_RVV must be OFF:
# ggml appends _xtheadvector to the march string after "v".
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$here:/w" -w /w "$IMG" sh -euc "
cat > /tmp/riscv64.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF
cmake -S llama.cpp -B build/riscv64-$variant -DCMAKE_TOOLCHAIN_FILE=/tmp/riscv64.cmake \
	-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
	-DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON \
	-DGGML_RVV=OFF -DGGML_RV_ZFH=$zfh -DGGML_RV_ZVFH=OFF -DGGML_RV_ZICBOP=OFF \
	-DGGML_RV_ZIHINTPAUSE=OFF $vec
cmake --build build/riscv64-$variant -j\$(nproc) --target llama-bench llama-cli llama-completion llama-perplexity llama-simple test-quantize-fns
"
