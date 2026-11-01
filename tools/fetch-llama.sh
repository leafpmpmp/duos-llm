#!/bin/sh
# SPDX-License-Identifier: MIT
# Clone llama.cpp at the tested tag and apply the XTheadVector patch.
# Usage: tools/fetch-llama.sh
set -eu
cd "$(dirname "$0")/.."
if [ -d llama.cpp ]; then
	echo "llama.cpp exists"
	exit 0
fi
git clone --depth 1 -b b11951 https://github.com/ggml-org/llama.cpp
git -C llama.cpp apply ../patches/llama.cpp-b11951-c906-xtheadvector.patch
