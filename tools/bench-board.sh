#!/bin/sh
# SPDX-License-Identifier: MIT
# Run on the board: llama-bench for each build variant and model.
# Usage: sh bench-board.sh <model.gguf>...
# Env: VARIANTS (default "scalar xtv"), P (prompt tokens), N (generated tokens), R (repeats)
set -eu
cd /root/llm
for m in "$@"; do
	for v in ${VARIANTS:-scalar xtv}; do
		echo "== $v $(basename "$m")"
		"bin/$v/llama-bench" -m "$m" -t 1 -p "${P:-64}" -n "${N:-32}" -r "${R:-2}" -o md 2>/dev/null |
			grep -E '^\| [a-z]'
	done
done
