#!/bin/sh
# SPDX-License-Identifier: MIT
# Download WikiText-2 and cut the calibration and evaluation texts.
# Usage: tools/fetch-data.sh
# Output: data/wiki-20k.txt, data/wiki-eval.txt
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/dl
cd data/dl
[ -f wikitext-2-raw-v1.zip ] ||
	curl -fLO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -oq wikitext-2-raw-v1.zip
cd ..
raw=dl/wikitext-2-raw/wiki.test.raw
head -c 20000 "$raw" > wiki-20k.txt
tail -c +100001 "$raw" | head -c 10000 > wiki-eval.txt
