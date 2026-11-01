#!/bin/sh
# SPDX-License-Identifier: MIT
# Download the SmolLM2 weights and GGUF files at pinned revisions, then check them.
# Usage: tools/fetch-models.sh
set -eu
cd "$(dirname "$0")/.."
HF=https://huggingface.co

get() { # URL FILE
	[ -f "$2" ] && return
	curl -fL --retry 5 -o "$2.part" "$1"
	mv "$2.part" "$2"
}

REV=12fd25f77366fa6b3b4b768ec3050bf629380bac
mkdir -p hf/SmolLM2-135M-Instruct
for f in config.json generation_config.json special_tokens_map.json \
	tokenizer_config.json tokenizer.json model.safetensors; do
	get "$HF/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/$REV/$f" "hf/SmolLM2-135M-Instruct/$f"
done

mkdir -p models
for f in 135M:09816acd5d99df7be770d85ea30822623dab342c:Q8_0 \
	135M:09816acd5d99df7be770d85ea30822623dab342c:Q4_0 \
	135M:09816acd5d99df7be770d85ea30822623dab342c:Q4_K_M \
	360M:7be6f65f1db715fe5dc5a4634c0d459b4eed42ec:Q4_0 \
	360M:7be6f65f1db715fe5dc5a4634c0d459b4eed42ec:Q4_K_M; do
	size=${f%%:*}
	quant=${f##*:}
	rev=${f#*:}
	rev=${rev%:*}
	n=SmolLM2-$size-Instruct-$quant.gguf
	get "$HF/bartowski/SmolLM2-$size-Instruct-GGUF/resolve/$rev/$n" "models/$n"
done

(cd hf && sha256sum -c --quiet SHA256SUMS)
(cd models && sha256sum -c --quiet SHA256SUMS)
echo "models ok"
