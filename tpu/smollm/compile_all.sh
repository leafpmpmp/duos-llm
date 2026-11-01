#!/bin/sh
# SPDX-License-Identifier: MIT
# Compile exported models for cv181x, 6 at a time.
# Usage: [BUILD=dir] sh compile_all.sh MODE [QTABLE] [MODELS...]   (on the host; runs the tpu-mlir container)
#   BUILD: export directory (default build); QTABLE: path relative to this directory
#   (mix modes); MODELS default: l00..l29 head
set -eu
cd "$(dirname "$0")"
B=${BUILD:-build}
MODE=${1:-int8}
QT=${2:+../../$2}
[ $# -gt 2 ] && shift 2 || set --
U="$(id -u):$(id -g)"
docker image inspect tpu-mlir:1.30.2 >/dev/null 2>&1 ||
	docker build -t tpu-mlir:1.30.2 -f ../../docker/Dockerfile.tpu-mlir ../../docker
if [ $# -gt 0 ]; then echo "$@" | tr ' ' '\n'; else (cd $B && ls -d l?? head); fi |
xargs -P 6 -I{} sh -c "docker run --rm -u $U -e HOME=/tmp \
	-v $PWD:/workspace -w /workspace/$B/{} tpu-mlir:1.30.2 sh ../../compile.sh {} 256 $MODE $QT \
	> $B/{}/compile_$MODE.log 2>&1 && echo '{} ok' || echo '{} FAILED'"
