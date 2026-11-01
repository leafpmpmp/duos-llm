# SPDX-License-Identifier: MIT
# Cross-build image for one target. The build scripts make it on first use:
#   docker build -t duos-llm-build:arm64 --build-arg ARCH=arm64 docker
FROM debian:trixie-slim
ARG ARCH=arm64
RUN case "$ARCH" in \
	arm64) T=aarch64-linux-gnu ;; \
	riscv64) T=riscv64-linux-gnu ;; \
	*) echo "ARCH must be arm64 or riscv64" >&2; exit 1 ;; \
	esac && \
	apt-get update && \
	apt-get install -y --no-install-recommends gcc-$T g++-$T \
		cmake ninja-build make gcc g++ git ca-certificates python3 && \
	rm -rf /var/lib/apt/lists/*
