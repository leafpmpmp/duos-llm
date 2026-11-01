// SPDX-License-Identifier: MIT
// Single-core DRAM read and copy bandwidth.
// Usage: bw [MiB]   (default 64)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
	size_t mib = argc > 1 ? strtoul(argv[1], NULL, 0) : 64;
	size_t n = mib << 20, words = n / sizeof(uint64_t);
	uint64_t *a = malloc(n), *b = malloc(n);
	volatile uint64_t sink;
	double t, best_rd = 1e9, best_cp = 1e9;

	if (!a || !b)
		return 1;
	memset(a, 1, n);
	memset(b, 2, n);
	for (int r = 0; r < 5; r++) {
		uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

		t = now();
		for (size_t i = 0; i < words; i += 4) {
			s0 += a[i];
			s1 += a[i + 1];
			s2 += a[i + 2];
			s3 += a[i + 3];
		}
		t = now() - t;
		sink = s0 + s1 + s2 + s3;
		if (t < best_rd)
			best_rd = t;

		t = now();
		memcpy(b, a, n);
		t = now() - t;
		if (t < best_cp)
			best_cp = t;
	}
	(void)sink;
	printf("read %.0f MB/s, memcpy %.0f MB/s (read+write %.0f MB/s), %zu MiB\n",
	       n / best_rd / 1e6, n / best_cp / 1e6, 2 * n / best_cp / 1e6, mib);
	return 0;
}
