// SPDX-License-Identifier: MIT
// Q8_0 dot-product throughput on the C906 (XTheadVector), cached vs DRAM.
// Same per-block inline asm as the llama.cpp kernel; optional scalar
// "touch" loads ahead of the vector loads act as a software prefetch.
// Usage: dotbench [MiB]   (default 64)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32
#define ROW 576 // SmolLM2-135M hidden size

typedef struct {
	uint16_t d;
	int8_t qs[QK];
} __attribute__((packed)) block_q8_0;

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline int dot_block(const int8_t *qx, const int8_t *qy)
{
	int sumi, vl32 = QK, vl4 = 4;

	__asm__ __volatile__(
		"th.vsetvli zero, %[vl32], e8, m2\n\t"
		"th.vlb.v v0, (%[qx])\n\t"
		"th.vlb.v v2, (%[qy])\n\t"
		"th.vwmul.vv v4, v0, v2\n\t"
		"th.vsetvli zero, %[vl4], e32, m1\n\t"
		"th.vmv.v.x v8, zero\n\t"
		"th.vsetvli zero, %[vl32], e16, m4\n\t"
		"th.vwredsum.vs v8, v4, v8\n\t"
		"th.vsetvli zero, %[vl4], e32, m1\n\t"
		"th.vmv.x.s %[sumi], v8"
		: [sumi] "=&r" (sumi)
		: [vl32] "r" (vl32), [vl4] "r" (vl4), [qx] "r" (qx), [qy] "r" (qy)
		: "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8");
	return sumi;
}

// One pass over all rows; dist = touch distance in bytes (0 = none).
static float run(const block_q8_0 *x, size_t nrows, const block_q8_0 *y, int dist)
{
	const int nb = ROW / QK;
	float total = 0;

	for (size_t r = 0; r < nrows; r++) {
		const block_q8_0 *xr = x + r * nb;
		float sumf = 0;

		for (int i = 0; i < nb; i++) {
			if (dist) {
				const volatile uint8_t *p = (const uint8_t *)&xr[i] + dist;
				uint8_t t;

				__asm__ __volatile__("lbu %0, 0(%1)" : "=r" (t) : "r" (p));
				(void)t;
			}
			sumf += dot_block(xr[i].qs, y[i].qs) * (float)(xr[i].d & 0xff);
		}
		total += sumf;
	}
	return total;
}

int main(int argc, char **argv)
{
	size_t mib = argc > 1 ? strtoul(argv[1], NULL, 0) : 64;
	const int nb = ROW / QK;
	size_t rowbytes = nb * sizeof(block_q8_0);
	size_t nrows = (mib << 20) / rowbytes, hot_rows = 16384 / rowbytes;
	block_q8_0 *x = malloc((nrows + 64) * rowbytes), y[ROW / QK];
	static const int dists[] = { 0, 64, 128, 256, 512, 1024 };

	if (!x)
		return 1;
	for (size_t i = 0; i < (nrows + 64) * rowbytes; i++)
		((uint8_t *)x)[i] = (uint8_t)(i * 2654435761u >> 13);
	for (int i = 0; i < nb; i++) {
		y[i].d = 1;
		for (int j = 0; j < QK; j++)
			y[i].qs[j] = (int8_t)(i * 7 + j * 3);
	}

	// Cached: the same 16 KB of rows again and again.
	double t = now();
	volatile float sink = 0;
	size_t reps = nrows / hot_rows;
	for (size_t k = 0; k < reps; k++)
		sink += run(x, hot_rows, y, 0);
	t = now() - t;
	printf("cached 16 KB:          %6.0f MB/s of weights\n", reps * hot_rows * rowbytes / t / 1e6);

	for (size_t k = 0; k < sizeof(dists) / sizeof(dists[0]); k++) {
		double best = 1e9;

		for (int rep = 0; rep < 3; rep++) {
			t = now();
			sink += run(x, nrows, y, dists[k]);
			t = now() - t;
			if (t < best)
				best = t;
		}
		printf("DRAM %3zu MiB, touch %4d: %6.0f MB/s of weights\n", mib, dists[k],
		       nrows * rowbytes / best / 1e6);
	}
	(void)sink;
	return 0;
}
