// SPDX-License-Identifier: MIT
// The final LayerNorm and the connector on the CPU: the last block's output to
// the 64 image rows of the decoder.
//
// Usage: post WEIGHTS_DIR IN.bin OUT.bin
// IN.bin: float32 [768][1024]. OUT.bin: float32 [64][576].
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GRID 32
#define N (GRID * GRID)
#define D 768
#define SCALE 4
#define ROWS 64
#define WIDE (D * SCALE * SCALE)
#define OUT 576
#define EPS 1e-6f

static float *load(const char *dir, const char *name, size_t n)
{
	char path[1024];
	float *x = malloc(n * sizeof(float));
	FILE *f;

	snprintf(path, sizeof(path), "%s/%s", dir, name);
	f = fopen(path, "rb");
	if (!x || !f || fread(x, sizeof(float), n, f) != n) {
		fprintf(stderr, "post: cannot read %s\n", path);
		exit(1);
	}
	fclose(f);
	return x;
}

static float dot(const float *restrict a, const float *restrict b, int n)
{
	float s = 0;

	for (int i = 0; i < n; i++)
		s += a[i] * b[i];
	return s;
}

int main(int argc, char **argv)
{
	static float x[D][N], wide[ROWS][WIDE], out[ROWS][OUT];
	FILE *f;

	if (argc != 4) {
		fprintf(stderr, "usage: post WEIGHTS_DIR IN.bin OUT.bin\n");
		return 1;
	}
	float *lw = load(argv[1], "ln_w.f32", D);
	float *lb = load(argv[1], "ln_b.f32", D);
	float *proj = load(argv[1], "proj.f32", (size_t)OUT * WIDE);

	f = fopen(argv[2], "rb");
	if (!f || fread(x, 1, sizeof(x), f) != sizeof(x)) {
		fprintf(stderr, "post: cannot read %s\n", argv[2]);
		return 1;
	}
	fclose(f);

	for (int p = 0; p < N; p++) {
		int y = p / GRID, u = p % GRID;
		int row = y / SCALE * (GRID / SCALE) + u / SCALE;
		float *o = wide[row] + (y % SCALE) * D * SCALE + (u % SCALE) * D;
		float mean = 0, var = 0, r;

		for (int d = 0; d < D; d++)
			mean += x[d][p];
		mean /= D;
		for (int d = 0; d < D; d++)
			var += (x[d][p] - mean) * (x[d][p] - mean);
		r = 1 / sqrtf(var / D + EPS);
		for (int d = 0; d < D; d++)
			o[d] = (x[d][p] - mean) * r * lw[d] + lb[d];
	}
	for (int r = 0; r < ROWS; r++)
		for (int i = 0; i < OUT; i++)
			out[r][i] = dot(proj + (size_t)i * WIDE, wide[r], WIDE);

	f = fopen(argv[3], "wb");
	if (!f || fwrite(out, 1, sizeof(out), f) != sizeof(out)) {
		fprintf(stderr, "post: cannot write %s\n", argv[3]);
		return 1;
	}
	fclose(f);
	return 0;
}
