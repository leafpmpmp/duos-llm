// SPDX-License-Identifier: MIT
// Patch embedding on the CPU: a 512x512 canvas to the first block's input.
// Patches of padding give the same result, so they are computed once.
//
// Usage: pre WEIGHTS_DIR IN.rgb OUT.bin
// OUT.bin: float32 [768][1024], the layout of the encoder's first block.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 512
#define P 16
#define GRID (SIZE / P)
#define N (GRID * GRID)
#define D 768
#define IN (3 * P * P)

static float *load(const char *dir, const char *name, size_t n)
{
	char path[1024];
	float *x = malloc(n * sizeof(float));
	FILE *f;

	snprintf(path, sizeof(path), "%s/%s", dir, name);
	f = fopen(path, "rb");
	if (!x || !f || fread(x, sizeof(float), n, f) != n) {
		fprintf(stderr, "pre: cannot read %s\n", path);
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
	static uint8_t rgb[SIZE * SIZE * 3];
	static float out[D][N], in[IN], grey[D];
	FILE *f;

	if (argc != 4) {
		fprintf(stderr, "usage: pre WEIGHTS_DIR IN.rgb OUT.bin\n");
		return 1;
	}
	float *w = load(argv[1], "patch.f32", (size_t)D * IN);
	float *bias = load(argv[1], "patch_bias.f32", D);
	float *pos = load(argv[1], "pos.f32", (size_t)N * D);

	f = fopen(argv[2], "rb");
	if (!f || fread(rgb, 1, sizeof(rgb), f) != sizeof(rgb)) {
		fprintf(stderr, "pre: cannot read %s\n", argv[2]);
		return 1;
	}
	fclose(f);

	int have_grey = 0;

	for (int p = 0; p < N; p++) {
		int y0 = p / GRID * P, x0 = p % GRID * P, flat = 1;

		for (int c = 0; c < 3; c++)
			for (int y = 0; y < P; y++)
				for (int x = 0; x < P; x++) {
					int v = rgb[((y0 + y) * SIZE + x0 + x) * 3 + c];

					flat &= v == rgb[(size_t)(SIZE * SIZE - 1) * 3];
					in[(c * P + y) * P + x] = (v / 255.0f - 0.5f) / 0.5f;
				}
		if (flat && have_grey) {
			for (int d = 0; d < D; d++)
				out[d][p] = grey[d] + pos[(size_t)p * D + d];
			continue;
		}
		for (int d = 0; d < D; d++) {
			float v = dot(w + (size_t)d * IN, in, IN) + bias[d];

			if (flat)
				grey[d] = v;
			out[d][p] = v + pos[(size_t)p * D + d];
		}
		have_grey |= flat;
	}
	f = fopen(argv[3], "wb");
	if (!f || fwrite(out, 1, sizeof(out), f) != sizeof(out)) {
		fprintf(stderr, "pre: cannot write %s\n", argv[3]);
		return 1;
	}
	fclose(f);
	return 0;
}
