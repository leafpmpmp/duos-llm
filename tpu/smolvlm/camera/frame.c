// SPDX-License-Identifier: MIT
// Turn one packed RAW12 BGGR frame into the encoder's 512x512 canvas: 2x2 bins,
// black level, grey-world white balance, gain, gamma, then the long side scaled
// to 512 and black padding below, as in the scored tests.
//
// Usage: frame RAW W H OUT.rgb
// OUT.rgb: 512x512 RGB, 8 bits. Prints the gains it used.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 512
#define PAD 0
#define BLACK 256.0f // 12-bit scale

static float clamp1(float v)
{
	return v < 0 ? 0 : v > 1 ? 1 : v;
}

int main(int argc, char **argv)
{
	if (argc != 5) {
		fprintf(stderr, "usage: frame RAW W H OUT.rgb\n");
		return 1;
	}
	int w = atoi(argv[2]), h = atoi(argv[3]);
	int stride = (w * 3 / 2 + 15) / 16 * 16;

	uint8_t *raw = malloc((size_t)stride * h);
	FILE *f = fopen(argv[1], "rb");

	if (!raw || !f || fread(raw, stride, h, f) != (size_t)h) {
		fprintf(stderr, "frame: cannot read %s\n", argv[1]);
		return 1;
	}
	fclose(f);

	// One RGB pixel per 2x2 Bayer group, linear, 0 to 1.
	int bw = w / 2, bh = h / 2;
	float *lin = malloc((size_t)bw * bh * 3 * sizeof(float));
	int hist[256] = { 0 }, total = bw * bh;
	double mean[3] = { 0 };

	for (int y = 0; y < bh; y++) {
		const uint8_t *r0 = raw + (size_t)y * 2 * stride, *r1 = r0 + stride;
		float *o = lin + (size_t)y * bw * 3;

		for (int x = 0; x < bw; x++, o += 3) {
			const uint8_t *a = r0 + x * 3, *b = r1 + x * 3;
			float blue = (a[0] << 4 | (a[2] & 15)) - BLACK;
			float g1 = (a[1] << 4 | a[2] >> 4) - BLACK;
			float g2 = (b[0] << 4 | (b[2] & 15)) - BLACK;
			float red = (b[1] << 4 | b[2] >> 4) - BLACK;

			o[0] = red / (4095 - BLACK);
			o[1] = (g1 + g2) / 2 / (4095 - BLACK);
			o[2] = blue / (4095 - BLACK);
			for (int c = 0; c < 3; c++)
				mean[c] += clamp1(o[c]);
			hist[(int)(255 * clamp1(o[1]))]++;
		}
	}

	// Grey world: scale red and blue to the mean of green.
	float gains[3];

	for (int c = 0; c < 3; c++) {
		gains[c] = (float)(mean[1] / (mean[c] > 1 ? mean[c] : 1));
		gains[c] = gains[c] < 0.5f ? 0.5f : gains[c] > 4 ? 4 : gains[c];
	}
	for (size_t i = 0; i < (size_t)bw * bh * 3; i++)
		lin[i] *= gains[i % 3];

	// Gain: put the 99.5th percentile of green near white, at most 16 times.
	int top = 255, seen = 0;

	while (top > 0 && seen + hist[top] < total / 200)
		seen += hist[top--];
	float gain = 0.95f * 255 / (top + 1);

	if (gain > 16)
		gain = 16;
	if (gain < 1)
		gain = 1;

	static uint8_t lut[4097];

	for (int i = 0; i <= 4096; i++)
		lut[i] = (uint8_t)lrintf(255 * powf(i / 4096.0f, 1 / 2.2f));

	uint8_t *rgb = malloc((size_t)bw * bh * 3);

	for (size_t i = 0; i < (size_t)bw * bh * 3; i++)
		rgb[i] = lut[(int)(4096 * clamp1(lin[i] * gain))];

	// Long side to 512 by box averaging, canvas filled below.
	static uint8_t out[SIZE * SIZE * 3];
	double s = (double)SIZE / (bw > bh ? bw : bh);
	int ow = (int)lrint(bw * s), oh = (int)lrint(bh * s);

	memset(out, PAD, sizeof(out));
	for (int y = 0; y < oh; y++) {
		int y0 = (int)(y / s), y1 = (int)((y + 1) / s);

		if (y1 <= y0)
			y1 = y0 + 1;
		if (y1 > bh)
			y1 = bh;
		for (int x = 0; x < ow; x++) {
			int x0 = (int)(x / s), x1 = (int)((x + 1) / s), sum[3] = { 0 }, n = 0;

			if (x1 <= x0)
				x1 = x0 + 1;
			if (x1 > bw)
				x1 = bw;
			for (int j = y0; j < y1; j++)
				for (int i = x0; i < x1; i++, n++)
					for (int c = 0; c < 3; c++)
						sum[c] += rgb[((size_t)j * bw + i) * 3 + c];
			for (int c = 0; c < 3; c++)
				out[(y * SIZE + x) * 3 + c] = (uint8_t)((sum[c] + n / 2) / n);
		}
	}
	f = fopen(argv[4], "wb");
	if (!f || fwrite(out, 1, sizeof(out), f) != sizeof(out)) {
		fprintf(stderr, "frame: cannot write %s\n", argv[4]);
		return 1;
	}
	fclose(f);
	printf("canvas %dx%d, white balance %.2f %.2f %.2f, gain %.2f\n", ow, oh, gains[0], gains[1], gains[2], gain);
	return 0;
}
