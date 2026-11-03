// SPDX-License-Identifier: MIT
// DuoSight in one resident process on the SG2000 TPU: camera frame to answer.
// Every model loads once, so a question pays no load time:
//   canvas (CPU) -> patch embedding (TPU) -> 12 SigLIP blocks (TPU)
//   -> final LayerNorm and connector (TPU) -> 64 image rows
//   -> SmolVLM decoder, 30 layers and head (TPU).
// The detector is a program of its own and shares the TPU.
//
// The decoder part comes from decoder/vlm.c (itself cut down from
// tpu/runtime/llm.c in leafpmpmp/duos-llm), the encoder part from
// encoder/chain.c, the canvas from camera/frame.c.
//
// Usage: duosight ENC_DIR DEC_DIR PREFIX.kv [MAX_NEW] < commands > replies
//   ENC_DIR: patch.cvimodel, l00..l11.cvimodel, post.cvimodel; "-" for no encoder (rows only)
//   DEC_DIR: l00..l29.cvimodel, head.cvimodel, embed.f16
// Commands, one per line:
//   raw PATH W H [OUT.rgb]   encode a packed RAW12 BGGR frame from the camera
//                            (and write its canvas, a check against camera/frame)
//   rgb PATH       encode a 512x512 RGB canvas
//   rows PATH      take 64x576 float32 image rows from a file instead
//   dump PATH      write the current image rows to a file (a check against the PC)
//   ask N ID...    answer a prompt about the last image; id 49190 marks the image rows
//   say TEXT       answer a question in words about the last image (DUOSIGHT_BPE)
//   bg raw PATH W H [OUT.rgb] | bg rgb PATH
//                  encode a frame in steps, answering questions about the last frame
//                  between them, then prefill the image part of the prompt, so that a
//                  question about this frame runs only its own tokens (and the last
//                  token of the image part); answers are the same as with rgb and ask
// Replies, one line each:
//   image CANVAS_MS PATCH_MS BLOCKS_MS POST_MS
//   bg CANVAS_MS PATCH_MS BLOCKS_MS POST_MS PREFILL_MS   (when the frame is ready)
//   dumped
//   ans PREFILL_MS DECODE_MS REUSED ID...   (REUSED: prompt tokens kept from the last prompt)
//   txt TEXT       after each ans, with DUOSIGHT_VOCAB set to a make_vocab.py table
//   err TEXT
// DUOSIGHT_STOP=ID,...: end an answer right after one of these ids, without the pass
// that would only produce EOS. With 30 ("."): 148 of the 150 scored answers end with it,
// and none has it inside.
// DUOSIGHT_BPE: make_bpe.py tables for say (bpe.h).
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cviruntime.h>

#include "bpe.h"

#define D 576
#define NKV 3
#define G 3
#define HD 64
#define LAYERS 30
#define THETA 100000.0
#define NEG -10000.0f
#define TMAX 16
#define MAXTOK 512
#define ROWS 64
#define IMAGE_ID 49190
#define AROUND_IMAGE 49189 // <fake_token_around_image>
#define EOS 49279

#define BLOCKS 12
#define SIZE 512
#define DV 768
#define NPATCH 1024
#define BLACK 256.0f // 12-bit scale

// A model tensor, or a CPU shadow buffer in the same format.
struct tens {
	CVI_TENSOR *t;
	CVI_FMT fmt;
	float qs;
	void *p;
	size_t n, bytes;
};

struct layer {
	CVI_MODEL_HANDLE m;
	CVI_TENSOR *in, *out;
	int32_t nin, nout;
	struct tens h, cq, sq, ck, sk, k, v, mask, dh, kn, vn;
	void *kbuf, *vbuf, *mbuf; // shadows of k_cache, v_cache, mask (resident inputs)
	struct {
		char *dst;
		const char *src;
		size_t n;
	} ovl[9]; // output bytes over resident inputs: restored from the shadows
	int novl, mvalid; // mvalid: cache slots unmasked in mask
	float kpend[NKV * TMAX * HD], vpend[NKV * TMAX * HD]; // k_new, v_new of the last pass
};

struct model {
	CVI_MODEL_HANDLE m;
	CVI_TENSOR *in, *out;
	int32_t nin, nout;
};

static int S, T, HT, VOCAB; // cache slots, tokens per call, head rows, vocabulary size
static struct layer lay[LAYERS], head;
static struct tens logits_t;
static _Float16 *embed;
static float image[ROWS][D]; // ids >= VOCAB are rows of this table
static struct model patch, enc[BLOCKS], post;

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void die(const char *msg, const char *arg)
{
	fprintf(stderr, "duosight: %s%s%s\n", msg, arg ? " " : "", arg ? arg : "");
	exit(1);
}

static long cma_free_kb(void)
{
	char line[128];
	long kb = -1;
	FILE *f = fopen("/proc/meminfo", "r");

	while (f && fgets(line, sizeof(line), f))
		if (sscanf(line, "CmaFree: %ld kB", &kb) == 1)
			break;
	if (f)
		fclose(f);
	return kb;
}

// Write back and drop the clean page cache, then compact memory (needs root; best effort).
// Page cache and scattered pages in CMA can make a contiguous allocation fail.
static void tidy_memory(void)
{
	const char *files[2] = { "/proc/sys/vm/drop_caches", "/proc/sys/vm/compact_memory" };

	sync();
	for (int i = 0; i < 2; i++) {
		int fd = open(files[i], O_WRONLY);

		if (fd >= 0) {
			if (write(fd, "1", 1) != 1)
				fprintf(stderr, "duosight: cannot write %s\n", files[i]);
			close(fd);
		}
	}
}

// CMA really allocated, in MiB, from debugfs (mounted on the board); -1 if unreadable.
// CmaFree also counts the movable pages the kernel parks in the pool.
static long cma_used_mib(void)
{
	long pages = -1;
	FILE *f = fopen("/sys/kernel/debug/cma/reserved/used", "r");

	if (f) {
		if (fscanf(f, "%ld", &pages) != 1)
			pages = -1;
		fclose(f);
	}
	return pages < 0 ? -1 : pages * 4096 / (1 << 20);
}

static void load_model(const char *path, CVI_MODEL_HANDLE *m, CVI_TENSOR **in, int32_t *nin, CVI_TENSOR **out, int32_t *nout)
{
	static int trace = -1, tidy = -1;
	int fd;

	if (trace < 0) {
		trace = getenv("DUOSIGHT_CMA") != NULL;
		tidy = getenv("DUOSIGHT_TIDY") != NULL;
	}
	if (tidy)
		tidy_memory();
	if (trace)
		fprintf(stderr, "CMA %ld MiB used, %ld MiB free, loading %s\n", cma_used_mib(), cma_free_kb() / 1024, path);
	if (CVI_NN_RegisterModel(path, m) != CVI_RC_SUCCESS)
		die("cannot load", path);
	// The file's page cache can sit in free CMA and block the next model's
	// contiguous buffers (EBUSY): drop it.
	fd = open(path, O_RDONLY);
	if (fd >= 0) {
		posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
		close(fd);
	}
	if (CVI_NN_GetInputOutputTensors(*m, in, nin, out, nout) != CVI_RC_SUCCESS)
		die("no tensors in", path);
}

static void run_model(struct model *m)
{
	if (CVI_NN_Forward(m->m, m->in, m->nin, m->out, m->nout) != CVI_RC_SUCCESS)
		die("forward failed", NULL);
}

// ---- Encoder ----

// All blocks share two I/O buffers: even blocks go A -> B, odd blocks B -> A, so block 11
// ends in A. Patch embedding writes A, the connector reads A.
// The runtime sizes the shared activation buffer by the models loaded so far, and models
// already loaded keep the old buffer when a larger one comes. So the block that needs the
// most loads first: going by file size, block 5, whose BF16 fc2 adds both weights and
// activations. On the board this saved 14 MiB. Its I/O buffers become A and B.
// Every model gets I/O buffers of its own when it loads. Each block gives its 6 MiB back
// at once, so the next model can use the gap; freed only at the end, they left gaps too
// small for anything loaded after them.
static void share_io(CVI_TENSOR *t, uint64_t paddr)
{
	if (CVI_NN_SetTensorPhysicalAddr(t, paddr) != CVI_RC_SUCCESS)
		die("cannot share the encoder buffers", NULL);
}

static void enc_init(const char *dir)
{
	char path[1024];
	size_t bytes = (size_t)DV * NPATCH * sizeof(float);
	off_t size[BLOCKS];
	int order[BLOCKS];
	struct stat st;
	uint64_t a = 0, b = 0;

	for (int i = 0; i < BLOCKS; i++) {
		snprintf(path, sizeof(path), "%s/l%02d.cvimodel", dir, i);
		size[i] = stat(path, &st) ? 0 : st.st_size;
		order[i] = i;
	}
	for (int i = 1; i < BLOCKS; i++)
		for (int j = i; j > 0 && size[order[j]] > size[order[j - 1]]; j--) {
			int t = order[j];

			order[j] = order[j - 1];
			order[j - 1] = t;
		}
	for (int k = 0; k < BLOCKS; k++) {
		int i = order[k];

		snprintf(path, sizeof(path), "%s/l%02d.cvimodel", dir, i);
		load_model(path, &enc[i].m, &enc[i].in, &enc[i].nin, &enc[i].out, &enc[i].nout);
		if (CVI_NN_TensorSize(&enc[i].in[0]) != bytes || CVI_NN_TensorSize(&enc[i].out[0]) != bytes)
			die("unexpected tensor size in", path);
		if (k == 0) {
			a = i & 1 ? enc[i].out[0].paddr : enc[i].in[0].paddr;
			b = i & 1 ? enc[i].in[0].paddr : enc[i].out[0].paddr;
		} else {
			share_io(&enc[i].in[0], i & 1 ? b : a);
			share_io(&enc[i].out[0], i & 1 ? a : b);
		}
	}
	snprintf(path, sizeof(path), "%s/patch.cvimodel", dir);
	load_model(path, &patch.m, &patch.in, &patch.nin, &patch.out, &patch.nout);
	if (CVI_NN_TensorSize(&patch.in[0]) != (size_t)3 * SIZE * SIZE * sizeof(float) ||
	    CVI_NN_TensorSize(&patch.out[0]) != bytes)
		die("unexpected tensor size in", path);
	share_io(&patch.out[0], a);
	snprintf(path, sizeof(path), "%s/post.cvimodel", dir);
	load_model(path, &post.m, &post.in, &post.nin, &post.out, &post.nout);
	if (CVI_NN_TensorSize(&post.in[0]) != bytes || CVI_NN_TensorSize(&post.out[0]) != (size_t)D * ROWS * sizeof(float))
		die("unexpected tensor size in", path);
	share_io(&post.in[0], (BLOCKS - 1) & 1 ? a : b);
}

static float clamp1(float v)
{
	return v < 0 ? 0 : v > 1 ? 1 : v;
}

// The 12-bit samples of one 2x2 Bayer group: red, the sum of the two greens, blue.
static void bayer(const uint8_t *raw, int stride, int x, int y, int *red, int *green, int *blue)
{
	const uint8_t *a = raw + (size_t)y * 2 * stride + x * 3, *b = a + stride;

	*blue = a[0] << 4 | (a[2] & 15);
	*green = (a[1] << 4 | a[2] >> 4) + (b[0] << 4 | (b[2] & 15));
	*red = b[1] << 4 | b[2] >> 4;
}

// Linear values, 0 to 1, as camera/frame.c computes them: one sample, or a green sum.
static float lin1(int v)
{
	return ((float)v - BLACK) / (4095 - BLACK);
}

static float lin2(int s)
{
	return ((float)s - 2 * BLACK) / 2 / (4095 - BLACK);
}

// camera/frame.c: one RGB pixel per 2x2 Bayer group, black level, grey-world white
// balance, gain, gamma, then the long side scaled to 512 and black padding below. The
// same arithmetic, so the same bytes, but through tables of the 4096 sample values and
// 8191 green sums, and with the raw file mapped and read twice instead of kept as
// floats: the CPU is slow and memory is what limits the board.
static int canvas(const char *file, int w, int h, uint8_t *out)
{
	static uint8_t *rgb, lut[4097], to_r[4096], to_b[4096], to_g[8191];
	static float c1[4096], c2[8191];
	static int h2[8191];
	static size_t have;
	int stride = (w * 3 / 2 + 15) / 16 * 16, bw = w / 2, bh = h / 2;
	size_t need = (size_t)stride * h;
	struct stat st;
	int fd = open(file, O_RDONLY);

	if (fd < 0)
		return -1;
	if (fstat(fd, &st) || st.st_size < (off_t)need) {
		close(fd);
		return -1;
	}
	const uint8_t *raw = mmap(NULL, need, PROT_READ, MAP_PRIVATE, fd, 0);

	close(fd);
	if (raw == MAP_FAILED)
		return -1;
	if ((size_t)bw * bh * 3 > have) {
		free(rgb);
		rgb = malloc((size_t)bw * bh * 3);
		if (!rgb)
			die("out of memory", NULL);
		have = (size_t)bw * bh * 3;
	}
	if (!lut[4096]) {
		for (int i = 0; i <= 4096; i++)
			lut[i] = (uint8_t)lrintf(255 * powf(i / 4096.0f, 1 / 2.2f));
		for (int v = 0; v < 4096; v++)
			c1[v] = clamp1(lin1(v));
		for (int s = 0; s < 8191; s++) {
			c2[s] = clamp1(lin2(s));
			h2[s] = (int)(255 * c2[s]);
		}
	}

	int hist[256] = { 0 }, total = bw * bh, r, g, bl;
	double mean[3] = { 0 };

	for (int y = 0; y < bh; y++)
		for (int x = 0; x < bw; x++) {
			bayer(raw, stride, x, y, &r, &g, &bl);
			mean[0] += c1[r];
			mean[1] += c2[g];
			mean[2] += c1[bl];
			hist[h2[g]]++;
		}
	float gains[3];

	for (int c = 0; c < 3; c++) {
		gains[c] = (float)(mean[1] / (mean[c] > 1 ? mean[c] : 1));
		gains[c] = gains[c] < 0.5f ? 0.5f : gains[c] > 4 ? 4 : gains[c];
	}
	int top = 255, seen = 0;

	while (top > 0 && seen + hist[top] < total / 200)
		seen += hist[top--];
	float gain = 0.95f * 255 / (top + 1);

	gain = gain > 16 ? 16 : gain < 1 ? 1 : gain;
	for (int v = 0; v < 4096; v++) {
		to_r[v] = lut[(int)(4096 * clamp1(lin1(v) * gains[0] * gain))];
		to_b[v] = lut[(int)(4096 * clamp1(lin1(v) * gains[2] * gain))];
	}
	for (int s = 0; s < 8191; s++)
		to_g[s] = lut[(int)(4096 * clamp1(lin2(s) * gains[1] * gain))];
	for (int y = 0; y < bh; y++)
		for (int x = 0; x < bw; x++) {
			uint8_t *p = rgb + ((size_t)y * bw + x) * 3;

			bayer(raw, stride, x, y, &r, &g, &bl);
			p[0] = to_r[r];
			p[1] = to_g[g];
			p[2] = to_b[bl];
		}
	munmap((void *)raw, need);

	double s = (double)SIZE / (bw > bh ? bw : bh);
	int ow = (int)lrint(bw * s), oh = (int)lrint(bh * s);

	memset(out, 0, (size_t)SIZE * SIZE * 3);
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
	return 0;
}

// Step s of a canvas to image rows: 0 patch embedding, 1..BLOCKS the blocks, BLOCKS + 1
// the connector, which writes rows. Between steps the state is in the shared I/O buffers
// only, so the decoder can run in between.
static void encode_step(int s, const uint8_t *rgb, float (*rows)[D])
{
	if (s == 0) {
		float *x = CVI_NN_TensorPtr(&patch.in[0]);

		for (int c = 0; c < 3; c++)
			for (int i = 0; i < SIZE * SIZE; i++)
				x[c * SIZE * SIZE + i] = rgb[i * 3 + c];
		run_model(&patch);
	} else if (s <= BLOCKS) {
		run_model(&enc[s - 1]);
	} else {
		run_model(&post);
		const float *y = CVI_NN_TensorPtr(&post.out[0]); // [576][64]: column r is image row r

		for (int r = 0; r < ROWS; r++)
			for (int i = 0; i < D; i++)
				rows[r][i] = y[i * ROWS + r];
	}
}

// Canvas to image rows. ms: patch, blocks, connector.
static void encode(const uint8_t *rgb, double *ms)
{
	double t = now();

	encode_step(0, rgb, image);
	ms[0] = 1e3 * (now() - t);
	t = now();
	for (int s = 1; s <= BLOCKS; s++)
		encode_step(s, rgb, image);
	ms[1] = 1e3 * (now() - t);
	t = now();
	encode_step(BLOCKS + 1, rgb, image);
	ms[2] = 1e3 * (now() - t);
}

// ---- Decoder (decoder/vlm.c) ----

static struct tens find(CVI_TENSOR *ts, int32_t n, const char *name, int exact)
{
	for (int32_t i = 0; i < n; i++) {
		const char *tn = CVI_NN_TensorName(&ts[i]);

		if (exact ? !strcmp(tn, name) : !strncmp(tn, name, strlen(name))) {
			struct tens t = {
				.t = &ts[i], .fmt = ts[i].fmt, .qs = CVI_NN_TensorQuantScale(&ts[i]),
				.p = CVI_NN_TensorPtr(&ts[i]), .n = CVI_NN_TensorCount(&ts[i]),
				.bytes = CVI_NN_TensorSize(&ts[i]),
			};
			if (t.fmt != CVI_FMT_INT8 && t.fmt != CVI_FMT_FP32 && t.fmt != CVI_FMT_BF16)
				die("unsupported tensor format:", tn);
			return t;
		}
	}
	die("tensor not found:", name);
	return (struct tens){ 0 };
}

// Write n floats at element offset off of buffer p in t's format.
static void put(const struct tens *t, void *p, size_t off, const float *x, size_t n)
{
	if (t->fmt == CVI_FMT_FP32) {
		memcpy((float *)p + off, x, n * sizeof(float));
	} else if (t->fmt == CVI_FMT_INT8) {
		int8_t *q = (int8_t *)p + off;

		for (size_t i = 0; i < n; i++) {
			float v = x[i] * t->qs;
			v = v >= 0 ? floorf(v + 0.5f) : ceilf(v - 0.5f); // TPU-MLIR rounding
			q[i] = v > 127 ? 127 : v < -128 ? -128 : (int8_t)v;
		}
	} else {
		uint16_t *b = (uint16_t *)p + off;

		for (size_t i = 0; i < n; i++) {
			uint32_t u;

			memcpy(&u, &x[i], 4);
			b[i] = (u + 0x7fff + ((u >> 16) & 1)) >> 16;
		}
	}
}

// Read n floats at element offset off of buffer p (t's format).
static void get(const struct tens *t, const void *p, size_t off, float *x, size_t n)
{
	if (t->fmt == CVI_FMT_FP32) {
		memcpy(x, (const float *)p + off, n * sizeof(float));
	} else if (t->fmt == CVI_FMT_INT8) {
		const int8_t *q = (const int8_t *)p + off;

		for (size_t i = 0; i < n; i++)
			x[i] = q[i] * t->qs;
	} else {
		const uint16_t *b = (const uint16_t *)p + off;

		for (size_t i = 0; i < n; i++) {
			uint32_t u = (uint32_t)b[i] << 16;

			memcpy(&x[i], &u, 4);
		}
	}
}

// Resident inputs are marked as device memory, so Forward neither copies nor
// flushes them: the runtime writes CPU cache lines back itself.
static uintptr_t cache_line(void)
{
	static uintptr_t line;
	uint64_t ctr;

	if (!line) {
		__asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
		line = (uintptr_t)4 << ((ctr >> 16) & 15);
	}
	return line;
}

// Write the cache lines of [p, p + n) back to DRAM (forward() waits for it).
static void flush(const void *p, size_t n)
{
	uintptr_t a = (uintptr_t)p & ~(cache_line() - 1), e = (uintptr_t)p + n;

	for (; a < e; a += cache_line())
		__asm__ volatile("dc cvac, %0" : : "r"(a) : "memory");
}

// Copy elements [off, off + n) of resident input t from its shadow to the model buffer.
static void sync_res(const struct tens *t, const void *shadow, size_t off, size_t n)
{
	size_t es = t->bytes / t->n;

	memcpy((char *)t->p + off * es, (const char *)shadow + off * es, n * es);
	flush((char *)t->p + off * es, n * es);
}

// Write n floats at element offset off of resident input t (shadow and model buffer).
static void put_res(const struct tens *t, void *shadow, size_t off, const float *x, size_t n)
{
	put(t, shadow, off, x, n);
	sync_res(t, shadow, off, n);
}

static void load(struct layer *l, const char *path)
{
	load_model(path, &l->m, &l->in, &l->nin, &l->out, &l->nout);
}

// Make k_cache, v_cache and mask resident: empty cache, all cache slots
// masked, causal mask over the T new tokens. Mask rows: (kv, token, group);
// columns: cache slots, then the new tokens.
static void resident(struct layer *l)
{
	static float *m;
	size_t mrow = S + T;
	struct tens *res[3] = { &l->k, &l->v, &l->mask };
	void **sh[3] = { &l->kbuf, &l->vbuf, &l->mbuf };

	if (!m) {
		m = malloc(l->mask.n * sizeof(float));
		for (size_t r = 0; r < (size_t)NKV * T * G; r++)
			for (size_t c = 0; c < mrow; c++)
				m[r * mrow + c] = c >= (size_t)S && c - S <= r / G % T ? 0 : NEG;
	}
	for (int j = 0; j < 3; j++) {
		*sh[j] = calloc(1, res[j]->bytes);
		res[j]->t->mem_type = CVI_MEM_DEVICE;
	}
	sync_res(&l->k, l->kbuf, 0, l->k.n);
	sync_res(&l->v, l->vbuf, 0, l->v.n);
	put_res(&l->mask, l->mbuf, 0, m, l->mask.n);
	l->mvalid = 0;
	for (int32_t i = 0; i < l->nout; i++) {
		char *o = CVI_NN_TensorPtr(&l->out[i]), *oe = o + CVI_NN_TensorSize(&l->out[i]);

		for (int j = 0; j < 3; j++) {
			char *r = res[j]->p, *re = r + res[j]->bytes;
			char *a = o > r ? o : r, *b = oe < re ? oe : re;

			if (a >= b)
				continue;
			if (l->novl == 9)
				die("too many outputs over resident inputs", NULL);
			l->ovl[l->novl].dst = a;
			l->ovl[l->novl].src = (char *)*sh[j] + (a - r);
			l->ovl[l->novl++].n = b - a;
		}
	}
}

static void dec_init_head(const char *dir)
{
	char path[1024];

	snprintf(path, sizeof(path), "%s/head.cvimodel", dir);
	load(&head, path);
	head.h = find(head.in, head.nin, "h", 1);
	logits_t = find(head.out, head.nout, "logits", 0);
	HT = (int)(head.h.n / D);
	if (HT < 1 || HT > TMAX)
		die("unexpected head shapes in", path);
	VOCAB = (int)(logits_t.n / HT);
}

static void dec_init_layers(const char *dir)
{
	char path[1024];
	struct stat st;
	int fd;

	for (int i = 0; i < LAYERS; i++) {
		struct layer *l = &lay[i];

		snprintf(path, sizeof(path), "%s/l%02d.cvimodel", dir, i);
		load(l, path);
		l->h = find(l->in, l->nin, "h", 1);
		l->cq = find(l->in, l->nin, "cos_q", 1);
		l->sq = find(l->in, l->nin, "sin_q", 1);
		l->ck = find(l->in, l->nin, "cos_k", 1);
		l->sk = find(l->in, l->nin, "sin_k", 1);
		l->k = find(l->in, l->nin, "k_cache", 1);
		l->v = find(l->in, l->nin, "v_cache", 1);
		l->mask = find(l->in, l->nin, "mask", 1);
		l->dh = find(l->out, l->nout, "dh", 0);
		l->kn = find(l->out, l->nout, "k_new", 0);
		l->vn = find(l->out, l->nout, "v_new", 0);
		int s = (int)(l->k.n / (NKV * HD)), t = (int)(l->h.n / D);
		if (i && (s != S || t != T))
			die("cache size or tokens per call differ in", path);
		S = s;
		T = t;
		if (T < 1 || T > TMAX || (HT != 1 && HT != T) || l->mask.n != (size_t)NKV * T * G * (S + T) ||
		    l->kn.n != (size_t)NKV * T * HD)
			die("layer shapes do not match in", path);
	}
	for (int i = 0; i < LAYERS; i++) // CPU memory after all CMA buffers
		resident(&lay[i]);

	snprintf(path, sizeof(path), "%s/embed.f16", dir);
	fd = open(path, O_RDONLY);
	if (fd < 0 || fstat(fd, &st) || st.st_size != (off_t)VOCAB * D * 2)
		die("bad embedding table", path);
	embed = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (embed == MAP_FAILED)
		die("cannot map", path);
}

static void forward(struct layer *l)
{
	__asm__ volatile("dsb sy" : : : "memory"); // flush() done before the TPU reads
	if (CVI_NN_Forward(l->m, l->in, l->nin, l->out, l->nout) != CVI_RC_SUCCESS)
		die("forward failed", NULL);
}

// Per-pass input tables in float, converted once per (format, scale) and
// copied into each layer.
struct conv {
	CVI_FMT fmt;
	float qs;
	void *buf;
};

static void put_cached(struct conv *c, int nc, const struct tens *t, const float *x)
{
	for (int i = 0; i < nc; i++) {
		if (!c[i].buf) {
			c[i] = (struct conv){ t->fmt, t->qs, malloc(t->bytes) };
			put(t, c[i].buf, 0, x, t->n);
		}
		if (c[i].fmt == t->fmt && c[i].qs == t->qs) {
			memcpy(t->p, c[i].buf, t->bytes);
			return;
		}
	}
	put(t, t->p, 0, x, t->n); // more formats than cache slots
}

static void conv_reset(struct conv *c, int nc)
{
	for (int i = 0; i < nc; i++) {
		free(c[i].buf);
		c[i].buf = NULL;
	}
}

// Unmask cache slots 0..n-1 of l's mask (and mask the slots from n on).
static void set_valid(struct layer *l, int n)
{
	static float *val; // S zeros, then S x NEG
	int a = n < l->mvalid ? n : l->mvalid, b = n < l->mvalid ? l->mvalid : n;

	if (!val) {
		val = malloc(2 * S * sizeof(float));
		for (int i = 0; i < 2 * S; i++)
			val[i] = i < S ? 0 : NEG;
	}
	for (size_t r = 0; a < b && r < (size_t)NKV * T * G; r++)
		put_res(&l->mask, l->mbuf, r * (S + T) + a, val + (n > l->mvalid ? 0 : S), b - a);
	l->mvalid = n;
}

// Run n <= T tokens at positions pos..pos+n-1 over cache slots 0..pos-1. Rows
// n..T-1 are padding. The head runs only when logits are wanted, and then on the
// last token: logits gets VOCAB floats. The new K/V rows wait in kpend/vpend
// until commit().
static void pass(const int *tok, int n, int pos, float *logits)
{
	static float h[TMAX * D], dh[TMAX * D], cs[TMAX][HD], sn[TMAX][HD];
	static float cq[NKV * TMAX * G * HD], sq[NKV * TMAX * G * HD], ck[NKV * TMAX * HD], sk[NKV * TMAX * HD];
	struct conv ccq[4] = { 0 }, csq[4] = { 0 }, cck[4] = { 0 }, csk[4] = { 0 };

	if (n < 1 || n > T || pos + n > S)
		die("context full", NULL);
	memset(h, 0, sizeof(h));
	for (int t = 0; t < n; t++)
		for (int i = 0; i < D; i++)
			h[t * D + i] = tok[t] >= VOCAB ? image[tok[t] - VOCAB][i] : embed[(size_t)tok[t] * D + i];
	for (int t = 0; t < T; t++)
		for (int j = 0; j < HD / 2; j++) {
			double a = (pos + t) * pow(THETA, -2.0 * j / HD);

			cs[t][j] = cs[t][j + HD / 2] = (float)cos(a);
			sn[t][j] = sn[t][j + HD / 2] = (float)sin(a);
		}
	for (int kv = 0; kv < NKV; kv++)
		for (int t = 0; t < T; t++) {
			memcpy(&ck[(kv * T + t) * HD], cs[t], sizeof(cs[t]));
			memcpy(&sk[(kv * T + t) * HD], sn[t], sizeof(sn[t]));
			for (int j = 0; j < G; j++) {
				memcpy(&cq[((kv * T + t) * G + j) * HD], cs[t], sizeof(cs[t]));
				memcpy(&sq[((kv * T + t) * G + j) * HD], sn[t], sizeof(sn[t]));
			}
		}

	for (int i = 0; i < LAYERS; i++) {
		struct layer *l = &lay[i];

		put(&l->h, l->h.p, 0, h, (size_t)T * D);
		put_cached(ccq, 4, &l->cq, cq);
		put_cached(csq, 4, &l->sq, sq);
		put_cached(cck, 4, &l->ck, ck);
		put_cached(csk, 4, &l->sk, sk);
		set_valid(l, pos);
		forward(l);
		get(&l->dh, l->dh.p, 0, dh, (size_t)T * D);
		for (int j = 0; j < T * D; j++)
			h[j] += dh[j];
		get(&l->kn, l->kn.p, 0, l->kpend, (size_t)NKV * T * HD);
		get(&l->vn, l->vn.p, 0, l->vpend, (size_t)NKV * T * HD);
		for (int j = 0; j < l->novl; j++) {
			memcpy(l->ovl[j].dst, l->ovl[j].src, l->ovl[j].n);
			flush(l->ovl[j].dst, l->ovl[j].n);
		}
	}
	conv_reset(ccq, 4);
	conv_reset(csq, 4);
	conv_reset(cck, 4);
	conv_reset(csk, 4);
	if (logits) {
		if (HT == 1) {
			put(&head.h, head.h.p, 0, h + (size_t)(n - 1) * D, D);
			forward(&head);
			get(&logits_t, logits_t.p, 0, logits, VOCAB);
		} else {
			put(&head.h, head.h.p, 0, h, (size_t)T * D);
			forward(&head);
			get(&logits_t, logits_t.p, (size_t)(n - 1) * VOCAB, logits, VOCAB);
		}
	}
}

// Keep the K/V of the first m tokens of the last pass in slots slot...
static void commit(int slot, int m)
{
	for (int i = 0; i < LAYERS; i++) {
		struct layer *l = &lay[i];

		for (int kv = 0; kv < NKV; kv++) {
			size_t dst = ((size_t)kv * S + slot) * HD, src = (size_t)kv * T * HD;

			put_res(&l->k, l->kbuf, dst, l->kpend + src, (size_t)m * HD);
			put_res(&l->v, l->vbuf, dst, l->vpend + src, (size_t)m * HD);
		}
	}
}

// Empty slots from..to-1 of every layer. A masked slot still gets a tiny weight
// on the TPU, so K/V left by the last prompt would leak into the next one.
static void clear(int from, int to)
{
	for (int i = 0; i < LAYERS; i++) {
		struct layer *l = &lay[i];
		const struct tens *t[2] = { &l->k, &l->v };
		char *sh[2] = { l->kbuf, l->vbuf };

		for (int j = 0; j < 2; j++)
			for (int kv = 0; kv < NKV; kv++) {
				size_t es = t[j]->bytes / t[j]->n, off = ((size_t)kv * S + from) * HD;
				size_t len = (size_t)(to - from) * HD;

				memset(sh[j] + off * es, 0, len * es);
				sync_res(t[j], sh[j], off, len);
			}
	}
}

// Fixed prompt prefix with float K/V (make_prefix.py): slots 0..P-1.
// File: int32 P, int32 ids[P], float32 k, v [layers][nkv][P][hd].
static int prefix_n;
static int32_t prefix_ids[MAXTOK];

static void load_prefix(const char *file)
{
	FILE *f = fopen(file, "rb");
	int32_t p;

	if (!f || fread(&p, 4, 1, f) != 1 || p < 1 || p >= S || p > MAXTOK || fread(prefix_ids, 4, p, f) != (size_t)p)
		die("bad prefix file", file);
	size_t n = (size_t)LAYERS * NKV * p * HD;
	float *k = malloc(n * sizeof(float)), *v = malloc(n * sizeof(float));
	if (fread(k, sizeof(float), n, f) != n || fread(v, sizeof(float), n, f) != n)
		die("short prefix file", file);
	fclose(f);
	for (int i = 0; i < LAYERS; i++)
		for (int kv = 0; kv < NKV; kv++)
			for (int s = 0; s < p; s++) {
				size_t src = (((size_t)i * NKV + kv) * p + s) * HD, dst = ((size_t)kv * S + s) * HD;

				put_res(&lay[i].k, lay[i].kbuf, dst, k + src, HD);
				put_res(&lay[i].v, lay[i].vbuf, dst, v + src, HD);
			}
	free(k);
	free(v);
	prefix_n = p;
}

// Run tokens from..n-1 in passes of T; logits of the last token into x.
static void prefill(const int32_t *tok, int from, int n, float *x)
{
	int buf[TMAX];

	for (int i = from; i < n; i += T) {
		int m = n - i < T ? n - i : T;

		for (int t = 0; t < m; t++)
			buf[t] = tok[i + t];
		pass(buf, m, i, i + m == n ? x : NULL);
		commit(i, m);
	}
}

// Answer ids to text (make_vocab.py): the bytes of each id, then trimmed as tok.py does.
static char *words;
static uint32_t *word_at, words_n;

static void load_words(const char *file)
{
	FILE *f = fopen(file, "rb");
	int32_t n;
	struct stat st;

	if (!f || fstat(fileno(f), &st) || fread(&n, 4, 1, f) != 1 || n < 1)
		die("bad vocabulary file", file);
	words = malloc(st.st_size);
	word_at = malloc((n + 1) * sizeof(*word_at));
	uint32_t at = 0;
	int c;

	for (int32_t i = 0; i < n; i++) {
		word_at[i] = at;
		if ((c = fgetc(f)) == EOF || fread(words + at, 1, c, f) != (size_t)c)
			die("short vocabulary file", file);
		at += c;
	}
	word_at[n] = at;
	words_n = n;
	fclose(f);
}

static void print_text(const int *ids, int n)
{
	static char text[MAXTOK * 256];
	size_t len = 0, a = 0;

	for (int i = 0; i < n; i++)
		if ((uint32_t)ids[i] < words_n) {
			memcpy(text + len, words + word_at[ids[i]], word_at[ids[i] + 1] - word_at[ids[i]]);
			len += word_at[ids[i] + 1] - word_at[ids[i]];
		}
	while (a < len && (text[a] == ' ' || text[a] == '\n' || text[a] == '\t' || text[a] == '\r'))
		a++;
	while (len > a && (text[len - 1] == ' ' || text[len - 1] == '\n' || text[len - 1] == '\t' || text[len - 1] == '\r'))
		len--;
	for (size_t i = a; i < len; i++)
		if (text[i] == '\n')
			text[i] = ' ';
	printf("txt %.*s\n", (int)(len - a), text + a);
}

static int argmax(const float *x)
{
	int best = 0;

	for (int i = 1; i < VOCAB; i++)
		if (x[i] > x[best])
			best = i;
	return best;
}

// ---- Commands ----

static uint8_t rgb[SIZE * SIZE * 3];
static int32_t run[MAXTOK], last[MAXTOK]; // this prompt, and the one whose K/V is in the cache
static int has_enc, have_image, last_n, used, max_new, stop[16], nstop;
static float *logits;

// Commands come through a buffer of our own, not stdio: between the steps of a frame the
// loop must see whether a line waits, without blocking. Returns 1 with a line, 0 when none
// waits (wait 0), -1 at the end of input.
static int read_line(char *line, size_t size, int wait)
{
	static char buf[1 << 16];
	static size_t len;
	static int eof;

	for (;;) {
		char *nl = memchr(buf, '\n', len);

		if (nl || (len && (eof || len == sizeof(buf)))) { // a line, or the rest of the input
			size_t n = nl ? (size_t)(nl - buf) : len, m = n < size - 1 ? n : size - 1;

			memcpy(line, buf, m);
			line[m] = 0;
			n += nl != NULL;
			memmove(buf, buf + n, len - n);
			len -= n;
			return 1;
		}
		if (eof)
			return -1;
		struct pollfd p = { .fd = 0, .events = POLLIN };

		if (!wait && poll(&p, 1, 0) <= 0)
			return 0;
		ssize_t r = read(0, buf + len, sizeof(buf) - len);

		if (r > 0)
			len += r;
		else if (r == 0 || errno != EINTR)
			eof = 1;
	}
}

// A canvas from a packed raw camera frame (also written to save, if given) or an RGB file.
static int read_canvas(int raw, const char *path, int w, int h, const char *save)
{
	FILE *f = NULL;
	int ok;

	if (raw) {
		if (canvas(path, w, h, rgb))
			return -1;
		if (save && (!(f = fopen(save, "wb")) || fwrite(rgb, sizeof(rgb), 1, f) != 1))
			fprintf(stderr, "duosight: cannot write %s\n", save);
		if (f)
			fclose(f);
		return 0;
	}
	f = fopen(path, "rb");
	ok = f && fread(rgb, sizeof(rgb), 1, f) == 1;
	if (f)
		fclose(f);
	return ok ? 0 : -1;
}

// New image rows: the last prompt's K/V stays valid only before its first image row.
static void new_image(void)
{
	for (int i = 0; i < last_n; i++)
		if (last[i] >= VOCAB) {
			last_n = i;
			break;
		}
	have_image = 1;
}

// The prompt up to the end of the image rows: the prefix, the rows, <fake_token_around_image>.
static int image_part(void)
{
	int n = 0;

	for (; n < prefix_n; n++)
		run[n] = prefix_ids[n];
	for (int r = 0; r < ROWS; r++)
		run[n++] = VOCAB + r;
	run[n++] = AROUND_IMAGE;
	return n;
}

// The last start of a pass, in a prefill from the prefix, at or before token n.
static int pass_start(int n)
{
	return prefix_n + (n - prefix_n) / T * T;
}

// Answer run[0..n-1]. The K/V of the tokens this prompt shares with the last one stays:
// the prefix, the image rows and the text around them, when the image is the same. It stays
// only up to the start of a pass, so the passes are the ones of a prefill from the prefix
// and the answer the same bytes: the INT8 decoder answers differently when a token's K/V
// comes from the cache instead of the same pass ("SITTING." against "Rain." once).
static void answer(int n)
{
	static int out[MAXTOK];
	int keep = prefix_n;

	while (keep < n && keep < last_n && run[keep] == last[keep])
		keep++;
	if (keep == n)
		keep = n - 1; // the last token's logits are needed again
	keep = pass_start(keep);
	double t0 = now();

	if (used > keep)
		clear(keep, used);
	prefill(run, keep, n, logits);
	double t_prefill = now() - t0;
	int gen = 0, pos = n, tk = argmax(logits), limit = max_new < S - n ? max_new : S - n;

	t0 = now();
	while (gen < limit && tk != EOS) {
		int in[TMAX];

		out[gen++] = tk;
		int end = gen == limit;

		for (int j = 0; j < nstop; j++)
			end |= tk == stop[j];
		if (end)
			break;
		in[0] = tk;
		pass(in, 1, pos, logits);
		commit(pos, 1);
		pos++;
		tk = argmax(logits);
	}
	used = pos;
	memcpy(last, run, n * sizeof(run[0]));
	last_n = n;
	printf("ans %.0f %.0f %d", 1e3 * t_prefill, 1e3 * (now() - t0), keep);
	for (int i = 0; i < gen; i++)
		printf(" %d", out[i]);
	printf("\n");
	if (words)
		print_text(out, gen);
}

// Replace the last frame's K/V with this frame's: the passes of the image part that end
// before its end (the prefix and the 64 rows, 4 passes), so that a question starts a pass.
static double prefill_image(void)
{
	int n = pass_start(image_part());
	double t = now();

	if (used > prefix_n)
		clear(prefix_n, used);
	prefill(run, prefix_n, n, NULL);
	used = n;
	memcpy(last, run, n * sizeof(run[0]));
	last_n = n;
	return 1e3 * (now() - t);
}

// bg: a frame in steps. Step 0 the canvas, 1..BLOCKS + 2 encode_step, then the prefill
// that replaces the last frame's K/V. Questions between the steps get the last frame.
static struct {
	int on, step, raw, w, h;
	char path[512], save[512];
	double ms[5];
} job;
static float next_rows[ROWS][D];

static void job_step(void)
{
	double t = now();
	int s = job.step++;

	if (s == 0) {
		if (read_canvas(job.raw, job.path, job.w, job.h, job.save[0] ? job.save : NULL)) {
			printf("err cannot read %s\n", job.path);
			job.on = 0;
			return;
		}
		job.ms[0] = 1e3 * (now() - t);
	} else if (s <= BLOCKS + 2) {
		encode_step(s - 1, rgb, next_rows);
		job.ms[s == 1 ? 1 : s <= BLOCKS + 1 ? 2 : 3] += 1e3 * (now() - t);
	} else {
		memcpy(image, next_rows, sizeof(image));
		have_image = 1;
		job.ms[4] = prefill_image();
		printf("bg %.0f %.0f %.0f %.0f %.0f\n", job.ms[0], job.ms[1], job.ms[2], job.ms[3], job.ms[4]);
		job.on = 0;
	}
}

static void command(char *line)
{
	if (!strncmp(line, "say ", 4)) {
		int n = image_part(), m;

		if (!bpe_merges)
			printf("err no tokenizer: set DUOSIGHT_BPE\n");
		else if (!have_image)
			printf("err no image yet\n");
		else if ((m = bpe_question(line + 4, run + n, MAXTOK - n)) < 0 || n + m >= S)
			printf("err question too long\n");
		else
			answer(n + m);
		return;
	}
	char *cmd = strtok(line, " \n"), *arg = strtok(NULL, " \n");

	if (!cmd)
		return;
	if (!strcmp(cmd, "dump")) {
		FILE *f = arg ? fopen(arg, "wb") : NULL;

		if (!f || fwrite(image, sizeof(image), 1, f) != 1)
			printf("err cannot write %s\n", arg ? arg : "");
		else
			printf("dumped\n");
		if (f)
			fclose(f);
		return;
	}
	if (!strcmp(cmd, "bg")) {
		char *path = strtok(NULL, " \n");
		int raw = arg && !strcmp(arg, "raw");
		char *w = raw ? strtok(NULL, " \n") : NULL, *h = raw ? strtok(NULL, " \n") : NULL;
		char *save = raw ? strtok(NULL, " \n") : NULL;

		if (job.on)
			printf("err busy\n");
		else if (!has_enc)
			printf("err no encoder loaded\n");
		else if (!path || (raw ? !w || !h : strcmp(arg, "rgb")))
			printf("err usage: bg raw PATH W H [OUT.rgb] | bg rgb PATH\n");
		else {
			memset(&job, 0, sizeof(job));
			job.on = 1;
			job.raw = raw;
			job.w = raw ? atoi(w) : 0;
			job.h = raw ? atoi(h) : 0;
			snprintf(job.path, sizeof(job.path), "%s", path);
			snprintf(job.save, sizeof(job.save), "%s", save ? save : "");
		}
		return;
	}
	if (!strcmp(cmd, "raw") || !strcmp(cmd, "rgb") || !strcmp(cmd, "rows")) {
		double ms[4] = { 0 }, t0 = now();

		if (!arg) {
			printf("err no file\n");
			return;
		}
		if (job.on) {
			printf("err busy\n");
			return;
		}
		if (!strcmp(cmd, "rows")) {
			FILE *f = fopen(arg, "rb");
			int ok = f && fread(image, sizeof(image), 1, f) == 1;

			if (f)
				fclose(f);
			if (!ok) {
				printf("err cannot read %s\n", arg);
				return;
			}
		} else if (!has_enc) {
			printf("err no encoder loaded\n");
			return;
		} else {
			int raw = !strcmp(cmd, "raw");
			char *w = raw ? strtok(NULL, " \n") : NULL, *h = raw ? strtok(NULL, " \n") : NULL;
			char *save = raw ? strtok(NULL, " \n") : NULL;

			if ((raw && (!w || !h)) || read_canvas(raw, arg, w ? atoi(w) : 0, h ? atoi(h) : 0, save)) {
				printf("err cannot read %s\n", arg);
				return;
			}
			ms[0] = 1e3 * (now() - t0);
			encode(rgb, ms + 1);
		}
		new_image();
		printf("image %.0f %.0f %.0f %.0f\n", ms[0], ms[1], ms[2], ms[3]);
		return;
	}
	if (strcmp(cmd, "ask")) {
		printf("err unknown command %s\n", cmd);
		return;
	}
	int n = arg ? atoi(arg) : 0, rows = 0, ok = n > prefix_n && n < S && n <= MAXTOK;

	for (int i = 0; ok && i < n; i++) {
		char *s = strtok(NULL, " \n");

		if (!s) {
			ok = 0;
			break;
		}
		int id = atoi(s);

		ok = i >= prefix_n || id == prefix_ids[i];
		run[i] = id == IMAGE_ID ? VOCAB + rows++ : id;
	}
	if (!ok || rows > ROWS)
		printf("err bad prompt\n");
	else
		answer(n);
}

int main(int argc, char **argv)
{
	static char line[16384];
	double t0;

	if (argc < 4 || argc > 5)
		die("usage: duosight ENC_DIR DEC_DIR PREFIX.kv [MAX_NEW] < commands > replies", NULL);
	max_new = argc > 4 ? atoi(argv[4]) : 24;
	setvbuf(stdout, NULL, _IOLBF, 0);

	// Page cache in CMA that the kernel cannot migrate makes contiguous
	// allocations fail with EBUSY: write back and drop it first (needs root).
	sync();
	int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
	if (fd >= 0) {
		if (write(fd, "1", 1) != 1)
			fprintf(stderr, "duosight: cannot drop the page cache\n");
		close(fd);
	}
	long cma0 = cma_free_kb();

	has_enc = strcmp(argv[1], "-") != 0;

	// The head first, while its 27 MB weight block still finds contiguous CMA, then the
	// small decoder layers, packed together, and the encoder last (see enc_init). The
	// decoder's shared buffers are under 1 MB, so the encoder's needs a new one anyway.
	t0 = now();
	dec_init_head(argv[2]);
	dec_init_layers(argv[2]);
	if (has_enc)
		enc_init(argv[1]);
	load_prefix(argv[3]);
	if (getenv("DUOSIGHT_VOCAB"))
		load_words(getenv("DUOSIGHT_VOCAB"));
	if (getenv("DUOSIGHT_BPE") && bpe_load(getenv("DUOSIGHT_BPE")))
		die("bad BPE file", getenv("DUOSIGHT_BPE"));
	fprintf(stderr, "loaded in %.1f s: encoder, %d layers + head (S=%d, T=%d, head rows %d, vocab %d); "
		"CMA %ld -> %ld MiB free, %ld MiB used\n",
		now() - t0, LAYERS, S, T, HT, VOCAB, cma0 / 1024, cma_free_kb() / 1024, cma_used_mib());

	logits = malloc(VOCAB * sizeof(float));
	for (char *s = getenv("DUOSIGHT_STOP"); s && *s && nstop < 16; s = strchr(s, ',') ? strchr(s, ',') + 1 : NULL)
		stop[nstop++] = atoi(s);

	for (;;) {
		int got = read_line(line, sizeof(line), !job.on);

		if (got > 0)
			command(line); // questions first: a frame in steps waits for them
		else if (job.on)
			job_step();
		else
			break;
	}
	return 0;
}
