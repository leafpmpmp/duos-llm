// SPDX-License-Identifier: MIT
// Answer prompts on the SG2000 TPU from token ids and image embeddings.
// Cut down from tpu/runtime/llm.c in leafpmpmp/duos-llm: the same models and
// the same K/V handling, without the llama.cpp vocabulary. Ids go in and come
// out, and the PC handles the text. The vocabulary size comes from the head.
//
// Usage: vlm MODEL_DIR PREFIX.kv IMAGE_ID MAX_NEW EOS < prompts.bin > answers.txt
// prompts.bin, for each prompt: int32 n, int32 ids[n], float32 rows[m][576] for
// the m ids equal to IMAGE_ID.
// answers.txt, one line per prompt: index, prefill ms, decode ms, then the ids.
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cviruntime.h>

#define D 576
#define NKV 3
#define G 3
#define HD 64
#define LAYERS 30
#define THETA 100000.0
#define NEG -10000.0f
#define TMAX 16
#define MAXTOK 512
#define MAXROWS 256

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

static int S, T, VOCAB; // cache slots, tokens per call, vocabulary size
static struct layer lay[LAYERS], head;
static struct tens logits_t;
static _Float16 *embed;
static float image[MAXROWS][D]; // ids >= VOCAB are rows of this table

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void die(const char *msg, const char *arg)
{
	fprintf(stderr, "vlm: %s%s%s\n", msg, arg ? " " : "", arg ? arg : "");
	exit(1);
}

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
	int fd;

	if (CVI_NN_RegisterModel(path, &l->m) != CVI_RC_SUCCESS)
		die("cannot load", path);
	// The file's page cache can sit in free CMA and block the next model's
	// contiguous buffers (EBUSY): drop it.
	fd = open(path, O_RDONLY);
	if (fd >= 0) {
		posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
		close(fd);
	}
	if (CVI_NN_GetInputOutputTensors(l->m, &l->in, &l->nin, &l->out, &l->nout) != CVI_RC_SUCCESS)
		die("no tensors in", path);
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

static void init(const char *dir)
{
	char path[1024];
	struct stat st;
	int fd;

	// Page cache in CMA that the kernel cannot migrate (dirty or busy pages)
	// makes contiguous allocations fail with EBUSY: write back and drop the
	// clean page cache first (needs root; best effort).
	sync();
	fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
	if (fd >= 0) {
		if (write(fd, "1", 1) != 1)
			fprintf(stderr, "vlm: cannot drop the page cache\n");
		close(fd);
	}

	// The head first: its 27 MB weight block needs CMA before the layers fragment it.
	snprintf(path, sizeof(path), "%s/head.cvimodel", dir);
	load(&head, path);
	head.h = find(head.in, head.nin, "h", 1);
	logits_t = find(head.out, head.nout, "logits", 0);
	T = (int)(head.h.n / D);
	if (T < 1 || T > TMAX)
		die("unexpected head shapes in", path);
	VOCAB = (int)(logits_t.n / T);

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
		int s = (int)(l->k.n / (NKV * HD));
		if (i && s != S)
			die("cache size differs in", path);
		S = s;
		if (l->h.n != (size_t)T * D || l->mask.n != (size_t)NKV * T * G * (S + T) ||
		    l->kn.n != (size_t)NKV * T * HD)
			die("layer shapes do not match the head in", path);
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
// n..T-1 are padding. The head runs only when logits are wanted, so most of a
// prefill skips it. The new K/V rows wait in kpend/vpend until commit().
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
		put(&head.h, head.h.p, 0, h, (size_t)T * D);
		forward(&head);
		get(&logits_t, logits_t.p, 0, logits, (size_t)n * VOCAB);
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
static void prefill(const int32_t *tok, int from, int n, float *x, float *lg)
{
	int buf[TMAX];

	for (int i = from; i < n; i += T) {
		int m = n - i < T ? n - i : T;

		for (int t = 0; t < m; t++)
			buf[t] = tok[i + t];
		pass(buf, m, i, i + m == n ? lg : NULL);
		commit(i, m);
		if (i + m == n)
			memcpy(x, lg + (size_t)(m - 1) * VOCAB, VOCAB * sizeof(float));
	}
}

static int argmax(const float *x)
{
	int best = 0;

	for (int i = 1; i < VOCAB; i++)
		if (x[i] > x[best])
			best = i;
	return best;
}

int main(int argc, char **argv)
{
	static int32_t ids[MAXTOK], run[MAXTOK];
	int32_t n;
	double t0;

	if (argc != 6)
		die("usage: vlm MODEL_DIR PREFIX.kv IMAGE_ID MAX_NEW EOS < prompts.bin > answers.txt", NULL);
	int image_id = atoi(argv[3]), max_new = atoi(argv[4]), eos = atoi(argv[5]);

	t0 = now();
	init(argv[1]);
	fprintf(stderr, "loaded %d layers + head (S=%d, T=%d, vocab %d) in %.1f s\n", LAYERS, S, T, VOCAB, now() - t0);
	load_prefix(argv[2]);
	if (getenv("VLM_CMA"))
		system("grep CmaFree /proc/meminfo >&2");

	float *x = malloc(VOCAB * sizeof(float)), *lg = malloc((size_t)T * VOCAB * sizeof(float));

	int used = 0;

	for (int idx = 0; fread(&n, 4, 1, stdin) == 1; idx++) {
		int rows = 0, gen = 0, tk, pos;

		if (n <= prefix_n || n + max_new > S || n > MAXTOK || fread(ids, 4, n, stdin) != (size_t)n)
			die("bad prompt", NULL);
		for (int i = 0; i < n; i++) {
			if (i < prefix_n && ids[i] != prefix_ids[i])
				die("prompt does not start with the prefix tokens", NULL);
			run[i] = ids[i] == image_id ? VOCAB + rows++ : ids[i];
		}
		if (rows > MAXROWS || fread(image, D * sizeof(float), rows, stdin) != (size_t)rows)
			die("bad image rows", NULL);

		t0 = now();
		if (used > prefix_n)
			clear(prefix_n, used);
		prefill(run, prefix_n, n, x, lg);
		double t_prefill = now() - t0;

		printf("%d", idx);
		static int out[MAXTOK];
		t0 = now();
		tk = argmax(x);
		pos = n;
		while (gen < max_new && tk != eos) {
			int in[TMAX];

			out[gen++] = tk;
			if (gen == max_new)
				break;
			in[0] = tk;
			pass(in, 1, pos, lg);
			commit(pos, 1);
			pos++;
			tk = argmax(lg);
		}
		used = pos;
		printf(" %.0f %.0f", 1e3 * t_prefill, 1e3 * (now() - t0));
		for (int i = 0; i < gen; i++)
			printf(" %d", out[i]);
		printf("\n");
		fflush(stdout);
	}
	return 0;
}
