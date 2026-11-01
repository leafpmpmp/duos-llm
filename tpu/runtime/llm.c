// SPDX-License-Identifier: MIT
// SmolLM2-135M on the SG2000 TPU: one cvimodel per layer + LM head.
// Layer models take T tokens per call (export_t.py; T = 1 also for the older
// 1-token export from export_all.py): inputs h, RoPE tables, k_cache,
// v_cache, mask; outputs dh, k_new, v_new (see tpu/smollm/model.py). The
// residual stream stays fp32 on the CPU. INT8, BF16 and mixed models can be
// combined. With T > 1 the runtime prefills T tokens per call and verifies
// prompt-lookup drafts (greedy speculative decoding).
// k_cache, v_cache and mask stay in the models' input buffers between calls:
// the runtime writes only changed rows and restores the bytes that outputs
// overwrite. When the cache is full, the oldest generated tokens leave it and
// the prompt stays (sliding window).
//
// Usage:
//   llm -d MODEL_DIR -v VOCAB.gguf [-x PREFIX.kv] [-c] [-S SYSTEM] [-p PROMPT]
//       [-n N] [-t TEMP] [-k TOPK] [-s SEED] [--draft N]
//   llm -d MODEL_DIR -v VOCAB.gguf [-x PREFIX.kv] -S SYSTEM --snapshot OUT.kv
//   llm -d MODEL_DIR [-x PREFIX.kv] --ppl TOKENS.i32
//   llm --layertest MODEL.cvimodel DIR [PROGRAM]
// MODEL_DIR holds l00..l29 and head as <name>.cvimodel (any variant, via
// symlinks) and embed.f16. -c applies the model's chat template (with the
// system message -S). -x loads K/V for the first tokens (make_prefix.py or
// --snapshot). --draft N: draft length, 0 = no speculation (default T-1).
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
#include <llama.h>

#define D 576
#define NKV 3
#define G 3
#define HD 64
#define LAYERS 30
#define VOCAB 49152
#define THETA 100000.0
#define NEG -10000.0f
#define TMAX 16
#define MAXTOK 8192 // max_position_embeddings

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
	int tq; // RoPE inputs cos_q/sin_q/cos_k/sin_k (else cos/sin, T = 1)
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

static int S, T; // cache slots, tokens per call
static struct layer lay[LAYERS], head;
static struct tens logits_t;
static _Float16 *embed;
static double t_tpu;
static int check_inputs; // LLM_CHECK_INPUTS=1: compare resident inputs with the shadows
static long changed[3], inputs_checked;
static float (*dbg_dh)[D]; // rowtest: row 0 of dh per layer, copied during pass()

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void die(const char *msg, const char *arg)
{
	fprintf(stderr, "llm: %s%s%s\n", msg, arg ? " " : "", arg ? arg : "");
	exit(1);
}

static int has(CVI_TENSOR *ts, int32_t n, const char *name)
{
	for (int32_t i = 0; i < n; i++)
		if (!strcmp(CVI_NN_TensorName(&ts[i]), name))
			return 1;
	return 0;
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
// flushes them: the runtime writes CPU cache lines back itself (arm64 lets
// user space clean by address). Elsewhere Forward flushes the whole inputs.
#ifdef __aarch64__
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
#endif

// Write the cache lines of [p, p + n) back to DRAM (forward() waits for it).
static void flush(const void *p, size_t n)
{
#ifdef __aarch64__
	uintptr_t a = (uintptr_t)p & ~(cache_line() - 1), e = (uintptr_t)p + n;

	for (; a < e; a += cache_line())
		__asm__ volatile("dc cvac, %0" : : "r"(a) : "memory");
#else
	(void)p;
	(void)n;
#endif
}

// Drop the (clean) cache lines of [p, p + n): reads then see what the TPU wrote.
static void invalidate(const void *p, size_t n)
{
#ifdef __aarch64__
	uintptr_t a = (uintptr_t)p & ~(cache_line() - 1), e = (uintptr_t)p + n;

	for (; a < e; a += cache_line())
		__asm__ volatile("dc civac, %0" : : "r"(a) : "memory");
	__asm__ volatile("dsb sy" : : : "memory");
#else
	(void)p;
	(void)n;
#endif
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

// prog: program of a merged cvimodel (0 = first).
static void load(struct layer *l, const char *path, int prog)
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
	if (prog)
		CVI_NN_SetConfig(l->m, OPTION_PROGRAM_INDEX, prog);
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
#ifdef __aarch64__
		res[j]->t->mem_type = CVI_MEM_DEVICE;
#endif
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
			fprintf(stderr, "llm: cannot drop the page cache\n");
		close(fd);
	}

	// The head first: its 27 MB weight block needs CMA before the layers fragment it.
	snprintf(path, sizeof(path), "%s/head.cvimodel", dir);
	load(&head, path, 0);
	head.h = find(head.in, head.nin, "h", 1);
	logits_t = find(head.out, head.nout, "logits", 0);
	T = (int)(head.h.n / D);
	if (T < 1 || T > TMAX || logits_t.n != (size_t)T * VOCAB)
		die("unexpected head shapes in", path);

	for (int i = 0; i < LAYERS; i++) {
		struct layer *l = &lay[i];

		snprintf(path, sizeof(path), "%s/l%02d.cvimodel", dir, i);
		load(l, path, 0);
		l->h = find(l->in, l->nin, "h", 1);
		l->tq = has(l->in, l->nin, "cos_q");
		if (l->tq) {
			l->cq = find(l->in, l->nin, "cos_q", 1);
			l->sq = find(l->in, l->nin, "sin_q", 1);
			l->ck = find(l->in, l->nin, "cos_k", 1);
			l->sk = find(l->in, l->nin, "sin_k", 1);
		} else {
			l->cq = find(l->in, l->nin, "cos", 1);
			l->sq = find(l->in, l->nin, "sin", 1);
		}
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
		    l->kn.n != (size_t)NKV * T * HD || (!l->tq && T != 1))
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
	double t0 = now();

#ifdef __aarch64__
	__asm__ volatile("dsb sy" : : : "memory"); // flush() done before the TPU reads
#endif
	if (CVI_NN_Forward(l->m, l->in, l->nin, l->out, l->nout) != CVI_RC_SUCCESS)
		die("forward failed", NULL);
	t_tpu += now() - t0;
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

// LLM_CHECK_INPUTS: resident inputs must equal their shadows after a pass.
static void check_resident(struct layer *l)
{
	const struct tens *t[3] = { &l->k, &l->v, &l->mask };
	const void *sh[3] = { l->kbuf, l->vbuf, l->mbuf };

	for (int j = 0; j < 3; j++) {
		invalidate(t[j]->p, t[j]->bytes);
		changed[j] += !!memcmp(t[j]->p, sh[j], t[j]->bytes);
	}
	inputs_checked++;
}

// Run n <= T tokens at positions pos..pos+n-1; cache slots 0..slot-1 hold
// the earlier tokens. Rows n..T-1 are padding. Fills logits for rows
// 0..n-1 if given. The new K/V rows wait in kpend/vpend until commit().
static void pass(const int *tok, int n, int pos, int slot, float *logits)
{
	static float h[TMAX * D], dh[TMAX * D], cs[TMAX][HD], sn[TMAX][HD];
	static float cq[NKV * TMAX * G * HD], sq[NKV * TMAX * G * HD], ck[NKV * TMAX * HD], sk[NKV * TMAX * HD];
	struct conv ccq[4] = { 0 }, csq[4] = { 0 }, cck[4] = { 0 }, csk[4] = { 0 };

	if (n < 1 || n > T || slot + n > S)
		die("context full", NULL);
	memset(h, 0, sizeof(h));
	for (int t = 0; t < n; t++)
		for (int i = 0; i < D; i++)
			h[t * D + i] = embed[(size_t)tok[t] * D + i];
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
		put_cached(ccq, 4, &l->cq, l->tq ? cq : cs[0]);
		put_cached(csq, 4, &l->sq, l->tq ? sq : sn[0]);
		if (l->tq) {
			put_cached(cck, 4, &l->ck, ck);
			put_cached(csk, 4, &l->sk, sk);
		}
		set_valid(l, slot);
		forward(l);
		get(&l->dh, l->dh.p, 0, dh, (size_t)T * D);
		if (dbg_dh)
			memcpy(dbg_dh[i], dh, sizeof(dbg_dh[i]));
		for (int j = 0; j < T * D; j++)
			h[j] += dh[j];
		get(&l->kn, l->kn.p, 0, l->kpend, (size_t)NKV * T * HD);
		get(&l->vn, l->vn.p, 0, l->vpend, (size_t)NKV * T * HD);
		for (int j = 0; j < l->novl; j++) {
			memcpy(l->ovl[j].dst, l->ovl[j].src, l->ovl[j].n);
			flush(l->ovl[j].dst, l->ovl[j].n);
		}
		if (check_inputs)
			check_resident(l);
	}
	conv_reset(ccq, 4);
	conv_reset(csq, 4);
	conv_reset(cck, 4);
	conv_reset(csk, 4);
	put(&head.h, head.h.p, 0, h, (size_t)T * D);
	forward(&head);
	if (logits)
		get(&logits_t, logits_t.p, 0, logits, (size_t)n * VOCAB);
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

// Drop cache slots keep..keep+drop-1 of n used slots: the later slots move
// down. RoPE positions stay: the cached K rows keep their rotation.
static void evict(int keep, int drop, int n)
{
	for (int i = 0; i < LAYERS; i++) {
		struct layer *l = &lay[i];
		const struct tens *t[2] = { &l->k, &l->v };
		char *sh[2] = { l->kbuf, l->vbuf };

		for (int j = 0; j < 2; j++)
			for (int kv = 0; kv < NKV; kv++) {
				size_t es = t[j]->bytes / t[j]->n, off = ((size_t)kv * S + keep) * HD;
				size_t len = (size_t)(n - keep - drop) * HD;

				memmove(sh[j] + off * es, sh[j] + (off + (size_t)drop * HD) * es, len * es);
				sync_res(t[j], sh[j], off, len);
			}
	}
}

// Fixed prompt prefix with float K/V (make_prefix.py or --snapshot): slots 0..P-1.
// File: int32 P, int32 ids[P], float32 k, v [layers][nkv][P][hd].
static int prefix_n;
static int32_t prefix_ids[MAXTOK];

static void load_prefix(const char *file)
{
	FILE *f = fopen(file, "rb");
	int32_t p;

	if (!f || fread(&p, 4, 1, f) != 1 || p < 1 || p >= S || fread(prefix_ids, 4, p, f) != (size_t)p)
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

static void save_prefix(const char *file, const int32_t *ids, int p)
{
	FILE *f = fopen(file, "wb");
	float x[HD];

	if (!f)
		die("cannot create", file);
	fwrite(&p, 4, 1, f);
	fwrite(ids, 4, p, f);
	for (int pass = 0; pass < 2; pass++)
		for (int i = 0; i < LAYERS; i++)
			for (int kv = 0; kv < NKV; kv++)
				for (int s = 0; s < p; s++) {
					const struct layer *l = &lay[i];
					size_t off = ((size_t)kv * S + s) * HD;

					get(pass ? &l->v : &l->k, pass ? l->vbuf : l->kbuf, off, x, HD);
					fwrite(x, sizeof(float), HD, f);
				}
	fclose(f);
}

static void check_prefix(const int32_t *tok, int n)
{
	if (n <= prefix_n)
		die("input shorter than the prefix", NULL);
	for (int i = 0; i < prefix_n; i++)
		if (tok[i] != prefix_ids[i])
			die("input does not start with the prefix tokens", NULL);
}

// Run tokens from..n-1 in passes of T; logits of the last token into x.
static void prefill(const int32_t *tok, int from, int n, float *x)
{
	static float lg[TMAX * VOCAB];
	int buf[TMAX];

	for (int i = from; i < n; i += T) {
		int m = n - i < T ? n - i : T;

		for (int t = 0; t < m; t++)
			buf[t] = tok[i + t];
		pass(buf, m, i, i, i + m == n ? lg : NULL);
		commit(i, m);
		if (i + m == n && x)
			memcpy(x, lg + (size_t)(m - 1) * VOCAB, VOCAB * sizeof(float));
	}
}

static uint64_t rng = 88172645463325252ull;

static double urand(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return (rng >> 11) * (1.0 / 9007199254740992.0);
}

static int argmax(const float *x)
{
	int best = 0;

	for (int i = 1; i < VOCAB; i++)
		if (x[i] > x[best])
			best = i;
	return best;
}

static int sample(const float *x, float temp, int topk)
{
	int best = argmax(x);

	if (temp <= 0 || topk <= 1)
		return best;

	int idx[256], n = 0;
	double p[256], sum = 0;

	if (topk > 256)
		topk = 256;
	for (int i = 0; i < VOCAB; i++) { // keep idx sorted, best first
		int j;

		if (n == topk && x[i] <= x[idx[n - 1]])
			continue;
		j = n < topk ? n++ : n - 1;
		for (; j > 0 && x[idx[j - 1]] < x[i]; j--)
			idx[j] = idx[j - 1];
		idx[j] = i;
	}
	for (int k = 0; k < topk; k++) {
		p[k] = exp((x[idx[k]] - x[best]) / temp);
		sum += p[k];
	}
	double r = urand() * sum;
	for (int k = 0; k < topk; k++) {
		r -= p[k];
		if (r <= 0)
			return idx[k];
	}
	return idx[topk - 1];
}

// Prompt lookup: the tokens that followed the latest earlier occurrence of
// the last 3, 2 or 1 tokens of ctx. Returns the draft length (<= max).
static int draft(const int32_t *ctx, int n, int *out, int max)
{
	for (int g = 3; g >= 1; g--) {
		if (n <= g)
			continue;
		for (int j = n - g - 1; j >= 0; j--) {
			if (memcmp(ctx + j, ctx + n - g, g * sizeof(*ctx)))
				continue;
			int d = 0;

			while (d < max && j + g + d < n)
				out[d] = ctx[j + g + d], d++;
			if (d)
				return d;
		}
	}
	return 0;
}

static int ppl(const char *file)
{
	static float lg[TMAX * VOCAB];
	FILE *f = fopen(file, "rb");
	int32_t tok[4096];
	int buf[TMAX];
	size_t n;
	double nll = 0, t0;

	if (!f)
		die("cannot open", file);
	n = fread(tok, sizeof(tok[0]), 4096, f);
	fclose(f);
	check_inputs = getenv("LLM_CHECK_INPUTS") != NULL;
	if (n > (size_t)S)
		n = S;
	check_prefix(tok, (int)n);
	t0 = now();
	for (size_t i = prefix_n; i + 1 < n; i += T) {
		int m = (int)(n - 1 - i) < T ? (int)(n - 1 - i) : T;

		for (int t = 0; t < m; t++)
			buf[t] = tok[i + t];
		pass(buf, m, (int)i, (int)i, lg);
		commit((int)i, m);
		for (int t = 0; t < m; t++) {
			const float *x = lg + (size_t)t * VOCAB;
			double mx = x[argmax(x)], z = 0;

			for (int j = 0; j < VOCAB; j++)
				z += exp(x[j] - mx);
			nll += mx + log(z) - x[tok[i + t + 1]];
		}
	}
	t0 = now() - t0;
	size_t m = n - 1 - prefix_n;
	printf("tokens %zu (scored %zu): PPL %.4f (mean NLL %.4f), %.1f tok/s (T=%d), TPU %.0f%%\n", n, m,
	       exp(nll / m), nll / m, m / t0, T, 100 * t_tpu / t0);
	if (check_inputs)
		printf("resident inputs differing from the shadows: k %ld, v %ld, mask %ld of %ld\n",
		       changed[0], changed[1], changed[2], inputs_checked);
	return 0;
}

// Row independence: run tok[32] after tok[..31] once alone (padding rows)
// and once followed by 3 more tokens; compare row 0 (dh per layer, k_new, logits).
static int rowtest(const char *file)
{
	static float a[TMAX * VOCAB], b[TMAX * VOCAB], dha[LAYERS][D], dhb[LAYERS][D];
	static float ka[LAYERS][NKV * TMAX * HD];
	FILE *f = fopen(file, "rb");
	int32_t tok[4096];
	int in[TMAX] = { 0 };
	size_t n;

	if (!f || T < 4)
		die("rowtest needs a token file and T >= 4 models", NULL);
	n = fread(tok, sizeof(tok[0]), 4096, f);
	fclose(f);
	if (n < 40)
		die("rowtest needs at least 40 tokens", NULL);
	check_prefix(tok, (int)n);
	prefill(tok, prefix_n, 32, NULL);
	in[0] = tok[32];
	dbg_dh = dha;
	pass(in, 1, 32, 32, a);
	for (int i = 0; i < LAYERS; i++)
		memcpy(ka[i], lay[i].kpend, sizeof(ka[i]));
	for (int t = 1; t < 4; t++)
		in[t] = tok[32 + t];
	dbg_dh = dhb;
	pass(in, 4, 32, 32, b);
	dbg_dh = NULL;
	double dl = 0;
	for (int j = 0; j < VOCAB; j++)
		dl = fmax(dl, fabs(a[j] - b[j]));
	printf("row 0 logits max|diff| %.4g, argmax %d vs %d\n", dl, argmax(a), argmax(b));
	for (int i = 0; i < LAYERS; i++) {
		double dd = 0, dk = 0;

		for (int j = 0; j < D; j++)
			dd = fmax(dd, fabs(dhb[i][j] - dha[i][j]));
		for (int kv = 0; kv < NKV; kv++)
			for (int j = 0; j < HD; j++)
				dk = fmax(dk, fabs(lay[i].kpend[kv * T * HD + j] - ka[i][kv * T * HD + j]));
		printf("layer %2d (%s): row 0 dh max|diff| %.4g, k_new %.4g\n", i,
		       lay[i].h.fmt == CVI_FMT_FP32 && lay[i].k.fmt == CVI_FMT_FP32 ? "bf16" : "mix", dd, dk);
	}
	return 0;
}

static float *readf(const char *dir, const char *name, size_t n)
{
	char path[1024];
	float *x = malloc(n * sizeof(float));
	FILE *f;

	snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
	f = fopen(path, "rb");
	if (!f || fread(x, sizeof(float), n, f) != n)
		die("cannot read", path);
	fclose(f);
	return x;
}

// Feed one model the test inputs DIR/<input>.f32, compare with DIR/ref_<output>.f32.
static int layertest(const char *model, const char *dir, int prog)
{
	static const char *outs[] = { "dh", "k_new", "v_new", "logits" };
	struct layer l;

	load(&l, model, prog);
	for (int32_t i = 0; i < l.nin; i++) {
		struct tens t = find(l.in, l.nin, CVI_NN_TensorName(&l.in[i]), 1);
		float *x = readf(dir, CVI_NN_TensorName(&l.in[i]), t.n);

		printf("in  %-8s fmt %d qscale %g n %zu\n", CVI_NN_TensorName(&l.in[i]), t.fmt, t.qs, t.n);
		put(&t, t.p, 0, x, t.n);
		free(x);
	}
	forward(&l);
	for (int32_t i = 0; i < l.nout; i++) {
		const char *name = CVI_NN_TensorName(&l.out[i]);
		char ref[64];

		for (size_t j = 0; j < sizeof(outs) / sizeof(outs[0]); j++)
			if (!strncmp(name, outs[j], strlen(outs[j])))
				snprintf(ref, sizeof(ref), "ref_%s", outs[j]);
		struct tens t = find(l.out, l.nout, name, 1);
		float *got = malloc(t.n * sizeof(float)), *want = readf(dir, ref, t.n);
		double ab = 0, aa = 0, bb = 0, md = 0;

		get(&t, t.p, 0, got, t.n);
		for (size_t j = 0; j < t.n; j++) {
			ab += (double)got[j] * want[j];
			aa += (double)got[j] * got[j];
			bb += (double)want[j] * want[j];
			if (fabs(got[j] - want[j]) > md)
				md = fabs(got[j] - want[j]);
		}
		printf("out %-16s fmt %d qscale %g: cos %.5f max|d| %.4g\n", name, t.fmt, t.qs,
		       ab / sqrt(aa * bb + 1e-30), md);
		free(got);
		free(want);
	}
	return 0;
}

static void quiet(enum ggml_log_level level, const char *text, void *data)
{
	(void)data;
	if (level >= GGML_LOG_LEVEL_ERROR)
		fputs(text, stderr);
}

int main(int argc, char **argv)
{
	const char *dir = NULL, *vocab_path = NULL, *prompt = "The capital of France is", *ppl_file = NULL,
		   *prefix_file = NULL, *system = NULL, *snapshot = NULL;
	int n_predict = 64, chat = 0, topk = 40, max_draft = -1;
	float temp = 0;
	double t0;

	if ((argc == 4 || argc == 5) && !strcmp(argv[1], "--layertest"))
		return layertest(argv[2], argv[3], argc == 5 ? atoi(argv[4]) : 0);
	for (int i = 1; i < argc; i++) {
		const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;

		if (!strcmp(a, "-c")) {
			chat = 1;
			continue;
		}
		if (!v)
			die("missing value for", a);
		if (!strcmp(a, "-d"))
			dir = v;
		else if (!strcmp(a, "-v"))
			vocab_path = v;
		else if (!strcmp(a, "-p"))
			prompt = v;
		else if (!strcmp(a, "-S"))
			system = v;
		else if (!strcmp(a, "-n"))
			n_predict = atoi(v);
		else if (!strcmp(a, "-t"))
			temp = atof(v);
		else if (!strcmp(a, "-k"))
			topk = atoi(v);
		else if (!strcmp(a, "-s"))
			rng = strtoull(v, NULL, 0) | 1;
		else if (!strcmp(a, "--ppl"))
			ppl_file = v;
		else if (!strcmp(a, "-x"))
			prefix_file = v;
		else if (!strcmp(a, "--snapshot"))
			snapshot = v;
		else if (!strcmp(a, "--draft"))
			max_draft = atoi(v);
		else
			die("unknown option", a);
		i++;
	}
	if (!dir)
		die("usage: see the comment at the top of llm.c", NULL);

	t0 = now();
	init(dir);
	fprintf(stderr, "loaded %d layers + head (S=%d, T=%d) in %.1f s\n", LAYERS, S, T, now() - t0);
	if (prefix_file) {
		t0 = now();
		load_prefix(prefix_file);
		fprintf(stderr, "prefix of %d tokens loaded in %.0f ms\n", prefix_n, 1e3 * (now() - t0));
	}
	if (ppl_file)
		return getenv("LLM_ROWTEST") ? rowtest(ppl_file) : ppl(ppl_file);
	if (!vocab_path)
		die("-v VOCAB.gguf needed to tokenize", NULL);
	if (max_draft < 0 || max_draft > T - 1)
		max_draft = T - 1;
	if (temp > 0)
		max_draft = 0; // speculation here is greedy only

	llama_log_set(quiet, NULL);
	llama_backend_init();
	struct llama_model_params mp = llama_model_default_params();
	mp.vocab_only = true;
	struct llama_model *vm = llama_model_load_from_file(vocab_path, mp);
	if (!vm)
		die("cannot load vocabulary", vocab_path);
	const struct llama_vocab *vocab = llama_model_get_vocab(vm);

	static char text[16384];
	if (chat || snapshot) {
		struct llama_chat_message msg[2];
		int nm = 0;

		if (system)
			msg[nm++] = (struct llama_chat_message){ "system", system };
		if (!snapshot)
			msg[nm++] = (struct llama_chat_message){ "user", prompt };
		if (!nm)
			die("--snapshot needs -S SYSTEM", NULL);
		int len = llama_chat_apply_template(llama_model_chat_template(vm, NULL), msg, nm, !snapshot,
						    text, sizeof(text));
		if (len < 0 || len >= (int)sizeof(text))
			die("chat template failed", NULL);
	} else {
		snprintf(text, sizeof(text), "%s", prompt);
	}
	static int32_t ctx[MAXTOK];
	int nt = llama_tokenize(vocab, text, strlen(text), ctx, MAXTOK, false, true);
	if (nt <= 0 || nt >= S)
		die("prompt too long or empty", NULL);
	check_prefix(ctx, nt);

	static float x[VOCAB], lg[TMAX * VOCAB];
	t0 = now();
	prefill(ctx, prefix_n, nt, x);
	double t_prompt = now() - t0, tpu_prompt = t_tpu;
	int n_prompt = nt - prefix_n; // tokens run on the TPU
	if (snapshot) {
		save_prefix(snapshot, ctx, nt);
		fprintf(stderr, "snapshot of %d tokens (%d computed in %.0f ms) -> %s\n", nt, n_prompt,
			1e3 * t_prompt, snapshot);
		llama_model_free(vm);
		return 0;
	}
	if (!chat)
		fputs(prompt, stdout);
	fflush(stdout);

	int pos = nt, slot = nt, gen = 0, passes = 0, drafted = 0, accepted = 0, slides = 0;
	int keep = nt <= S / 2 ? nt : S / 4; // prompt slots that stay when the cache is full
	t0 = now();
	t_tpu = 0;
	int tk = sample(x, temp, topk);
	while (gen < n_predict) {
		char piece[256];
		int len, in[TMAX], d, m;

		if (llama_vocab_is_eog(vocab, tk))
			break;
		len = llama_token_to_piece(vocab, tk, piece, sizeof(piece), 0, false);
		if (len > 0)
			fwrite(piece, 1, len, stdout);
		fflush(stdout);
		ctx[pos] = tk;
		gen++;
		if (gen >= n_predict || pos + 1 >= MAXTOK)
			break;
		// Verify [tk, draft...] in one pass: row i predicts the token after input i.
		in[0] = tk;
		d = draft(ctx, pos + 1, in + 1, pos + 1 + max_draft < MAXTOK ? max_draft : MAXTOK - pos - 1);
		if (slot + 1 + d > S) { // cache full: the oldest generated tokens leave it
			int drop = (S - keep) / 4;

			evict(keep, drop, slot);
			slot -= drop;
			slides++;
		}
		pass(in, 1 + d, pos, slot, lg);
		passes++;
		drafted += d;
		for (m = 0; m < d && argmax(lg + (size_t)m * VOCAB) == in[1 + m]; m++)
			;
		commit(slot, 1 + m);
		accepted += m;
		for (int j = 0; j < m && gen < n_predict; j++) { // matched draft tokens
			if (llama_vocab_is_eog(vocab, in[1 + j]))
				goto done;
			len = llama_token_to_piece(vocab, in[1 + j], piece, sizeof(piece), 0, false);
			if (len > 0)
				fwrite(piece, 1, len, stdout);
			ctx[pos + 1 + j] = in[1 + j];
			gen++;
		}
		fflush(stdout);
		pos += 1 + m;
		slot += 1 + m;
		tk = sample(lg + (size_t)m * VOCAB, temp, topk); // argmax when greedy
	}
done:;
	double t_gen = now() - t0;
	printf("\n");
	fprintf(stderr, "prompt %d tok %.1f tok/s (TPU %.0f%%); generated %d tok in %d passes, %.1f tok/s "
		"(TPU %.0f%%); drafts %d, accepted %d; window slides %d\n",
		n_prompt, n_prompt / t_prompt, 100 * tpu_prompt / t_prompt, gen, passes,
		t_gen > 0 ? gen / t_gen : 0.0, t_gen > 0 ? 100 * t_tpu / t_gen : 0.0, drafted, accepted, slides);
	llama_model_free(vm);
	return 0;
}
