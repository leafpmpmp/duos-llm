// Run the 12 SigLIP blocks in sequence on the TPU.
// Reads fp32 [768, 1, N] tensors from stdin, one per image, and writes the output of
// the last block to stdout. Timing goes to stderr.
//
// All blocks share the two I/O buffers of block 0, so the other 22 are freed and nothing
// is copied between blocks: block i reads what block i-1 wrote. Block 0 goes A -> B,
// block 1 B -> A, and block 11 ends in A. Only the TPU touches the buffers between the
// first and the last block, so the CPU maintains its cache only at the two ends.
//
// Usage: chain MODEL_DIR N < in.bin > out.bin
//   CHAIN_DUMP=1: write every block's output instead of only the last
//   CHAIN_CMA=1: print CmaFree after loading
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cviruntime.h>

#define BLOCKS 12

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// Drop the cache lines of [p, p + n): reads then see what the TPU wrote.
static void invalidate(const void *p, size_t n)
{
	uint64_t ctr;
	uintptr_t line, a, e = (uintptr_t)p + n;

	__asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
	line = (uintptr_t)4 << ((ctr >> 16) & 15);
	for (a = (uintptr_t)p & ~(line - 1); a < e; a += line)
		__asm__ volatile("dc civac, %0" : : "r"(a) : "memory");
	__asm__ volatile("dsb sy" : : : "memory");
}

// Page cache that sits in CMA can block the next contiguous buffer, which matters
// when a detector already holds part of the pool.
static void drop_page_cache(void)
{
	int fd;

	sync();
	fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
	if (fd >= 0) {
		if (write(fd, "1", 1) != 1)
			fprintf(stderr, "cannot drop the page cache\n");
		close(fd);
	}
}

int main(int argc, char **argv)
{
	CVI_MODEL_HANDLE m[BLOCKS];
	CVI_TENSOR *in[BLOCKS], *out[BLOCKS];
	int32_t nin[BLOCKS], nout[BLOCKS];
	char path[512];
	double t0, load, sum = 0, best = 1e30, worst = 0;
	size_t bytes;
	float *x;
	void *a, *b;
	int images = 0, dump = getenv("CHAIN_DUMP") != NULL;

	if (argc < 3) {
		fprintf(stderr, "usage: %s MODEL_DIR N < in.bin > out.bin\n", argv[0]);
		return 1;
	}
	bytes = (size_t)768 * atoi(argv[2]) * sizeof(float);
	x = malloc(bytes);
	t0 = now_ms();
	drop_page_cache();
	for (int i = 0; i < BLOCKS; i++) {
		int fd;

		snprintf(path, sizeof(path), "%s/l%02d.cvimodel", argv[1], i);
		if (CVI_NN_RegisterModel(path, &m[i]) != CVI_RC_SUCCESS ||
		    CVI_NN_GetInputOutputTensors(m[i], &in[i], &nin[i], &out[i], &nout[i]) != CVI_RC_SUCCESS) {
			fprintf(stderr, "cannot load %s\n", path);
			return 1;
		}
		fd = open(path, O_RDONLY);
		if (fd >= 0) {
			posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
			close(fd);
		}
		if (CVI_NN_TensorSize(&in[i][0]) != bytes || CVI_NN_TensorSize(&out[i][0]) != bytes) {
			fprintf(stderr, "%s: unexpected tensor size\n", path);
			return 1;
		}
	}
	a = CVI_NN_TensorPtr(&in[0][0]);
	b = CVI_NN_TensorPtr(&out[0][0]);
	for (int i = 1; i < BLOCKS; i++) {
		uint64_t pa = in[0][0].paddr, pb = out[0][0].paddr;

		if (CVI_NN_SetTensorPhysicalAddr(&in[i][0], i & 1 ? pb : pa) != CVI_RC_SUCCESS ||
		    CVI_NN_SetTensorPhysicalAddr(&out[i][0], i & 1 ? pa : pb) != CVI_RC_SUCCESS) {
			fprintf(stderr, "cannot share the buffers of block %d\n", i);
			return 1;
		}
	}
	load = now_ms() - t0;
	fprintf(stderr, "loaded %d blocks in %.0f ms\n", BLOCKS, load);
	if (getenv("CHAIN_CMA"))
		system("grep CmaFree /proc/meminfo >&2");

	while (fread(x, 1, bytes, stdin) == bytes) {
		t0 = now_ms();
		memcpy(a, x, bytes);
		for (int i = 0; i < BLOCKS; i++) {
			if (CVI_NN_Forward(m[i], in[i], nin[i], out[i], nout[i]) != CVI_RC_SUCCESS) {
				fprintf(stderr, "forward failed in block %d\n", i);
				return 1;
			}
			if (dump) {
				void *p = i & 1 ? a : b;

				invalidate(p, bytes);
				fwrite(p, 1, bytes, stdout);
			}
		}
		invalidate(a, bytes);
		memcpy(x, a, bytes);
		t0 = now_ms() - t0;
		sum += t0;
		if (t0 < best)
			best = t0;
		if (t0 > worst)
			worst = t0;
		images++;
		fwrite(x, 1, bytes, stdout);
		fflush(stdout);
	}
	fprintf(stderr, "%d images, encoder mean %.1f ms, min %.1f, max %.1f\n", images, images ? sum / images : 0, best, worst);
	return 0;
}
