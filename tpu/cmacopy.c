// SPDX-License-Identifier: MIT
// Time weight copies into a CMA buffer on the A53, alone and while the TPU runs.
// Usage: cmacopy FILE [model.cvimodel]
// FILE is the copy source. With a model, a second thread runs Forward
// (device I/O) during a second round of the copy tests.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <cviruntime.h>
#include <cviruntime_context.h>

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void clean(char *p, size_t n)
{
	uint64_t ctr;
	size_t line;

	asm volatile("mrs %0, ctr_el0" : "=r"(ctr));
	line = 4u << (ctr >> 16 & 15);
	for (char *q = (char *)((uintptr_t)p & ~(line - 1)); q < p + n; q += line)
		asm volatile("dc cvac, %0" : : "r"(q) : "memory");
	asm volatile("dsb sy" : : : "memory");
}

static CVI_RT_HANDLE ctx;
static CVI_RT_MEM mem;
static char *cma, *ram;
static size_t size;
static int fd, dfd;

static CVI_MODEL_HANDLE model;
static CVI_TENSOR *in, *out;
static int32_t nin, nout;
static atomic_int run;
static atomic_long calls;
static atomic_long busy_us;

static void *tpu_loop(void *arg)
{
	(void)arg;
	while (atomic_load(&run)) {
		double t0 = now_ms();

		CVI_NN_Forward(model, in, nin, out, nout);
		busy_us += (long)((now_ms() - t0) * 1e3);
		calls++;
	}
	return NULL;
}

enum { MEMCPY, CLEAN, MEMCPY_CLEAN, FLUSH, PREAD, DIRECT_CMA, DIRECT_RAM, NTEST };
static const char *names[NTEST] = {
	"memcpy RAM to CMA", "dc cvac on CMA", "memcpy + dc cvac", "CVI_RT_MemFlush",
	"pread page cache to CMA", "pread O_DIRECT to CMA", "pread O_DIRECT to RAM",
};

static int one(int test)
{
	switch (test) {
	case MEMCPY: memcpy(cma, ram, size); return 0;
	case CLEAN: clean(cma, size); return 0;
	case MEMCPY_CLEAN: memcpy(cma, ram, size); clean(cma, size); return 0;
	case FLUSH: return CVI_RT_MemFlush(ctx, mem) == CVI_RC_SUCCESS ? 0 : -1;
	case PREAD: return pread(fd, cma, size, 0) == (ssize_t)size ? 0 : -1;
	case DIRECT_CMA: return pread(dfd, cma, size, 0) == (ssize_t)size ? 0 : -1;
	case DIRECT_RAM: return pread(dfd, ram, size, 0) == (ssize_t)size ? 0 : -1;
	}
	return -1;
}

static void round_of_tests(const char *title)
{
	printf("%s\n", title);
	for (int t = 0; t < NTEST; t++) {
		int reps = t >= DIRECT_CMA ? 5 : 20;
		double t0, ms;
		long c0 = calls;
		long b0 = busy_us;

		t0 = now_ms();
		for (int r = 0; r < reps; r++)
			if (one(t)) {
				printf("  %-24s fails: %s\n", names[t], strerror(errno));
				goto next;
			}
		ms = now_ms() - t0;
		printf("  %-24s %6.0f MB/s", names[t], size * reps / ms / 1e3);
		if (calls > c0)
			printf("   TPU %.3f ms/call", (busy_us - b0) / 1e3 / (calls - c0));
		printf("\n");
next:;
	}
}

int main(int argc, char **argv)
{
	struct stat st;
	pthread_t th;

	if (argc < 2) {
		fprintf(stderr, "usage: %s FILE [model.cvimodel]\n", argv[0]);
		return 1;
	}
	fd = open(argv[1], O_RDONLY);
	dfd = open(argv[1], O_RDONLY | O_DIRECT);
	if (fd < 0 || dfd < 0 || fstat(fd, &st)) {
		perror(argv[1]);
		return 1;
	}
	size = st.st_size & ~(size_t)4095;
	if (CVI_RT_Init(&ctx) != CVI_RC_SUCCESS || !(mem = CVI_RT_MemAlloc(ctx, size))) {
		fprintf(stderr, "no CMA buffer of %zu bytes\n", size);
		return 1;
	}
	cma = (char *)CVI_RT_MemGetVAddr(mem);
	if (posix_memalign((void **)&ram, 4096, size) || pread(fd, ram, size, 0) != (ssize_t)size) {
		perror("read");
		return 1;
	}
	printf("%zu bytes\n", size);
	round_of_tests("CPU only:");

	if (argc < 3)
		return 0;
	if (CVI_NN_RegisterModel(argv[2], &model) != CVI_RC_SUCCESS ||
	    CVI_NN_GetInputOutputTensors(model, &in, &nin, &out, &nout) != CVI_RC_SUCCESS) {
		fprintf(stderr, "cannot load %s\n", argv[2]);
		return 1;
	}
	for (int i = 0; i < nin; i++)
		in[i].mem_type = CVI_MEM_DEVICE;
	for (int i = 0; i < nout; i++)
		out[i].mem_type = CVI_MEM_DEVICE;
	run = 1;
	pthread_create(&th, NULL, tpu_loop, NULL);
	usleep(300000);
	{
		long c0 = calls;
		long b0 = busy_us;

		usleep(1000000);
		printf("TPU alone: %.3f ms/call, %ld calls/s\n", (busy_us - b0) / 1e3 / (calls - c0), calls - c0);
	}
	round_of_tests("CPU copies while the TPU runs:");
	run = 0;
	pthread_join(th, NULL);
	CVI_NN_CleanupModel(model);
	return 0;
}
