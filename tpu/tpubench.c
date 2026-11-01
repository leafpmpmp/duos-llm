// SPDX-License-Identifier: MIT
// Time CVI_NN_Forward of a cvimodel on the SG2000 TPU.
// Usage: tpubench model.cvimodel [iterations] [weight_bytes] [program]
// weight_bytes (default or 0: model file size) gives the weight read rate;
// program selects one program of a merged (multi-program) cvimodel.
// DEVICE_IO=1 marks the I/O tensors as device memory: Forward then skips the
// input copies and cache maintenance (pure TPU time).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <cviruntime.h>

static double now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void show(const char *what, CVI_TENSOR *t, int n)
{
	for (int i = 0; i < n; i++) {
		CVI_SHAPE s = CVI_NN_TensorShape(&t[i]);

		printf("%s %d %s: [", what, i, CVI_NN_TensorName(&t[i]));
		for (size_t d = 0; d < s.dim_size; d++)
			printf("%s%d", d ? "," : "", s.dim[d]);
		printf("] %zu bytes\n", CVI_NN_TensorSize(&t[i]));
	}
}

int main(int argc, char **argv)
{
	CVI_MODEL_HANDLE model;
	CVI_TENSOR *in, *out;
	int32_t nin, nout;
	int iters = argc > 2 ? atoi(argv[2]) : 50;
	double bytes = argc > 3 ? atof(argv[3]) : 0, t0, t, sum = 0, best = 1e30;
	struct stat st;

	if (argc < 2) {
		fprintf(stderr, "usage: %s model.cvimodel [iterations] [weight_bytes]\n", argv[0]);
		return 1;
	}
	if (!bytes && stat(argv[1], &st) == 0)
		bytes = st.st_size;

	t0 = now_ms();
	if (CVI_NN_RegisterModel(argv[1], &model) != CVI_RC_SUCCESS) {
		fprintf(stderr, "cannot load %s\n", argv[1]);
		return 1;
	}
	printf("load %.1f ms\n", now_ms() - t0);
	if (argc > 4)
		CVI_NN_SetConfig(model, OPTION_PROGRAM_INDEX, atoi(argv[4]));
	if (CVI_NN_GetInputOutputTensors(model, &in, &nin, &out, &nout) != CVI_RC_SUCCESS)
		return 1;
	show("in", in, nin);
	show("out", out, nout);
	for (int i = 0; i < nin; i++)
		memset(CVI_NN_TensorPtr(&in[i]), 1, CVI_NN_TensorSize(&in[i]));
	if (getenv("DEVICE_IO")) {
		for (int i = 0; i < nin; i++)
			in[i].mem_type = CVI_MEM_DEVICE;
		for (int i = 0; i < nout; i++)
			out[i].mem_type = CVI_MEM_DEVICE;
	}

	for (int i = 0; i < 3; i++)
		CVI_NN_Forward(model, in, nin, out, nout);
	for (int i = 0; i < iters; i++) {
		t0 = now_ms();
		if (CVI_NN_Forward(model, in, nin, out, nout) != CVI_RC_SUCCESS) {
			fprintf(stderr, "forward failed\n");
			return 1;
		}
		t = now_ms() - t0;
		sum += t;
		if (t < best)
			best = t;
	}
	printf("forward: mean %.3f ms, min %.3f ms (%d runs); %.0f MB/s at min\n",
	       sum / iters, best, iters, bytes / best / 1e3);
	CVI_NN_CleanupModel(model);
	return 0;
}
