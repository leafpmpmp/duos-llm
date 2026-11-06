// SPDX-License-Identifier: MIT
// Print the ids bpe.h gives each question, one line per line of stdin, to compare with
// tok.py (the prompt after its 70 image tokens).
// Usage: tok_check BPE.bin < questions.txt
#include "bpe.h"

int main(int argc, char **argv)
{
	static char line[1024];
	static int32_t ids[1024];

	if (argc != 2 || bpe_load(argv[1])) {
		fprintf(stderr, "usage: tok_check BPE.bin < questions.txt\n");
		return 1;
	}
	while (fgets(line, sizeof(line), stdin)) {
		line[strcspn(line, "\n")] = 0;
		int n = bpe_question(line, ids, 1024);

		for (int i = 0; i < n; i++)
			printf(i ? " %d" : "%d", ids[i]);
		printf(n < 0 ? "error\n" : "\n");
	}
	return 0;
}
