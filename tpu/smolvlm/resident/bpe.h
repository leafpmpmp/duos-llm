// SPDX-License-Identifier: MIT
// The SmolVLM tokenizer of camera/tok.py in C, for questions in words on the board: the
// resident Python tokenizer held 23 MB, and a short-lived one takes about a second to start.
// The tables come from make_bpe.py. Bytes from 128 up count as letters, so text outside
// ASCII may split where tok.py would not; ASCII text gives the same ids (tok_check.c).
#ifndef BPE_H
#define BPE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct bpe_merge {
	int32_t a, b, rank, out;
};

static int32_t bpe_byte_id[256];
static struct bpe_merge *bpe_merges;
static int32_t bpe_nmerges;

// 0 on success.
static int bpe_load(const char *file)
{
	FILE *f = fopen(file, "rb");
	int ok = f && fread(bpe_byte_id, 4, 256, f) == 256 && fread(&bpe_nmerges, 4, 1, f) == 1 && bpe_nmerges > 0;

	if (ok) {
		bpe_merges = malloc((size_t)bpe_nmerges * sizeof(*bpe_merges));
		ok = bpe_merges && fread(bpe_merges, sizeof(*bpe_merges), bpe_nmerges, f) == (size_t)bpe_nmerges;
	}
	if (f)
		fclose(f);
	return ok ? 0 : -1;
}

static int bpe_cmp(const void *k, const void *e)
{
	const struct bpe_merge *x = k, *y = e;

	if (x->a != y->a)
		return x->a < y->a ? -1 : 1;
	return x->b < y->b ? -1 : x->b > y->b;
}

// Byte-level BPE over one word: the lowest-ranked pair merges everywhere, left to right,
// until no pair merges. Returns the number of ids, or -1 if they do not fit in room or a
// byte has no token.
static int bpe_word(const uint8_t *w, int len, int32_t *out, int room)
{
	int k = len;

	if (len > room)
		return -1;
	for (int i = 0; i < len; i++)
		out[i] = bpe_byte_id[w[i]];
	for (;;) {
		const struct bpe_merge *best = NULL;

		for (int i = 0; i + 1 < k; i++) {
			struct bpe_merge key = { .a = out[i], .b = out[i + 1] };
			const struct bpe_merge *m = bsearch(&key, bpe_merges, bpe_nmerges, sizeof(key), bpe_cmp);

			if (m && (!best || m->rank < best->rank))
				best = m;
		}
		if (!best) {
			for (int i = 0; i < k; i++)
				if (out[i] < 0) // a byte with no token: tok.py fails too
					return -1;
			return k;
		}
		int j = 0;

		for (int i = 0; i < k; j++)
			if (i + 1 < k && out[i] == best->a && out[i + 1] == best->b) {
				out[j] = best->out;
				i += 2;
			} else {
				out[j] = out[i++];
			}
		k = j;
	}
}

static int bpe_space(int c)
{
	return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31);
}

static int bpe_letter(int c)
{
	return ((c | 32) >= 'a' && (c | 32) <= 'z') || c >= 128;
}

static int bpe_digit(int c)
{
	return c >= '0' && c <= '9';
}

static int bpe_mark(int c) // [^\s\w]
{
	return !bpe_space(c) && !bpe_letter(c) && !bpe_digit(c) && c != '_';
}

static int bpe_under(int c)
{
	return c == '_';
}

// The end of tok.py's WORD match at s[i], in a piece of n bytes:
// 's|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+| ?_+|\s+(?!\S)|\s+
static int bpe_word_end(const uint8_t *s, int i, int n)
{
	static const char *const tail[7] = { "s", "t", "re", "ve", "m", "ll", "d" };
	static int (*const cls[4])(int) = { bpe_letter, bpe_digit, bpe_mark, bpe_under };
	int j;

	if (s[i] == '\'')
		for (int k = 0; k < 7; k++) {
			int l = (int)strlen(tail[k]);

			if (i + 1 + l <= n && !memcmp(s + i + 1, tail[k], l))
				return i + 1 + l;
		}
	for (int k = 0; k < 4; k++) {
		j = i + (s[i] == ' ');
		if (j < n && cls[k](s[j])) {
			while (j < n && cls[k](s[j]))
				j++;
			return j;
		}
	}
	for (j = i; j < n && bpe_space(s[j]); j++)
		;
	// At the end, or one space before a non-space, \s+ takes all; otherwise \s+(?!\S)
	// leaves the last space to the next word.
	return j == n || j - i < 2 ? j : j - 1;
}

// tok.py's text(): each digit stands alone, then WORD pieces, then BPE. Appends to out;
// returns the number of ids, or -1 if they do not fit in room.
static int bpe_text(const char *text, int32_t *out, int room)
{
	const uint8_t *s = (const uint8_t *)text;
	int len = (int)strlen(text), n = 0;

	for (int a = 0; a < len;) {
		int b = a + 1;

		if (!bpe_digit(s[a]))
			while (b < len && !bpe_digit(s[b]))
				b++;
		for (int i = 0; i < b - a;) {
			int e = bpe_word_end(s + a, i, b - a), m = bpe_word(s + a + i, e - i, out + n, room - n);

			if (m < 0)
				return -1;
			n += m;
			i = e;
		}
		a = b;
	}
	return n;
}

// The prompt after the image rows, as tok.py's prompt() builds it: the question and the
// short-answer instruction, <end_of_utterance>, then the assistant's turn.
static int bpe_question(const char *question, int32_t *out, int room)
{
	static char buf[1024];
	int n, m;

	if (snprintf(buf, sizeof(buf), "%s Answer using a single word or phrase.", question) >= (int)sizeof(buf))
		return -1;
	n = bpe_text(buf, out, room);
	if (n < 0 || n >= room)
		return -1;
	out[n++] = 49279;
	m = bpe_text("\nAssistant:", out + n, room - n);
	return m < 0 ? -1 : n + m;
}

#endif
