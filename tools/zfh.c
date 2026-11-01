// SPDX-License-Identifier: MIT
// Check that the CPU runs Zfh: conversions and half-precision arithmetic.
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
	volatile float f = 0.333251953125f;
	volatile _Float16 a = 1.5f16, b = 0.25f16;
	_Float16 h = (_Float16)f;
	unsigned short bits;
	float back = h;
	int ok;

	memcpy(&bits, &h, sizeof(bits));
	ok = bits == 0x3555 && back == f;
	printf("convert: fp16 bits 0x%04x, back %.9g: %s\n", bits, back, ok ? "OK" : "WRONG");

	_Float16 s = a + b, d = a - b, m = a * b, q = a / b;
	_Float16 r = __builtin_sqrtf16(a * a);
	int ok2 = (float)s == 1.75f && (float)d == 1.25f && (float)m == 0.375f &&
		  (float)q == 6.0f && (float)r == 1.5f;
	printf("arith: %g %g %g %g %g: %s\n", (float)s, (float)d, (float)m, (float)q, (float)r,
	       ok2 ? "OK" : "WRONG");
	return !(ok && ok2);
}
