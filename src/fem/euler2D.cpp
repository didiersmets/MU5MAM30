#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void Euler2D_apply_transport(const uint32_t *indices, size_t tri_count,
			     const double *omega, const double *psi, size_t N,
			     double *out)
{
	memset(out, 0, N * sizeof(double));

	for (size_t t = 0; t < tri_count; t++) {
		uint32_t a = indices[3 * t + 0];
		uint32_t b = indices[3 * t + 1];
		uint32_t c = indices[3 * t + 2];
		assert(a < N && b < N && c < N);
		double sum = omega[a] + omega[b] + omega[c];
		out[a] += sum * (psi[b] - psi[c]);
		out[b] += sum * (psi[c] - psi[a]);
		out[c] += sum * (psi[a] - psi[b]);
	}
	for (size_t v = 0; v < N; v++) {
		out[v] *= 1.0 / 12;
	}
}

void euler_evolve();

