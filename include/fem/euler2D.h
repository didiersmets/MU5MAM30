#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void Euler2D_apply_transport(const uint32_t *indices, size_t tri_count,
			     const double *omega, const double *psi, size_t N,
			     double *out);
