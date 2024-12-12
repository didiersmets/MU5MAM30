#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi/stb_image_write.h"

#include "math_utils.h"
#include "sparse_matrix.h"
#include "sys_utils.h"

void spy(const SKLMatrix &A, uint32_t w, const char *fname)
{
	uint8_t *data = (uint8_t *)safe_malloc(w * w);
	for (size_t k = 0; k < w * w; ++k) {
		data[k] = 255;
	}

	uint32_t n = A.rows;
	float invn = 1.f / n;
	for (uint32_t i = 0; i < n; ++i) {
		uint32_t py = (int)round(float(i) * invn * w);
		uint32_t start = A.row_start[i];
		uint32_t end = A.row_start[i + 1];
		uint32_t j = A.jmin[i];
		for (uint32_t k = start; k < end; ++k, ++j) {
			uint32_t px = (int)round(float(j) * invn * w);
			uint8_t *pix = data + w * py + px;
			if (A.data[k] != 0) {
				*pix = MIN(0, *pix);
			} else {
				*pix = MIN(128, *pix);
			}
		}
	}
	stbi_write_png(fname, w, w, 1, data, w);
}

void spy(const CSRPattern &P, uint32_t w, const char *fname)
{
	/* TODO non symmetric case too ? */

	uint8_t *data = (uint8_t *)safe_malloc(w * w);
	for (size_t k = 0; k < w * w; ++k) {
		data[k] = 255;
	}

	uint32_t n = P.rows;
	float invn = 1.f / n;
	for (size_t i = 0; i < n; ++i) {
		uint32_t py = (int)round(float(i) * invn * w);
		uint32_t start = P.row_start[i];
		uint32_t end = P.row_start[i + 1];
		for (uint32_t k = start; k < end; ++k) {
			uint32_t j = P.col[k];
			uint32_t px = (int)round(float(j) * invn * w);
			uint8_t *pix = data + w * py + px;
			*pix = 0;
		}
	}
	stbi_write_png(fname, w, w, 1, data, w);
}

