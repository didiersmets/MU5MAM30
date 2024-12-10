#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi/stb_image_write.h"

#include "sys_utils.h"
#include "math_utils.h"
#include "sparse_matrix.h"

void spy(const SKLMatrix &A, uint32_t w, const char *fname)
{
	uint8_t *data = (uint8_t *)safe_malloc(w * w);
	for (size_t k = 0; k < w * w; ++k) {
		data[k] = 255;
	}

	uint32_t n = A.rows;
	for (uint32_t i = 0; i < n; ++i) {
		int py = (w * i) / n;
		uint32_t start = A.row_start[i];
		uint32_t end = A.row_start[i + 1];
		uint32_t j = A.jmin[i];
		for (uint32_t k = start; k < end; ++k, ++j) {
			uint32_t px = (w * j) / n;
			uint8_t *pix = data + w * py + px;
			if (A.data[k] != 0) {
				*pix = MIN(0, *pix);
			} /*else {
				*pix = MIN(192, *pix);
			}*/
		}
	}
	stbi_write_png(fname, w, w, 1, data, w);
}

