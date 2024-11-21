#include "array.h"
#include "matrix.h"

/* Compressed Row Storage spagre matrix */
struct CRSMatrix : public Matrix {
	size_t nnz; /* Number of (non zero) entries */
	// Non zero entries on line i (0 <= i < rows)
	// are stored at indices K(i) <= k < K(i + 1).
	// Column indices are read into J(k), entry values
	// are read into AIJ(k).
	TArray<int> K; /* Size = rows + 1 */
	TArray<int> J; /* Size = nnz  */
	TArray<double> AIJ; /* Size = nnz  */
	void mvp(const double *__restrict x, double *__restrict y) const;
	double sum() const;
};
