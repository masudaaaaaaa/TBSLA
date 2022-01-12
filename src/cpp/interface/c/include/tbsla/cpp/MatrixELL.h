#ifndef TBSLA_CINTERFACE_CPP_MatrixELL
#define TBSLA_CINTERFACE_CPP_MatrixELL
#include <stdbool.h>
#include <tbsla/cpp/Vector.h>

#ifdef __cplusplus
extern "C" {
#endif

struct C_CPP_MatrixELL;
typedef struct C_CPP_MatrixELL C_CPP_MatrixELL_t;

C_CPP_MatrixELL_t *C_CPP_MatrixELL_create();
void C_CPP_MatrixELL_destroy(C_CPP_MatrixELL_t *m);

void C_CPP_MatrixELL_fill_cdiag(C_CPP_MatrixELL_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC);
void C_CPP_MatrixELL_fill_cqmat(C_CPP_MatrixELL_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
bool C_CPP_MatrixELL_read(C_CPP_MatrixELL_t *m, char *filename, int seek);
bool C_CPP_MatrixELL_write(C_CPP_MatrixELL_t *m, char *filename);
void C_CPP_MatrixELL_print(C_CPP_MatrixELL_t *m);
C_CPP_Vector_t *C_CPP_MatrixELL_spmv(C_CPP_MatrixELL_t *m, C_CPP_Vector_t *v);

#ifdef __cplusplus
}
#endif

#endif /* TBSLA_C_MatrixELL */
