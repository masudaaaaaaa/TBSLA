#ifndef TBSLA_CINTERFACE_CPP_MatrixCOO
#define TBSLA_CINTERFACE_CPP_MatrixCOO
#include <stdbool.h>
#include <tbsla/cpp/Vector.h>

#ifdef __cplusplus
extern "C" {
#endif

struct C_CPP_MatrixCOO;
typedef struct C_CPP_MatrixCOO C_CPP_MatrixCOO_t;

C_CPP_MatrixCOO_t *C_CPP_MatrixCOO_create();
void C_CPP_MatrixCOO_destroy(C_CPP_MatrixCOO_t *m);

void C_CPP_MatrixCOO_fill_cdiag(C_CPP_MatrixCOO_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC);
void C_CPP_MatrixCOO_fill_cqmat(C_CPP_MatrixCOO_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
bool C_CPP_MatrixCOO_read(C_CPP_MatrixCOO_t *m, char *filename, int seek);
bool C_CPP_MatrixCOO_write(C_CPP_MatrixCOO_t *m, char *filename);
void C_CPP_MatrixCOO_print(C_CPP_MatrixCOO_t *m);
C_CPP_Vector_t *C_CPP_MatrixCOO_spmv(C_CPP_MatrixCOO_t *m, C_CPP_Vector_t *v);

#ifdef __cplusplus
}
#endif

#endif /* TBSLA_C_MatrixCOO */
