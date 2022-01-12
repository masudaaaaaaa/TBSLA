#ifndef TBSLA_CINTERFACE_CPP_MatrixCSR
#define TBSLA_CINTERFACE_CPP_MatrixCSR
#include <stdbool.h>
#include <tbsla/cpp/Vector.h>

#ifdef __cplusplus
extern "C" {
#endif

struct C_CPP_MatrixCSR;
typedef struct C_CPP_MatrixCSR C_CPP_MatrixCSR_t;

C_CPP_MatrixCSR_t *C_CPP_MatrixCSR_create();
void C_CPP_MatrixCSR_destroy(C_CPP_MatrixCSR_t *m);

void C_CPP_MatrixCSR_fill_cdiag(C_CPP_MatrixCSR_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC);
void C_CPP_MatrixCSR_fill_cqmat(C_CPP_MatrixCSR_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
bool C_CPP_MatrixCSR_read(C_CPP_MatrixCSR_t *m, char *filename, int seek);
bool C_CPP_MatrixCSR_write(C_CPP_MatrixCSR_t *m, char *filename);
void C_CPP_MatrixCSR_print(C_CPP_MatrixCSR_t *m);
C_CPP_Vector_t *C_CPP_MatrixCSR_spmv(C_CPP_MatrixCSR_t *m, C_CPP_Vector_t *v);

#ifdef __cplusplus
}
#endif

#endif /* TBSLA_C_MatrixCSR */
