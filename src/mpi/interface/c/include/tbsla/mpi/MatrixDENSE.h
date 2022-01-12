#ifndef TBSLA_CINTERFACE_MPI_MatrixDENSE
#define TBSLA_CINTERFACE_MPI_MatrixDENSE
#include <stdbool.h>
#include <tbsla/cpp/Vector.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

struct C_MPI_MatrixDENSE;
typedef struct C_MPI_MatrixDENSE C_MPI_MatrixDENSE_t;

C_MPI_MatrixDENSE_t *C_MPI_MatrixDENSE_create();
void C_MPI_MatrixDENSE_destroy(C_MPI_MatrixDENSE_t *m);

void C_MPI_MatrixDENSE_fill_cdiag(C_MPI_MatrixDENSE_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC);
void C_MPI_MatrixDENSE_fill_cqmat(C_MPI_MatrixDENSE_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
bool C_MPI_MatrixDENSE_read(C_MPI_MatrixDENSE_t *m, char *filename, int seek);
bool C_MPI_MatrixDENSE_write(C_MPI_MatrixDENSE_t *m, char *filename);
void C_MPI_MatrixDENSE_print(C_MPI_MatrixDENSE_t *m);
C_CPP_Vector_t *C_MPI_MatrixDENSE_spmv(C_MPI_MatrixDENSE_t *m, MPI_Comm comm, C_CPP_Vector_t *v);

#ifdef __cplusplus
}
#endif

#endif /* TBSLA_C_MatrixDENSE */
