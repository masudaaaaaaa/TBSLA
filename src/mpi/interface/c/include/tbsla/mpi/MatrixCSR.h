#ifndef TBSLA_CINTERFACE_MPI_MatrixCSR
#define TBSLA_CINTERFACE_MPI_MatrixCSR
#include <stdbool.h>
#include <tbsla/cpp/Vector.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

struct C_MPI_MatrixCSR;
typedef struct C_MPI_MatrixCSR C_MPI_MatrixCSR_t;

C_MPI_MatrixCSR_t *C_MPI_MatrixCSR_create();
void C_MPI_MatrixCSR_destroy(C_MPI_MatrixCSR_t *m);

void C_MPI_MatrixCSR_fill_cdiag(C_MPI_MatrixCSR_t *m, int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC);
void C_MPI_MatrixCSR_fill_cqmat(C_MPI_MatrixCSR_t *m, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
bool C_MPI_MatrixCSR_read(C_MPI_MatrixCSR_t *m, char *filename, int seek);
bool C_MPI_MatrixCSR_write(C_MPI_MatrixCSR_t *m, char *filename);
void C_MPI_MatrixCSR_print(C_MPI_MatrixCSR_t *m);
C_CPP_Vector_t *C_MPI_MatrixCSR_spmv(C_MPI_MatrixCSR_t *m, MPI_Comm comm, C_CPP_Vector_t *v);

#ifdef __cplusplus
}
#endif

#endif /* TBSLA_C_MatrixCSR */
