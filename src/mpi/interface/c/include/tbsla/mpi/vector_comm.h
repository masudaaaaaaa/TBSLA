#ifndef TBSLA_CINTERFACE_MPI_vector_comm
#define TBSLA_CINTERFACE_MPI_vector_comm
#include <stdbool.h>
#include <tbsla/cpp/Vector.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

C_CPP_Vector_t *C_MPI_allgatherv(MPI_Comm comm, C_CPP_Vector_t *v, int bn_row, int lgr);
C_CPP_Vector_t *C_MPI_reduce_sum(MPI_Comm comm, C_CPP_Vector_t *v, int n);
C_CPP_Vector_t *C_MPI_reduce_gather(MPI_Comm comm, C_CPP_Vector_t *v, int bn_row, int bn_col, int lpr, int lpc, int lgr, int lgc);
C_CPP_Vector_t *C_MPI_redistribute(MPI_Comm comm, C_CPP_Vector_t *v, int bn_row, int bn_col, int lpr, int lpc, int lgr, int lgc);

#ifdef __cplusplus
}
#endif

#endif /* TBSLA_CINTERFACE_MPI_vector_comm */
