#ifndef TBSLA_PETSC_Matrix
#define TBSLA_PETSC_Matrix

#include <vector>

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

namespace tbsla { namespace petsc {

class Matrix {
  public:
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult);
    Vec spmv(MPI_Comm comm, Vec &v);
    Vec a_axpx_(MPI_Comm comm, Vec &v);
    int get_n_row() { return n_row; }
    int get_n_col() { return n_col; }

  protected:
    Mat m;
    int n_row;
    int n_col;
};

}}

#endif
