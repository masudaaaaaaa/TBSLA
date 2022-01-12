#ifndef TBSLA_MPI_MatrixSCOO
#define TBSLA_MPI_MatrixSCOO

#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/cpp/MatrixSCOO.hpp>
#include <iostream>
#include <fstream>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixSCOO : public tbsla::cpp::MatrixSCOO, public virtual tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
    using tbsla::cpp::MatrixSCOO::spmv;
    using tbsla::cpp::MatrixSCOO::Ax;
    using tbsla::cpp::MatrixSCOO::fill_cdiag;
    using tbsla::cpp::MatrixSCOO::fill_cqmat;
    using tbsla::cpp::MatrixSCOO::read;
    using tbsla::cpp::MatrixSCOO::write;
    using tbsla::mpi::Matrix::spmv_no_redist;
    using tbsla::mpi::Matrix::spmv;
    using tbsla::mpi::Matrix::Ax;
    using tbsla::mpi::Matrix::Ax_;
    using tbsla::mpi::Matrix::a_axpx_;
};

}}

#endif
