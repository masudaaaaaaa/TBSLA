#ifndef TBSLA_MPI_MatrixDENSE
#define TBSLA_MPI_MatrixDENSE

#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/mpi/Matrix.hpp>
#include <iostream>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixDENSE : public tbsla::cpp::MatrixDENSE, public tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
    using tbsla::cpp::MatrixDENSE::spmv;
    using tbsla::cpp::MatrixDENSE::Ax;
    using tbsla::cpp::MatrixDENSE::fill_cdiag;
    using tbsla::cpp::MatrixDENSE::fill_cqmat;
    using tbsla::cpp::MatrixDENSE::read;
    using tbsla::cpp::MatrixDENSE::write;
    using tbsla::mpi::Matrix::spmv_no_redist;
    using tbsla::mpi::Matrix::spmv;
    using tbsla::mpi::Matrix::Ax;
    using tbsla::mpi::Matrix::Ax_;
    using tbsla::mpi::Matrix::a_axpx_;
};

}}

#endif
