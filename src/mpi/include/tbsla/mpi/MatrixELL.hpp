#ifndef TBSLA_MPI_MatrixELL
#define TBSLA_MPI_MatrixELL

#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/mpi/Matrix.hpp>
#include <iostream>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixELL : public tbsla::cpp::MatrixELL, public tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC);
    void mpiio_read_lines(MPI_File &fh, int s, int n, int columns_start, int values_start);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
    using tbsla::cpp::MatrixELL::spmv;
    using tbsla::cpp::MatrixELL::Ax;
    using tbsla::cpp::MatrixELL::fill_cdiag;
    using tbsla::cpp::MatrixELL::fill_cqmat;
    using tbsla::cpp::MatrixELL::read;
    using tbsla::cpp::MatrixELL::write;
    using tbsla::mpi::Matrix::spmv_no_redist;
    using tbsla::mpi::Matrix::spmv;
    using tbsla::mpi::Matrix::Ax;
    using tbsla::mpi::Matrix::Ax_;
    using tbsla::mpi::Matrix::a_axpx_;
};

}}

#endif
