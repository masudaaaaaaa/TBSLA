#ifndef TBSLA_MPI_MatrixCSR
#define TBSLA_MPI_MatrixCSR

#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/mpi/Matrix.hpp>
#include <iostream>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixCSR : public tbsla::cpp::MatrixCSR, public tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
    void dense_multiply(const double* B_local, double* C_local, int B_cols, MPI_Comm comm);
    void reduce_row_sums(MPI_Comm comm, double* s);
    void reduce_row_max_abs(MPI_Comm comm, double* max_abs);
    void row_sum_reduction_for_dense_multiply(double* C_local, int ln_row, int B_cols, MPI_Comm row_comm);  
    std::ostream& print_dense(std::ostream& os, MPI_Comm comm);
    using tbsla::cpp::MatrixCSR::spmv;
    using tbsla::cpp::MatrixCSR::Ax;
    using tbsla::cpp::MatrixCSR::fill_cdiag;
    using tbsla::cpp::MatrixCSR::fill_cqmat;
    using tbsla::cpp::MatrixCSR::read;
    using tbsla::cpp::MatrixCSR::write;
    using tbsla::mpi::Matrix::spmv_no_redist;
    using tbsla::mpi::Matrix::spmv;
    using tbsla::mpi::Matrix::Ax;
    using tbsla::mpi::Matrix::Ax_;
    using tbsla::mpi::Matrix::a_axpx_;
private:
    void mpiio_read_lines(MPI_File &fh, int s, int n, int rowptr_start, int colidx_start, int values_start, size_t& mem_alloc);
    // void dense_multiply(const double* B_local, double* C_local, int B_cols, MPI_Comm comm);
};

}}

#endif
