#ifndef TBSLA_MPI_Matrix
#define TBSLA_MPI_Matrix

#include <tbsla/cpp/Matrix.hpp>

#include <fstream>

#include <mpi.h>

namespace tbsla { namespace mpi {

class Matrix : public virtual tbsla::cpp::Matrix {
  public:
    virtual int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) = 0;
    virtual double* spmv(MPI_Comm comm, const double* v, int vect_incr = 0);
    virtual double* spmv_no_redist(MPI_Comm comm, const double* v, int vect_incr = 0);
    virtual inline void Ax(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, int vect_incr = 0);
    virtual inline void Ax_(MPI_Comm comm, double* r, const double* v, int vect_incr = 0);
    virtual double* a_axpx_(MPI_Comm comm, const double* v, int vect_incr = 0);
    virtual inline void AAxpAx(MPI_Comm comm, double* r, const double* v, double *buffer, double* buffer2, double *buffer3, int vect_incr = 0);
    virtual inline void AAxpAxpx(MPI_Comm comm, double* r, const double* v, double *buffer, double* buffer2, double *buffer3, int vect_incr = 0);
    virtual double* page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done);
    virtual void CG(MPI_Comm comm, double* v, double* r, int max_iterations, int &nb_iterations_done);
    int const get_gnnz() {return gnnz;};
    long int const compute_sum_nnz(MPI_Comm comm);
    long int const compute_min_nnz(MPI_Comm comm);
    long int const compute_max_nnz(MPI_Comm comm);
    using tbsla::cpp::Matrix::fill_cdiag;
    using tbsla::cpp::Matrix::fill_cqmat;
    using tbsla::cpp::Matrix::spmv;
    using tbsla::cpp::Matrix::Ax;
    using tbsla::cpp::Matrix::read;
    using tbsla::cpp::Matrix::write;

  protected:
    int gnnz;
};

}}

#endif
