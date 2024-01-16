#ifndef TBSLA_MPI_Matrix
#define TBSLA_MPI_Matrix

#include <tbsla/cpp/Matrix.hpp>

#include <fstream>
#include <unordered_map>
#include <vector>

#include <mpi.h>

namespace tbsla { namespace mpi {

class Matrix : public virtual tbsla::cpp::Matrix {
  public:
    virtual int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) = 0;
    virtual double* spmv(MPI_Comm comm, const double* v, int vect_incr = 0);
    virtual double* spmv_no_redist(MPI_Comm comm, const double* v, int vect_incr = 0);
    virtual inline void Ax(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, int vect_incr = 0);
    virtual inline void Ax_local(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, int vect_incr = 0);
    virtual inline double pagerank_normalization(MPI_Comm comm, double* b, double* b_t, double beta);
    virtual void pagerank_norma_end(MPI_Comm comm, double* b);
    virtual std::unordered_map<int,std::vector<int> > find_senders_receivers(MPI_Comm comm);
    virtual std::vector<MPI_Comm> create_comms(MPI_Comm comm, std::unordered_map<int,std::vector<int> > recv_map);
    virtual void redistribute_vector(std::vector<MPI_Comm> comms, double* r, const double* v, double* buffer, double* buffer2, std::unordered_map<int,std::vector<int> > recv_map);
    virtual inline void make_stochastic(MPI_Comm comm, double* s, double* buffer, double* buffer2);
    virtual inline void make_diagonally_dominant(MPI_Comm comm, double* s, double* buffer);
    virtual inline void Ax_(MPI_Comm comm, double* r, const double* v, int vect_incr = 0);
    virtual double* a_axpx_(MPI_Comm comm, const double* v, int vect_incr = 0);
    virtual inline void AAxpAx(MPI_Comm comm, double* r, const double* v, double *buffer, double* buffer2, double *buffer3, int vect_incr = 0);
    virtual inline void AAxpAxpx(MPI_Comm comm, double* r, const double* v, double *buffer, double* buffer2, double *buffer3, int vect_incr = 0);
    virtual double* page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done);
    virtual double * page_rank_opticom(int maxIter, double beta, double epsilon, int &nb_iterations_done);
    virtual double * conjugate_gradient_opticom(int maxIter, double beta, double epsilon, int &nb_iterations_done);
    virtual void CG(MPI_Comm comm, double* v, double* r, int max_iterations, int &nb_iterations_done);
    int const get_gnnz() {return gnnz;};
    long int const compute_sum_nnz(MPI_Comm comm);
    long int const compute_min_nnz(MPI_Comm comm);
    long int const compute_max_nnz(MPI_Comm comm);
    using tbsla::cpp::Matrix::fill_cdiag;
    using tbsla::cpp::Matrix::fill_cqmat;
    using tbsla::cpp::Matrix::fill_random;
    using tbsla::cpp::Matrix::fill_brain;
    using tbsla::cpp::Matrix::fill_cdistrib;
    using tbsla::cpp::Matrix::spmv;
    using tbsla::cpp::Matrix::Ax;
	using tbsla::cpp::Matrix::get_row_sums;
	using tbsla::cpp::Matrix::normalize_rows;
	using tbsla::cpp::Matrix::get_col_sums;
	using tbsla::cpp::Matrix::normalize_cols;
    using tbsla::cpp::Matrix::set_diag;
    using tbsla::cpp::Matrix::read;
    using tbsla::cpp::Matrix::write;

  protected:
    int gnnz;
};

}}

#endif
