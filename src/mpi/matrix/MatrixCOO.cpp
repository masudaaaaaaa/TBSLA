#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <mpi.h>

int tbsla::mpi::MatrixCOO::read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);
  MPI_File_read_at_all(fh, 6 * sizeof(int), &this->gnnz, 1, MPI_LONG, &status);

  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;
  this->ln_row = this->n_row;
  this->ln_col = this->n_col;
  this->f_row = 0;
  this->f_col = 0;

  size_t vec_size, depla_general, depla_local;
  depla_general = 10 * sizeof(int) + sizeof(long int);

  /*
  *
  * read values vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  int s = tbsla::utils::range::pflv(vec_size, pr * NC + pc, NR * NC);
  this->nnz = tbsla::utils::range::lnv(vec_size, pr * NC + pc, NR * NC);

  if (this->values)
    delete[] this->values;
  this->values = new double[this->nnz]();
  depla_local = depla_general + s * sizeof(double);
  MPI_File_read_at_all(fh, depla_local, this->values, this->nnz, MPI_DOUBLE, &status);
  depla_general += vec_size * sizeof(double);

  /*
  *
  * read row vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  if (this->row)
    delete[] this->row;
  this->row = new int[this->nnz]();
  depla_local = depla_general + s * sizeof(int);
  MPI_File_read_at_all(fh, depla_local, this->row, this->nnz, MPI_INT, &status);
  depla_general += vec_size * sizeof(int);

  /*
  *
  * read col vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  if (this->col)
    delete[] this->col;
  this->col = new int[this->nnz]();
  depla_local = depla_general + s * sizeof(int);
  MPI_File_read_at_all(fh, depla_local, this->col, this->nnz, MPI_INT, &status);

  MPI_File_close(&fh);
  return 0;
}

double* tbsla::mpi::MatrixCOO::spmv(MPI_Comm comm, const double* v, int vect_incr) {
  double* send = this->spmv(v, vect_incr);
  double* recv = new double[this->n_row]();
  MPI_Allreduce(send, recv, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
  delete[] send;
  return recv;
}

/*
 * comm : MPI communicator
 * r : results (size : n_row)
 * v : input vector (size : n_col)
 * buffer : buffer for internal operations (size : ln_row(=n_row))
 * buffer2 : unused, for consistency with other formats
 *
 */
inline void tbsla::mpi::MatrixCOO::Ax(MPI_Comm comm, double* r, const double* v, double *buffer, double *buffer2, int vect_incr) {
  this->Ax(buffer, v, vect_incr);
  MPI_Allreduce(buffer, r, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
}

double* tbsla::mpi::MatrixCOO::a_axpx_(MPI_Comm comm, const double* v, int vect_incr) {
  double* r = this->spmv(comm, v, vect_incr);
  std::transform (r, r + this->n_row, v, r, std::plus<double>());
  return this->spmv(comm, r, vect_incr);
}

double* tbsla::mpi::MatrixCOO::page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done){
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  double* b = new double[n_col];
  double* buf1 = new double[ln_row];
  double* buf2 = new double[ln_row];
  double* b_t = new double[n_col];
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < n_col; i++){
    b[i] = 1;
  }
  bool converge = false;
  int nb_iterations = 0;
  double max_val, error, teleportation_sum;

  while(!converge && nb_iterations < max_iterations){
    #pragma omp parallel for schedule(static)
    for(int i = 0 ; i < n_col; i++) {
      b_t[i] = b[i];
      b[i] = 0;
      buf1[i] = 0;
      buf2[i] = 0;
    }
    this->Ax(comm, b, b_t, buf1, buf2);
    #pragma omp parallel for reduction(+ : teleportation_sum) schedule(static)
    for(int i = 0 ; i < n_col; i++){
      teleportation_sum += b_t[i];
    }
    teleportation_sum *= (1-beta)/n_col ;

    b[0] = beta*b[0] + teleportation_sum;
    max_val = b[0];
    #pragma omp parallel for reduction(max : max_val) schedule(static)
    for(int  i = 1 ; i < n_col;i++){
      b[i] = beta*b[i] + teleportation_sum;
      if(max_val < b[i])
        max_val = b[i];
    }

    error = 0.0;
    #pragma omp parallel for reduction(+ : error) schedule(static)
    for (int i = 0;i< n_col;i++){
      b[i] = b[i]/max_val;
      error += std::abs(b[i]- b_t[i]);
    }
    if(error < epsilon){
      converge = true;
    }
    nb_iterations++;
  }

  nb_iterations_done = nb_iterations;

  double sum = b[0];
  #pragma omp parallel for reduction(+ : sum) schedule(static)
  for(int i = 1; i < n_col; i++) {
    sum += b[i];
  }

  #pragma omp parallel for schedule(static)
  for(int i = 0 ; i < n_col; i++) {
    b[i] = b[i]/sum;
  }
  delete[] b_t;
  delete[] buf1;
  delete[] buf2;
  return b;
}

