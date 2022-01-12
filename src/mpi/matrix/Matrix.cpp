#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <mpi.h>
#include <iostream>

long int const tbsla::mpi::Matrix::compute_sum_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  return nnz;
}

long int const tbsla::mpi::Matrix::compute_min_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  return nnz;
}

long int const tbsla::mpi::Matrix::compute_max_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  return nnz;
}

double* tbsla::mpi::Matrix::spmv_no_redist(MPI_Comm comm, const double* v, int vect_incr) {
  return this->spmv(v, vect_incr);
}

inline void tbsla::mpi::Matrix::Ax_(MPI_Comm comm, double* r, const double* v, int vect_incr) {
  this->Ax(r, v, vect_incr);
}

/*
 * comm : MPI communicator
 * r : results (size : n_row)
 * v : input vector (size : n_col)
 * buffer : buffer for internal operations (size : ln_row)
 * buffer2 : buffer for internal operations (size : ln_row)
 *
 */
inline void tbsla::mpi::Matrix::Ax(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, int vect_incr) {
  this->Ax(buffer, v, vect_incr);
  if(this->NC == 1 && this->NR > 1) {
    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(buffer, this->ln_row, MPI_DOUBLE, r, recvcounts, displs, MPI_DOUBLE, comm);
  } else if(this->NC > 1 && this->NR == 1) {
    MPI_Allreduce(buffer, r, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    MPI_Allreduce(buffer, buffer2, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);

    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allgatherv(buffer2, this->ln_row, MPI_DOUBLE, r, recvcounts, displs, MPI_DOUBLE, col_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
  }
}



/*
 * comm : MPI communicator
 * s : sum of rows ; internal use (size : n_row)
 * buffer : buffer for internal operations (size : ln_row)
 * buffer2 : buffer for internal operations (size : ln_row)
 *
 */
/*inline void tbsla::mpi::Matrix::make_stochastic(MPI_Comm comm, double* s, double* buffer, double* buffer2) {
  this->get_row_sums(buffer);
  if(this->NC == 1 && this->NR > 1) {
	int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(buffer, this->ln_row, MPI_DOUBLE, s, recvcounts, displs, MPI_DOUBLE, comm);
  } else if(this->NC > 1 && this->NR == 1) {
    MPI_Allreduce(buffer, s, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    MPI_Allreduce(buffer, buffer2, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);

    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allgatherv(buffer2, this->ln_row, MPI_DOUBLE, s, recvcounts, displs, MPI_DOUBLE, col_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
  }
  this->normalize_rows(s);
}*/

/*
 * comm : MPI communicator
 * s : sum of cols ; internal use (size : n_col)
 * buffer : buffer for internal operations (size : ln_col)
 * buffer2 : buffer for internal operations (size : ln_col)
 *
 */
// normalize on columns instead

inline void tbsla::mpi::Matrix::make_stochastic(MPI_Comm comm, double* s, double* buffer, double* buffer2) {
  this->get_col_sums(buffer);
  std::cout << "computed col_sums" << std::endl;
  if(this->NR == 1) {
    for(int k=0; k<this->ln_col; k++)
      s[k] = buffer[k];
  }
  else if(this->NC == 1 && this->NR > 1) {
    MPI_Allreduce(buffer, s, this->n_col, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    std::cout << "NR > 1 and NC > 1" << std::endl;
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allreduce(buffer, s, this->ln_col, MPI_DOUBLE, MPI_SUM, col_comm);

    MPI_Comm_free(&col_comm);
    std::cout << "end" << std::endl;
  }
  double tot = 0;
  for(int i=0; i<this->ln_col; i++) {
    tot += s[i];
  }
  std::cout << "tot = " << tot << std::endl;
  this->normalize_cols(s);
}

double* tbsla::mpi::Matrix::spmv(MPI_Comm comm, const double* v, int vect_incr) {
  double* send = this->spmv(v, vect_incr);
  if(this->NC == 1 && this->NR == 1) {
    return send;
  } else if(this->NC == 1 && this->NR > 1) {
    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    double* recv = new double[this->n_row]();
    MPI_Allgatherv(send, this->ln_row, MPI_DOUBLE, recv, recvcounts, displs, MPI_DOUBLE, comm);
    return recv;
  } else if(this->NC > 1 && this->NR == 1) {
    double* recv = new double[this->n_row]();
    MPI_Allreduce(send, recv, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
    return recv;
  } else {
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    double* recv = new double[this->ln_row]();
    MPI_Allreduce(send, recv, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);

    double* recv2 = new double[this->n_row]();
    int* recvcounts = new int[this->NR]();
    int* displs = new int[this->NR]();
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
      displs[i] = 0;
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allgatherv(recv, this->ln_row, MPI_DOUBLE, recv2, recvcounts, displs, MPI_DOUBLE, col_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
    return recv2;
  }
}

double* tbsla::mpi::Matrix::a_axpx_(MPI_Comm comm, const double* v, int vect_incr) {
  double* r = this->spmv(comm, v + this->f_col, vect_incr);
  std::transform (r, r + this->n_row, v, r, std::plus<double>());
  double* r2 = this->spmv(comm, r + this->f_col, vect_incr);
  delete[] r;
  return r2;
}

inline void tbsla::mpi::Matrix::AAxpAx(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, double* buffer3, int vect_incr) {
  this->Ax(comm, buffer3, v + this->f_col, buffer, buffer2, vect_incr);
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < this->n_row; i++) {
    buffer3[i] += v[i];
  }
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < this->ln_row; i++) {
    buffer[i] = 0;
    buffer2[i] = 0;
  }
  this->Ax(comm, r, buffer3 + this->f_col, buffer, buffer2, vect_incr);
}

inline void tbsla::mpi::Matrix::AAxpAxpx(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, double* buffer3, int vect_incr) {
  this->AAxpAx(comm, r, v, buffer, buffer2, buffer3, vect_incr);
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < this->n_row; i++) {
    r[i] += v[i];
  }
}

double* tbsla::mpi::Matrix::page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done){
  //std::cout << "PageRank" << std::endl;
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  double* b = new double[n_col];
  double* buf1 = new double[ln_row];
  double* buf2 = new double[ln_row];
  double* b_t = new double[n_col];
  //std::cout << "outer-one" << std::endl;
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < n_col; i++){
    b[i] = 1;
  }
  bool converge = false;
  int nb_iterations = 0;
  double max_val, error, teleportation_sum;

  while(!converge && nb_iterations < max_iterations){
    //std::cout << "inner-one" << std::endl;
    //std::cout << nb_iterations << std::endl;
    #pragma omp parallel for schedule(static)
    for(int i = 0 ; i < n_col; i++) {
      b_t[i] = b[i];
      b[i] = 0;
    }
    //std::cout << "inner-two" << std::endl;
    #pragma omp parallel for schedule(static)
    for(int i = 0 ; i < ln_row; i++) {
      buf1[i] = 0;
      buf2[i] = 0;
    }
    //std::cout << "inner-three" << std::endl;
    this->Ax(comm, b, b_t + f_col, buf1, buf2);
    //std::cout << "inner-four" << std::endl;
    #pragma omp parallel for reduction(+ : teleportation_sum) schedule(static)
    for(int i = 0 ; i < n_col; i++){
      teleportation_sum += b_t[i];
    }
    teleportation_sum *= (1-beta)/n_col ;

    b[0] = beta*b[0] + teleportation_sum;
    max_val = b[0];
	error = 0.0;
    //std::cout << "inner-five" << std::endl;
    #pragma omp parallel for reduction(max : max_val) schedule(static)
    for(int  i = 1 ; i < n_col;i++){
      b[i] = beta*b[i] + teleportation_sum;
      if(max_val < b[i])
        max_val = b[i];
      //error += std::abs(b[i]- b_t[i]);
    }

	// no need to normalize here
    error = 0.0;
    //std::cout << "inner-six" << std::endl;
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

  //std::cout << "outer-two" << std::endl;
  double sum = b[0];
  #pragma omp parallel for reduction(+ : sum) schedule(static)
  for(int i = 1; i < n_col; i++) {
    sum += b[i];
  }

  //std::cout << "outer-three" << std::endl;
  #pragma omp parallel for schedule(static)
  for(int i = 0 ; i < n_col; i++) {
    b[i] = b[i]/sum;
  }
  delete[] b_t;
  delete[] buf1;
  delete[] buf2;
  return b;
}
