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
  std::transform (buffer3, buffer3 + this->n_row, v, buffer3, std::plus<double>());
  for(int i = 0; i < this->ln_row; i++) {
    buffer[i] = 0;
    buffer2[i] = 0;
  }
  this->Ax(comm, r, buffer3 + this->f_col, buffer, buffer2, vect_incr);
}

inline void tbsla::mpi::Matrix::AAxpAxpx(MPI_Comm comm, double* r, const double* v, double* buffer, double* buffer2, double* buffer3, int vect_incr) {
  this->AAxpAx(comm, r, v, buffer, buffer2, buffer3, vect_incr);
  std::transform (r, r + this->n_row, v, r, std::plus<double>());
}
