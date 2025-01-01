#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/array.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <mpi.h>  

#define TBSLA_MATRIX_CSR_READLINES 2048

void tbsla::mpi::MatrixCSR::dense_multiply(const double* B_local, double* C_local, int B_cols, MPI_Comm comm) {
    std::fill(C_local, C_local + this->ln_row * B_cols, 0.0); // Clear C_local

    for (int i = 0; i < this->ln_row; ++i) { // Iterate over rows
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; ++j) { // Non-zero elements
            int global_col = this->colidx[j]; // Global column index
            if (global_col >= this->f_col && global_col < this->f_col + this->ln_col) {
                int local_col = global_col - this->f_col; // Map global to local index
                for (int k = 0; k < B_cols; ++k) {
                    C_local[i * B_cols + k] += this->values[j] * B_local[local_col * B_cols + k];
                }
            }
        }
    }
}


void tbsla::mpi::MatrixCSR::row_sum_reduction(double* C_local, int ln_row, int B_cols, MPI_Comm row_comm) {
    // Perform in-place all-reduce operation for summing corresponding elements
    MPI_Allreduce(MPI_IN_PLACE, C_local, ln_row * B_cols, MPI_DOUBLE, MPI_SUM, row_comm);
}



void tbsla::mpi::MatrixCSR::reduce_row_max_abs(MPI_Comm comm, double* max_abs) {
    // Create a row communicator based on the process's row index (pr)
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    std::cout << "Reducing max absolute values..." << std::endl;

    // Temporary array to store reduced max values
    double* global_max = new double[this->ln_row];

    // Perform reduction (max) within the row communicator
    MPI_Allreduce(max_abs, global_max, this->ln_row, MPI_DOUBLE, MPI_MAX, row_comm);

    // Copy the reduced values back to max_abs
    std::copy(global_max, global_max + this->ln_row, max_abs);

    // Output the reduced max absolute values
    for (int i = 0; i < this->ln_row; ++i) {
        std::cout << "Row " << i + this->f_row << " -> Global max_abs: " << max_abs[i] << std::endl;
    }

    delete[] global_max;

    // Free the row communicator
    MPI_Comm_free(&row_comm);
}

void tbsla::mpi::MatrixCSR::reduce_row_sums(MPI_Comm comm, double* s) {
    // Create a row communicator based on the process's row index (pr)
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    std::cout << "Reducing row sums..." << std::endl;

    // Temporary array to store reduced values
    double* global_s = new double[this->ln_row];

    // Perform reduction (sum) within the row communicator
    MPI_Allreduce(s, global_s, this->ln_row, MPI_DOUBLE, MPI_SUM, row_comm);

    // Copy the reduced values back to (s)
    std::copy(global_s, global_s + this->ln_row, s);

    // Output the reduced row sums
    for (int i = 0; i < this->ln_row; ++i) {
        std::cout << "Row " << i + this->f_row << " -> Global sum: " << s[i] << std::endl;
    }

    delete[] global_s;

    // Free the row communicator
    MPI_Comm_free(&row_comm);
}


std::ostream& tbsla::mpi::MatrixCSR::print_dense(std::ostream& os, MPI_Comm comm) {
    // Get the rank of the process and the number of processes
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Prepare global data array (d) and initialize to zero
    double* d = new double[this->n_row * this->n_col]();
    
    // Populate the global data array for the local process's contribution
    if (this->nnz != 0) {
        for (int i = 0; i < this->ln_row; i++) {
            for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
                d[(this->f_row + i) * this->n_col + this->colidx[j]] += this->values[j];
            }
        }
    }

    // Rank 0 gathers data from all other ranks
    if (rank == 0) {
        // Receive and integrate contributions from other ranks
        for (int p = 1; p < size; p++) {
            double* recv_d = new double[this->n_row * this->n_col]();
            MPI_Recv(recv_d, this->n_row * this->n_col, MPI_DOUBLE, p, 0, comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < this->n_row * this->n_col; i++) {
                d[i] += recv_d[i];
            }
            delete[] recv_d;
        }

        // Print the global matrix
        tbsla::utils::array::print_dense_matrix(this->n_row, this->n_col, d, os);
    } else {
        // Other ranks send their local contributions to rank 0
        MPI_Send(d, this->n_row * this->n_col, MPI_DOUBLE, 0, 0, comm);
    }

    // Clean up
    delete[] d;
    return os;
}


int tbsla::mpi::MatrixCSR::read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);
  MPI_File_read_at_all(fh, 6 * sizeof(int), &this->gnnz, 1, MPI_LONG, &status);

  size_t vec_size, depla_general, values_start;
  depla_general = 10 * sizeof(int) + sizeof(long int);

  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  // skip values vector for now
  int values_size = vec_size;
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  values_start = depla_general;
  depla_general += vec_size * sizeof(double);

  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  if (this->rowptr)
    delete[] this->rowptr;
  this->rowptr = new int[this->ln_row + 1];
  int rowptr_start = depla_general + this->f_row * sizeof(int);
  depla_general += vec_size * sizeof(int);

  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  int colidx_start = depla_general;

  this->rowptr[0] = 0;
  this->nnz = 0;
  size_t mem_alloc = this->ln_row * 10;
  this->colidx = new int[mem_alloc];
  this->values = new double[mem_alloc];
  int mod = this->ln_row % TBSLA_MATRIX_CSR_READLINES;
  tbsla::mpi::MatrixCSR::mpiio_read_lines(fh, 0, mod, rowptr_start, colidx_start, values_start, mem_alloc);
  for(int i = mod; i < this->ln_row; i += TBSLA_MATRIX_CSR_READLINES) {
    tbsla::mpi::MatrixCSR::mpiio_read_lines(fh, i, TBSLA_MATRIX_CSR_READLINES, rowptr_start, colidx_start, values_start, mem_alloc);
  }
  MPI_File_close(&fh);
  return 0;
}

void tbsla::mpi::MatrixCSR::mpiio_read_lines(MPI_File &fh, int s, int n, int rowptr_start, int colidx_start, int values_start, size_t& mem_alloc) {
  MPI_Status status;
  std::vector<int> jtmp(n + 1);
  int idx, jmin, jmax, nv;
  MPI_File_read_at(fh, rowptr_start + s * sizeof(int), jtmp.data(), n + 1, MPI_INT, &status);
  jmin = jtmp[0];
  jmax = jtmp[n];
  nv = jmax - jmin;
  std::vector<int> ctmp(nv);
  std::vector<double> vtmp(nv);
  MPI_File_read_at(fh, colidx_start + jmin * sizeof(int), ctmp.data(), nv, MPI_INT, &status);
  MPI_File_read_at(fh, values_start + jmin * sizeof(double), vtmp.data(), nv, MPI_DOUBLE, &status);
  int incr = 0;
  for(int i = 0; i < n; i++) {
    jmin = jtmp[i];
    jmax = jtmp[i + 1];
    nv = jmax - jmin;
    for(int j = incr; j < incr + nv; j++) {
      idx = ctmp[j];
      if(this->nnz >= mem_alloc) {
        this->colidx = (int*)realloc(this->colidx, 2 * this->nnz * sizeof(int));
        this->values = (double*)realloc(this->values, 2 * this->nnz * sizeof(double));
        mem_alloc = 2 * this->nnz;
      }
      if(idx >= this->f_col && idx < this->f_col + this->ln_col) {
        this->colidx[this->nnz] = idx;
        this->values[this->nnz] = vtmp[j];
        this->nnz++;
      }
    }
    incr += nv;
    this->rowptr[s + i + 1] = this->nnz;
  }
}
