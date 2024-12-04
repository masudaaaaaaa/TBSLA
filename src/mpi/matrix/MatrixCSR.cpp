#include <tbsla/mpi/MatrixCSR.hpp>
//#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/array.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include <mpi.h>  

#define TBSLA_MATRIX_CSR_READLINES 2048

void tbsla::mpi::MatrixCSR::dense_multiply(const double* B_local, double* C_local, int B_cols, MPI_Comm comm) {
    // Initialize C_local
    std::fill(C_local, C_local + this->ln_row * B_cols, 0.0);

    // Perform local sparse-dense multiplication
    for (int i = 0; i < this->ln_row; ++i) {
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; ++j) {
            int col = this->colidx[j];
            double value = this->values[j];
            for (int k = 0; k < B_cols; ++k) {
                C_local[i * B_cols + k] += value * B_local[col * B_cols + k];
            }
        }
    }

    // Reduce results along rows of processors
    double* C_reduced = (this->pc == 0) ? new double[this->ln_row * B_cols] : nullptr;
    MPI_Reduce(C_local, C_reduced, this->ln_row * B_cols, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (this->pc == 0) {
        // Save or process reduced results as needed
        // e.g., write to file or output the result
    }

    if (C_reduced) delete[] C_reduced;
}

void tbsla::mpi::MatrixCSR::compute_and_reduce_row_sum(MPI_Comm comm, double* s, double* global_s, int base) {
  MPI_Comm row_comm; // New communicator for the row
  int row_rank, row_size;

  // Create a communicator based on the row group 'pr'
  MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_size(row_comm, &row_size);

  // Compute local row sums
  //this->tbsla::cpp::MatrixCSR::get_row_sums(s);
  std::cout << "Computing row-sums on rows " << this->f_row << " to " << this->f_row + this->ln_row << std::endl;
  for (int i = 0; i < this->ln_row; i++) {
    double sum = 0, z = 0;
    if (this->rowptr[i + 1] != this->rowptr[i])
    for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
      z = this->values[j];
      if (z != 0) {
        if (base <= 0) {
            sum += std::exp(z);
        } else
          sum += std::pow(base, z);
      }
    }
    s[i] = sum;
  }

  // Reduce row sums within the row communicator
  for (int i = 0; i < this->ln_row; ++i) {
    std::cout << "Process (" << pr << ", " << pc << ") computed row sum[" << i << "]: " << s[i] << std::endl;
    if (NC > 1) {
      // Perform a reduction to calculate the global row sum within the row communicator
      MPI_Allreduce(&s[i], &global_s[i], 1, MPI_DOUBLE, MPI_SUM, row_comm);
    } else {
      // If NC == 1, the local sum is the global sum
      global_s[i] = s[i];
    }
    std::cout << "Process (" << pr << ", " << pc << ") computed global row sum[" << i << "]: " << global_s[i] << std::endl;
  }

  // Free the row communicator
  MPI_Comm_free(&row_comm);
}


std::ostream& tbsla::mpi::MatrixCSR::print_dense(std::ostream& os, MPI_Comm comm) {
    // Get the rank of the process and the number of processes
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Calculate local size
    int local_size = this->ln_row * this->ln_col;

    // Prepare local data array
    double* local_d = new double[local_size]();
    for (int i = 0; i < local_size; i++) {
        local_d[i] = 0;
    }

    if (this->nnz != 0) {
        for (int i = 0; i < this->ln_row; i++) {
            for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
                int global_col = this->pc * this->ln_col + this->colidx[j];
                if (global_col >= 0 && global_col < this->n_col) {  // Check for valid column index
                    local_d[i * this->ln_col + global_col] += this->values[j];
                }
            }
        }
    }

    // Allocate memory for recv_counts and displs on all processes
    int* recv_counts = new int[size];
    int* displs = new int[size];

    // Gather the local sizes of each process
    MPI_Gather(&local_size, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);

    // Calculate displacements on rank 0
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
    }

    // Allocate memory for the global array on rank 0
    double* global_d = nullptr;
    if (rank == 0) {
        int global_size = 0;
        for (int i = 0; i < size; i++) {
            global_size += recv_counts[i];
        }
        global_d = new double[global_size]();
    }

    // Gather the data from all processes
    MPI_Gatherv(local_d, local_size, MPI_DOUBLE,
                global_d, recv_counts, displs, MPI_DOUBLE,
                0, comm);

    // Print the global matrix on rank 0
    if (rank == 0) {
        tbsla::utils::array::print_dense_matrix(this->n_row, this->n_col, global_d, os);
        delete[] global_d;
        delete[] recv_counts;
        delete[] displs;
    }

    // Clean up local data
    delete[] local_d;
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
