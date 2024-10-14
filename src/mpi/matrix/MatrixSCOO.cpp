#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <algorithm>
#include <vector>
#include <mpi.h>

#define TBSLA_MATRIX_COO_READ 2048

int tbsla::mpi::MatrixSCOO::read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);
  MPI_File_read_at_all(fh, 6 * sizeof(int), &this->gnnz, 1, MPI_LONG, &status);

  size_t vec_size, depla_general, values_start, row_start, col_start;
  depla_general = 10 * sizeof(int) + sizeof(long int);

  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);


  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  this->nnz = 0;
  values_start = depla_general;
  row_start = values_start + sizeof(size_t) + this->gnnz * sizeof(double);
  col_start = row_start + sizeof(size_t) + this->gnnz * sizeof(int);
  int c, r;

  size_t mem_alloc = this->gnnz / NR / NC;
  this->values = new double[mem_alloc];
  this->col = new int[mem_alloc];
  this->row = new int[mem_alloc];
  std::vector<int> ctmp(TBSLA_MATRIX_COO_READ);
  std::vector<int> rtmp(TBSLA_MATRIX_COO_READ);
  std::vector<double> vtmp(TBSLA_MATRIX_COO_READ);

  int mod = this->gnnz % TBSLA_MATRIX_COO_READ;
  MPI_File_read_at(fh, row_start, rtmp.data(), mod, MPI_INT, &status);
  MPI_File_read_at(fh, col_start, ctmp.data(), mod, MPI_INT, &status);
  MPI_File_read_at(fh, values_start, vtmp.data(), mod, MPI_DOUBLE, &status);
  for(int idx = 0; idx < mod; idx++) {
    r = rtmp[idx];
    c = ctmp[idx];
    if(this->nnz >= mem_alloc) {
      this->col = (int *)realloc(this->col, 2 * this->nnz * sizeof(int));
      this->row = (int *)realloc(this->row, 2 * this->nnz * sizeof(int));
      this->values = (double*)realloc(this->values, 2 * this->nnz * sizeof(double));
      mem_alloc = 2 * this->nnz;
    }
    if(r >= f_row && r < f_row + ln_row && c >= f_col && c < f_col + ln_col) {
      this->row[this->nnz] = r;
      this->col[this->nnz] = c;
      this->values[this->nnz] = vtmp[idx];
      this->nnz++;
    }
  }

  for(int i = mod; i < this->gnnz; i += TBSLA_MATRIX_COO_READ) {
    MPI_File_read_at(fh, row_start + i * sizeof(int), rtmp.data(), TBSLA_MATRIX_COO_READ, MPI_INT, &status);
    MPI_File_read_at(fh, col_start + i * sizeof(int), ctmp.data(), TBSLA_MATRIX_COO_READ, MPI_INT, &status);
    MPI_File_read_at(fh, values_start + i * sizeof(double), vtmp.data(), TBSLA_MATRIX_COO_READ, MPI_DOUBLE, &status);
    for(int idx = 0; idx < TBSLA_MATRIX_COO_READ; idx++) {
      r = rtmp[idx];
      c = ctmp[idx];
      if(this->nnz >= mem_alloc) {
        this->col = (int *)realloc(this->col, 2 * this->nnz * sizeof(int));
        this->row = (int *)realloc(this->row, 2 * this->nnz * sizeof(int));
        this->values = (double*)realloc(this->values, 2 * this->nnz * sizeof(double));
        mem_alloc = 2 * this->nnz;
      }
      if(r >= f_row && r < f_row + ln_row && c >= f_col && c < f_col + ln_col) {
        this->row[this->nnz] = r;
        this->col[this->nnz] = c;
        this->values[this->nnz] = vtmp[idx];
        this->nnz++;
      }
    }
  }

  return 0;
}

