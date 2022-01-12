#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <vector>
#include <mpi.h>

#define TBSLA_MATRIX_CSR_READLINES 2048

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

