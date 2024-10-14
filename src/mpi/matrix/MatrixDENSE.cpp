#include <tbsla/mpi/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <mpi.h>

int tbsla::mpi::MatrixDENSE::read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);
  MPI_File_read_at_all(fh, 6 * sizeof(int), &this->gnnz, 1, MPI_LONG, &status);

  size_t vec_size, depla_general;
  depla_general = 10 * sizeof(int) + sizeof(long int);

  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  int values_size = vec_size;
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  this->nnz = ln_row * ln_col;

  if(this->values)
    delete[] this->values;
  this->values = new double[this->nnz];
  for(int i = 0; i < this->ln_row; i++) {
    int idx, val, jmin, jmax;
    MPI_File_read_at(fh, depla_general + ((this->f_row + i) * this->n_col + this->f_col) * sizeof(double), this->values + i * ln_col, ln_col, MPI_DOUBLE, &status);
  }

  MPI_File_close(&fh);
  return 0;
}

