#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/mpi/MatrixDENSE.hpp>

#include <tbsla/cpp/utils/array.hpp>

#include <mpi.h>

#include <numeric>
#include <iostream>

void test_matrix_split_vector(tbsla::mpi::Matrix & m, int nr, int nc, int cdiag, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  m.fill_cdiag(nr, nc, cdiag, pr, pc, NR, NC);

  double* v = new double[nc];
  std::iota (v, v + nc, 0);
  double* vl = new double[m.get_ln_col()];
  std::iota (vl, vl + m.get_ln_col(), m.get_f_col());
  double* r = m.spmv(MPI_COMM_WORLD, vl);
  int res = tbsla::utils::array::test_spmv_cdiag(nr, nc, cdiag, v, r, false);
  delete[] r;
  int res0;
  MPI_Allreduce(&res, &res0, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "return : " << res0 << std::endl;
  }
  if(res0) {
    int res;
    tbsla::utils::array::stream<double>(std::cout, "vl ", vl, m.get_ln_col());
    std::cout << std::endl;
    double* r = m.spmv(MPI_COMM_WORLD, vl);
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << m << std::endl;
        res = tbsla::utils::array::test_spmv_cdiag(nr, nc, cdiag, v, r, true);
        tbsla::utils::array::stream<double>(std::cout, "r ", r, nr);
        std::cout << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    delete[] r;
    if(rank == 0) {
      tbsla::cpp::MatrixCOO ml;
      ml.fill_cdiag(nr, nc, cdiag);
    }
    exit(res);
  }
  delete[] v;
  delete[] vl;
  MPI_Barrier(MPI_COMM_WORLD);
}


void test_matrix(tbsla::mpi::Matrix & m, int nr, int nc, int cdiag, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  m.fill_cdiag(nr, nc, cdiag, pr, pc, NR, NC);

  double* v = new double[nc];
  std::iota (v, v + nc, 0);
  double* r = m.spmv(MPI_COMM_WORLD, v);
  int res = tbsla::utils::array::test_spmv_cdiag(nr, nc, cdiag, v, r, false);
  delete[] r;
  int res0;
  MPI_Allreduce(&res, &res0, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "return : " << res0 << std::endl;
  }
  if(res0) {
    int res;
    double* r = m.spmv(MPI_COMM_WORLD, v);
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << m << std::endl;
        res = tbsla::utils::array::test_spmv_cdiag(nr, nc, cdiag, v, r, true);
        tbsla::utils::array::stream<double>(std::cout, "r ", r, nr);
        std::cout << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    delete[] r;
    if(rank == 0) {
      tbsla::cpp::MatrixCOO ml;
      ml.fill_cdiag(nr, nc, cdiag);
    }
    exit(res);
  }
  delete[] v;
  MPI_Barrier(MPI_COMM_WORLD);
}

void test_cdiag(int nr, int nc, int cdiag) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  tbsla::mpi::MatrixCOO mcoo;
  tbsla::mpi::MatrixSCOO mscoo;
  tbsla::mpi::MatrixCSR mcsr;
  tbsla::mpi::MatrixELL mell;
  tbsla::mpi::MatrixDENSE mdense;

  if(rank == 0)
    std::cout << "--- row ---- nr : " << nr << "; nc : " << nc << "; c : " << cdiag << " ----" <<  std::endl;
  std::cout << std::flush;
  MPI_Barrier(MPI_COMM_WORLD);
  test_matrix(mcoo, nr, nc, cdiag, rank, 0, world, 1);
  test_matrix_split_vector(mscoo, nr, nc, cdiag, rank, 0, world, 1);
  test_matrix_split_vector(mcsr, nr, nc, cdiag, rank, 0, world, 1);
  test_matrix_split_vector(mell, nr, nc, cdiag, rank, 0, world, 1);
  test_matrix_split_vector(mdense, nr, nc, cdiag, rank, 0, world, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
    std::cout << "--- col ---- nr : " << nr << "; nc : " << nc << "; c : " << cdiag << " ----" <<  std::endl;
  std::cout << std::flush;
  MPI_Barrier(MPI_COMM_WORLD);
  test_matrix(mcoo, nr, nc, cdiag, 0, rank, 1, world);
  test_matrix_split_vector(mscoo, nr, nc, cdiag, 0, rank, 1, world);
  test_matrix_split_vector(mcsr, nr, nc, cdiag, 0, rank, 1, world);
  test_matrix_split_vector(mell, nr, nc, cdiag, 0, rank, 1, world);
  test_matrix_split_vector(mdense, nr, nc, cdiag, 0, rank, 1, world);

  if(world % 2 == 0 && world / 2 > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
      std::cout << "--- mix ---- nr : " << nr << "; nc : " << nc << "; c : " << cdiag << " ----" <<  std::endl;
    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
    test_matrix(mcoo, nr, nc, cdiag, rank / 2, rank % 2, world / 2, 2);
    test_matrix_split_vector(mscoo, nr, nc, cdiag, rank / 2, rank % 2, world / 2, 2);
    test_matrix_split_vector(mcsr, nr, nc, cdiag, rank / 2, rank % 2, world / 2, 2);
    test_matrix_split_vector(mell, nr, nc, cdiag, rank / 2, rank % 2, world / 2, 2);
    test_matrix_split_vector(mdense, nr, nc, cdiag, rank / 2, rank % 2, world / 2, 2);
  }

  if(world % 3 == 0 && world / 3 > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
      std::cout << "--- mix %3 ---- nr : " << nr << "; nc : " << nc << "; c : " << cdiag << " ----" <<  std::endl;
    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
    test_matrix(mcoo, nr, nc, cdiag, rank / 3, rank % 3, world / 3, 3);
    test_matrix_split_vector(mscoo, nr, nc, cdiag, rank / 3, rank % 3, world / 3, 3);
    test_matrix_split_vector(mcsr, nr, nc, cdiag, rank / 3, rank % 3, world / 3, 3);
    test_matrix_split_vector(mell, nr, nc, cdiag, rank / 3, rank % 3, world / 3, 3);
    test_matrix_split_vector(mdense, nr, nc, cdiag, rank / 3, rank % 3, world / 3, 3);

    test_matrix(mcoo, nr, nc, cdiag, rank % 3, rank / 3, 3, world / 3);
    test_matrix_split_vector(mscoo, nr, nc, cdiag, rank % 3, rank / 3, 3, world / 3);
    test_matrix_split_vector(mcsr, nr, nc, cdiag, rank % 3, rank / 3, 3, world / 3);
    test_matrix_split_vector(mell, nr, nc, cdiag, rank % 3, rank / 3, 3, world / 3);
    test_matrix_split_vector(mdense, nr, nc, cdiag, rank % 3, rank / 3, 3, world / 3);
  }

}

int main(int argc, char** argv) {

  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int t = 0;
  for(int i = 0; i <= 12; i++) {
    if(rank == 0)
      std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(10, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    if(rank == 0)
      std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(5, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    if(rank == 0)
      std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(10, 5, i);
  }
  for(int i = 0; i <= 12; i++) {
    if(rank == 0)
      std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(30, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    if(rank == 0)
      std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(20, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    if(rank == 0)
      std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(30, 20, 2 * i);
  }
  std::cout << "=== finished without error === " << std::endl;

  MPI_Finalize();
}
