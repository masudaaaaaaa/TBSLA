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

void test_cqmat(int nr, int nc, int c, double q, unsigned int seed, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double* v = new double[nc];
  std::iota (v, v + nc, 0);

  tbsla::mpi::MatrixCOO mcoo;
  mcoo.fill_cqmat(nr, nc, c, q, seed, pr, pc, NR, NC);
  double* rcoo = mcoo.spmv(MPI_COMM_WORLD, v);

  tbsla::mpi::MatrixSCOO mscoo;
  mscoo.fill_cqmat(nr, nc, c, q, seed, pr, pc, NR, NC);
  double* vl = new double[mscoo.get_ln_col()];
  std::iota (vl, vl + mscoo.get_ln_col(), mscoo.get_f_col());
  double* rscoo = mscoo.spmv(MPI_COMM_WORLD, vl);
  if(tbsla::utils::array::compare_arrays(rscoo, rcoo, nr)) {
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << mcoo << std::endl;
        std::cout << mscoo << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::utils::array::stream<double>(std::cout, "v ", v, nc);
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "vl ", vl, mscoo.get_ln_col());
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "rcoo ", rcoo, mscoo.get_n_row());
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "rscoo ", rscoo, mscoo.get_n_row());
      std::cout << std::endl;
    }
    exit(1);
  }

  tbsla::mpi::MatrixCSR mcsr;
  mcsr.fill_cqmat(nr, nc, c, q, seed, pr, pc, NR, NC);
  double* rcsr = mcsr.spmv(MPI_COMM_WORLD, vl);
  if(tbsla::utils::array::compare_arrays(rcsr, rcoo, nr)) {
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << mcoo << std::endl;
        std::cout << mcsr << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::utils::array::stream<double>(std::cout, "v ", v, nc);
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "vl ", vl, mscoo.get_ln_col());
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "rcoo ", rcoo, mscoo.get_n_row());
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "rcsr ", rcsr, mscoo.get_n_row());
      std::cout << std::endl;
    }
    exit(1);
  }

  tbsla::mpi::MatrixELL mell;
  mell.fill_cqmat(nr, nc, c, q, seed, pr, pc, NR, NC);
  double* rell = mell.spmv(MPI_COMM_WORLD, vl);
  if(tbsla::utils::array::compare_arrays(rell, rcoo, nr)) {
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << mcoo << std::endl;
        std::cout << mell << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::utils::array::stream<double>(std::cout, "v ", v, nc);
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "vl ", vl, mscoo.get_ln_col());
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "rcoo ", rcoo, mscoo.get_n_row());
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "rell ", rell, mscoo.get_n_row());
      std::cout << std::endl;
    }
    exit(1);
  }

  tbsla::mpi::MatrixDENSE mdense;
  mdense.fill_cqmat(nr, nc, c, q, seed, pr, pc, NR, NC);
  double* rdense = mdense.spmv(MPI_COMM_WORLD, vl);
  if(tbsla::utils::array::compare_arrays(rdense, rcoo, nr)) {
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << mcoo << std::endl;
        std::cout << mdense << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::utils::array::stream<double>(std::cout, "v ", v, nc);
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "vl ", vl, mscoo.get_ln_col());
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "rcoo ", rcoo, mscoo.get_n_row());
      std::cout << std::endl;
      tbsla::utils::array::stream<double>(std::cout, "rdense ", rdense, mscoo.get_n_row());
      std::cout << std::endl;
    }
    exit(1);
  }
}

void test_mat(int nr, int nc, int c) {
  for(double s = 0; s < 4; s++) {
    for(double q = 0; q <= 1; q += 0.1) {
      int world, rank;
      MPI_Comm_size(MPI_COMM_WORLD, &world);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0)
        std::cout << "--- row ---- nr : " << nr << "; nc : " << nc << "; c : " << q << ": " << q << "; s : " << s << "----" << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      test_cqmat(nr, nc, c, q, s, rank, 0, world, 1);
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
        std::cout << "--- col ---- nr : " << nr << "; nc : " << nc << "; c : " << q << ": " << q << "; s : " << s << "----" << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      test_cqmat(nr, nc, c, q, s, 0, rank, 1, world);
      MPI_Barrier(MPI_COMM_WORLD);
      if(world % 2 == 0 && world / 2 > 1) {
        if (rank == 0)
          std::cout << "--- mix ---- nr : " << nr << "; nc : " << nc << "; c : " << q << ": " << q << "; s : " << s << "----" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        test_cqmat(nr, nc, c, q, s, rank / 2, rank % 2, world / 2, 2);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(10, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(5, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(10, 5, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(30, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(20, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(30, 20, 2 * i);
  }
  std::cout << "=== finished without error === " << std::endl;

  MPI_Finalize();
}
