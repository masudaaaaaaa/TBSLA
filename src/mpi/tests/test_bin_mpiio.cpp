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
#include <string>

void print(tbsla::mpi::Matrix & m) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for(int i = 0; i < world; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(i == rank) {
      std::cout << m << std::endl << std::flush;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void test_mpiio(int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  tbsla::mpi::MatrixCOO mcoo;
  mcoo.read_bin_mpiio(MPI_COMM_WORLD, "mcoo.bin", pr, pc, NR, NC);
  double* v = new double[mcoo.get_n_col()];
  std::iota (v, v + mcoo.get_n_col(), 0);
  double* rg_coo = mcoo.spmv(MPI_COMM_WORLD, v);

  tbsla::mpi::MatrixSCOO mscoo;
  mscoo.read_bin_mpiio(MPI_COMM_WORLD, "mcoo.bin", pr, pc, NR, NC);
  double* vl = new double[mscoo.get_ln_col()];
  std::iota (vl, vl + mscoo.get_ln_col(), mscoo.get_f_col());
  double* rl_scoo = mscoo.spmv(vl);
  double* rg_scoo = mscoo.spmv(MPI_COMM_WORLD, vl);

  tbsla::mpi::MatrixCSR mcsr;
  mcsr.read_bin_mpiio(MPI_COMM_WORLD, "mcsr.bin", pr, pc, NR, NC);
  double* rl_csr = mcsr.spmv(vl, 0);

  tbsla::mpi::MatrixELL mell;
  mell.read_bin_mpiio(MPI_COMM_WORLD, "mell.bin", pr, pc, NR, NC);
  double* rl_ell = mell.spmv(vl, 0);

  tbsla::mpi::MatrixDENSE mdense;
  mdense.read_bin_mpiio(MPI_COMM_WORLD, "mdense.bin", pr, pc, NR, NC);
  double* rl_dense = mdense.spmv(vl, 0);

  if(tbsla::utils::array::compare_arrays(rg_coo, rg_scoo, mcoo.get_n_row())) {
    print(mcoo);
    print(mscoo);
    exit(1);
  }

  if(tbsla::utils::array::compare_arrays(rl_csr, rl_scoo, mcsr.get_ln_row())) {
    print(mscoo);
    print(mcsr);
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        tbsla::utils::array::stream<double>(std::cout, "vl ", vl, mcsr.get_ln_col());
        std::cout << std::endl << std::flush;
        tbsla::utils::array::stream<double>(std::cout, "rl_csr ", rl_csr, mcsr.get_ln_row());
        std::cout << std::endl << std::flush;
        tbsla::utils::array::stream<double>(std::cout, "rl_scoo ", rl_scoo, mcsr.get_ln_row());
        std::cout << std::endl << std::flush;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    exit(2);
  }

  if(tbsla::utils::array::compare_arrays(rl_ell, rl_scoo, mcsr.get_ln_row())) {
    print(mscoo);
    print(mell);
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        tbsla::utils::array::stream<double>(std::cout, "vl ", vl, mcsr.get_ln_col());
        std::cout << std::endl << std::flush;
        tbsla::utils::array::stream<double>(std::cout, "rl_ell ", rl_ell, mcsr.get_ln_row());
        std::cout << std::endl << std::flush;
        tbsla::utils::array::stream<double>(std::cout, "rl_scoo ", rl_scoo, mcsr.get_ln_row());
        std::cout << std::endl << std::flush;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    exit(3);
  }

  if(tbsla::utils::array::compare_arrays(rl_dense, rl_scoo, mcsr.get_ln_row())) {
    print(mscoo);
    print(mdense);
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        tbsla::utils::array::stream<double>(std::cout, "vl ", vl, mcsr.get_ln_col());
        std::cout << std::endl << std::flush;
        tbsla::utils::array::stream<double>(std::cout, "rl_dense ", rl_dense, mcsr.get_ln_row());
        std::cout << std::endl << std::flush;
        tbsla::utils::array::stream<double>(std::cout, "rl_scoo ", rl_scoo, mcsr.get_ln_row());
        std::cout << std::endl << std::flush;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    exit(4);
  }
}

void test_(int nr, int nc, int cdiag) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    std::ofstream os;
    tbsla::cpp::MatrixCOO mcoo;
    os.open("mcoo.bin", std::ofstream::binary);
    mcoo.fill_cdiag(nr, nc, cdiag, 0, 0, 1, 1);
    mcoo.write(os);
    os.close();

    tbsla::cpp::MatrixCSR mcsr;
    os.open("mcsr.bin", std::ofstream::binary);
    mcsr.fill_cdiag(nr, nc, cdiag, 0, 0, 1, 1);
    mcsr.write(os);
    os.close();

    tbsla::cpp::MatrixELL mell;
    os.open("mell.bin", std::ofstream::binary);
    mell.fill_cdiag(nr, nc, cdiag, 0, 0, 1, 1);
    mell.write(os);
    os.close();

    tbsla::cpp::MatrixDENSE mdense;
    os.open("mdense.bin", std::ofstream::binary);
    mdense.fill_cdiag(nr, nc, cdiag, 0, 0, 1, 1);
    mdense.write(os);
    os.close();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  test_mpiio(rank, 0, world, 1);
  test_mpiio(0, rank, 1, world);
}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int t = 0;
  for(int i = 0; i <= 3; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_(30, 30, 3 * i);
  }
  for(int i = 0; i <= 3; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_(20, 30, 3 * i);
  }
  for(int i = 0; i <= 3; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_(30, 20, 3 * i);
  }
  std::cout << "=== finished without error === " << std::endl;

  MPI_Finalize();
}
