#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/petsc/Matrix.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/vector.hpp>

#include <mpi.h>

#include <numeric>
#include <iostream>
#include <petsc.h>

static char help[] = "test spmv cqmat";

void test_cqmat(int nr, int nc, int c, double q, unsigned int seed) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << q << ": " << q << "; s : " << seed << "----  r : " << rank << "/" << world << std::endl;

  std::vector<double> v(nc);
  std::iota (std::begin(v), std::end(v), 0);

  tbsla::mpi::MatrixCOO mcoo;
  mcoo.fill_cqmat(MPI_COMM_WORLD, nr, nc, c, q, seed);
  std::vector<double> rcoo = mcoo.spmv(MPI_COMM_WORLD, v);

  tbsla::petsc::Matrix m;
  m.fill_cqmat(MPI_COMM_WORLD, nr, nc, c, q, seed);
  int v_start = tbsla::utils::range::pflv(nc, rank, world);
  int v_n = tbsla::utils::range::lnv(nc, rank, world);
  Vec petscv;
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, v_n, v.size(), v.data() + v_start, &petscv);
  Vec petscr = m.spmv(MPI_COMM_WORLD, petscv);
  double * rptr;
  VecGetArray(petscr, &rptr);
  int size;
  VecGetLocalSize(petscr, &size);
  std::vector<double> r(nr);

  int nbR = nr / world;
  int mod = nr % world;
  int counts[world], displs[world];
  displs[0] = 0;
  counts[world - 1] = nbR;
  for (int i = 0; i < world - 1; i++) {
    if (i < mod) {
      counts[i] = nbR + 1;
      displs[i + 1] = displs[i] + nbR + 1;
    } else {
      counts[i] = nbR;
      displs[i + 1] = displs[i] + nbR;
    }
  }
  if (rank < mod)
    nbR++;
  MPI_Gatherv(rptr, nbR, MPI_DOUBLE, r.data(), counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  VecRestoreArray(petscr, &rptr);
  if(rank == 0) {
    if(r != rcoo) {
      tbsla::utils::vector::streamvector<double>(std::cout, "v ", v);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "petscr ", r);
      std::cout << std::endl;
      exit(1);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  free(rptr);
}

void test_mat(int nr, int nc, int c) {
  for(double s = 0; s < 4; s++) {
    for(double q = 0; q <= 1; q += 0.1) {
      test_cqmat(nr, nc, c, q, s);
    }
  }
}

int main(int argc, char** argv) {

  PetscInitialize(&argc, &argv, (char*)0, help);

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

  PetscFinalize();
}
