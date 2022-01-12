#include <tbsla/petsc/Matrix.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/range.hpp>

#include <mpi.h>

#include <numeric>
#include <iostream>
#include <petsc.h>

static char help[] = "test a_axpx_ cdiag";

void test_matrix(tbsla::petsc::Matrix & m, int nr, int nc, int cdiag) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << cdiag << " ----  r : " << rank << "/" << world << std::endl;

  std::vector<double> v(nc);
  std::iota (std::begin(v), std::end(v), 0);
  int v_start = tbsla::utils::range::pflv(nc, rank, world);
  int v_n = tbsla::utils::range::lnv(nc, rank, world);
  Vec petscv;
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, v_n, v.size(), v.data() + v_start, &petscv);
  Vec petscr = m.a_axpx_(MPI_COMM_WORLD, petscv);
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
    int res = tbsla::utils::vector::test_a_axpx__cdiag(nr, nc, cdiag, v, r, false);
    std::cout << "return : " << res << std::endl;
    if(res) {
      tbsla::utils::vector::test_a_axpx__cdiag(nr, nc, cdiag, v, r, true);
      tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "r", r);
      std::cout << std::endl;

      tbsla::cpp::MatrixCOO ml;
      ml.fill_cdiag(nr, nc, cdiag);
      ml.print_infos(std::cout);
      exit(res);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  free(rptr);
}

void test_cdiag(int nr, int nc, int cdiag) {
  tbsla::petsc::Matrix m;
  m.fill_cdiag(MPI_COMM_WORLD, nr, nc, cdiag);
  test_matrix(m, nr, nc, cdiag);
}

int main(int argc, char** argv) {

  PetscInitialize(&argc, &argv, (char*)0, help);

  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(10, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(30, 30, 2 * i);
  }
  std::cout << "=== finished without error === " << std::endl;

  PetscFinalize();
}
