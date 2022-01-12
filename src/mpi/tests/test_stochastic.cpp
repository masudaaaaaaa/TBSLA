#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/mpi/MatrixDENSE.hpp>

#include <tbsla/cpp/utils/array.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>

#include <mpi.h>

#include <random>

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


void test_random_stochastic(tbsla::mpi::Matrix & m, int nr, int nc, double nnz_ratio, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  m.fill_random(nr, nc, nnz_ratio, 0, pr, pc, NR, NC);
  std::cout << "Generated local " << m.get_ln_row() << "x" << m.get_ln_col() << " random matrix at position [" << pr << "," << pc << std::endl;
  
  double* s = new double[nr];
  //double* b = new double[2 * m.get_ln_row()];
  double* b1 = new double[m.get_ln_row()];
  double* b2 = new double[m.get_ln_row()];
  for(int i = 0; i < nr; i++) {
    s[i] = 0;
    //b[i] = 0;
  }
  for(int i = 0; i < m.get_ln_row(); i++) {
    b1[i] = 0;
    b2[i] = 0;
  }
  //m.make_stochastic(MPI_COMM_WORLD, s, b, NULL);
  m.make_stochastic(MPI_COMM_WORLD, s, b1, b2);
  std::cout << "Normalized matrix\n";
  delete[] s;
  delete[] b1;
  delete[] b2;
  MPI_Barrier(MPI_COMM_WORLD);
}


/*void test_random_stochastic_pagerank(tbsla::mpi::Matrix & m, int nr, int nc, double nnz_ratio, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand(12);
  m.fill_random(nr, nc, nnz_ratio, 0, pr, pc, NR, NC);
  std::cout << "Generated local " << m.get_ln_row() << "x" << m.get_ln_col() << " random matrix at position [" << pr << "," << pc << std::endl;
  
  double* s = new double[nr];
  double* b1 = new double[m.get_ln_row()];
  double* b2 = new double[m.get_ln_row()];
  for(int i = 0; i < nr; i++) {
    s[i] = 0;
  }
  for(int i = 0; i < m.get_ln_row(); i++) {
    b1[i] = 0;
    b2[i] = 0;
  }
  //m.make_stochastic(MPI_COMM_WORLD, s, b1, b2);
  //std::cout << "Normalized matrix\n";
  delete[] s;
  delete[] b1;
  delete[] b2;
  MPI_Barrier(MPI_COMM_WORLD);
  
  double epsilon = 0.001;
  double beta = 0.85;
  int max_iteration = 1000;
  int nb_iterations_done;
  double* res = m.page_rank(MPI_COMM_WORLD, beta, epsilon, max_iteration, nb_iterations_done);
  std::cout << "PageRank converged in " << nb_iterations_done << " iterations\n";
  for(int i=0; i<10; i++)
	  std::cout << res[i] << " ";
  std::cout << std::endl;
  delete[] res;
}*/

void test_random_stochastic_pagerank(tbsla::mpi::Matrix & m, int nr, int nc, double nnz_ratio, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand(12);
  m.fill_random(nr, nc, nnz_ratio, 0, pr, pc, NR, NC);
  std::cout << "Generated local " << m.get_ln_row() << "x" << m.get_ln_col() << " random matrix at position [" << pr << "," << pc << std::endl;
  
  double* s = new double[nc];
  double* b1 = new double[m.get_ln_col()];
  double* b2 = new double[m.get_ln_col()];
  for(int i = 0; i < nr; i++) {
    s[i] = 0;
  }
  for(int i = 0; i < m.get_ln_col(); i++) {
    b1[i] = 0;
    b2[i] = 0;
  }
  m.make_stochastic(MPI_COMM_WORLD, s, b1, b2);
  std::cout << "Normalized matrix\n";
  delete[] s;
  delete[] b1;
  delete[] b2;
  MPI_Barrier(MPI_COMM_WORLD);
  
  double epsilon = 0.001;
  double beta = 0.85;
  int max_iteration = 1000;
  int nb_iterations_done;
  double* res = m.page_rank(MPI_COMM_WORLD, beta, epsilon, max_iteration, nb_iterations_done);
  std::cout << "PageRank converged in " << nb_iterations_done << " iterations\n";
  for(int i=0; i<10; i++)
	  std::cout << res[i] << " ";
  std::cout << std::endl;
  delete[] res;
}


void test_random_wrapper(int nr, int nc, double nnz_ratio) {
  int NR = 2, NC = 2;
  
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //tbsla::mpi::MatrixCOO mcoo;
  //tbsla::mpi::MatrixSCOO mscoo;
  tbsla::mpi::MatrixCSR mcsr;
  //tbsla::mpi::MatrixELL mell;
  //tbsla::mpi::MatrixDENSE mdense;
  
  if(rank == 0)
    std::cout << "--- row ---- nr : " << nr << "; nc : " << nc << "; c : " << nnz_ratio << " ----" <<  std::endl;
  std::cout << std::flush;
  MPI_Barrier(MPI_COMM_WORLD);
  //test_random_stochastic(mcsr, nr, nc, nnz_ratio, rank / NR, rank % NC, NR, NC);
  test_random_stochastic(mcsr, nr, nc, nnz_ratio, rank, 0, world, 1);
  //test_random_stochastic(mell, nr, nc, nnz_ratio, NR, NC);
  //test_random_stochastic(mdense, nr, nc, nnz_ratio, NR, NC);
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
    std::cout << "--- col ---- nr : " << nr << "; nc : " << nc << "; c : " << nnz_ratio << " ----" <<  std::endl;
  std::cout << std::flush;
  test_random_stochastic(mcsr, nr, nc, nnz_ratio, 0, rank, 1, world);
  
  if(world % 2 == 0 && world / 2 > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
      std::cout << "--- mix ---- nr : " << nr << "; nc : " << nc << "; c : " << nnz_ratio << " ----" <<  std::endl;
    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
	test_random_stochastic(mcsr, nr, nc, nnz_ratio, rank / 2, rank % 2, world / 2, 2);
  }

  if(world % 3 == 0 && world / 3 > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
      std::cout << "--- mix %3 ---- nr : " << nr << "; nc : " << nc << "; c : " << nnz_ratio << " ----" <<  std::endl;
    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
	test_random_stochastic(mcsr, nr, nc, nnz_ratio, rank / 3, rank % 3, world / 3, 3);

	test_random_stochastic(mcsr, nr, nc, nnz_ratio, rank % 3, rank / 3, 3, world / 3);
  }
}


void test_pagerank_wrapper(int nr, int nc, double nnz_ratio) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  tbsla::mpi::MatrixCSR mcsr;
  tbsla::mpi::MatrixELL mell;
  tbsla::mpi::MatrixSCOO mscoo;
  
  if(rank == 0)
    std::cout << "--- row ---- nr : " << nr << "; nc : " << nc << "; c : " << nnz_ratio << " ----" <<  std::endl;
  std::cout << std::flush;
  MPI_Barrier(MPI_COMM_WORLD);
  test_random_stochastic_pagerank(mcsr, nr, nc, nnz_ratio, rank, 0, world, 1);
  test_random_stochastic_pagerank(mell, nr, nc, nnz_ratio, rank, 0, world, 1);
  test_random_stochastic_pagerank(mscoo, nr, nc, nnz_ratio, rank, 0, world, 1);
  
  /*MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
    std::cout << "--- col ---- nr : " << nr << "; nc : " << nc << "; c : " << nnz_ratio << " ----" <<  std::endl;
  std::cout << std::flush;
  test_random_stochastic_pagerank(mcsr, nr, nc, nnz_ratio, 0, rank, 1, world);*/
  
  if(world % 2 == 0 && world / 2 > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
      std::cout << "--- mix ---- nr : " << nr << "; nc : " << nc << "; c : " << nnz_ratio << " ----" <<  std::endl;
    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
	test_random_stochastic_pagerank(mcsr, nr, nc, nnz_ratio, rank / 2, rank % 2, world / 2, 2);
	test_random_stochastic_pagerank(mell, nr, nc, nnz_ratio, rank / 2, rank % 2, world / 2, 2);
	test_random_stochastic_pagerank(mscoo, nr, nc, nnz_ratio, rank / 2, rank % 2, world / 2, 2);
  }

  if(world % 3 == 0 && world / 3 > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
      std::cout << "--- mix %3 ---- nr : " << nr << "; nc : " << nc << "; c : " << nnz_ratio << " ----" <<  std::endl;
    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
	test_random_stochastic_pagerank(mcsr, nr, nc, nnz_ratio, rank / 3, rank % 3, world / 3, 3);
	test_random_stochastic_pagerank(mell, nr, nc, nnz_ratio, rank / 3, rank % 3, world / 3, 3);
	test_random_stochastic_pagerank(mscoo, nr, nc, nnz_ratio, rank / 3, rank % 3, world / 3, 3);

	test_random_stochastic_pagerank(mcsr, nr, nc, nnz_ratio, rank % 3, rank / 3, 3, world / 3);
	test_random_stochastic_pagerank(mell, nr, nc, nnz_ratio, rank % 3, rank / 3, 3, world / 3);
	test_random_stochastic_pagerank(mscoo, nr, nc, nnz_ratio, rank % 3, rank / 3, 3, world / 3);
  }
}


void test_generator_wrapper() {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  std::default_random_engine generator(12);
  std::uniform_real_distribution<double> distr_ind(0, 100);
  srand(12);
  
  int nb_random = 10;
  for(int k=0; k<10; k++) {
	std::cout << "Iteration " << k << std::endl;
	//int* random_cols = tbsla::utils::values_generation::random_columns(nb_random, 240, distr_ind, generator);
	int* random_cols = tbsla::utils::values_generation::random_columns(1, nb_random, 240, 12);
	for(int z=0; z<nb_random; z++)
		std::cout << random_cols[z] << " ";
	std::cout << std::endl;
  }
}


int main(int argc, char** argv) {

  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //test_random_wrapper(240, 240, 0.05);
  test_pagerank_wrapper(240, 240, 0.05);
  //test_generator_wrapper();
  
  std::cout << "=== finished without error === " << std::endl;

  MPI_Finalize();
}
