#include <tbsla/mpi/Matrix.hpp>

#include <cstring>
#include <cmath>


void vector_combination_arr(double scalar_one, double* v_one, double scalar_two, double* v_two, int n, double* res) {
  #pragma omp parallel for schedule(static)
  for(int k=0; k<n; k++)
    res[k] = scalar_one*v_one[k] + scalar_two*v_two[k];
}


double inner_product_arr(double* v_one, double* v_two, int n) {
  double res = 0.0;
  #pragma omp parallel for schedule(static) reduction(+ : res)
  for(int k=0; k<n; k++)
    res += v_one[k]*v_two[k];
  return res;
}


double vector_norm_arr(double* v, int n) {
  double res = 0;
  #pragma omp parallel for schedule(static) reduction(+ : res)
  for(int k=0; k<n; k++)
    res += v[k]*v[k];
  res = sqrt(res);
  return res;
}

void tbsla::mpi::Matrix::CG(MPI_Comm comm, double* v, double* res, int max_iterations, int &nb_iterations_done)
{
  double TOLERANCE = 1.0e-10;
  double NEARZERO = 1.0e-10;

  int n = this->n_col;

  double* R = new double[n];
  std::memcpy(R, v, n * sizeof(double));
  double* P = new double[n];
  std::memcpy(P, R, n * sizeof(double));
  double* Rold = new double[n];
  double* AP = new double[n];
  double* buf1 = new double[ln_row];
  double* buf2 = new double[ln_row];
  nb_iterations_done = 0;

  while ( nb_iterations_done < max_iterations )
  {
    //std::cout << "Iteration " << k << std::endl;
    std::memcpy(Rold, R, n * sizeof(double));
    #pragma omp parallel for schedule(static)
    for(int z=0; z<n; z++)
      AP[z] = 0.0;
    #pragma omp parallel for schedule(static)
    for(int i = 0 ; i < ln_row; i++) {
      buf1[i] = 0;
      buf2[i] = 0;
    }
    this->Ax(comm, P + f_col, AP, buf1, buf2);

    double alpha = inner_product_arr( R, R, n ) / std::max( inner_product_arr( P, AP, n ), NEARZERO );
    vector_combination_arr( 1.0, res, alpha, P, n, res);            // Next estimate of solution
    vector_combination_arr( 1.0, R, -alpha, AP, n, R );          // Residual

    if ( vector_norm_arr( R, n ) < TOLERANCE ) break;             // Convergence test

    double beta = inner_product_arr( R, R, n ) / std::max( inner_product_arr( Rold, Rold, n ), NEARZERO );
    vector_combination_arr( 1.0, R, beta, P, n, P );             // Next gradient
    nb_iterations_done++;
  }
}
