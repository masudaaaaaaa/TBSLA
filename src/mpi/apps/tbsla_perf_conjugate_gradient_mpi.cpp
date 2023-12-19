#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/mpi/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>
#include <tbsla/Configs.h>

#if TBSLA_COMPILED_WITH_OMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <string>
#include <cstdlib>
#include <vector>

#include <mpi.h>

static std::uint64_t now() {
  std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(ns.count());
}

int* fix_list(int* list, int n_vals, int nc) {
  int* fixed_list = new int[n_vals];
  bool* is_used = new bool[nc];
  for(int z=0; z<nc; z++)
    is_used[z] = false;
  bool* to_fix = new bool[n_vals];
  for(int z=0; z<n_vals; z++)
    to_fix[z] = false;
  for(int k=0; k<n_vals; k++) {
    int col_ind = list[k];
    if(is_used[col_ind]) {
      to_fix[k] = true;
      //std::cout << col_ind << " already used ; need to fix element " << k << std::endl;
    }
    is_used[col_ind] = true;
    //std::cout << col_ind << " is used\n";
  }
  for(int k=0; k<n_vals; k++) {
      if(to_fix[k]) {
	int new_col = list[k];
	int mod = 1;
	if(new_col>(nc/2))
	  mod = -1;
	while(is_used[new_col] && new_col > 0 && new_col < nc-1) {
	  //std::cout << new_col << " is used ; moving to ";
	  new_col += mod;
	  //std::cout << new_col << std::endl;
	}
	if(!is_used[new_col]) {
	  fixed_list[k] = new_col;
	  is_used[new_col] = true;
	}
	else {
	  //std::cout << "Ran out of values to shift, in 'fix_list'\n";
	}
      }
      else {
	fixed_list[k] = list[k];
      }
  }
  return fixed_list;
}

double compute_median(std::vector<double> _values)
{
  double median = 0;
  int n = _values.size();
  std::vector<double> sorted(_values);
  std::sort(sorted.begin(), sorted.end());
  if(n>0) {
    if(n % 2 != 0) {
      int ind = ((n+1)/2)-1;
      median = sorted[ind];
    }
    else {
      int ind_one = (n/2)-1, ind_two = (n/2);
      median = (sorted[ind_one]+sorted[ind_two])/2;
    }
  }
  return median;
}

double compute_gflops_pagerank(double runtime, int n, int nnz, int n_iters) {
  unsigned long long int a = 2*nnz+2*n;
  unsigned long long int b = a*n_iters;
  unsigned long long int c = 2*n;
  unsigned long long int n_ops = b+c;
  double gfl = (double)(n_ops/(runtime*1000000000));
  return gfl;
}


void generate_brain_structs(int n, int nnz, std::vector<std::vector<double> > &proba_conn, std::vector<std::unordered_map<int,std::vector<int> > > &brain_struct, int* neuron_type) {
  std::cout << "n = " << n << std::endl;
  // Dummy example w/ 2 parts and 2 neuron types
  std::vector<double> pc_one, pc_two;
  /*pc_one.push_back(0.0001); pc_one.push_back(0.0004);
  pc_one.push_back(0.0003); pc_one.push_back(0.00005);
  pc_two.push_back(0.0007); pc_two.push_back(0.0001);
  pc_two.push_back(0.0004); pc_two.push_back(0.00002);*/
  pc_one.push_back(1*nnz); pc_one.push_back(4*nnz);
  pc_one.push_back(3*nnz); pc_one.push_back(0.5);
  pc_two.push_back(7*nnz); pc_two.push_back(1*nnz);
  pc_two.push_back(4*nnz); pc_two.push_back(0.2*nnz);
  proba_conn.push_back(pc_one); proba_conn.push_back(pc_two);
  std::unordered_map<int,std::vector<int> > map_one, map_two;
  std::vector<int> v_one_one, v_one_two, v_two_one, v_two_two;
  double ratio_parts = 0.3;
  double split_one = 0.4, split_two = 0.8;
  int n_part_one = (int)(n*ratio_parts); int n_part_two = n-n_part_one;
  int n_one_one = (int)(n_part_one*split_one); int n_one_two = n_part_one-n_one_one;
  v_one_one.push_back(0); v_one_one.push_back(n_one_one);
  v_one_two.push_back(n_one_one); v_one_two.push_back(n_one_two);
  std::cout << "v_1_1 : " << v_one_one[0] << " , " << v_one_one[1] << std::endl;
  std::cout << "v_1_2 : " << v_one_two[0] << " , " << v_one_two[1] << std::endl;

  int n_two_one = (int)(n_part_two*split_two); int n_two_two = n_part_two-n_two_one;
  v_two_one.push_back(n_part_one+0); v_two_one.push_back(n_two_one);
  v_two_two.push_back(n_part_one+n_two_one); v_two_two.push_back(n_two_two);
  std::cout << "v_2_1 : " << v_two_one[0] << " , " << v_two_one[1] << std::endl;
  std::cout << "v_2_2 : " << v_two_two[0] << " , " << v_two_two[1] << std::endl;

  map_one[0] = v_one_one; map_one[1] = v_one_two;
  map_two[0] = v_two_one; map_two[1] = v_two_two;
  brain_struct.push_back(map_one); brain_struct.push_back(map_two);

  std::cout << "n_type 0 from " << brain_struct[0][0][0] << " to " << brain_struct[0][0][0]+brain_struct[0][0][1] << std::endl;
  for(int k=brain_struct[0][0][0]; k<brain_struct[0][0][0]+brain_struct[0][0][1]; k++)
    neuron_type[k] = 0;
  std::cout << "n_type 1 from " << brain_struct[0][1][0] << " to " << brain_struct[0][1][0]+brain_struct[0][1][1] << std::endl;
  for(int k=brain_struct[0][1][0]; k<brain_struct[0][1][0]+brain_struct[0][1][1]; k++)
    neuron_type[k] = 1;
  std::cout << "n_type 0 from " << brain_struct[1][0][0] << " to " << brain_struct[1][0][0]+brain_struct[1][0][1] << std::endl;
  for(int k=brain_struct[1][0][0]; k<brain_struct[1][0][0]+brain_struct[1][0][1]; k++)
    neuron_type[k] = 0;
  std::cout << "n_type 1 from " << brain_struct[1][1][0] << " to " << brain_struct[1][1][0]+brain_struct[1][1][1] << std::endl;
  for(int k=brain_struct[1][1][0]; k<brain_struct[1][1][0]+brain_struct[1][1][1]; k++)
    neuron_type[k] = 1;
}


int main(int argc, char** argv) {
  InputParser input(argc, argv);
  MPI_Init(&argc, &argv);
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string format = input.get_opt("--format");
  if(format == "") {
    std::cerr << "A file format has to be given with the parameter --format format" << std::endl;
    exit(1);
  }

  std::string matrix_dim_string = input.get_opt("--matrix_dim", "1024");

  std::string gr_string = input.get_opt("--GR", "1");
  std::string gc_string = input.get_opt("--GC", "1");

  std::string beta_string = input.get_opt("--beta", "0.85");
  std::string epsilon_string = input.get_opt("--epsilon", "0.00001");
  std::string max_iterations_string = input.get_opt("--max-iterations", "10000");

  double epsilon = std::stod(epsilon_string);
  double beta = std::stod(beta_string);
  int max_iterations = std::stoi(max_iterations_string);
  int matrix_dim = std::stoi(matrix_dim_string);
  int GR = std::stoi(gr_string);
  int GC = std::stoi(gc_string);
  int C = -1;
  double Q = -1;
  int S = -1;
  double NNZ = -1;

  if(world != GR * GC) {
    printf("The number of processes (%d) does not match the grid dimensions (%d x %d = %d).\n", world, GR, GC, GR * GC);
    exit(99);
  }

  std::string matrix = input.get_opt("--matrix");
  std::string matrix_folder = input.get_opt("--matrix_folder", ".");
  if(matrix == "cdiag") {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
  } else if(matrix == "cqmat") {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
    std::string q_string = input.get_opt("--Q", "0.1");
    Q = std::stod(q_string);
    std::string s_string = input.get_opt("--S", "0");
    S = std::stoi(s_string);
  } else if(matrix == "random_stoch") {
    std::string nnz_string = input.get_opt("--NNZ", "0.0001");
    NNZ = std::stod(nnz_string);
  } 
  else if(matrix == "cdistrib") {
    std::string nnz_string = input.get_opt("--NNZ", "0.0001");
    NNZ = std::stod(nnz_string);
  }else if(matrix == "brain") {
    std::string nnz_string = input.get_opt("--NNZ", "10");
    NNZ = std::stod(nnz_string);
  } else if (matrix == "") {
    if(rank == 0) {
      std::cerr << "No matrix has been given with the parameter --matrix matrix." << std::endl;
    }
    exit(1);
  }

  tbsla::mpi::Matrix *m;

  if(format == "COO" | format == "coo") {
    m = new tbsla::mpi::MatrixCOO();
  } else if(format == "SCOO" | format == "scoo") {
    m = new tbsla::mpi::MatrixSCOO();
  } else if(format == "CSR" | format == "csr") {
    m = new tbsla::mpi::MatrixCSR();
  } else if(format == "ELL" | format == "ell") {
    m = new tbsla::mpi::MatrixELL();
  } else if(format == "DENSE" | format == "dense") {
    m = new tbsla::mpi::MatrixDENSE();
  } else {
    if(rank == 0) {
      std::cerr << format << " unrecognized!" << std::endl;
    }
    exit(1);
  }
  auto t_app_start = now();

  if(matrix == "cdiag") {
    auto t_one = now();
    m->fill_cdiag(matrix_dim, matrix_dim, C, rank / GC, rank % GC, GR, GC);
    auto t_two = now();
    double* s = new double[m->get_ln_row()];
    double* b1 = new double[m->get_ln_row()];
    double* b2 = new double[1];
    for(int i = 0; i < m->get_ln_row(); i++) {
      s[i] = 0;
      b1[i] = 0;
    }
    auto t_three = now();
    std::cout << "Make diagonally dominante with buffers sizes = " << matrix_dim << " and " << m->get_ln_col() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_diagonally_dominant(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Diagonally dominante matrix" << std::endl;
    auto t_four = now();
    delete[] s;
    delete[] b1;
    delete[] b2;
    double* s = new double[m->get_ln_col()];
    double* b1 = new double[m->get_ln_col()];
    double* b2 = new double[1];
    for(int i = 0; i < m->get_ln_col(); i++) {
      s[i] = 0;
      b1[i] = 0;
    }
    std::cout << "Normalizing with buffers sizes = " << matrix_dim << " and " << m->get_ln_col() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_stochastic(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_five = now();
    std::cout << "Normalized matrix" << std::endl;
    delete[] s;
    delete[] b1;
    delete[] b2;
    std::cout << "Matrix generation complete" << std::endl;
    std::cout << "Time random filling = " << std::to_string((t_two-t_one) / 1e9) << std::endl;
    std::cout << "Time making diagonally dominante = " << std::to_string((t_four-t_three) / 1e9) << std::endl;
    std::cout << "Time normalization = " << std::to_string((t_five-t_four) / 1e9) << std::endl;
  } else if(matrix == "cqmat") {
    m->fill_cqmat(matrix_dim, matrix_dim, C, Q, S, rank / GC, rank % GC, GR, GC);
    double* s = new double[m->get_ln_row()];
    double* b1 = new double[m->get_ln_row()];
    double* b2 = new double[1];
    std::cout << "Make diagonally dominante with buffers sizes = " << matrix_dim << " and " << m->get_ln_col() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_diagonally_dominant(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Diagonally dominante matrix" << std::endl;
    delete[] s;
    delete[] b1;
    delete[] b2;
    double* s = new double[m->get_ln_col()];
    double* b1 = new double[m->get_ln_col()];
    double* b2 = new double[1];
    for(int i = 0; i < m->get_ln_col(); i++) {
      s[i] = 0;
      b1[i] = 0;
    }
    std::cout << "Normalizing cqmat with buffers sizes = " << matrix_dim << " and " << m->get_ln_col() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_stochastic(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Normalized matrix" << std::endl;
    delete[] s;
    delete[] b1;
    delete[] b2;
    std::cout << "Matrix generation complete" << std::endl;
  } else if(matrix == "random_stoch") {
    auto t_one = now();
    m->fill_random(matrix_dim, matrix_dim, NNZ,  rank / GC, rank % GC, GR, GC);
    auto t_two = now();
    //double* s = new double[matrix_dim];
    double* s = new double[m->get_ln_row()];
    double* b1 = new double[m->get_ln_row()];
    //double* b2 = new double[m->get_ln_col()];
    double* b2 = new double[1];
    /*for(int i = 0; i < matrix_dim; i++) {
      s[i] = 0;
    }
    for(int i = 0; i < m->get_ln_col(); i++) {
      b1[i] = 0;
      b2[i] = 0;
    }*/
    for(int i = 0; i < m->get_ln_row(); i++) {
      s[i] = 0;
      b1[i] = 0;
    }
    auto t_three = now();
    std::cout << "Make diagonally dominante with buffers sizes = " << matrix_dim << " and " << m->get_ln_col() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_diagonally_dominant(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    m->print(std::cout) << std::endl;
    m->print_as_dense(std::cout) << std::endl;
    std::cout << "Diagonally dominante matrix" << std::endl;
    auto t_four = now();
    delete[] s;
    delete[] b1;
    delete[] b2;
    double* s = new double[m->get_ln_col()];
    double* b1 = new double[m->get_ln_col()];
    double* b2 = new double[1];
    for(int i = 0; i < m->get_ln_col(); i++) {
      s[i] = 0;
      b1[i] = 0;
    }
    std::cout << "Normalizing with buffers sizes = " << matrix_dim << " and " << m->get_ln_col() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_stochastic(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_five = now();
    std::cout << "Normalized matrix" << std::endl;
    std::cout << "Matrix" << std::endl;
    m->print(std::cout) << std::endl;
    m->print_as_dense(std::cout) << std::endl;
    delete[] s;
    delete[] b1;
    delete[] b2;
    std::cout << "Matrix generation complete" << std::endl;
    std::cout << "Time random filling = " << std::to_string((t_two-t_one) / 1e9) << std::endl;
    std::cout << "Time making diagonally dominante = " << std::to_string((t_four-t_three) / 1e9) << std::endl;
    std::cout << "Time normalization = " << std::to_string((t_five-t_four) / 1e9) << std::endl;
    } else if(matrix == "cdistrib") {
    auto t_one = now();
    m->fill_cdistrib(matrix_dim, matrix_dim, NNZ, S, rank / GC, rank % GC, GR, GC);
    auto t_two = now();
    //double* s = new double[matrix_dim];
    double* s = new double[m->get_ln_col()];
    double* b1 = new double[m->get_ln_col()];
    //double* b2 = new double[m->get_ln_col()];
    double* b2 = new double[1];
    /*for(int i = 0; i < matrix_dim; i++) {
      s[i] = 0;
    }
    for(int i = 0; i < m->get_ln_col(); i++) {
      b1[i] = 0;
      b2[i] = 0;
    }*/
    for(int i = 0; i < m->get_ln_col(); i++) {
      s[i] = 0;
      b1[i] = 0;
    }
    auto t_three = now();
    std::cout << "Make diagonally dominante with buffers sizes = " << matrix_dim << " and " << m->get_ln_col() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_diagonally_dominant(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    m->print(std::cout) << std::endl;
    m->print_as_dense(std::cout) << std::endl;
    std::cout << "Diagonally dominante matrix" << std::endl;
    auto t_four = now();
    std::cout << "Matrix" << std::endl;
    m->print(std::cout) << std::endl;
    m->print_as_dense(std::cout) << std::endl;
    delete[] s;
    delete[] b1;
    delete[] b2;
    std::cout << "Matrix generation complete" << std::endl;
    std::cout << "Time random filling = " << std::to_string((t_two-t_one) / 1e9) << std::endl;
    std::cout << "Time making diagonally dominante = " << std::to_string((t_four-t_three) / 1e9) << std::endl;
    }else if(matrix == "brain") {
    std::cout << "Init brain structure..." << std::endl;
    std::vector<std::vector<double> > proba_conn;
    std::vector<std::unordered_map<int,std::vector<int> > > brain_struct;
    int* neuron_type = new int[matrix_dim]();
    generate_brain_structs(matrix_dim, NNZ, proba_conn, brain_struct, neuron_type);
    std::cout << "...done" << std::endl;
    auto t_one = now();
    m->fill_brain(matrix_dim, matrix_dim, neuron_type, proba_conn, brain_struct, S, rank / GC, rank % GC, GR, GC);
    auto t_two = now();
    double* s = new double[m->get_ln_row()];
    double* b1 = new double[m->get_ln_row()];
    double* b2 = new double[1];
    for(int i = 0; i < m->get_ln_row(); i++) {
      s[i] = 0;
      b1[i] = 0;
    }
    auto t_three = now();
    std::cout << "Make diagonally dominante with buffers sizes = " << matrix_dim << " and " << m->get_ln_col() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_diagonally_dominant(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Diagonally dominante matrix" << std::endl;
    auto t_four = now();
    std::cout << "Normalizing with buffers sizes = " << matrix_dim << " and " << m->get_ln_row() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    m->make_stochastic(MPI_COMM_WORLD, s, b1, b2);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_five = now();
    std::cout << "Normalized matrix" << std::endl;
    delete[] s;
    delete[] b1;
    delete[] b2;
    delete[] neuron_type;
    std::cout << "Matrix generation complete" << std::endl;
    std::cout << "Time random filling = " << std::to_string((t_two-t_one) / 1e9) << std::endl;
    std::cout << "Time making diagonally dominante = " << std::to_string((t_four-t_three) / 1e9) << std::endl;
    std::cout << "Time normalization = " << std::to_string((t_five-t_four) / 1e9) << std::endl;
  } else {
    std::string filepath = matrix_folder + "/" + matrix + "." + format;
    std::ifstream f(filepath);
    if(!f.good()) {
      std::cerr << filepath << " cannot be open!" << std::endl;
    }
    m->read_bin_mpiio(MPI_COMM_WORLD, filepath, rank / GC, rank % GC, GR, GC);
    f.close();
  }

  if(input.has_opt("--numa-init")) {
    m->NUMAinit();
  }

  long int sum_nnz = m->compute_sum_nnz(MPI_COMM_WORLD);

  auto t_op_start = now();

  int nb_iterations_total = 0;
  int n_runs = 10;
  std::vector<double> runtimes;
  std::vector<double> gflops;
  std::cout << "Matrix" << std::endl;
  m->print_as_dense(std::cout) << std::endl;
  
  /*for(int ir=0; ir<n_runs; ir++) {
    int nb_iterations_done;
    std::cout << "Running PageRank - Iteration " << ir << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_start_iter = now();
    //double* res = m->page_rank(MPI_COMM_WORLD, beta, epsilon, max_iterations, nb_iterations_done);
    double* res = m->conjugate_gradient_opticom(max_iterations, beta, epsilon, nb_iterations_done);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_end_iter = now();
    std::cout << "...finished" << std::endl;
    double rt = (t_op_end_iter - t_op_start_iter) / 1e9;
    runtimes.push_back(rt);
    double gfl_local = compute_gflops_pagerank(rt, m->get_n_col(), m->get_nnz(), nb_iterations_done);
    long double gfl = compute_gflops_pagerank(rt, m->get_n_col(), sum_nnz, nb_iterations_done);
    gflops.push_back(gfl);
    std::cout << "runtime op = " << rt << std::endl;
    std::cout << "gflops local op = " << gfl_local << std::endl;
    std::cout << "gflops total op = " << gfl << std::endl;
    for(int ires=0; ires<10; ires++)
      std::cout << res[ires] << "  ";
    std::cout << ".....";
    //for(int ires=(matrix_dim-10); ires<matrix_dim; ires++)
    //  std::cout << res[ires] << "  ";
    std::cout << "converged in " << nb_iterations_done << " iterations" << std::endl;
    nb_iterations_total += nb_iterations_done;
    delete[] res;
  }
  auto t_op_end = now();
  double median_op_time = compute_median(runtimes);
  double median_gflops = compute_median(gflops);

  long int min_nnz = m->compute_min_nnz(MPI_COMM_WORLD);
  long int max_nnz = m->compute_max_nnz(MPI_COMM_WORLD);

  if(rank == 0) {
    auto t_app_end = now();

    std::map<std::string, std::string> outmap;
    outmap["test"] = "page_rank";
    outmap["matrix_format"] = format;
    outmap["matrix_dim"] = std::to_string(m->get_n_col());
    outmap["g_row"] = gr_string;
    outmap["g_col"] = gc_string;
    outmap["nnz"] = std::to_string(sum_nnz);
    outmap["nnz_min"] = std::to_string(min_nnz);
    outmap["nnz_max"] = std::to_string(max_nnz);
    outmap["nb_iterations"] = std::to_string(nb_iterations_total);
    outmap["time_app_in"] = std::to_string((t_app_end - t_app_start) / 1e9);
    //outmap["time_op"] = std::to_string((t_op_end - t_op_start) / 1e9);
    outmap["time_op"] = std::to_string(median_op_time);
    outmap["gflops"] = std::to_string(median_gflops);
    outmap["time_gen_mat"] = std::to_string((t_op_start - t_app_start) / 1e9);
    outmap["gnnz"] = std::to_string(m->get_gnnz());
    outmap["processes"] = std::to_string(world);
#if TBSLA_COMPILED_WITH_OMP
    outmap["lang"] = "MPIOMP";
    outmap["omp_threads"] = std::to_string(omp_get_max_threads());
#else
    outmap["lang"] = "MPI";
#endif
    outmap["matrix_type"] = matrix;
    outmap["compiler"] = std::string(CMAKE_CXX_COMPILER_ID) + " " + std::string(CMAKE_CXX_COMPILER_VERSION);
    outmap["compile_options"] = std::string(CMAKE_BUILD_TYPE);
    if (std::string(CMAKE_CXX_FLAGS).length() > 0) {
      outmap["compile_options"] += " " + std::string(CMAKE_CXX_FLAGS);
    }
    if (std::string(CMAKE_BUILD_TYPE) == "Release" && std::string(CMAKE_CXX_FLAGS_RELEASE).length() > 0) {
      outmap["compile_options"] += " " + std::string(CMAKE_CXX_FLAGS_RELEASE);
    } else if (std::string(CMAKE_BUILD_TYPE) == "Debug" && std::string(CMAKE_CXX_FLAGS_DEBUG).length() > 0) {
      outmap["compile_options"] += " " + std::string(CMAKE_CXX_FLAGS_DEBUG);
    }
#if TBSLA_COMPILED_WITH_OMP
    outmap["compile_options"] += " " + std::string(OpenMP_CXX_FLAGS);
#endif
    outmap["vectorization"] = m->get_vectorization();
    if(matrix == "cdiag") {
      outmap["cdiag_c"] = std::to_string(C);
    } else if(matrix == "cqmat") {
      outmap["cqmat_c"] = std::to_string(C);
      outmap["cqmat_q"] = std::to_string(Q);
      outmap["cqmat_s"] = std::to_string(S);
    }
    if(input.has_opt("--numa-init")) {
      outmap["numa_init"] = "true";
    }

    std::map<std::string, std::string>::iterator it=outmap.begin();
    std::cout << "{\"" << it->first << "\":\"" << it->second << "\"";
    it++;
    for (; it != outmap.end(); ++it) {
      std::cout << ",\"" << it->first << "\":\"" << it->second << "\"";
    }
    std::cout << "}\n";
  }*/

  MPI_Finalize();
}
