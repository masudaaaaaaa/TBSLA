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

#include <mpi.h>

static std::uint64_t now() {
  std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(ns.count());
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
  int max_iterations = std::stoi(max_iterations_string);;
  int nb_iterations_done;
  int matrix_dim = std::stoi(matrix_dim_string);
  int GR = std::stoi(gr_string);
  int GC = std::stoi(gc_string);
  int C = -1;
  double Q = -1;
  int S = -1;

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
    m->fill_cdiag(matrix_dim, matrix_dim, C, rank / GC, rank % GC, GR, GC);
  } else if(matrix == "cqmat") {
    m->fill_cqmat(matrix_dim, matrix_dim, C, Q, S, rank / GC, rank % GC, GR, GC);
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

  MPI_Barrier(MPI_COMM_WORLD);
  double* v = new double[matrix_dim];
  double* r = new double[matrix_dim];
  for(int i = 0; i < matrix_dim; i++) {
    v[i] = 1;
    r[i] = 0;
  }
  auto t_op_start = now();
  m->CG(MPI_COMM_WORLD, v, r, max_iterations, nb_iterations_done);
  MPI_Barrier(MPI_COMM_WORLD);
  auto t_op_end = now();

  long int sum_nnz = m->compute_sum_nnz(MPI_COMM_WORLD);
  long int min_nnz = m->compute_min_nnz(MPI_COMM_WORLD);
  long int max_nnz = m->compute_max_nnz(MPI_COMM_WORLD);

  if(rank == 0) {
    auto t_app_end = now();

    std::map<std::string, std::string> outmap;
    outmap["test"] = "conjugate_gradient";
    outmap["matrix_format"] = format;
    outmap["matrix_dim"] = std::to_string(m->get_n_col());
    outmap["g_row"] = gr_string;
    outmap["g_col"] = gc_string;
    outmap["nnz"] = std::to_string(sum_nnz);
    outmap["nnz_min"] = std::to_string(min_nnz);
    outmap["nnz_max"] = std::to_string(max_nnz);
    outmap["nb_iterations"] = std::to_string(nb_iterations_done);
    outmap["time_app_in"] = std::to_string((t_app_end - t_app_start) / 1e9);
    outmap["time_op"] = std::to_string((t_op_end - t_op_start) / 1e9);
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
  }

  MPI_Finalize();
}
