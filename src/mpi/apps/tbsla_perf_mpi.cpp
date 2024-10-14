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

  std::string nr_string = input.get_opt("--NR", "1024");
  std::string nc_string = input.get_opt("--NC", "1024");

  std::string gr_string = input.get_opt("--GR", "1");
  std::string gc_string = input.get_opt("--GC", "1");

  int NR = std::stoi(nr_string);
  int NC = std::stoi(nc_string);
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

  std::string op = input.get_opt("--op");
  if(op == "") {
    if(rank == 0) {
      std::cerr << "An operation (spmv, a_axpx) has to be given with the parameter --op op" << std::endl;
    }
    exit(1);
  }
  if(op != "spmv" && op != "a_axpx" && op != "spmv_no_redist" && op != "Ax" && op != "Ax_" && op != "AAxpAx" && op != "AAxpAxpx") {
    if(rank == 0) {
      std::cerr << "OP : " << op << " unrecognized!" << std::endl;
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
    m->fill_cdiag(NR, NC, C, rank / GC, rank % GC, GR, GC);
  } else if(matrix == "cqmat") {
    m->fill_cqmat(NR, NC, C, Q, S, rank / GC, rank % GC, GR, GC);
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

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_real_distribution<double> dist {-1, 1};
  auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
  double* vec = new double[m->get_n_col()];
  std::generate(vec, vec + m->get_n_col(), gen);

  MPI_Barrier(MPI_COMM_WORLD);
  std::uint64_t t_op = 0;
  if(op == "spmv") {
    auto t_op_start = now();
    MPI_Barrier(MPI_COMM_WORLD);
    double* res = m->spmv(MPI_COMM_WORLD, vec);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_end = now();
    t_op = t_op_end - t_op_start;
  } else if(op == "spmv_no_redist") {
    auto t_op_start = now();
    MPI_Barrier(MPI_COMM_WORLD);
    double* res = m->spmv_no_redist(MPI_COMM_WORLD, vec);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_end = now();
    t_op = t_op_end - t_op_start;
  } else if(op == "Ax") {
    double* res = new double[m->get_n_row()]();
    double* buffer = new double[m->get_ln_row()]();
    double* buffer2 = new double[m->get_ln_row()]();
    auto t_op_start = now();
    MPI_Barrier(MPI_COMM_WORLD);
    m->Ax(MPI_COMM_WORLD, res, vec, buffer, buffer2);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_end = now();
    t_op = t_op_end - t_op_start;
  } else if(op == "AAxpAx") {
    double* res = new double[m->get_n_row()]();
    double* buffer = new double[m->get_ln_row()]();
    double* buffer2 = new double[m->get_ln_row()]();
    double* buffer3 = new double[m->get_n_row()]();
    auto t_op_start = now();
    MPI_Barrier(MPI_COMM_WORLD);
    m->AAxpAx(MPI_COMM_WORLD, res, vec, buffer, buffer2, buffer3);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_end = now();
    t_op = t_op_end - t_op_start;
  } else if(op == "AAxpAxpx") {
    double* res = new double[m->get_n_row()]();
    double* buffer = new double[m->get_ln_row()]();
    double* buffer2 = new double[m->get_ln_row()]();
    double* buffer3 = new double[m->get_n_row()]();
    auto t_op_start = now();
    MPI_Barrier(MPI_COMM_WORLD);
    m->AAxpAxpx(MPI_COMM_WORLD, res, vec, buffer, buffer2, buffer3);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_end = now();
    t_op = t_op_end - t_op_start;
  } else if(op == "Ax_") {
    double* res = new double[m->get_ln_row()]();
    auto t_op_start = now();
    MPI_Barrier(MPI_COMM_WORLD);
    m->Ax_(MPI_COMM_WORLD, res, vec);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_end = now();
    t_op = t_op_end - t_op_start;
  } else if(op == "a_axpx") {
    auto t_op_start = now();
    MPI_Barrier(MPI_COMM_WORLD);
    double* res = m->a_axpx_(MPI_COMM_WORLD, vec);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_op_end = now();
    t_op = t_op_end - t_op_start;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  long int sum_nnz = m->compute_sum_nnz(MPI_COMM_WORLD);
  long int min_nnz = m->compute_min_nnz(MPI_COMM_WORLD);
  long int max_nnz = m->compute_max_nnz(MPI_COMM_WORLD);

  if(rank == 0) {
    auto t_app_end = now();

    std::map<std::string, std::string> outmap;
    outmap["test"] = op;
    outmap["matrix_format"] = format;
    outmap["n_row"] = std::to_string(m->get_n_row());
    outmap["n_col"] = std::to_string(m->get_n_col());
    outmap["g_row"] = gr_string;
    outmap["g_col"] = gc_string;
    outmap["nnz"] = std::to_string(sum_nnz);
    outmap["gnnz"] = std::to_string(m->get_gnnz());
    outmap["nnz_min"] = std::to_string(min_nnz);
    outmap["nnz_max"] = std::to_string(max_nnz);
    outmap["time_app_in"] = std::to_string((t_app_end - t_app_start) / 1e9);
    outmap["time_op"] = std::to_string(t_op / 1e9);
    outmap["processes"] = std::to_string(world);
    if(op == "spmv" or op == "spmv_no_redist" or op == "Ax" or op == "Ax_") {
      outmap["gflops"] = std::to_string(2.0 * m->get_nnz() / t_op);
    } else if(op == "a_axpx" or op == "AAxpAx") {
      outmap["gflops"] = std::to_string(4.0 * m->get_nnz() / t_op);
    } else if(op == "AAxpAxpx") {
      outmap["gflops"] = std::to_string((4.0 * m->get_nnz() + m->get_n_col()) / t_op);
    }
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
    } else {
      outmap["numa_init"] = "false";
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
