#include <tbsla/petsc/Matrix.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>
#include <tbsla/cpp/utils/range.hpp>

#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <string>
#include <iostream>

#include <mpi.h>
#include <petsc.h>

static char help[] = "tbsla perf petsc";

static std::uint64_t now() {
  std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(ns.count());
}

int main(int argc, char** argv) {
  InputParser input(argc, argv);
  PetscInitialize(&argc, &argv, (char*)0, help);
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string nr_string = input.get_opt("--NR", "1024");
  std::string nc_string = input.get_opt("--NC", "1024");

  int NR = std::stoi(nr_string);
  int NC = std::stoi(nc_string);
  int C = -1;
  double Q = -1;
  int S = -1;

  if(input.has_opt("--cdiag")) {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
  } else if(input.has_opt("--cqmat")) {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
    std::string q_string = input.get_opt("--Q", "0.1");
    Q = std::stod(q_string);
    std::string s_string = input.get_opt("--S", "0");
    S = std::stoi(s_string);
  } else {
    if(rank == 0) {
      std::cerr << "No matrix type has been given (--cdiag or --cqmat)." << std::endl;
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
  if(op != "spmv" && op != "a_axpx") {
    if(rank == 0) {
      std::cerr << "OP : " << op << " unrecognized!" << std::endl;
    }
    exit(1);
  }

  tbsla::petsc::Matrix m;

  auto t_app_start = now();


  if(input.has_opt("--cdiag")) {
    m.fill_cdiag(MPI_COMM_WORLD, NR, NC, C);
  } else if(input.has_opt("--cqmat")) {
    m.fill_cqmat(MPI_COMM_WORLD, NR, NC, C, Q, S);
  }

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_real_distribution<double> dist {-1, 1};
  auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
  std::vector<double> vec(m.get_n_col());
  std::generate(begin(vec), end(vec), gen);

  int v_start = tbsla::utils::range::pflv(vec.size(), rank, world);
  int v_n = tbsla::utils::range::lnv(vec.size(), rank, world);
  Vec petscv;
  VecCreateMPIWithArray(MPI_COMM_WORLD, 1, v_n, vec.size(), vec.data() + v_start, &petscv);

  MPI_Barrier(MPI_COMM_WORLD);
  auto t_op_start = now();
  if(op == "spmv") {
    m.spmv(MPI_COMM_WORLD, petscv);
  } else if(op == "a_axpx") {
    m.a_axpx_(MPI_COMM_WORLD, petscv);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  auto t_op_end = now();


  if(rank == 0) {
    auto t_app_end = now();

    std::map<std::string, std::string> outmap;
    outmap["test"] = op;
    outmap["matrix_format"] = "PETSC";
    outmap["n_row"] = std::to_string(m.get_n_row());
    outmap["n_col"] = std::to_string(m.get_n_col());
    outmap["time_app_in"] = std::to_string((t_app_end - t_app_start) / 1e9);
    outmap["time_op"] = std::to_string((t_op_end - t_op_start) / 1e9);
    outmap["lang"] = "MPI";
    if(input.has_opt("--cdiag")) {
      outmap["matrix_type"] = "cdiag";
      outmap["cdiag_c"] = std::to_string(C);
    } else if(input.has_opt("--cqmat")) {
      outmap["matrix_type"] = "cqmat";
      outmap["cqmat_c"] = std::to_string(C);
      outmap["cqmat_q"] = std::to_string(Q);
      outmap["cqmat_s"] = std::to_string(S);
    }

    std::map<std::string, std::string>::iterator it=outmap.begin();
    std::cout << "{\"" << it->first << "\":\"" << it->second << "\"";
    it++;
    for (; it != outmap.end(); ++it) {
      std::cout << ",\"" << it->first << "\":\"" << it->second << "\"";
    }
    std::cout << "}\n";
  }

  PetscFinalize();
}
