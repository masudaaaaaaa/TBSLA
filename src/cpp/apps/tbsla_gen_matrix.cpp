#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixSCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>

int main(int argc, char** argv) {
  InputParser input(argc, argv);
  std::string matrix_input = input.get_opt("--matrix_input");

  std::string format = input.get_opt("--format");
  if(format == "") {
    std::cerr << "A file format has to be given with the parameter --format format" << std::endl;
    exit(1);
  }

  std::string nr_string = input.get_opt("--NR", "1024");
  std::string nc_string = input.get_opt("--NC", "1024");

  int NR = std::stoi(nr_string);
  int NC = std::stoi(nc_string);
  int C = -1;
  double Q = -1;
  int S = -1;
  int pr = -1;
  int pc = -1;
  int GR = -1;
  int GC = -1;

  if(input.has_opt("--cdiag")) {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
  }

  if(input.has_opt("--cqmat")) {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
    std::string q_string = input.get_opt("--Q", "0.1");
    Q = std::stod(q_string);
    std::string s_string = input.get_opt("--S", "0");
    S = std::stoi(s_string);
  }

  std::string str = input.get_opt("--pr", "0");
  pr = std::stoi(str);
  str = input.get_opt("--pc", "0");
  pc = std::stoi(str);
  str = input.get_opt("--GC", "1");
  GC = std::stoi(str);
  str = input.get_opt("--GR", "1");
  GR = std::stoi(str);

  tbsla::cpp::Matrix *m;

  if(format == "COO" | format == "coo") {
    m = new tbsla::cpp::MatrixCOO();
  } else if(format == "SCOO" | format == "scoo") {
    m = new tbsla::cpp::MatrixSCOO();
  } else if(format == "CSR" | format == "csr") {
    m = new tbsla::cpp::MatrixCSR();
  } else if(format == "ELL" | format == "ell") {
    m = new tbsla::cpp::MatrixELL();
  } else if(format == "DENSE" | format == "dense") {
    m = new tbsla::cpp::MatrixDENSE();
  } else {
    std::cerr << format << " unrecognized!" << std::endl;
    exit(1);
  }

  if(input.has_opt("--cdiag")) {
    m->fill_cdiag(NR, NC, C, pr, pc, GR, GC);
  }
  if(input.has_opt("--cqmat")) {
    m->fill_cqmat(NR, NC, C, Q, S, pr, pc, GR, GC);
  }

  if(input.has_opt("--print-matrix")) {
    std::cout << *m << std::endl;
  }
  if(input.has_opt("--print-dense")) {
    m->print_as_dense(std::cout);
  }
}
