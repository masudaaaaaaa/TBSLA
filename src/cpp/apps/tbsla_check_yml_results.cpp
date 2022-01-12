#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixSCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/array.hpp>

#include <glob.h>
#include <numeric>

#define MATRIX_FORMAT_COO 1
#define MATRIX_FORMAT_SCOO 2
#define MATRIX_FORMAT_CSR 3
#define MATRIX_FORMAT_ELL 4
#define MATRIX_FORMAT_DENSE 5

struct metadata {
  int n_row, n_col; // global matrix size
  int ln_row, ln_col; // local matrix size
  int bn_row, bn_col; // block matrix size
  int pr, pc, gr, gc; // positionning in the fine grain grid
  int bpr, bpc, bgr, bgc; // positionning in the coarse grain grid
  int lpr, lpc, lgr, lgc; // positionning in the task fine grain grid
  int matrixformat;
};

struct vector {
  struct metadata data;
  std::vector<double> v;
};

struct matrix {
  struct metadata data;
  int matrixformat;
  tbsla::cpp::Matrix *m;
};

struct metadata read_metadata(std::istream & is) {
  struct metadata m;
  is.read(reinterpret_cast<char*>(&m.n_row), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.n_col), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.ln_row), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.ln_col), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.bn_row), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.bn_col), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.pr), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.pc), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.gr), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.gc), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.bpr), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.bpc), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.bgr), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.bgc), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.lpr), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.lpc), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.lgr), sizeof(int));
  is.read(reinterpret_cast<char*>(&m.lgc), sizeof(int));
  return m;
}

void print_metadata(std::ostream & os, struct metadata & m) {
  os << "nrow: " << m.n_row << "; ";
  os << "ncol: " << m.n_col << "; ";
  os << "lnrow: " << m.ln_row << "; ";
  os << "lncol: " << m.ln_col << "; ";
  os << "bnrow: " << m.bn_row << "; ";
  os << "bncol: " << m.bn_col << "; ";
  os << "pr: " << m.pr << "; ";
  os << "pc: " << m.pc << "; ";
  os << "gr: " << m.gr << "; ";
  os << "gc: " << m.gc << "; ";
  os << "bpr: " << m.bpr << "; ";
  os << "bpc: " << m.bpc << "; ";
  os << "bgr: " << m.bgr << "; ";
  os << "bgc: " << m.bgc << "; ";
  os << "lpr: " << m.lpr << "; ";
  os << "lpc: " << m.lpc << "; ";
  os << "lgr: " << m.lgr << "; ";
  os << "lgc: " << m.lgc;
}

struct vector read_vector(std::string filename) {
  struct vector sv;
  std::ifstream is(filename, std::ios::binary);
  sv.data = read_metadata(is);
  size_t size = -1;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  sv.v.resize(size);
  is.read(reinterpret_cast<char*>(sv.v.data()), size * sizeof(double));
  return sv;
}

struct matrix read_matrix(std::string filename) {
  struct matrix sm;
  sm.matrixformat = -1;
  std::ifstream is(filename, std::ios::binary);
  is.read(reinterpret_cast<char*>(&(sm.matrixformat)), sizeof(int));
  sm.data = read_metadata(is);

  if(sm.matrixformat == MATRIX_FORMAT_COO) {
    sm.m = new tbsla::cpp::MatrixCOO();
  } else if(sm.matrixformat == MATRIX_FORMAT_SCOO) {
    sm.m = new tbsla::cpp::MatrixSCOO();
  } else if(sm.matrixformat == MATRIX_FORMAT_CSR) {
    sm.m = new tbsla::cpp::MatrixCSR();
  } else if(sm.matrixformat == MATRIX_FORMAT_ELL) {
    sm.m = new tbsla::cpp::MatrixELL();
  } else if(sm.matrixformat == MATRIX_FORMAT_DENSE) {
    sm.m = new tbsla::cpp::MatrixDENSE();
  }

  sm.m->read(is);
  return sm;
}

int main(int argc, char** argv) {
  InputParser input(argc, argv);
  std::string matrix_input = input.get_opt("--matrix_input");
  std::string vector_input = input.get_opt("--vector_input");
  glob_t glob_result;
  if(input.has_opt("--matrix_input")) {
    glob(matrix_input.append("*").c_str(), GLOB_TILDE, NULL, &glob_result);
    for(unsigned int i = 0; i < glob_result.gl_pathc; i++){
      std::cout << glob_result.gl_pathv[i] << std::endl;
      struct matrix sm = read_matrix(std::string(glob_result.gl_pathv[i]));
      if(input.has_opt("--print-matrix")) {
        print_metadata(std::cout, sm.data);
        std::cout << std::endl << std::flush;
        std::cout << *sm.m << std::endl << std::flush;
      }
      if(input.has_opt("--print-dense")) {
        sm.m->print_as_dense(std::cout);
      }
    }
  }
  if(input.has_opt("--vector_input")) {
    glob(vector_input.append("*").c_str(), GLOB_TILDE, NULL, &glob_result);
    for(unsigned int i = 0; i < glob_result.gl_pathc; i++){
      std::cout << glob_result.gl_pathv[i] << std::endl;
      struct vector sv = read_vector(std::string(glob_result.gl_pathv[i]));
      if(input.has_opt("--print-vector")) {
        tbsla::utils::vector::streamvector<double>(std::cout, "v", sv.v);
        std::cout << std::endl << std::flush;
        print_metadata(std::cout, sv.data);
        std::cout << std::endl << std::flush;
      }
    }
  }
  if(input.has_opt("--compute-spmv")) {
    std::string nr_string = input.get_opt("--NR", "100");
    std::string nc_string = input.get_opt("--NC", "100");

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
      std::cerr << "Needs a matrix type (--cdiag or --cqmat)" << std::endl << std::flush;
      exit(1);
    }

    tbsla::cpp::MatrixCOO m;

    if(input.has_opt("--cdiag")) {
      m.fill_cdiag(NR, NC, C, 0, 0, 1, 1);
    } else if(input.has_opt("--cqmat")) {
      m.fill_cqmat(NR, NC, C, Q, S, 0, 0, 1, 1);
    }
    std::vector<double> v(NC);
    std::iota (std::begin(v), std::end(v), 0);
    std::string op = input.get_opt("--op");
    double* r;
    if(op == "spmv") {
      r = m.spmv(v.data(), 0);
    } else if(op == "a_axpx") {
      r = m.a_axpx_(v.data(), 0);
    }
    tbsla::utils::array::stream<double>(std::cout, "r (" + op + ")", r, NR);
    std::cout << std::endl << std::flush;
  }
}
