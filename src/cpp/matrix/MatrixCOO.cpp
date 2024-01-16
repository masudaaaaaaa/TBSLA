#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/reduction.hpp>
#include <tbsla/cpp/utils/array.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/split.hpp>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <sstream>

void tbsla::cpp::MatrixCOO::init(int n_row, int n_col, long int n_values) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  this->pr = 0;
  this->pc = 0;
  this->NR = 1;
  this->NC = 1;
  if (this->values)
    delete[] this->values;
  if (this->row)
    delete[] this->row;
  if (this->col)
    delete[] this->col;
  this->values = new double[n_values]();
  this->row = new int[n_values]();
  this->col = new int[n_values]();
}

tbsla::cpp::MatrixCOO::~MatrixCOO() {
  if (this->values)
    delete[] this->values;
  if (this->row)
    delete[] this->row;
  if (this->col)
    delete[] this->col;
}

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col, double* values, int* row,  int* col) {
  if (this->values)
    delete[] this->values;
  if (this->row)
    delete[] this->row;
  if (this->col)
    delete[] this->col;
  this->n_row = n_row;
  this->n_col = n_col;
  this->values = values;
  this->row = row;
  this->col = col;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  this->pr = 0;
  this->pc = 0;
  this->NR = 1;
  this->NC = 1;
}

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col, long int n_values) {
  this->init(n_row, n_col, n_values);
}

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  this->pr = 0;
  this->pc = 0;
  this->NR = 1;
  this->NC = 1;
}

std::ostream& tbsla::cpp::MatrixCOO::print_as_dense(std::ostream& os) {
  double* d = new double[this->n_row * this->n_col];
  for(int i = 0; i < this->n_row * this->n_col; i++) {
    d[i] = 0;
  }
  for(int i = 0; i < this->nnz; i++) {
    d[row[i] * this->n_col + col[i]] += this->values[i];
  }
  tbsla::utils::array::print_dense_matrix(this->n_row, this->n_col, d, os);
  return os;
}

std::ostream& tbsla::cpp::MatrixCOO::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "------ COO ------" << std::endl;
  os << "-----------------" << std::endl;
  os << "number of rows             -  n_row   : " << this->n_row << std::endl;
  os << "number of columns          -  n_col   : " << this->n_col << std::endl;
  os << "local number of rows       -  ln_row  : " << this->ln_row << std::endl;
  os << "local number of columns    -  ln_col  : " << this->ln_col << std::endl;
  os << "first row                  -  f_row   : " << this->f_row << std::endl;
  os << "first column               -  f_col   : " << this->f_col << std::endl;
  os << "number of non-zero elts    -  nnz     : " << this->nnz << std::endl;
  os << "block position (row)       -  pr      : " << this->pr << std::endl;
  os << "block position (column)    -  pc      : " << this->pc << std::endl;
  os << "number of blocks (row)     -  NR      : " << this->NR << std::endl;
  os << "number of blocks (column)  -  NC      : " << this->NC << std::endl;
  tbsla::utils::array::stream<double>(os, "val", this->values, this->nnz);
  os << std::endl;
  tbsla::utils::array::stream<int>(os, "row", this->row, this->nnz);
  os << std::endl;
  tbsla::utils::array::stream<int>(os, "col", this->col, this->nnz);
  os << std::endl;
  os << "-----------------" << std::endl << std::flush;
  return os;
}

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixCOO &m) {
  return m.print(os);
}

double* tbsla::cpp::MatrixCOO::spmv(const double* v, int vect_incr) const {
  double* r = new double[this->n_row]();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->n_row; i++) {
    r[i] = 0;
  }
  this->Ax(r, v, vect_incr);
  return r;
}

inline void tbsla::cpp::MatrixCOO::Ax(double* r, const double* v, int vect_incr) const {
  #pragma omp declare reduction(add_arr: tbsla::cpp::reduction::array<double> : omp_out.add(omp_in)) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
  tbsla::cpp::reduction::array<double> s(r, this->n_row);
  #pragma omp parallel for reduction(add_arr:s) schedule(static)
  for (std::size_t i = 0; i < this->nnz; i++) {
     s[this->row[i] + vect_incr] += this->values[i] * v[this->col[i]];
  }
}

std::ostream & tbsla::cpp::MatrixCOO::write(std::ostream &os) {
  os.write(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  os.write(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));
  os.write(reinterpret_cast<char*>(&this->ln_row), sizeof(this->ln_row));
  os.write(reinterpret_cast<char*>(&this->ln_col), sizeof(this->ln_col));
  os.write(reinterpret_cast<char*>(&this->f_row), sizeof(this->f_row));
  os.write(reinterpret_cast<char*>(&this->f_col), sizeof(this->f_col));
  os.write(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));
  os.write(reinterpret_cast<char*>(&this->pr), sizeof(this->pr));
  os.write(reinterpret_cast<char*>(&this->pc), sizeof(this->pc));
  os.write(reinterpret_cast<char*>(&this->NR), sizeof(this->NR));
  os.write(reinterpret_cast<char*>(&this->NC), sizeof(this->NC));

  size_t size_v = this->nnz;
  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->values), this->nnz * sizeof(double));

  size_t size_r = this->nnz;
  os.write(reinterpret_cast<char*>(&size_r), sizeof(size_r));
  os.write(reinterpret_cast<char*>(this->row), this->nnz * sizeof(int));

  size_t size_c = this->nnz;
  os.write(reinterpret_cast<char*>(&size_c), sizeof(size_c));
  os.write(reinterpret_cast<char*>(this->col), this->nnz * sizeof(int));
  return os;
}

std::istream & tbsla::cpp::MatrixCOO::read(std::istream &is, std::size_t pos, std::size_t n) {
  is.read(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  is.read(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));
  is.read(reinterpret_cast<char*>(&this->ln_row), sizeof(this->ln_row));
  is.read(reinterpret_cast<char*>(&this->ln_col), sizeof(this->ln_col));
  is.read(reinterpret_cast<char*>(&this->f_row), sizeof(this->f_row));
  is.read(reinterpret_cast<char*>(&this->f_col), sizeof(this->f_col));
  is.read(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));
  is.read(reinterpret_cast<char*>(&this->pr), sizeof(this->pr));
  is.read(reinterpret_cast<char*>(&this->pc), sizeof(this->pc));
  is.read(reinterpret_cast<char*>(&this->NR), sizeof(this->NR));
  is.read(reinterpret_cast<char*>(&this->NC), sizeof(this->NC));


  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  if (this->values)
    delete[] this->values;
  this->values = new double[size];
  is.read(reinterpret_cast<char*>(this->values), size * sizeof(double));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  if (this->row)
    delete[] this->row;
  this->row = new int[size];
  is.read(reinterpret_cast<char*>(this->row), size * sizeof(int));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  if (this->col)
    delete[] this->col;
  this->col = new int[size];
  is.read(reinterpret_cast<char*>(this->col), size * sizeof(int));
  return is;
}

void tbsla::cpp::MatrixCOO::readMM(std::string fname) {
  std::ifstream is(fname);
  std::string line;
  std::string delim(" ");

  if (is.is_open()) {
    getline(is, line);
    std::vector<std::string> splits = tbsla::utils::io::split(line, delim);
    std::cout << line << "\n";
    if(splits[0].compare(std::string("%%MatrixMarket")) == 0
       && splits[1].compare(std::string("matrix")) == 0
       && splits[2].compare(std::string("coordinate")) == 0
       && splits[3].compare(std::string("real")) == 0 ) {
      if(splits[4].compare(std::string("general")) == 0) {
        while(line.rfind("%", 0) == 0) {
          getline(is, line);
        }
        std::cout << line << "\n";
        std::stringstream ss(line);
        int nc, nr, nv;
        ss >> nr;
        ss >> nc;
        ss >> nv;
        this->init(nr, nc, nv);
        this->nnz = 0;
        int r, c;
        double v;
        for(int i = 0; i < nv && !is.eof(); i++) {
          is >> r;
          is >> c;
          is >> v;
          this->values[i] = v;
          this->row[i] = r - 1;
          this->col[i] = c - 1;
          this->nnz++;
        }
      } else if(splits[4].compare(std::string("symmetric")) == 0) {
        while(line.rfind("%", 0) == 0) {
          getline(is, line);
        }
        std::cout << line << "\n";
        std::stringstream ss(line);
        int nc, nr, nv;
        ss >> nr;
        ss >> nc;
        ss >> nv;
        this->init(nr, nc, nv * 2);
        this->nnz = 0;
        int r, c;
        double v;
        for(int i = 0; i < nv && !is.eof(); i++) {
          is >> r;
          is >> c;
          is >> v;
          this->values[i] = v;
          this->row[i] = r - 1;
          this->col[i] = c - 1;
          this->nnz++;
          if(r != c) {
            this->values[i] = v;
            this->row[i] = c - 1;
            this->col[i] = r - 1;
            this->nnz++;
          }
        }
      } else {
        throw tbsla::cpp::MatrixFormatReadException();
      }
    } else {
      throw tbsla::cpp::MatrixFormatReadException();
    }
  }
}

void tbsla::cpp::MatrixCOO::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  long int gnv = std::max(std::min(n_row, n_col - cdiag), 0) + std::max(std::min(n_row - cdiag, n_col), 0);
  if(cdiag == 0)
    gnv /= 2;
  this->nnz = 0;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->n_row = n_row;
  this->n_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  if(gnv == 0)
    return;

  long int s = tbsla::utils::range::pflv(gnv, pr * NC + pc, NR * NC);
  long int n = tbsla::utils::range::lnv(gnv, pr * NC + pc, NR * NC);
  this->nnz = n;

  this->init(n_row, n_col, n);
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;

  for(int i = 0; i < n; i++) {
    auto tuple = tbsla::utils::values_generation::cdiag_value(i + s, gnv, n_row, n_col, cdiag);
    this->row[i] = std::get<0>(tuple);
    this->col[i] = std::get<1>(tuple);
    this->values[i] = std::get<2>(tuple);
  }
}

void tbsla::cpp::MatrixCOO::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  long int gnv = 0;
  for(long int i = 0; i < std::min(n_col - std::min(c, n_col) + 1, n_row); i++) {
    gnv += std::min(c, n_col);
  }
  for(long int i = 0; i < std::min(n_row, n_col) - std::min(n_col - std::min(c, n_col) + 1, n_row); i++) {
    gnv += std::min(c, n_col) - i - 1;
  }
  this->nnz = 0;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->n_row = n_row;
  this->n_col = n_col;
  this->f_row = 0;
  this->f_col = 0;

  if(gnv == 0)
    return;

  long int s = tbsla::utils::range::pflv(gnv, pr * NC + pc, NR * NC);
  long int n = tbsla::utils::range::lnv(gnv, pr * NC + pc, NR * NC);
  this->nnz = n;
  this->init(n_row, n_col, n);

  for(long int i = 0; i < n; i++) {
    auto tuple = tbsla::utils::values_generation::cqmat_value(i + s, n_row, n_col, c, q, seed_mult);
    this->row[i] = std::get<0>(tuple);
    this->col[i] = std::get<1>(tuple);
    this->values[i] = std::get<2>(tuple);
  }
}

void tbsla::cpp::MatrixCOO::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->nnz = 0;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->n_row = n_row;
  this->n_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  
}

void tbsla::cpp::MatrixCOO::fill_brain(int n_row, int n_col, int* neuron_type, std::vector<std::vector<double> > proba_conn, std::vector<std::unordered_map<int,std::vector<int> > > brain_struct, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

}
//TODO : fill_cdistribute COO

void tbsla::cpp::MatrixCOO::fill_cdistrib(int n_row, int n_col, int nnz, int pr, int pc, int NR, int NC) {

}
// TODO : normalization for COO
void tbsla::cpp::MatrixCOO::get_row_sums(double* s) {
  
}

void tbsla::cpp::MatrixCOO::normalize_rows(double* s) {

}

void tbsla::cpp::MatrixCOO::get_col_sums(double* s) {
  
}

void tbsla::cpp::MatrixCOO::normalize_cols(double* s) {

}

void tbsla::cpp::MatrixCOO::set_diag(double* s) {

}

void tbsla::cpp::MatrixCOO::NUMAinit() {
  double* newVal = new double[this->nnz];
  int* newCol = new int[this->nnz];
  int* newRow = new int[this->nnz];

  //NUMA init
#pragma omp parallel for schedule(static)
  for(int idx = 0; idx < this->nnz; ++idx) {
    newCol[idx] = this->col[idx];
    newRow[idx] = this->row[idx];
    newVal[idx] = this->values[idx];
  }

  delete[] this->values;
  delete[] this->row;
  delete[] this->col;

  this->values = newVal;
  this->row = newRow;
  this->col = newCol;
}
