#include <tbsla/cpp/MatrixSCOO.hpp>
#include <tbsla/cpp/reduction.hpp>
#include <tbsla/cpp/utils/array.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/csr.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>

tbsla::cpp::MatrixSCOO::~MatrixSCOO() {
  if (this->values)
    delete[] this->values;
  if (this->row)
    delete[] this->row;
  if (this->col)
    delete[] this->col;
}

tbsla::cpp::MatrixSCOO::MatrixSCOO(int n_row, int n_col, double* values, int* row,  int* col) : values(NULL), row(NULL), col(NULL) {
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

tbsla::cpp::MatrixSCOO::MatrixSCOO(int n_row, int n_col, int n_values) : values(NULL), row(NULL), col(NULL) {
  if (this->values)
    delete[] this->values;
  if (this->row)
    delete[] this->row;
  if (this->col)
    delete[] this->col;
  this->values = new double[n_values];
  this->row = new int[n_values];
  this->col = new int[n_values];
  this->n_row = n_row;
  this->n_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  this->pr = 0;
  this->pc = 0;
  this->NR = 1;
  this->NC = 1;
}

tbsla::cpp::MatrixSCOO::MatrixSCOO(int n_row, int n_col) : values(NULL), row(NULL), col(NULL) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  this->pr = 0;
  this->pc = 0;
  this->NR = 1;
  this->NC = 1;
}

tbsla::cpp::MatrixSCOO::MatrixSCOO(const tbsla::cpp::MatrixCOO & m) : values(NULL), row(NULL), col(NULL) {
  this->n_row = m.get_n_row();
  this->n_col = m.get_n_col();
  this->ln_row = m.get_n_row();
  this->ln_col = m.get_n_col();
  this->f_row = m.get_f_row();
  this->f_col = m.get_f_col();
  this->pr = m.get_pr();
  this->pc = m.get_pc();
  this->NR = m.get_NR();
  this->NC = m.get_NC();
  this->nnz = m.get_nnz();
  if (this->values)
    delete[] this->values;
  this->values = new double[this->nnz]();
  if (this->row)
    delete[] this->row;
  this->row = new int[this->nnz]();
  if (this->col)
    delete[] this->col;
  this->col = new int[this->nnz]();
  for(int i = 0; i < this->nnz; i++) {
    this->values[i] = m.get_values()[i];
    this->col[i] = m.get_col()[i];
    this->row[i] = m.get_row()[i];
  }
}

std::ostream& tbsla::cpp::MatrixSCOO::print_as_dense(std::ostream& os) {
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

std::ostream& tbsla::cpp::MatrixSCOO::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "----- SCOO ------" << std::endl;
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

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixSCOO &m) {
  return m.print(os);
}

double* tbsla::cpp::MatrixSCOO::spmv(const double* v, int vect_incr) const {
  double* r = new double[this->ln_row];
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    r[i] = 0;
  }
  this->Ax(r, v, vect_incr);
  return r;
}

inline void tbsla::cpp::MatrixSCOO::Ax(double* r, const double* v, int vect_incr) const {
  if (this->nnz == 0) {
    return;
  }
  #pragma omp declare reduction(add_arr: tbsla::cpp::reduction::array<double> : omp_out.add(omp_in)) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
  tbsla::cpp::reduction::array<double> s(r, this->ln_row);
  #pragma omp parallel for reduction(add_arr:s) schedule(static)
  for (int i = 0; i < this->nnz; i++) {
     s[this->row[i] + vect_incr - this->f_row] += this->values[i] * v[this->col[i] - this->f_col];
  }
}

std::ostream & tbsla::cpp::MatrixSCOO::write(std::ostream &os) {
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

std::istream & tbsla::cpp::MatrixSCOO::read(std::istream &is, std::size_t pos, std::size_t n) {
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

void tbsla::cpp::MatrixSCOO::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  this->nnz = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    long int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      this->nnz++;
    }
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->nnz++;
      }
    }
  }

  if(this->nnz == 0)
    return;

  if (this->values)
    delete[] this->values;
  if (this->row)
    delete[] this->row;
  if (this->col)
    delete[] this->col;
  this->values = new double[this->nnz];
  this->row = new int[this->nnz];
  this->col = new int[this->nnz];

  long int idx = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    long int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      this->row[idx] = ii;
      this->col[idx] = jj;
      this->values[idx] = 1;
      idx++;
    }
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->row[idx] = ii;
        this->col[idx] = jj;
        this->values[idx] = 1;
        idx++;
      }
    }
  }
}

void tbsla::cpp::MatrixSCOO::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  long int incr = 0, nv = 0;
  for(long int i = 0; i < min_; i++) {
    if(i < f_row) {
      incr += std::min(c, n_col);
    }
    if(i >= f_row && i < f_row + ln_row) {
      nv += std::min(c, n_col);
    }
    if(i >= f_row + ln_row) {
      break;
    }
  }
  for(long int i = 0; i < std::min(n_row, n_col) - min_; i++) {
    if(i + min_ < f_row) {
      incr += std::min(c, n_col) - i - 1;
    }
    if(i + min_ >= f_row && i + min_ < f_row + ln_row) {
      nv += std::min(c, n_col) - i - 1;
    }
    if(i + min_ >= f_row + ln_row) {
      break;
    }
  }

  this->nnz = 0;
  long int incr_save = incr;

  long int i;
  for(i = f_row; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    long int j = 0;
    for(; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->nnz++;
      }
      incr++;
    }
  }

  incr = incr_save;

  if(nv == 0)
    return;

  if (this->values)
    delete[] this->values;
  if (this->row)
    delete[] this->row;
  if (this->col)
    delete[] this->col;
  this->values = new double[nv];
  this->row = new int[nv];
  this->col = new int[nv];

  long int idx = 0;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    for(long int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      double v;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      v = std::get<2>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->row[idx] = ii;
        this->col[idx] = jj;
        this->values[idx] = v;
        idx++;
      }
      incr++;
    }
  }
  for(; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    for(long int j = 0; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      double v;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      v = std::get<2>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->row[idx] = ii;
        this->col[idx] = jj;
        this->values[idx] = v;
        idx++;
      }
      incr++;
    }
  }
  this->nnz = idx;
}

void tbsla::cpp::MatrixSCOO::NUMAinit() {
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
