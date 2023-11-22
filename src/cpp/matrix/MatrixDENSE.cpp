#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/array.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>

tbsla::cpp::MatrixDENSE::~MatrixDENSE() {
  if (this->values)
    delete[] this->values;
}

tbsla::cpp::MatrixDENSE::MatrixDENSE(const tbsla::cpp::MatrixCOO & m) : values(NULL) {
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
  this->values = new double[this->ln_row * this->ln_col]();
  for(int i = 0; i < this->ln_row * this->ln_col; i++) {
    this->values[i] = 0;
  }
  for(int i = 0; i < m.get_nnz(); i++) {
    this->values[m.get_row()[i] * this->ln_col + m.get_col()[i]] += m.get_values()[i];
  }
}

std::ostream& tbsla::cpp::MatrixDENSE::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "----- DENSE -----" << std::endl;
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
  tbsla::utils::array::print_dense_matrix(this->ln_row, this->ln_col, values, os);
  os << "-----------------" << std::endl << std::flush;
  return os;
}

std::ostream& tbsla::cpp::MatrixDENSE::print_as_dense(std::ostream& os) {
  return this->print(os);
}

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixDENSE &m) {
  return m.print(os);
}

double* tbsla::cpp::MatrixDENSE::spmv(const double* v, int vect_incr) const {
  double* r = new double[this->ln_row]();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    r[i] = 0;
  }
  this->Ax(r, v, vect_incr);
  return r;
}

inline void tbsla::cpp::MatrixDENSE::Ax(double* r, const double* v, int vect_incr) const {
  if(this->nnz == 0 || this->values == NULL)
    return;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->ln_col; j++) {
      r[i] += this->values[i * this->ln_col + j] * v[j];
    }
  }
}

std::ostream & tbsla::cpp::MatrixDENSE::write(std::ostream &os) {
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

  size_t size_v = this->ln_row * this->ln_col;
  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->values), size_v * sizeof(double));
  return os;
}

std::istream & tbsla::cpp::MatrixDENSE::read(std::istream &is, std::size_t pos, std::size_t n) {
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
  this->values = new double[this->ln_row * this->ln_col];
  is.read(reinterpret_cast<char*>(this->values), size * sizeof(double));
  return is;
}

void tbsla::cpp::MatrixDENSE::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
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

  long int nv = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    long int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      nv++;
    }
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        nv++;
      }
    }
  }

  this->nnz = nv;
  if(nv == 0)
    return;

  if (this->values)
    delete[] this->values;
  this->values = new double[this->ln_row * this->ln_col];
  for(int i = 0; i < this->ln_row * this->ln_col; i++) {
    this->values[i] = 0;
  }

  for(long int i = f_row; i < f_row + ln_row; i++) {
    long int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      this->values[(ii - f_row) * this->ln_col + (jj - f_col)] = 1;
    }
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->values[(ii - f_row) * this->ln_col + (jj - f_col)] = 1;
      }
    }
  }
}

void tbsla::cpp::MatrixDENSE::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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


  if(nv == 0)
    return;

  this->nnz = 0;
  if (this->values)
    delete[] this->values;
  this->values = new double[this->ln_row * this->ln_col];
  for(int i = 0; i < this->ln_row * this->ln_col; i++) {
    this->values[i] = 0;
  }

  long int lincr;
  long int i;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    lincr = 0;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->values[(ii - f_row) * this->ln_col + (jj - f_col)] += std::get<2>(tuple);
        this->nnz++;
        lincr++;
      }
      incr++;
    }
  }
  for(; i < std::min({n_row, f_row + ln_row}); i++) {
    lincr = 0;
    for(long int j = 0; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->values[(ii - f_row) * this->ln_col + (jj - f_col)] += std::get<2>(tuple);
        this->nnz++;
        lincr++;
      }
      incr++;
    }
  }
}

void tbsla::cpp::MatrixDENSE::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;
  
}

void tbsla::cpp::MatrixDENSE::fill_brain(int n_row, int n_col, int* neuron_type, std::vector<std::vector<double> > proba_conn, std::vector<std::unordered_map<int,std::vector<int> > > brain_struct, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

}

void tbsla::cpp::MatrixDENSE::get_row_sums(double* s) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
	double sum = 0;
    for (int j = 0; j < this->ln_col; j++) {
      sum += this->values[i * this->ln_col + j];
    }
	s[i] = sum;
  }
}

void tbsla::cpp::MatrixDENSE::normalize_rows(double* s) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->ln_col; j++) {
      this->values[i * this->ln_col + j] /= s[i];
    }
  }
}

void tbsla::cpp::MatrixDENSE::get_col_sums(double* s) {
  
}

void tbsla::cpp::MatrixDENSE::normalize_cols(double* s) {

}

void tbsla::cpp::MatrixDENSE::set_diag(double* s) {

}

void tbsla::cpp::MatrixDENSE::NUMAinit() {
  double* newVal = new double[this->ln_row * this->ln_col];

  //NUMA init
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->ln_col; j++) {
      newVal[i * this->ln_col + j] = this->values[i * this->ln_col + j];
    }
  }

  delete[] this->values;
  this->values = newVal;
}
