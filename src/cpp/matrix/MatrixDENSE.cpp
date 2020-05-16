#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

std::ostream& tbsla::cpp::MatrixDENSE::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "----- DENSE -----" << std::endl;
  os << "-----------------" << std::endl;
  os << "n_row : " << this->n_row << std::endl;
  os << "n_col : " << this->n_col << std::endl;
  os << "n_values : " << this->nnz << std::endl;
  tbsla::utils::vector::print_dense_matrix(this->ln_row, this->ln_col, values, os);
  os << "-----------------" << std::endl;
  return os;
}

std::ostream& tbsla::cpp::MatrixDENSE::print_as_dense(std::ostream& os) {
  return this->print(os);
}

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixDENSE &m) {
  return m.print(os);
}

std::vector<double> tbsla::cpp::MatrixDENSE::spmv(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r (this->ln_row, 0);
  if(this->nnz == 0 || this->values.size() == 0)
    return r;
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->ln_col; j++) {
      r[i] += this->values[i * this->ln_col + j] * v[j];
    }
  }
  return r;
}

std::vector<double> tbsla::cpp::MatrixDENSE::a_axpx_(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r = this->spmv(v, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  r = this->spmv(r, vect_incr);
  return r;
}

std::ostream & tbsla::cpp::MatrixDENSE::print_infos(std::ostream &os) {
  os << "-----------------" << std::endl;
  os << "----- DENSE -----" << std::endl;
  os << "--- general   ---" << std::endl;
  os << "n_row : " << n_row << std::endl;
  os << "n_col : " << n_col << std::endl;
  os << "nnz : " << nnz << std::endl;
  os << "--- capacity  ---" << std::endl;
  os << "values : " << values.capacity() << std::endl;
  os << "--- size      ---" << std::endl;
  os << "values : " << values.size() << std::endl;
  os << "-----------------" << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixDENSE::print_stats(std::ostream &os) {
  int s = 0, u = 0, d = 0;
  os << "upper values : " << u << std::endl;
  os << "lower values : " << s << std::endl;
  os << "diag  values : " << d << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixDENSE::write(std::ostream &os) {
  os.write(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  os.write(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));
  os.write(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));

  size_t size_v = this->values.size();
  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->values.data()), this->values.size() * sizeof(double));
  return os;
}

std::istream & tbsla::cpp::MatrixDENSE::read(std::istream &is, std::size_t pos, std::size_t n) {
  is.read(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  is.read(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));
  is.read(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));

  size_t vec_size, depla_general, depla_local;
  depla_general = 4 * sizeof(int);

  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  depla_general += sizeof(size_t);

  return is;
}

void tbsla::cpp::MatrixDENSE::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->values.clear();

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  int nv = 0;
  for(int i = f_row; i < f_row + ln_row; i++) {
    int ii, jj;
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

  if(nv == 0)
    return;

  this->nnz = nv;
  this->values.resize(this->ln_col * this->ln_row, 0);

  for(int i = f_row; i < f_row + ln_row; i++) {
    int ii, jj;
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

  this->values.clear();

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  int incr = 0, nv = 0;
  for(int i = 0; i < min_; i++) {
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
  for(int i = 0; i < std::min(n_row, n_col) - min_; i++) {
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
  this->values.resize(this->ln_col * this->ln_row, 0);
  int lincr;
  int i;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    lincr = 0;
    for(int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
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
    for(int j = 0; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
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
