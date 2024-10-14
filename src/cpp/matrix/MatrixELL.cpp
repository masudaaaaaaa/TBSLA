#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/utils/array.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/csr.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>

#if TBSLA_ENABLE_VECTO
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#endif

tbsla::cpp::MatrixELL::~MatrixELL() {
  if (this->values)
    delete[] this->values;
  if (this->columns)
    delete[] this->columns;
}

tbsla::cpp::MatrixELL::MatrixELL(const tbsla::cpp::MatrixCOO & m) : values(NULL), columns(NULL) {
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


  if(m.get_nnz() == 0) {
    this->max_col = 0;
    this->nnz = 0;
  } else {
    int* p = new int[this->nnz]();
    std::iota(p, p + this->nnz, 0);
    std::sort(p, p + this->nnz, [&](unsigned i, unsigned j){ return tbsla::cpp::utils::csr::compare_row(m.get_row(), m.get_col(), i, j); });

    int* srow = tbsla::cpp::utils::csr::applyPermutation<int>(p, m.get_row(), this->nnz);
    int* scol = tbsla::cpp::utils::csr::applyPermutation<int>(p, m.get_col(), this->nnz);
    double* sval = tbsla::cpp::utils::csr::applyPermutation<double>(p, m.get_values(), this->nnz);
  
    int* nvrow = new int[this->n_row]();
    for(int i = 0; i < this->n_row; i++) {
      nvrow[i] = 0;
    }
    for(int i = 0; i < this->nnz; i++) {
      nvrow[srow[i]]++;
    }
    this->max_col = *std::max_element(nvrow, nvrow + this->n_row);
    if (this->values)
      delete[] this->values;
    if (this->columns)
      delete[] this->columns;
    this->values = new double[this->n_row * this->max_col]();
    this->columns = new int[this->n_row * this->max_col]();
    std::size_t incr = 0;
    for(int i = 0; i < this->n_row; i++) {
      for(int j = 0; j < nvrow[i]; j++) {
        this->values[i * this->max_col + j] = sval[incr];
        this->columns[i * this->max_col + j] = scol[incr];
        incr++;
      }
    }
    delete[] p;
    delete[] nvrow;
    delete[] srow;
    delete[] scol;
    delete[] sval;
  }
}

std::ostream& tbsla::cpp::MatrixELL::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "------ ELL ------" << std::endl;
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
  os << "maximum number of columns  -  max_col : " << this->max_col << std::endl;
  tbsla::utils::array::stream<double>(os, "val", this->values, this->ln_row * this->max_col);
  os << std::endl;
  tbsla::utils::array::stream<int>(os, "col", this->columns, this->ln_row * this->max_col);
  os << std::endl;
  os << "-----------------" << std::endl << std::flush;
  return os;
}

std::ostream& tbsla::cpp::MatrixELL::print_as_dense(std::ostream& os) {
  double* d = new double[this->n_row * this->n_col];
  for (int i = 0; i < std::min((long int)this->n_row, this->nnz / this->max_col); i++) {
    for (int j = 0; j < this->max_col; j++) {
      d[i * this->n_col + this->columns[i * this->max_col + j]] += this->values[i * this->max_col + j];
    }
  }
  tbsla::utils::array::print_dense_matrix(this->n_row, this->n_col, d, os);
  return os;
}

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixELL &m) {
  return m.print(os);
}

double* tbsla::cpp::MatrixELL::spmv(const double* v, int vect_incr) const {
  double* r = new double[this->ln_row]();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    r[i] = 0;
  }
  this->Ax(r, v, vect_incr);
  return r;
}

#if TBSLA_ENABLE_VECTO
#ifdef __ARM_FEATURE_SVE
std::string tbsla::cpp::MatrixELL::get_vectorization() const {
  return "ARM_SVE";
}

inline void tbsla::cpp::MatrixELL::Ax(double* r, const double* v, int vect_incr) const {
  if (this->nnz == 0)
    return;
  if (this->f_col == 0) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < this->ln_row; i++) {
      svfloat64_t tmp; tmp = svadd_z(svpfalse(), tmp, tmp);
      int j = 0;
      svbool_t pg = svwhilelt_b64(j, this->max_col);
      do {
        svfloat64_t values_vec = svld1(pg, &(this->values[i * this->max_col + j]));
        svuint64_t col = svld1sw_u64(pg, &(this->columns[i * this->max_col + j]));
        svfloat64_t v_vec = svld1_gather_index(pg, v, col);
        tmp = svmla_m(pg, tmp, values_vec, v_vec);
        j += svcntd();
        pg = svwhilelt_b64(j, this->max_col);
      } while (svptest_any(svptrue_b64(), pg));
      r[i] = svaddv(svptrue_b64(), tmp);
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < this->ln_row; i++) {
      svfloat64_t tmp; tmp = svadd_z(svpfalse(), tmp, tmp);
      int j = 0;
      uint64_t fix = this->f_col;
      svbool_t pg = svwhilelt_b64(j, this->max_col);
      do {
        svfloat64_t values_vec = svld1(pg, &(this->values[i * this->max_col + j]));
        svuint64_t col = svld1sw_u64(pg, &(this->columns[i * this->max_col + j]));
        svuint64_t col_fix = svsub_z(pg, col, fix);
        svfloat64_t v_vec = svld1_gather_index(pg, v, col_fix);
        tmp = svmla_m(pg, tmp, values_vec, v_vec);
        j += svcntd();
        pg = svwhilelt_b64(j, this->max_col);
      } while (svptest_any(svptrue_b64(), pg));
      r[i] = svaddv(svptrue_b64(), tmp);
    }
  }
}
#endif // __ARM_FEATURE_SVE
#else
std::string tbsla::cpp::MatrixELL::get_vectorization() const {
  return "None";
}

inline void tbsla::cpp::MatrixELL::Ax(double* r, const double* v, int vect_incr) const {
  if(this->nnz == 0 || this->max_col == 0 || this->values == NULL || this->columns == NULL)
    return;
  if (this->f_col == 0) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < this->ln_row; i++) {
      double tmp = 0;
      for (int j = 0; j < this->max_col; j++) {
        tmp += this->values[i * this->max_col + j] * v[this->columns[i * this->max_col + j]];
      }
      r[i] = tmp;
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < this->ln_row; i++) {
      double tmp = 0;
      for (int j = 0; j < this->max_col; j++) {
        int idx = this->columns[i * this->max_col + j] - this->f_col;
        if(idx < 0) idx = 0;
        tmp += this->values[i * this->max_col + j] * v[idx];
      }
      r[i] = tmp;
    }
  }
}
#endif // TBSLA_ENABLE_VECTO

std::ostream & tbsla::cpp::MatrixELL::write(std::ostream &os) {
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
  os.write(reinterpret_cast<char*>(&this->max_col), sizeof(this->max_col));

  size_t size_v = this->ln_row * this->max_col;
  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->values), size_v * sizeof(double));

  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->columns), size_v * sizeof(int));
  return os;
}

std::istream & tbsla::cpp::MatrixELL::read(std::istream &is, std::size_t pos, std::size_t n) {
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
  is.read(reinterpret_cast<char*>(&this->max_col), sizeof(this->max_col));

  if (this->values)
    delete[] this->values;
  if (this->columns)
    delete[] this->columns;
  this->values = new double[this->ln_row * this->max_col];
  this->columns = new int[this->ln_row * this->max_col];

  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  is.read(reinterpret_cast<char*>(this->values), size * sizeof(double));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  is.read(reinterpret_cast<char*>(this->columns), size * sizeof(int));
  return is;
}

void tbsla::cpp::MatrixELL::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  if(cdiag == 0)
    this->max_col = 1;
  else
    this->max_col = 2;

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
  if (this->columns)
    delete[] this->columns;
  this->values = new double[this->ln_row * this->max_col]();
  this->columns = new int[this->ln_row * this->max_col]();

  int idx = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    long int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      this->columns[idx] = jj;
      this->values[idx] = 1;
    } else {
      this->columns[idx] = 0;
      this->values[idx] = 0;
    }
    idx++;
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->columns[idx] = jj;
        this->values[idx] = 1;
      } else {
        this->columns[idx] = 0;
        this->values[idx] = 0;
      }
      idx++;
    }
  }
}

void tbsla::cpp::MatrixELL::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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
  this->max_col = 0;
  long int incr_save = incr;

  long int i;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    int nbc = 0;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->nnz++;
        nbc++;
      }
      incr++;
    }
    this->max_col = std::max(this->max_col, nbc);
  }
  for(; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    int nbc = 0;
    long int j = 0;
    for(; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        nbc++;
        this->nnz++;
      }
      incr++;
    }
    for(; j < std::min(c, n_col); j++) {
      if(j >= f_col && j < f_col + ln_col) {
        nbc++;
      }
    }
    this->max_col = std::max(this->max_col, nbc);
  }

  if(nv == 0)
    return;

  if (this->values)
    delete[] this->values;
  if (this->columns)
    delete[] this->columns;
  this->values = new double[this->ln_row * this->max_col]();
  this->columns = new int[this->ln_row * this->max_col]();

  incr = incr_save;
  long int lincr;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    lincr = 0;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->columns[(i - f_row) * this->max_col + lincr] = jj;
        this->values[(i - f_row) * this->max_col + lincr] = std::get<2>(tuple);
        lincr++;
      }
      incr++;
    }
    for(long int j = lincr; j < max_col; j++) {
      this->columns[(i - f_row) * this->max_col + j] = 0;
      this->values[(i - f_row) * this->max_col + j] = 0;
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
        this->columns[(i - f_row) * this->max_col + lincr] = jj;
        this->values[(i - f_row) * this->max_col + lincr] = std::get<2>(tuple);
        lincr++;
      }
      incr++;
    }
    for(long int j = lincr; j < max_col; j++) {
      this->columns[(i - f_row) * this->max_col + j] = 0;
      this->values[(i - f_row) * this->max_col + j] = 0;
    }
  }
}

void tbsla::cpp::MatrixELL::NUMAinit() {
  double* newVal = new double[this->ln_row * this->max_col];
  int* newCol = new int[this->ln_row * this->max_col];

  //NUMA init
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->max_col; j++) {
      newCol[i * this->max_col + j] = this->columns[i * this->max_col + j];
      newVal[i * this->max_col + j] = this->values[i * this->max_col + j];
    }
  }

  delete[] this->values;
  delete[] this->columns;

  this->values = newVal;
  this->columns = newCol;
}
