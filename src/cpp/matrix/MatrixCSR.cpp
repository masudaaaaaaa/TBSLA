#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/array.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/csr.hpp>
#include <numeric>
#include <iostream>
#include <string>
#include <algorithm>

#if TBSLA_ENABLE_VECTO
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#endif

tbsla::cpp::MatrixCSR::MatrixCSR(int n_row, int n_col, double* values, int* rowptr,  int* colidx) : values(NULL), rowptr(NULL), colidx(NULL) {
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
  if (this->rowptr)
    delete[] this->rowptr;
  if (this->colidx)
    delete[] this->colidx;
}

tbsla::cpp::MatrixCSR::~MatrixCSR() {
  if (this->values)
    delete[] this->values;
  if (this->rowptr)
    delete[] this->rowptr;
  if (this->colidx)
    delete[] this->colidx;
}

tbsla::cpp::MatrixCSR::MatrixCSR(const tbsla::cpp::MatrixCOO & m) : values(NULL), rowptr(NULL), colidx(NULL) {
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

  int* p = new int[this->nnz];
  std::iota(p, p + this->nnz, 0);
  std::sort(p, p + this->nnz, [&](unsigned i, unsigned j){ return tbsla::cpp::utils::csr::compare_row(m.get_row(), m.get_col(), i, j); });

  int* pr = tbsla::cpp::utils::csr::applyPermutation<int>(p, m.get_row(), this->nnz);
  this->colidx = tbsla::cpp::utils::csr::applyPermutation<int>(p, m.get_col(), this->nnz);
  this->values = tbsla::cpp::utils::csr::applyPermutation<double>(p, m.get_values(), this->nnz);

  if(this->nnz == 0) {
    this->rowptr = new int[this->n_row + 1]();
  } else {
    this->rowptr = new int[this->n_row + 1]();
    this->rowptr[0] = 0;
    size_t incr = 1;
    for(int i = 0; i < pr[0]; i++) {
      incr++;
      this->rowptr[incr] = 0;
    }
    for(int i = 0; i < this->nnz - 1; i++) {
      this->rowptr[incr]++;
      if(pr[i] != pr[i + 1]) {
        for(int j = 0; j < pr[i + 1] - pr[i]; j++) {
          incr++;
          this->rowptr[incr] = this->rowptr[incr - 1];
        }
      }
    }
    this->rowptr[incr]++;
    for(int i = incr; i < this->n_row; i++) {
      this->rowptr[i + 1] = this->rowptr[i];
    }
  }
  delete[] p;
  delete[] pr;
}


std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixCSR &m) {
  return m.print(os);
}

std::ostream& tbsla::cpp::MatrixCSR::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "------ CSR ------" << std::endl;
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
  tbsla::utils::array::stream<double>(os, "values", this->values, this->nnz);
  os << std::endl;
  tbsla::utils::array::stream<int>(os, "rowptr", this->rowptr, this->ln_row + 1);
  os << std::endl;
  tbsla::utils::array::stream<int>(os, "colidx", this->colidx, this->nnz);
  os << std::endl;
  os << "-----------------" << std::endl << std::flush;
  return os;
}

std::ostream& tbsla::cpp::MatrixCSR::print_as_dense(std::ostream& os) {
  double* d = new double[this->ln_row * this->ln_col];
  if(this->nnz != 0) {
    for(int i = 0; i < this->ln_row * this->ln_col; i++) {
      d[i] = 0;
    }
    for (int i = 0; i < this->ln_row; i++) {
      for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
         d[i * this->n_col + this->colidx[j]] += this->values[j];
      }
    }
  }
  tbsla::utils::array::print_dense_matrix(this->n_row, this->n_col, d, os);
  return os;
}

double* tbsla::cpp::MatrixCSR::spmv(const double* v, int vect_incr) const {
  double* r = new double[this->ln_row]();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < ln_row; i++) {
    r[i] = 0;
  }
  this->Ax(r, v, vect_incr);
  return r;
}

#if TBSLA_ENABLE_VECTO
#ifdef __ARM_FEATURE_SVE
std::string tbsla::cpp::MatrixCSR::get_vectorization() const {
  return "ARM_SVE";
}

inline void tbsla::cpp::MatrixCSR::Ax(double* r, const double* v, int vect_incr) const {
  if (this->nnz == 0)
    return;
  if (this->f_col == 0) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < this->ln_row; i++) {
      svfloat64_t tmp; tmp = svadd_z(svpfalse(), tmp, tmp);
      int start = this->rowptr[i];
      int end = this->rowptr[i + 1];
      int j = start;
      svbool_t pg = svwhilelt_b64(j, end);
      do {
        svfloat64_t values_vec = svld1(pg, &(this->values[j]));
        svuint64_t col = svld1sw_u64(pg, &(this->colidx[j]));
        svfloat64_t v_vec = svld1_gather_index(pg, v, col);
        tmp = svmla_m(pg, tmp, values_vec, v_vec);
        j += svcntd();
        pg = svwhilelt_b64(j, end);
      } while (svptest_any(svptrue_b64(), pg));
      r[i] = svaddv(svptrue_b64(), tmp);
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < this->ln_row; i++) {
      svfloat64_t tmp; tmp = svadd_z(svpfalse(), tmp, tmp);
      int start = this->rowptr[i];
      int end = this->rowptr[i + 1];
      int j = start;
      uint64_t fix = this->f_col;
      svbool_t pg = svwhilelt_b64(j, end);
      do {
        svfloat64_t values_vec = svld1(pg, &(this->values[j]));
        svuint64_t col = svld1sw_u64(pg, &(this->colidx[j]));
        svuint64_t col_fix = svsub_z(pg, col, fix);
        svfloat64_t v_vec = svld1_gather_index(pg, v, col_fix);
        tmp = svmla_m(pg, tmp, values_vec, v_vec);
        j += svcntd();
        pg = svwhilelt_b64(j, end);
      } while (svptest_any(svptrue_b64(), pg));
      r[i] = svaddv(svptrue_b64(), tmp);
    }
  }
}
#endif // __ARM_FEATURE_SVE
#else
std::string tbsla::cpp::MatrixCSR::get_vectorization() const {
  return "None";
}

inline void tbsla::cpp::MatrixCSR::Ax(double* r, const double* v, int vect_incr) const {
  if (this->nnz == 0)
    return;
  if (this->f_col == 0) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < this->ln_row; i++) {
      double tmp = 0;
      // front ?
      for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
         tmp += this->values[j] * v[this->colidx[j]];
      }
      r[i] = tmp;
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < this->ln_row; i++) {
      double tmp = 0;
      // front ?
      for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
         tmp += this->values[j] * v[this->colidx[j] - this->f_col];
      }
      r[i] = tmp;
    }
  }
}
#endif // TBSLA_ENABLE_VECTO

std::ostream & tbsla::cpp::MatrixCSR::write(std::ostream &os) {
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

  size_t size_r = this->ln_row + 1;
  os.write(reinterpret_cast<char*>(&size_r), sizeof(size_r));
  os.write(reinterpret_cast<char*>(this->rowptr), (this->ln_row + 1) * sizeof(int));

  size_t size_c = this->nnz;
  os.write(reinterpret_cast<char*>(&size_c), sizeof(size_c));
  os.write(reinterpret_cast<char*>(this->colidx), this->nnz * sizeof(int));
  return os;
}

std::istream & tbsla::cpp::MatrixCSR::read(std::istream &is, std::size_t pos, std::size_t n) {
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
  if (this->rowptr)
    delete[] this->rowptr;
  this->rowptr = new int[size];
  is.read(reinterpret_cast<char*>(this->rowptr), size * sizeof(int));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  if (this->colidx)
    delete[] this->colidx;
  this->colidx = new int[size];
  is.read(reinterpret_cast<char*>(this->colidx), size * sizeof(int));
  return is;
}

void tbsla::cpp::MatrixCSR::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  if (this->values) {
    delete[] this->values;
    this->values = NULL;
  }
  if (this->rowptr) {
    delete[] this->rowptr;
    this->rowptr = NULL;
  }
  if (this->colidx) {
    delete[] this->colidx;
    this->colidx = NULL;
  }

  ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  f_col = tbsla::utils::range::pflv(n_col, pc, NC);

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

  this->values = new double[this->nnz];
  this->colidx = new int[this->nnz];
  this->rowptr = new int[ln_row + 1];

  size_t incr = 0;
  this->rowptr[0] = incr;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    long int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      this->colidx[incr] = jj;
      this->values[incr] = 1;
      incr++;
    }
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->colidx[incr] = jj;
        this->values[incr] = 1;
        incr++;
      }
    }
    this->rowptr[i - f_row + 1] = incr;
  }
}

void tbsla::cpp::MatrixCSR::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  if (this->values)
    delete[] this->values;
  if (this->rowptr)
    delete[] this->rowptr;
  if (this->colidx)
    delete[] this->colidx;

  ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  f_col = tbsla::utils::range::pflv(n_col, pc, NC);

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
    for(long int j = 0; j < std::min(c, n_col); j++) {
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

  this->values = new double[this->nnz];
  this->colidx = new int[this->nnz];
  this->rowptr = new int[ln_row + 1];

  long int lincr = 0;
  this->rowptr[0] = lincr;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    for(long int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      double v;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      v = std::get<2>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->colidx[lincr] = jj;
        this->values[lincr] = v;
        lincr++;
      }
      incr++;
    }
    this->rowptr[i - f_row + 1] = lincr;
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
        this->colidx[lincr] = jj;
        this->values[lincr] = v;
        lincr++;
      }
      incr++;
    }
    this->rowptr[i - f_row + 1] = lincr;
  }
  for(; i < f_row + ln_row; i++) {
    this->rowptr[i - f_row + 1] = lincr;
  }
}

void tbsla::cpp::MatrixCSR::NUMAinit() {
  double* newVal = new double[this->nnz];
  int* newCol = new int[this->nnz];
  int* newRowPtr = new int[this->ln_row + 1];

  //NUMA init
#pragma omp parallel for schedule(static)
  for(int row = 0; row < this->ln_row + 1; ++row)
  {
    newRowPtr[row] = this->rowptr[row];
  }
#pragma omp parallel for schedule(static)
  for(int row = 0; row < this->ln_row; ++row)
  {
    for(int idx = newRowPtr[row]; idx < newRowPtr[row + 1]; ++idx)
    {
      newCol[idx] = this->colidx[idx];
      newVal[idx] = this->values[idx];
    }
  }

  delete[] this->values;
  delete[] this->rowptr;
  delete[] this->colidx;

  this->values = newVal;
  this->rowptr = newRowPtr;
  this->colidx = newCol;
}
