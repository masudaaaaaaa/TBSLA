#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/utils/array.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/csr.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <string>
#include <omp.h>

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

/*void tbsla::cpp::MatrixELL::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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
  if (this->columns) {
    delete[] this->columns;
    this->columns = NULL;
  }

  ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  f_col = tbsla::utils::range::pflv(n_col, pc, NC);
  
  int exp_nnz_per_row = (int)(ln_col*nnz_ratio);
  int stdev = (int)(sqrt(sqrt(exp_nnz_per_row)));
  std::default_random_engine generator(seed_mult);
  std::normal_distribution<double> distribution(exp_nnz_per_row, stdev);
  //std::uniform_real_distribution<double> distr_ind(f_col, f_col+ln_col);
  // not actually used anymore ; rand() is simply used to generate column indexes
  std::uniform_real_distribution<double> distr_ind(0, ln_col-1);
  int* nnz_per_row = new int[ln_row];

  this->nnz = 0;
  this->max_col = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    int nnz_in_row = (int)(distribution(generator));
    //if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
    this->nnz += nnz_in_row;
	nnz_per_row[i-f_row] = nnz_in_row;
	this->max_col = std::max(this->max_col, nnz_in_row);
  }

  if(this->nnz == 0)
    return;

  this->values = new double[this->ln_row * this->max_col]();
  this->columns = new int[this->ln_row * this->max_col]();

  size_t incr = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
	//std::cout << "Generating " << nnz_per_row[i-f_row] << " values in range " << this->ln_col << std::endl;
    int* random_cols = tbsla::utils::values_generation::random_columns(nnz_per_row[i-f_row], this->ln_col, distr_ind, generator);
	for(int k=0; k<nnz_per_row[i-f_row]; k++) {
		this->columns[incr] = random_cols[k]+f_col;
		this->values[incr] = 1;
		incr++;
	}
	for(int k=nnz_per_row[i-f_row]; k<this->max_col; k++) {
		this->columns[incr] = 0;
		this->values[incr] = 0;
		incr++;
	}
  }
}*/

/*void tbsla::cpp::MatrixELL::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  int c = std::max(1, (int)(n_col * nnz_ratio));
  if (nnz_ratio > 1)
    c = (int)nnz_ratio;

  //int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  std::cout << "ELL-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;
  std::cout << "nnz ratio = " << nnz_ratio << std::endl;
  std::cout << "c = " << c << std::endl;

  long int incr = 0, nv = 0;
  for(long int i = 0; i < n_row; i++) {
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

  this->nnz = 0;
  this->max_col = 0;
  long int incr_save = incr;

  long int i;
  for(i = f_row; i < std::min(n_row, f_row + ln_row); i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr, std::min(c, n_col), n_col, seed_mult);
    int nbc = 0;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      ii = i;
      jj = random_cols[j];
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->nnz++;
        nbc++;
      }
      incr++;
    }
    delete[] random_cols;
    this->max_col = std::max(this->max_col, nbc);
  }
  std::cout << "max_col = " << this->max_col << std::endl;

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
  for(i = f_row; i < std::min(n_row, f_row + ln_row); i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr, std::min(c, n_col), n_col, seed_mult);
    lincr = 0;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      ii = i;
      jj = random_cols[j];
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->columns[(i - f_row) * this->max_col + lincr] = jj;
        this->values[(i - f_row) * this->max_col + lincr] = 1;
        lincr++;
      }
      incr++;
    }
    delete[] random_cols;
    for(long int j = lincr; j < max_col; j++) {
      this->columns[(i - f_row) * this->max_col + j] = 0;
      this->values[(i - f_row) * this->max_col + j] = 0;
    }
  }
  
}*/

void tbsla::cpp::MatrixELL::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  int c = std::max(1, (int)(n_col * nnz_ratio));
  if (nnz_ratio > 1)
    c = (int)nnz_ratio;

  //int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  std::cout << "ELL-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;
  std::cout << "nnz ratio = " << nnz_ratio << std::endl;
  std::cout << "c = " << c << std::endl;

  long int incr = 0;

  for(long int i = 0; i < n_row; i++) {
    if(i < f_row) {
      incr += std::min(c, n_col);
    }
    else {
      break;
    }
  }
  std::cout << "incr = " << incr << std::endl;

  this->nnz = 0;
  long int incr_save = incr;
  int n_threads = 1;
  #pragma omp parallel
  {
    n_threads = omp_get_num_threads();
  }
  std::vector<std::vector<std::vector<int> > > col_inds_t(n_threads);
  std::vector<int> max_col_t(n_threads);
  std::vector<int> nnz_t(n_threads);

  int upper_bound = std::min(n_row, f_row + ln_row);

  std::cout << "one" << std::endl;
#pragma omp parallel
  {
  int t_n = omp_get_thread_num();
  int chunk_size = std::floor((upper_bound-f_row) / n_threads);
  int start = f_row + (t_n * chunk_size);
  int end = std::min((start+chunk_size), upper_bound);
  if(t_n == n_threads-1)
    end = upper_bound;
  int incr_part = incr + (t_n * chunk_size * std::min(c, n_col));
  std::vector<std::vector<int> > col_inds_part;
  int max_col_part = 0;
  int nnz_part = 0;
  long int i;
  for(i = start; i < end; i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr_part, std::min(c, n_col), n_col, seed_mult);
    std::vector<int> col_inds_inner;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      double v;
      ii = i;
      jj = random_cols[j];
      v = 1;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        col_inds_inner.push_back(jj);
        nnz_part++;
      }
      incr_part++;
    }
    col_inds_part.push_back(col_inds_inner);
    if(col_inds_inner.size() > max_col_part)
      max_col_part = col_inds_inner.size();
    delete[] random_cols;
  }
  col_inds_t[t_n] = col_inds_part;
  max_col_t[t_n] = max_col_part;
  nnz_t[t_n] = nnz_part;
  std::cout << "finished thread " << t_n << std::endl;
  }
  this->max_col = 0;
  for(int it=0; it<n_threads; it++) {
    //this->nnz += col_inds_t[it].size();
    this->nnz += nnz_t[it];
    if(max_col_t[it] > this->max_col)
      this->max_col = max_col_t[it];
  }

  std::cout << "nnz = " << this->nnz << std::endl;
  std::cout << "max_col = " << this->max_col << std::endl;

  std::cout << "init..." << std::endl;
  if (this->values)
    delete[] this->values;
  if (this->columns)
    delete[] this->columns;
  int arr_size = this->ln_row * this->max_col;
  this->values = new double[arr_size]();
  this->columns = new int[arr_size]();
  std::cout << "=> array size = " << arr_size << std::endl;
  std::vector<int> updated_t(n_threads);
  #pragma omp parallel for schedule(static)
  for(int it=0; it<n_threads; it++) {
    int t_n = omp_get_thread_num();
    int chunk_size = std::floor((upper_bound-f_row) / n_threads);
    int start = (t_n * chunk_size) * this->max_col;
    int pos = 0;
    for(int itt=0; itt<col_inds_t[it].size(); itt++) {
      for(int ittt=0; ittt<col_inds_t[it][itt].size(); ittt++) {
        if((start+pos)>arr_size)
          std::cout << "oob : " << (start+pos) << std::endl;
        this->columns[start+pos] = col_inds_t[it][itt][ittt];
        this->values[start+pos] = 1;
        pos++;
      }
      for(int ittt=col_inds_t[it][itt].size(); ittt<this->max_col; ittt++) {
        if((start+pos)>arr_size)
          std::cout << "oob : " << (start+pos) << std::endl;
        this->columns[start+pos] = 0;
        this->values[start+pos] = 0;
        pos++;
      }
    }
    updated_t[it] = pos;
  }
  int updated_total = 0;
  for(int it=0; it<n_threads; it++) {
    //std::cout << "thread " << it << " : " << updated_t[it] << " nnz" << std::endl;
    updated_total += updated_t[it];
  }
  std::cout << "total nnz updated = " << updated_total;

  std::cout << "Done" << std::endl;
  std::cout << " ; incr = " << incr << std::endl;
  updated_t.resize(0);
  col_inds_t.resize(0);
}


void tbsla::cpp::MatrixELL::fill_brain(int n_row, int n_col, int* neuron_type, std::vector<std::vector<double> > proba_conn, std::vector<std::unordered_map<int,std::vector<int> > > brain_struct, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

}


void tbsla::cpp::MatrixELL::get_row_sums(double* s) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
	double sum = 0;
    for (int j = 0; j < this->max_col; j++) {
      sum += this->values[i * this->max_col + j];
    }
	s[i+this->f_row] = sum;
  }
}

void tbsla::cpp::MatrixELL::normalize_rows(double* s) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->max_col; j++) {
      this->values[i * this->max_col + j] /= s[i+this->f_row];
    }
  }
}

void tbsla::cpp::MatrixELL::get_col_sums(double* s) {
  std::cout << "Computing col-sums on cols " << this->f_col << " to " << this->f_col+this->ln_col << std::endl;
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->max_col; j++) {
      double val = this->values[i * this->max_col + j];
      if(val>0)
        s[this->columns[i * this->max_col + j] - this->f_col] += val;
    }
  }
}

void tbsla::cpp::MatrixELL::normalize_cols(double* s) {
  std::cout << "Normalizing on cols " << this->f_col << " to " << this->f_col+this->ln_col << std::endl;
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->max_col; j++) {
       //this->values[i * this->max_col + j] /= s[this->columns[i * this->max_col + j]];
      double val = this->values[i * this->max_col + j];
      double sval = s[this->columns[i * this->max_col + j] - this->f_col];
      if(val>0 && sval>0)
        this->values[i * this->max_col + j] /= sval;
    }
  }
}

void tbsla::cpp::MatrixELL::set_diag(double* s) {
  std::cout << "diagonalised on cols " << this->f_col << " to " << this->f_col+this->ln_col << std::endl;
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = 0; j < this->max_col; j++) {
       //this->values[i * this->max_col + j] /= s[this->columns[i * this->max_col + j]];
      double val = this->values[i * this->max_col + j];
      double sval = s[this->columns[i * this->max_col + j] - this->f_col];
      if(val>0 && sval>0 && this->columns[i * this->max_col + j]==i)
        this->values[i * this->max_col + j] = sval;
    }
  }
}

void tbsla::cpp::MatrixELL::NUMAinit() {
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
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
