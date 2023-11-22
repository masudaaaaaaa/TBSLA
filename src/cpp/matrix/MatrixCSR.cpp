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
#include <vector>
#include <omp.h>

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
  std::cout << "Done" << std::endl;
  std::cout << "incr = " << incr << std::endl;
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
  std::cout << "nnz = " << this->nnz << " ; incr = " << incr << " ; nv = " << nv << std::endl;

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
  std::cout << "Done\n";
}

/*void tbsla::cpp::MatrixCSR::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  std::cout << "CSR-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;
  std::cout << "nnz ratio = " << nnz_ratio << std::endl;
  
  int exp_nnz_per_row = (int)(ln_col*nnz_ratio);
  int stdev = (int)(sqrt(sqrt(exp_nnz_per_row)));
  std::default_random_engine generator(seed_mult);
  std::normal_distribution<double> distribution(exp_nnz_per_row, stdev);
  //std::uniform_real_distribution<double> distr_ind(f_col, f_col+ln_col);
  std::uniform_real_distribution<double> distr_ind(0, ln_col-1);
  int* nnz_per_row = new int[ln_row];

  this->nnz = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    int nnz_in_row = (int)(distribution(generator));
    //std::cout << nnz_in_row << " ";
    if(nnz_in_row<0)
      nnz_in_row = 0;
    //if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
    this->nnz += nnz_in_row;
	nnz_per_row[i-f_row] = nnz_in_row;
  }
  std::cout << "nnz = " << this->nnz << std::endl;

  if(this->nnz == 0)
    return;

  this->values = new double[this->nnz];
  this->colidx = new int[this->nnz];
  this->rowptr = new int[ln_row + 1];

  size_t incr = 0;
  this->rowptr[0] = incr;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    //std::cout << "Generating " << nnz_per_row[i-f_row] << " values in range " << this->ln_col << std::endl;
    int* random_cols = tbsla::utils::values_generation::random_columns(nnz_per_row[i-f_row], this->ln_col, distr_ind, generator);
	for(int k=0; k<nnz_per_row[i-f_row]; k++) {
		// Should be no need to check ; values are generated within valid range
		this->colidx[incr] = random_cols[k]+f_col;
		this->values[incr] = 1;
		incr++;
	}
    this->rowptr[i - f_row + 1] = incr;
  }
  delete[] nnz_per_row;
  std::cout << "Filled random\n";
  std::cout << "Filled random ; columns :\n";
  for(int i=0; i<15; i++)
	  std::cout << this->colidx[i] << " ";
  std::cout << std::endl;
}*/

/*void tbsla::cpp::MatrixCSR::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  int c = std::max(1, (int)(n_col * nnz_ratio));

  int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  std::cout << "CSR-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;
  std::cout << "nnz ratio = " << nnz_ratio << std::endl;
  std::cout << "c = " << c << " ; min_ = " << min_ << std::endl;

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
  std::cout << "incr = " << incr << " ; nv = " << nv << std::endl;
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
  std::cout << "incr = " << incr << " ; nv = " << nv << std::endl;

  this->nnz = 0;
  long int incr_save = incr;

  long int i;
  for(i = f_row; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr, std::min(c, n_col), n_col, seed_mult);
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      ii = i;
      jj = random_cols[j];
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->nnz++;
      }
      incr++;
    }
  }
  std::cout << "nnz = " << this->nnz << " ; incr = " << incr << " ; nv = " << nv << std::endl;

  incr = incr_save;

  //if(nv == 0)
    //return;

  std::cout << "init...\n";
  this->values = new double[this->nnz];
  this->colidx = new int[this->nnz];
  this->rowptr = new int[ln_row + 1];
  std::cout << "...done\n";

  long int lincr = 0;
  this->rowptr[0] = lincr;
  std::cout << "one\n";
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr, std::min(c, n_col), n_col, seed_mult);
    //std::cout << (std::min(c, n_col)) << std::endl;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      double v;
      ii = i;
      jj = random_cols[j];
      v = 1;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->colidx[lincr] = jj;
        this->values[lincr] = v;
        lincr++;
	//if((i - f_row + 1)<10)
	  //std::cout << jj << " ";
      }
      incr++;
    }
    if((i - f_row + 1)<10)
      std::cout << (i - f_row + 1) << "=>" << lincr << "\n";
    this->rowptr[i - f_row + 1] = lincr;
  }
  for(int zz=0; zz<500; zz++)
    std::cout << this->rowptr[zz] << "  ";
  std::cout << std::endl;
  std::cout << "two\n";
  for(; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr, std::min(c, n_col), n_col, seed_mult);
    for(long int j = 0; j < std::min(c, n_col) - i + min_ - 1; j++) {
      int ii, jj;
      double v;
      ii = i;
      jj = random_cols[j];
      v = 1;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->colidx[lincr] = jj;
        this->values[lincr] = v;
        lincr++;
      }
      incr++;
    }
    if((i - f_row + 1)<10)
      std::cout << (i - f_row + 1) << "=>" << lincr << "\n";
    this->rowptr[i - f_row + 1] = lincr;
  }
  for(int zz=0; zz<500; zz++)
    std::cout << this->rowptr[zz] << "  ";
  std::cout << std::endl;
  std::cout << "three\n";
  for(; i < f_row + ln_row; i++) {
    if((i - f_row + 1)<10)
      std::cout << (i - f_row + 1) << "=>" << lincr << "\n";
    this->rowptr[i - f_row + 1] = lincr;
  }
  std::cout << "Done\n";
  for(int zz=0; zz<500; zz++)
    std::cout << this->rowptr[zz] << "  ";
  std::cout << std::endl;
  for(int zz=0; zz<500; zz++)
    std::cout << this->colidx[zz] << "  ";
  std::cout << std::endl;
  for(int zz=0; zz<500; zz++)
    std::cout << this->values[zz] << "  ";
  std::cout << std::endl;
}*/

/*void tbsla::cpp::MatrixCSR::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  int c = std::max(1, (int)(n_col * nnz_ratio));
  if (nnz_ratio > 1)
    c = (int)nnz_ratio;

  //int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  std::cout << "CSR-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;
  std::cout << "min = " << (std::min(n_row, f_row + ln_row)) << std::endl;
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
  //std::cout << "incr = " << incr << " ; nv = " << nv << std::endl;

  this->nnz = 0;
  long int incr_save = incr;
  int n_threads = 1;
  #pragma omp parallel
  {
    n_threads = omp_get_num_threads();
  }
  int* nnz_per_thread = new int[n_threads]();
  int* nnz_start_thread = new int[n_threads]();
  int* incr_per_thread = new int[n_threads]();

  int upper_bound = std::min(n_row, f_row + ln_row);
  #pragma omp parallel
  {
  int t_n = omp_get_thread_num();
  int nnz_part = 0;
  int chunk_size = std::floor((upper_bound-f_row) / n_threads);
  int start = f_row + (t_n * chunk_size);
  int end = std::min((start+chunk_size), upper_bound);
  if(t_n == n_threads-1)
    end = upper_bound;
  int incr_part = incr + (t_n * chunk_size * std::min(c, n_col));
  long int i;
  for(i = start; i < end; i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr_part, std::min(c, n_col), n_col, seed_mult);
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      ii = i;
      jj = random_cols[j];
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        nnz_part++;
      }
    }
    incr_part += std::min(c, n_col);
    delete[] random_cols;
  }
  nnz_per_thread[t_n] = nnz_part;
  incr_per_thread[t_n] = incr_part;
  }
  for(int k=0; k<n_threads; k++) {
    this->nnz += nnz_per_thread[k];
    incr += incr_per_thread[k];
  }
  nnz_start_thread[0] = 0;
  for(int k=1; k<n_threads; k++)
    nnz_start_thread[k] = nnz_per_thread[k] + nnz_start_thread[k-1];
  std::cout << "nnz = " << this->nnz << " ; incr = " << incr << std::endl;

  incr = incr_save;

  //if(nv == 0)
    //return;

  std::cout << "init..." << std::endl;
  this->values = new double[this->nnz];
  this->colidx = new int[this->nnz];
  this->rowptr = new int[ln_row + 1];
  std::cout << "...done" << std::endl;

  long int lincr = 0;
  this->rowptr[0] = lincr;

  std::cout << "one" << std::endl;
#pragma omp parallel
  {
  int t_n = omp_get_thread_num();
  int nnz_ind = nnz_start_thread[t_n];
  int chunk_size = std::floor((upper_bound-f_row) / n_threads);
  int start = f_row + (t_n * chunk_size);
  int end = std::min((start+chunk_size), upper_bound);
  if(t_n == n_threads-1)
    end = upper_bound;
  int incr_part = incr + (t_n * chunk_size * std::min(c, n_col));
  int lincr_part = 0;
  long int i;
  for(i = start; i < end; i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr_part, std::min(c, n_col), n_col, seed_mult);
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      double v;
      ii = i;
      jj = random_cols[j];
      v = 1;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->colidx[nnz_ind] = jj;
        this->values[nnz_ind] = v;
        lincr++;
      }
      //incr_part++;
    }
    incr_part += std::min(c, n_col);
    this->rowptr[i - f_row + 1] = lincr;
    delete[] random_cols;
  }
  std::cout << "finished thread " << t_n << std::endl;
  }
  std::cout << "Done" << std::endl;
  std::cout << "lincr = " << lincr << " ; incr = " << incr << std::endl;
}*/


void tbsla::cpp::MatrixCSR::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  int c = std::max(1, (int)(n_col * nnz_ratio));
  if (nnz_ratio > 1)
    c = (int)nnz_ratio;

  //int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  std::cout << "CSR-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;
  std::cout << "min = " << (std::min(n_row, f_row + ln_row)) << std::endl;
  std::cout << "nnz ratio = " << nnz_ratio << std::endl;
  std::cout << "c = " << c /*<< " ; min_ = " << min_*/ << std::endl;

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
  std::vector<std::vector<int> > row_ptrs_t(n_threads);
  std::vector<std::vector<int> > col_inds_t(n_threads);

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
  long int incr_part = incr + (t_n * chunk_size * std::min(c, n_col));
  int lincr_part = 0;
  std::vector<int> row_ptrs_part;
  std::vector<int> col_inds_part;
  row_ptrs_part.push_back(0);
  long int i;
  //std::cout << "thread " << t_n << " from " << start << " to " << end << std::endl;
  for(i = start; i < end; i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr_part, std::min(c, n_col), n_col, seed_mult);
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      ii = i;
      jj = random_cols[j];
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        col_inds_part.push_back(jj);
        lincr_part++;
      }
      incr_part++;
    }
    //if(lincr_part%10000==0)
      //std::cout << i << "|" << t_n << "|" << lincr_part << std::endl;
    row_ptrs_part.push_back(lincr_part);
    delete[] random_cols;
  }
  row_ptrs_t[t_n] = row_ptrs_part;
  col_inds_t[t_n] = col_inds_part;
  std::cout << "finished thread " << t_n << std::endl;
  }
  std::cout << "aggregating..." << std::endl;
  std::vector<int> row_ptrs_full;
  int cumul = 0;
  for(int it=0; it<n_threads; it++) {
    //std::cout << "row_ptrs_t[" << it << "] : # = " << row_ptrs_t[it].size() << std::endl;
    for(int itt=0; itt<row_ptrs_t[it].size()-1; itt++) {
      row_ptrs_full.push_back(row_ptrs_t[it][itt]+cumul);
    }
    //std::cout << "cumul = " << cumul << " + " << row_ptrs_t[it][row_ptrs_t[it].size()-1] << std::endl;
    cumul += row_ptrs_t[it][row_ptrs_t[it].size()-1];
    //std::cout << "=> cumul = " << cumul  << std::endl;
    row_ptrs_t[it].clear();
  }
  row_ptrs_full.push_back(cumul);
  row_ptrs_t.resize(0);

  std::cout << "row_ptrs_full.size() = " << row_ptrs_full.size() << std::endl;
  std::cout << "nnz = " << cumul << std::endl;

  this->nnz = cumul;

  std::cout << "init..." << std::endl;
  this->rowptr = new int[ln_row + 1]();
  //this->rowptr = &row_ptrs_full[0];
  for(int k=0; k<row_ptrs_full.size(); k++)
    this->rowptr[k] = row_ptrs_full[k];

  this->values = new double[this->nnz]();
  this->colidx = new int[this->nnz]();
  std::vector<int> offsets;
  offsets.push_back(0);
  for(int it=1; it<n_threads; it++) {
    int no = offsets[it-1] + col_inds_t[it-1].size();
    offsets.push_back(no);
    //std::cout << "offsets[" << it << "] = " << no << std::endl;
  }
  //std::cout << "last col_inds_t = " << col_inds_t[n_threads-1].size() << std::endl;
  std::cout << "=> total nnz = " << (col_inds_t[n_threads-1].size() + offsets[n_threads-1]) << std::endl;
  #pragma omp parallel for schedule(static)
  for(int it=0; it<n_threads; it++) {
    int of = offsets[it];
    //std::cout << it << " => adding " << col_inds_t[it].size() << " nnzs starting at position " << of << std::endl;
    for(int itt=0; itt<col_inds_t[it].size(); itt++) {
      this->colidx[of+itt] = col_inds_t[it][itt];
      this->values[of+itt] = 1;
    }
  }
  col_inds_t.resize(0);
  offsets.resize(0);

  std::cout << "Done" << std::endl;
  std::cout << " ; incr = " << incr << std::endl;
}


void tbsla::cpp::MatrixCSR::fill_brain(int n_row, int n_col, int* neuron_type, std::vector<std::vector<double> > proba_conn, std::vector<std::unordered_map<int,std::vector<int> > > brain_struct, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  std::cout << "CSR-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;

  // arbitrary value for RNG (experimental)
  int c = std::max(10, (int)(n_col / 100000));
  //int c = std::max(1, );

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
  std::vector<std::vector<int> > row_ptrs_t(n_threads);
  std::vector<std::vector<int> > col_inds_t(n_threads);

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
  long int incr_part = incr + (t_n * chunk_size * std::min(c, n_col));
  int lincr_part = 0;
  std::vector<int> row_ptrs_part;
  std::vector<int> col_inds_part;
  row_ptrs_part.push_back(0);
  long int i;
  //std::cout << "thread " << t_n << " from " << start << " to " << end << std::endl;
  for(i = start; i < end; i++) {
    std::vector<int> random_cols = tbsla::utils::values_generation::brain_columns(incr_part, proba_conn, brain_struct, neuron_type[i], seed_mult);
    for(long int j = 0; j < random_cols.size(); j++) {
      int ii, jj;
      ii = i;
      jj = random_cols[j];
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        col_inds_part.push_back(jj);
        lincr_part++;
      }
      incr_part++;
    }
    row_ptrs_part.push_back(lincr_part);
  }
  row_ptrs_t[t_n] = row_ptrs_part;
  col_inds_t[t_n] = col_inds_part;
  std::cout << "finished thread " << t_n << std::endl;
  }
  std::cout << "aggregating..." << std::endl;
  std::vector<int> row_ptrs_full;
  int cumul = 0;
  for(int it=0; it<n_threads; it++) {
    //std::cout << "row_ptrs_t[" << it << "] : # = " << row_ptrs_t[it].size() << std::endl;
    for(int itt=0; itt<row_ptrs_t[it].size()-1; itt++) {
      row_ptrs_full.push_back(row_ptrs_t[it][itt]+cumul);
    }
    //std::cout << "cumul = " << cumul << " + " << row_ptrs_t[it][row_ptrs_t[it].size()-1] << std::endl;
    cumul += row_ptrs_t[it][row_ptrs_t[it].size()-1];
    //std::cout << "=> cumul = " << cumul  << std::endl;
    row_ptrs_t[it].clear();
  }
  row_ptrs_full.push_back(cumul);
  row_ptrs_t.resize(0);

  std::cout << "row_ptrs_full.size() = " << row_ptrs_full.size() << std::endl;
  std::cout << "nnz = " << cumul << std::endl;

  this->nnz = cumul;

  std::cout << "init..." << std::endl;
  this->rowptr = new int[ln_row + 1]();
  //this->rowptr = &row_ptrs_full[0];
  for(int k=0; k<row_ptrs_full.size(); k++)
    this->rowptr[k] = row_ptrs_full[k];

  this->values = new double[this->nnz]();
  this->colidx = new int[this->nnz]();
  std::vector<int> offsets;
  offsets.push_back(0);
  for(int it=1; it<n_threads; it++) {
    int no = offsets[it-1] + col_inds_t[it-1].size();
    offsets.push_back(no);
    //std::cout << "offsets[" << it << "] = " << no << std::endl;
  }
  //std::cout << "last col_inds_t = " << col_inds_t[n_threads-1].size() << std::endl;
  std::cout << "=> total nnz = " << (col_inds_t[n_threads-1].size() + offsets[n_threads-1]) << std::endl;
  #pragma omp parallel for schedule(static)
  for(int it=0; it<n_threads; it++) {
    int of = offsets[it];
    //std::cout << it << " => adding " << col_inds_t[it].size() << " nnzs starting at position " << of << std::endl;
    for(int itt=0; itt<col_inds_t[it].size(); itt++) {
      this->colidx[of+itt] = col_inds_t[it][itt];
      this->values[of+itt] = 1;
    }
  }
  col_inds_t.resize(0);
  offsets.resize(0);

  std::cout << "Done" << std::endl;
  std::cout << " ; incr = " << incr << std::endl;
}


void tbsla::cpp::MatrixCSR::get_row_sums(double* s) {
  std::cout << "Computing row-sums on rows " << this->f_row << " to " << this->f_row+this->ln_row << std::endl;
  #pragma omp parallel for schedule(static)
  //for (int i = this->f_row; i < this->f_row+this->ln_row; i++) {
  for (int i = 0; i < this->ln_row; i++) {
	double sum = 0;
    for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
      sum += this->values[j];
    }
	s[i+this->f_row] = sum;
	//std::cout << "sum[" << i << "] = " << sum << std::endl;
  }
}

void tbsla::cpp::MatrixCSR::normalize_rows(double* s) {
  std::cout << "Normalizing on rows " << this->f_row << " to " << this->f_row+this->ln_row << std::endl;
  #pragma omp parallel for schedule(static)
  //for (int i = this->f_row; i < this->f_row+this->ln_row; i++) {
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
      this->values[j] /= s[i+this->f_row];
    }
  }
}

void tbsla::cpp::MatrixCSR::get_col_sums(double* s) {
  std::cout << "Computing col-sums on cols " << this->f_col << " to " << this->f_col+this->ln_col << std::endl;
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  else
    std::cout << "nnz = " << this->nnz << std::endl;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
      s[this->colidx[j] - this->f_col] += this->values[j];
	   //s[this->colidx[j]] += this->values[j];
    }
  }
  double tot = 0;
  for(int i=0; i<this->ln_col; i++) {
    tot += s[i];
  }
  std::cout << std::endl;
  std::cout << "total = " << tot << std::endl;
}

void tbsla::cpp::MatrixCSR::normalize_cols(double* s) {
  std::cout << "Normalizing on cols " << this->f_col << " to " << this->f_col+this->ln_col << std::endl;
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
      //if(s[this->colidx[j] - this->f_col]==0)
	//std::cout << "sum = 0 at " << (this->colidx[j] - this->f_col) << std::endl;
      //this->values[j] /= s[this->colidx[j]];
      double sval = s[this->colidx[j] - this->f_col];
      if(sval > 0)
        this->values[j] /= s[this->colidx[j] - this->f_col];
    }
  }
}

void tbsla::cpp::MatrixCSR::set_diag(double* s) {
  std::cout << "diagonalised on cols " << this->f_col << " to " << this->f_col+this->ln_col << std::endl;
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_row; i++) {
    for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
      
      double sval = s[this->colidx[j] - this->f_col];
      if(sval > 0 &&this->colidx[j]==(this->f_row+i))
        this->values[j] = s[this->colidx[j] - this->f_col];
    }
  }
}

void tbsla::cpp::MatrixCSR::NUMAinit() {
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  //std::cout << "numa init" << std::endl;
  //std::cout << this->nnz << " ; " << this->ln_row << std::endl;
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
