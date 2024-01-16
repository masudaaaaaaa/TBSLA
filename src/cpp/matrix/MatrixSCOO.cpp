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
#include <omp.h>

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

/*void tbsla::cpp::MatrixSCOO::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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
  if (this->row) {
    delete[] this->row;
    this->row = NULL;
  }
  if (this->col) {
    delete[] this->col;
    this->col = NULL;
  }

  ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  f_col = tbsla::utils::range::pflv(n_col, pc, NC);
  
  int exp_nnz_per_row = (int)(ln_col*nnz_ratio);
  int stdev = (int)(sqrt(sqrt(exp_nnz_per_row)));
  std::default_random_engine generator(seed_mult);
  std::normal_distribution<double> distribution(exp_nnz_per_row, stdev);
  std::uniform_real_distribution<double> distr_ind(0, ln_col-1);
  int* nnz_per_row = new int[ln_row];

  this->nnz = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    int nnz_in_row = (int)(distribution(generator));
    this->nnz += nnz_in_row;
	nnz_per_row[i-f_row] = nnz_in_row;
  }

  if(this->nnz == 0)
    return;

  this->values = new double[this->nnz];
  this->col = new int[this->nnz];
  this->row = new int[this->nnz];

  size_t incr = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
	//std::cout << "Generating " << nnz_per_row[i-f_row] << " values in range " << this->ln_col << std::endl;
    int* random_cols = tbsla::utils::values_generation::random_columns(nnz_per_row[i-f_row], this->ln_col, distr_ind, generator);
	for(int k=0; k<nnz_per_row[i-f_row]; k++) {
		this->row[incr] = i;
		this->col[incr] = random_cols[k]+f_col;
		this->values[incr] = 1;
		incr++;
	}
  }
  
}*/

/*void tbsla::cpp::MatrixSCOO::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  std::cout << "SCOO-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;
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
  long int incr_save = incr;

  long int i;
  for(i = f_row; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr, std::min(c, n_col), n_col, seed_mult);
    long int j = 0;
    for(; j < std::min(c, n_col); j++) {
      int ii, jj;
      ii = i;
      jj = random_cols[j];
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->nnz++;
      }
      incr++;
    }
    delete[] random_cols;
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
  for(i = f_row; i < std::min(n_row, f_row + ln_row); i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr, std::min(c, n_col), n_col, seed_mult);
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      double v;
      ii = i;
      jj = random_cols[j];
      v = 1;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->row[idx] = ii;
        this->col[idx] = jj;
        this->values[idx] = v;
        idx++;
      }
      incr++;
    }
    delete[] random_cols;
  }
  this->nnz = idx;
}*/

void tbsla::cpp::MatrixSCOO::fill_random(int n_row, int n_col, double nnz_ratio, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
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

  if (this->values)
    delete[] this->values;
  if (this->row)
    delete[] this->row;
  if (this->col)
    delete[] this->col;


  std::cout << "SCOO-MPI : " << ln_row << " " << f_row << " " << ln_col << " " << f_col << std::endl;
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
  std::vector<std::vector<int> > row_inds_t(n_threads);
  std::vector<std::vector<int> > col_inds_t(n_threads);
  std::vector<int> offsets_t(n_threads);

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
  int lincr_part = 0;
  std::vector<int> row_inds_part;
  std::vector<int> col_inds_part;
  long int i;
  for(i = start; i < end; i++) {
    int* random_cols = tbsla::utils::values_generation::random_columns(incr_part, std::min(c, n_col), n_col, seed_mult);
    for(long int j = 0; j < std::min(c, n_col); j++) {
      int ii, jj;
      ii = i;
      jj = random_cols[j];
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        row_inds_part.push_back(ii);
        col_inds_part.push_back(jj);
        lincr_part++;
      }
      incr_part++;
    }
    delete[] random_cols;
  }
  row_inds_t[t_n] = row_inds_part;
  col_inds_t[t_n] = col_inds_part;
  std::cout << "finished thread " << t_n << std::endl;
  }
  int nv = 0;
  for(int it=0; it<n_threads; it++) {
    nv += row_inds_t[it].size();
  }
  std::cout << "nv = " << nv << std::endl;
  this->nnz = nv;
  offsets_t[0] = 0;
  for(int it=1; it<n_threads; it++)
    offsets_t[it] = offsets_t[it-1] + row_inds_t[it-1].size();

  this->values = new double[nv]();
  this->row = new int[nv]();
  this->col = new int[nv]();

  std::vector<int> updated_t(n_threads);
  #pragma omp parallel for schedule(static)
  for(int it=0; it<n_threads; it++) {
    int start = offsets_t[it];
    //std::cout << "start at " << start << std::endl;
    int pos = 0;
    for(int itt=0; itt<row_inds_t[it].size(); itt++) {
      this->row[start+pos] = row_inds_t[it][itt];
      this->col[start+pos] = col_inds_t[it][itt];
      this->values[start+pos] = 1;
      pos++;
    }
    //row_inds_t[it].resize(0);
    //col_inds_t[it].resize(0);
    updated_t[it] = pos;
  }
  row_inds_t.resize(0);
  col_inds_t.resize(0);
  offsets_t.resize(0);
  std::cout << "Done" << std::endl;
  int updated_tot = 0;
  for(int it=0; it<n_threads; it++)
    updated_tot += updated_t[it];
  std::cout << "updated n = " << updated_tot << std::endl;
  updated_t.resize(0);
}


void tbsla::cpp::MatrixSCOO::fill_brain(int n_row, int n_col, int* neuron_type, std::vector<std::vector<double> > proba_conn, std::vector<std::unordered_map<int,std::vector<int> > > brain_struct, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

}


// TODO : normalization for SCOO
void tbsla::cpp::MatrixSCOO::get_row_sums(double* s) {
  for(int k=0; k<this->nnz; k++) {
	s[this->row[k]] += this->values[k];
  }
}

void tbsla::cpp::MatrixSCOO::normalize_rows(double* s) {
  for(int k=0; k<this->nnz; k++) {
	this->values[k] /= s[this->row[k]];
  }
}

void tbsla::cpp::MatrixSCOO::get_col_sums(double* s) {
  std::cout << "Computing col-sums on cols " << this->f_col << " to " << this->f_col+this->ln_col << std::endl;
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  for(int k=0; k<this->nnz; k++) {
    //if((this->col[k] - this->f_col)<0 || (this->col[k] - this->f_col)>=this->ln_col)
      //std::cout << "oob : " << (this->col[k] - this->f_col) << std::endl;
	s[this->col[k] - this->f_col] += this->values[k];
  }
}

void tbsla::cpp::MatrixSCOO::normalize_cols(double* s) {
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  for(int k=0; k<this->nnz; k++) {
    //this->values[k] /= s[this->col[k]];
    double sval = s[this->col[k] - this->f_col];
    if(sval>0)
      this->values[k] /= sval;
  }
  std::cout << "normalized ; nnz = " << this->nnz << std::endl;
}

void tbsla::cpp::MatrixSCOO::set_diag(double* s) {
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
  #pragma omp parallel for schedule(static)
  for(int k=0; k<this->nnz; k++) {
    double sval = s[this->row[k] - this->f_row];
    if(sval>0 && this->col[k]==this->row[k])
      this->values[k] = sval;
  }
  std::cout << "diagonalised ; nnz = " << this->nnz << std::endl;
}

void tbsla::cpp::MatrixSCOO::NUMAinit() {
  if(this->nnz==0) {
    std::cout << "Nothing to do ; block matrix is empty" << std::endl;
    return;
  }
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

//TODO : fill_cdistribute SCOO

void tbsla::cpp::MatrixSCOO::fill_cdistrib(int n_row, int n_col, int nnz, int pr, int pc, int NR, int NC) {

}
