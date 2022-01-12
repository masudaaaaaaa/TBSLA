#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

static std::uint64_t now() {
  std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(ns.count());
}

class MatrixCSRVector {
  public:
    std::vector<double> values;
    std::vector<int> colidx;
    std::vector<int> rowptr;
    int n_row, n_col, f_row, f_col, ln_row, ln_col, pr, pc, NR, NC;
    long int nnz;


    MatrixCSRVector(std::string filepath) {
      std::ifstream is(filepath);
      if(!is.good()) {
        std::cerr << filepath << " cannot be open!" << std::endl;
      }
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
      this->values.resize(size);
      is.read(reinterpret_cast<char*>(this->values.data()), size * sizeof(double));

      is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
      this->rowptr.resize(size);
      is.read(reinterpret_cast<char*>(this->rowptr.data()), size * sizeof(int));

      is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
      this->colidx.resize(size);
      is.read(reinterpret_cast<char*>(this->colidx.data()), size * sizeof(int));
      is.close();
    }

    void spmv1(std::vector<double> &b, std::vector<double> &x){
      #pragma omp parallel for
      for (int i = 0; i < this->rowptr.size() - 1; i++) {
        double tmp = 0;
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
           tmp += this->values[j] * x[this->colidx[j]];
        }
        b[i] = tmp;
      }
    }

    std::vector<double> spmv2(std::vector<double> &x){
      std::vector<double> b(this->n_col, 0);
      #pragma omp parallel for
      for (int i = 0; i < this->rowptr.size() - 1; i++) {
        double tmp = 0;
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
           tmp += this->values[j] * x[this->colidx[j]];
        }
        b[i] = tmp;
      }
      return b;
    }

    std::vector<double> spmv3(std::vector<double> &x){
      std::vector<double> b(this->n_col);
      #pragma omp parallel for
      for (int i = 0; i < this->rowptr.size() - 1; i++) {
        double tmp = 0;
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
           tmp += this->values[j] * x[this->colidx[j]];
        }
        b[i] = tmp;
      }
      return b;
    }

    void spmv_sched_static(std::vector<double> &b, std::vector<double> &x){
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < this->rowptr.size() - 1; i++) {
        double tmp = 0;
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
           tmp += this->values[j] * x[this->colidx[j]];
        }
        b[i] = tmp;
      }
    }

    void spmv_sched_dynamic(std::vector<double> &b, std::vector<double> &x){
      #pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < this->rowptr.size() - 1; i++) {
        double tmp = 0;
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
           tmp += this->values[j] * x[this->colidx[j]];
        }
        b[i] = tmp;
      }
    }
};

class MatrixCSRCArray {
  public:
    double *values;
    int *colidx;
    int *rowptr;
    int n_row, n_col, f_row, f_col, ln_row, ln_col, pr, pc, NR, NC;
    long int nnz;


    MatrixCSRCArray(std::string filepath) {
      std::ifstream is(filepath);
      if(!is.good()) {
        std::cerr << filepath << " cannot be open!" << std::endl;
      }
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
      this->values = new double[size];
      is.read(reinterpret_cast<char*>(this->values), size * sizeof(double));

      is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
      this->rowptr = new int[size];
      is.read(reinterpret_cast<char*>(this->rowptr), size * sizeof(int));

      is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
      this->colidx = new int[size];
      is.read(reinterpret_cast<char*>(this->colidx), size * sizeof(int));
      is.close();
    }

    ~MatrixCSRCArray() {
      delete [] values;
      delete [] rowptr;
      delete [] colidx;
    }

    void NUMAinit() {
      double* newVal = new double[this->nnz];
      int* newCol = new int[this->nnz];
      int* newRowPtr = new int[this->n_row + 1];
    
      //NUMA init
    #pragma omp parallel for schedule(static)
      for(int row = 0; row < this->n_row + 1; ++row)
      {
        newRowPtr[row] = this->rowptr[row];
      }
    #pragma omp parallel for schedule(static)
      for(int row = 0; row < this->n_row; ++row)
      {
        for(int idx = newRowPtr[row]; idx < newRowPtr[row + 1]; ++idx)
        {
          newCol[idx] = this->colidx[idx];
          newVal[idx] = this->values[idx];
        }
      }

      delete[] values;
      delete[] rowptr;
      delete[] colidx;

      values = newVal;
      rowptr = newRowPtr;
      colidx = newCol;
    }

    void spmv1(double *b, double *x){
      #pragma omp parallel for
      for (int i = 0; i < this->n_row - 1; i++) {
        double tmp = 0;
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
           tmp += this->values[j] * x[this->colidx[j]];
        }
        b[i] = tmp;
      }
    }

};

class MatrixCSRCMalloc {
  public:
    double *values;
    int *colidx;
    int *rowptr;
    int n_row, n_col, f_row, f_col, ln_row, ln_col, pr, pc, NR, NC;
    long int nnz;


    MatrixCSRCMalloc(std::string filepath) {
      std::ifstream is(filepath);
      if(!is.good()) {
        std::cerr << filepath << " cannot be open!" << std::endl;
      }
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
      this->values = (double *) malloc(size * sizeof(double));
      is.read(reinterpret_cast<char*>(this->values), size * sizeof(double));

      is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
      this->rowptr = (int *) malloc(size * sizeof(int));
      is.read(reinterpret_cast<char*>(this->rowptr), size * sizeof(int));

      is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
      this->colidx = (int *) malloc(size * sizeof(int));
      is.read(reinterpret_cast<char*>(this->colidx), size * sizeof(int));
      is.close();
    }

    ~MatrixCSRCMalloc() {
      std::cout << "call ~MatrixCSRCMalloc" << std::endl;
      free(values);
      free(rowptr);
      free(colidx);
    }

    void spmv1(double *b, double *x){
      #pragma omp parallel for
      for (int i = 0; i < this->n_row - 1; i++) {
        double tmp = 0;
        for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
           tmp += this->values[j] * x[this->colidx[j]];
        }
        b[i] = tmp;
      }
    }

};



inline void spmv_array_no_class(double *b, double *x, int *rowptr, int *colidx, double *values, int s){
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < s - 1; i++) {
    double tmp = 0;
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
       tmp += values[j] * x[colidx[j]];
    }
    b[i] = tmp;
  }
}

#ifdef __ARM_FEATURE_SVE
inline void spmv_array_no_class_sve1(double *b, double *x, int *rowptr, int *colidx, double *values, int s){
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < s - 1; i++) {
    double tmp = 0;
    int start = rowptr[i];
    int end = rowptr[i + 1];
    int j = start;
    svbool_t pg = svwhilelt_b64(j, end);
    do {
      svfloat64_t values_vec = svld1(pg, &(values[j]));
      svuint64_t col = svld1sw_u64(pg, &(colidx[j]));
      svfloat64_t v_vec = svld1_gather_index(pg, x, col);
      tmp += svaddv(pg, svmul_x(pg, values_vec, v_vec));
      j += svcntd();
      pg = svwhilelt_b64(j, end);
    } while (svptest_any(svptrue_b64(), pg));
    b[i] = tmp;
  }
}

inline void spmv_array_no_class_sve2(double *b, double *x, int *rowptr, int *colidx, double *values, int s){
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < s - 1; i++) {
    double tmp = 0;
    int start = rowptr[i];
    int end = rowptr[i + 1];
    int j = start;
    svbool_t pg = svwhilelt_b64(j, end);
    do {
      svfloat64_t values_vec = svld1(pg, values + j);
      svuint64_t col = svld1sw_u64(pg, colidx + j);
      svfloat64_t v_vec = svld1_gather_index(pg, x, col);
      tmp += svaddv(pg, svmul_x(pg, values_vec, v_vec));
      j += svcntd();
      pg = svwhilelt_b64(j, end);
    } while (svptest_any(svptrue_b64(), pg));
    b[i] = tmp;
  }
}

inline void spmv_array_no_class_sve3(double *b, double *x, int *rowptr, int *colidx, double *values, int s){
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < s - 1; i++) {
    svfloat64_t tmp; tmp = svadd_z(svpfalse(), tmp, tmp);
    int start = rowptr[i];
    int end = rowptr[i + 1];
    int j = start;
    svbool_t pg = svwhilelt_b64(j, end);
    do {
      svfloat64_t values_vec = svld1(pg, &(values[j]));
      svuint64_t col = svld1sw_u64(pg, &(colidx[j]));
      svfloat64_t v_vec = svld1_gather_index(pg, x, col);
      tmp = svmla_m(pg, tmp, values_vec, v_vec);
      j += svcntd();
      pg = svwhilelt_b64(j, end);
    } while (svptest_any(svptrue_b64(), pg));
    b[i] = svaddv(pg, tmp);
  }
}
#endif

void spmv1(std::vector<double> &b, std::vector<double> &x, std::vector<double> &values, std::vector<int> &colidx, std::vector<int> &rowptr){
  #pragma omp parallel for
  for (int i = 0; i < rowptr.size() - 1; i++) {
    double tmp = 0;
    for (int j = rowptr[i]; j < rowptr[i + 1]; j++) {
       tmp += values[j] * x[colidx[j]];
    }
    b[i] = tmp;
  }
}

int main(int argc, char** argv) {
  if(argc != 2) {
    std::cerr << "Error : Need the file containing a CSR matrix as only input !" << std::endl;
    exit(1);
  }
  std::string in(argv[1]);
  MatrixCSRVector mvec(in);
  MatrixCSRCArray mc(in);
  MatrixCSRCMalloc mm(in);
  std::vector<double> x(mc.n_col), b(mc.n_col);
  for(int i = 0; i < mc.n_col; i++) {
    x[i] = i/mc.n_col;
  }
  double *x2 = new double[mc.n_col];
  double *b2 = new double[mc.n_col];

  int ITERATIONS = 100;
  double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0;
  for(int it = 0; it < ITERATIONS; it++) {
    auto start = now();
    mvec.spmv1(b, x);
    auto end = now();
    t1 += (end - start) / 1e9;

    start = now();
    std::vector<double> r2 = mvec.spmv2(x);
    end = now();
    t2 += (end - start) / 1e9;

    start = now();
    std::vector<double> r3 = mvec.spmv3(x);
    end = now();
    t3 += (end - start) / 1e9;

    start = now();
    mvec.spmv_sched_static(b, x);
    end = now();
    t4 += (end - start) / 1e9;

    start = now();
    mvec.spmv_sched_dynamic(b, x);
    end = now();
    t5 += (end - start) / 1e9;

    start = now();
    spmv1(b, x, mvec.values, mvec.colidx, mvec.rowptr);
    end = now();
    t6 += (end - start) / 1e9;

    start = now();
    mc.spmv1(b2, x2);
    end = now();
    t7 += (end - start) / 1e9;

    start = now();
    spmv_array_no_class(b2, x2, mc.rowptr, mc.colidx, mc.values, mc.n_row);
    end = now();
    t8 += (end - start) / 1e9;
  }

  mc.NUMAinit();

  for(int it = 0; it < ITERATIONS; it++) {
    auto start = now();
    spmv_array_no_class(b2, x2, mc.rowptr, mc.colidx, mc.values, mc.n_row);
    auto end = now();
    t9 += (end - start) / 1e9;

    start = now();
    spmv_array_no_class(b2, x2, mm.rowptr, mm.colidx, mm.values, mm.n_row);
    end = now();
    t10 += (end - start) / 1e9;

#ifdef __ARM_FEATURE_SVE
    start = now();
    spmv_array_no_class_sve1(b2, x2, mc.rowptr, mc.colidx, mc.values, mc.n_row);
    end = now();
    t11 += (end - start) / 1e9;

    start = now();
    spmv_array_no_class_sve2(b2, x2, mc.rowptr, mc.colidx, mc.values, mc.n_row);
    end = now();
    t12 += (end - start) / 1e9;

    start = now();
    spmv_array_no_class_sve3(b2, x2, mc.rowptr, mc.colidx, mc.values, mc.n_row);
    end = now();
    t13 += (end - start) / 1e9;
#endif
  }

  std::cout << "spmv vec class                    --> time (s) : " << t1 / ITERATIONS << std::endl;
  std::cout << "spmv vec class                    --> GFlops   : " << 2 * mvec.nnz / t1 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv2 vec class                   --> time (s) : " << t2 / ITERATIONS << std::endl;
  std::cout << "spmv2 vec class                   --> GFlops   : " << 2 * mvec.nnz / t2 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv3 vec class                   --> time (s) : " << t3 / ITERATIONS << std::endl;
  std::cout << "spmv3 vec class                   --> GFlops   : " << 2 * mvec.nnz / t3 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv_sched_static vec class       --> time (s) : " << t4 / ITERATIONS << std::endl;
  std::cout << "spmv_sched_static vec class       --> GFlops   : " << 2 * mvec.nnz / t4 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv_sched_dynamic vec class      --> time (s) : " << t5 / ITERATIONS << std::endl;
  std::cout << "spmv_sched_dynamic vec class      --> GFlops   : " << 2 * mvec.nnz / t5 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv vec no class                 --> time (s) : " << t6 / ITERATIONS << std::endl;
  std::cout << "spmv vec no class                 --> GFlops   : " << 2 * mvec.nnz / t6 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv array class                  --> time (s) : " << t7 / ITERATIONS << std::endl;
  std::cout << "spmv array class                  --> GFlops   : " << 2 * mvec.nnz / t7 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv array no class               --> time (s) : " << t8 / ITERATIONS << std::endl;
  std::cout << "spmv array no class               --> GFlops   : " << 2 * mvec.nnz / t8 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv array no class NUMAinit      --> time (s) : " << t9 / ITERATIONS << std::endl;
  std::cout << "spmv array no class NUMAinit      --> GFlops   : " << 2 * mvec.nnz / t9 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv alloc no class               --> time (s) : " << t10 / ITERATIONS << std::endl;
  std::cout << "spmv alloc no class               --> GFlops   : " << 2 * mvec.nnz / t10 * ITERATIONS / 1e9 << std::endl;
#ifdef __ARM_FEATURE_SVE
  std::cout << "spmv array no class NUMAinit SVE1 --> time (s) : " << t11 / ITERATIONS << std::endl;
  std::cout << "spmv array no class NUMAinit SVE1 --> GFlops   : " << 2 * mvec.nnz / t11 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv array no class NUMAinit SVE2 --> time (s) : " << t12 / ITERATIONS << std::endl;
  std::cout << "spmv array no class NUMAinit SVE2 --> GFlops   : " << 2 * mvec.nnz / t12 * ITERATIONS / 1e9 << std::endl;
  std::cout << "spmv array no class NUMAinit SVE3 --> time (s) : " << t13 / ITERATIONS << std::endl;
  std::cout << "spmv array no class NUMAinit SVE3 --> GFlops   : " << 2 * mvec.nnz / t13 * ITERATIONS / 1e9 << std::endl;
#endif

  return 0;
}
