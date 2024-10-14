#include <tbsla/cpp/Matrix.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
double* tbsla::cpp::Matrix::a_axpx_(const double* v, int vect_incr) const {
  double* r = this->spmv(v, vect_incr);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_col; i++) {
    r[i] += v[i];
  }
  double* r2 = this->spmv(r, vect_incr);
  delete[] r;
  return r2;
}

void tbsla::cpp::Matrix::AAxpAx(double* r, double* v, double* buffer, int vect_incr) const {
  this->Ax(buffer, v, vect_incr);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_col; i++) {
    buffer[i] += v[i];
  }
  this->Ax(r, buffer, vect_incr);
}

void tbsla::cpp::Matrix::AAxpAxpx(double* r, double* v, double* buffer, int vect_incr) const {
  this->AAxpAx(r, v, buffer, vect_incr);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < this->ln_col; i++) {
    r[i] += v[i];
  }
}
