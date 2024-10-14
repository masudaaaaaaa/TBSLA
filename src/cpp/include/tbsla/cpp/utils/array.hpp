#ifndef TBSLA_CPP_ARRAY
#define TBSLA_CPP_ARRAY

#include <iostream>

namespace tbsla { namespace utils { namespace array {

template <class myType>
void stream(std::ostream &os, const std::string name, myType* v, int size) {
  os << name << " : ";
  if (size > 0) {
    for (int i = 0; i < size - 1; i++) {
      os << v[i] << ", ";
    }
    os << v[size - 1];
  }
}

int test_spmv_cdiag(int nr, int nc, int c, double* v, double* r, bool debug);
int test_a_axpx__cdiag(int nr, int nc, int c, double* v, double* r, bool debug);
void print_dense_matrix(int nr, int nc, const double* m, std::ostream& os);
int compare_arrays(double* v1, double* v2, int size);
int check(int i, double v, double exp, int return_value, bool debug);

}}}

#endif
