#ifndef TBSLA_CPP_UTILS
#define TBSLA_CPP_UTILS

#include <vector>
#include <iostream>

namespace tbsla { namespace utils { namespace vector {

template <class myType>
void streamvector(std::ostream &os, const std::string name, const std::vector<myType> & v) {
  os << name << " : ";
  auto it = v.cbegin();
  if (it != v.cend() && v.size() != 0) {
    for (; it != v.cend() - 1; it++) {
      os << *it << ", ";
    }
    os << *it;
  }
}

int test_spmv_cdiag(int nr, int nc, int c, std::vector<double> & v, std::vector<double> & r, bool debug);
int test_a_axpx__cdiag(int nr, int nc, int c, std::vector<double> & v, std::vector<double> & r, bool debug);
void print_dense_matrix(int nr, int nc, const std::vector<double>& m, std::ostream& os);
int compare_vectors(std::vector<double> v1, std::vector<double> v2);

}}}

#endif
